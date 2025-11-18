import os
import json
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from baselines.monkeypatch import set_model
from baselines.gemfilter.utils import gemfilter_generate_selection

context_length_list = [8192, 16384]

datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}


model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 7950,
    "llama-3": 7950,
    "mistral": 31500,
    "ministral": 127500,
    "llama-3.1": 127500
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(tokenizer, prompt):

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    
    return prompt


def main(model, args):
    

    print("Loading data...")
    
    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []
    
    input_max_len = 0
    model_path = args.model_path.lower()
    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]

    if args.method.lower() in ["h2o", "pyramidinfer"]:
        model_max_len = 7950
        print(f"[{model_path}] Model Max Length ignored due to H2O. Current Max Length is {model_max_len}")
    else:

        print(f"[{model_path}] Model Max Length is {model_max_len}")
    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file, "r", encoding="utf-8") as fp:
        for line in fp:

            example = json.loads(line)
            length = example["length"]
            if length > input_max_len: 
                input_max_len = length

            prompt = example["input"] #TODO tokenizer.apply_chat_template ?
           
            prompt = build_chat(tokenizer, prompt)
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    for example in test_data:
        prompt_list.append(example["prompt"])
        input_list.append(example["input"])
        outputs_list.append(example["outputs"])
        length_list.append(example["length"])
        index_list.append(example["index"])

    print("Finish loading model and tokenizer")
    model_name = model_path.split("/")[-1]

    
    if args.eviction_mode == "constant": 
        os.makedirs(os.path.join(args.save_dir, args.dataset), exist_ok=True)
        fout = open(os.path.join(args.save_dir, args.dataset, f"{args.method}.json"), "w")
        desc_string = f"Predicting {args.dataset} with {args.method}, max_capacity_prompt={args.max_capacity_prompts}"
    else:
        os.makedirs(os.path.join(args.save_dir, args.dataset), exist_ok=True)
        fout = open(os.path.join(args.save_dir, args.dataset, f"{args.method}.json"), "w")
        desc_string = f"Predicting {args.dataset} with {args.method}, retrain_rate={args.retain_rate}"

    
    for i in tqdm(range(0, len(prompt_list), args.eval_batch_size), desc=desc_string, ncols=100):
        
        batch_prompts = prompt_list[i:i+args.eval_batch_size]
        batch_inputs = input_list[i:i+args.eval_batch_size]
        batch_answers = outputs_list[i:i+args.eval_batch_size]
        batch_lengths = length_list[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        # Set model
        set_model(model, args)
 
        context_length = batch_input_ids.shape[-1]

        if args.method.lower() == "gemfilter":
            output = gemfilter_generate_selection(tokenized_prompts['input_ids'], tokenized_prompts['attention_mask'], 
            model, tokenizer, max_gen_len=output_max_len, select_layer_idx=args.filter_idx)
            batch_outputs = [output] # only single batch
            batch_generations = batch_outputs
        else:         
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id]
            )
            batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
            batch_generations = batch_outputs
    
        torch.cuda.empty_cache()
        
        for j in range(args.eval_batch_size):
            
            example = {}
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["answers"] = batch_answers[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]

            fout.write(json.dumps(example) + "\n")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Base settings
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="outputs/results_ruler")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    
    # Eval settings
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--context_length", type=int, default=8192, help="")
    
    # KV cache compression
    parser.add_argument("--method", type=str,  default=None, choices=["fullkv", "fastkv", "snapkv", "h2o", "streamingllm", "gemfilter", "pyramidinfer"])
    parser.add_argument("--eviction_mode", type=str, default="constant", choices=["constant", "proportional"])
    parser.add_argument("--retain_rate", type=float, default=0.1, help="retain rate of KV entries")
    parser.add_argument("--max_capacity_prompts", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="maxpool")
    parser.add_argument("--merge", type=str, default=None, help="Does not support")

    # FastKV
    parser.add_argument("--tsp_len", type=int, default=2048, help="tsp_len used for constant eviction mode")
    parser.add_argument("--tsp_rate", type=float, default=0.2, help="tsp_rate used for proportional eviction mode")
    parser.add_argument("--tsp_idx", type=int, default=15, help="")

    # GemFilter
    parser.add_argument("--filter_idx", type=int, default=13, help="")

    # PyramidInfer
    parser.add_argument("--pyramidinfer_config", type=str, default="")

    args = parser.parse_args()
    set_seed(args.seed)

    from baselines.monkeypatch import replace_llama, replace_mistral
    replace_llama(args.method)
    replace_mistral(args.method)
    
    if args.method == "pyramidinfer":
        args.attn_implementation = "eager"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    
    model.eval()
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    save_dir = args.save_dir
    
    for idx, dataset in enumerate(datasets):
        if args.eviction_mode == "constant":
            print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        else:
            print(f"Working on retain_rate {args.retain_rate} dataset {dataset} - {idx}/{len(datasets)}")
            
        args.dataset = dataset
        args.data_file = f"data/RULER/{args.context_length}/{args.dataset}.jsonl"

        main(model, args)
