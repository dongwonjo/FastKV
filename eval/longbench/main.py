import sys
sys.path.append(".")

import argparse
import os
import time
import json
import logging
import pprint
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from accelerate import dispatch_model, load_checkpoint_in_model, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

from utils import utils


def build_chat(tokenizer, prompt, model_name):
    if "Llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    else:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    
    return prompt
            
@torch.inference_mode()
def generate_longbench(data, max_length, max_gen, prompt_format, 
                       dataset, model_name, model, tokenizer,
                       out_path, args):
    device = model.device
    for json_obj in tqdm(data, desc="Generating Responses..."):
        # load compress args
        if args.mode == 'fastkv':
            from baseline.fastkv.fastkv_utils import compress
            compress(model, args)
        elif args.mode == 'snapkv':
            from baseline.snapkv.snapkv_utils import compress
            compress(model, args)
        elif args.mode == 'gemfilter':
            from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
            set_topk(model, args.max_capacity_prompt, mode='gemfilter')
        elif args.mode == 'adakv':
            from baseline.adakv.adaptive_snapkv.snapkv_utils import compress
            compress(model, args)
        elif args.mode == 'headkv':
            from baseline.headkv.headkv.snapkv_utils import compress
            compress(model, args)

        prompt = prompt_format.format(**json_obj)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        # chat models are better off without build prompts on these tasks
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if args.mode == 'gemfilter':
            pred = gemfilter_generate_selection(input['input_ids'], input['attention_mask'], 
                model, tokenizer, max_gen_len=max_gen, select_layer_idx=args.filter_idx)
        else:
            output = model.generate(
                    **input,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    min_length=context_length+1,
                    max_new_tokens=max_gen,
                    )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

def main(args):
    set_seed(args.seed)
 
    if args.save_path:
        save_path = os.path.join(f"outputs/{args.model}/longbench", args.save_path)
        Path(save_path).mkdir(parents=True, exist_ok=True)  
    else:
        tm = time.localtime(time.time())
        f_name = f"{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}"
        save_path = os.path.join(f"outputs/{args.model}/longbench", f_name)
        Path(save_path).mkdir(parents=True, exist_ok=True)

    utils.config_logging(os.path.join(save_path, f'process.log'))
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)

    # Setup for Configuration
    model2maxlen = json.load(open("eval/longbench/config/model2maxlen.json", "r"))
    max_length = model2maxlen[args.model]

    if args.longbench_type == "longbench-e":
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    elif args.longbench_type == "longbench":
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

    if args.dataset not in datasets:
        raise ValueError(f"Dataset {args.dataset} not found in datasets")
    
    # We design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("eval/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/longbench/config/dataset2maxlen.json", "r"))

    dataset = args.dataset

    # KV compression
    if args.mode == 'fullkv':
        from baseline.fullkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args.mode == 'fastkv':
        from baseline.fastkv.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args.mode == 'snapkv':
        from baseline.snapkv.monkeypatch import replace_llama, replace_mistral, replace_phi3
        replace_llama()
        replace_mistral()
        replace_phi3()
    elif args.mode == 'gemfilter':
        from baseline.gemfilter.monkeypatch import replace_llama, replace_mistral
        replace_llama()
        replace_mistral()
    elif args.mode == 'adakv':
        from baseline.adakv.adaptive_snapkv.monkeypatch import replace_llama_adaptive, replace_mistral_adaptive
        replace_llama_adaptive()
        replace_mistral_adaptive()
    elif args.mode == 'headkv':
        from baseline.headkv.headkv.monkeypatch import replace_llama, replace_mistral
        replace_llama(args.method)
        replace_mistral(args.method)
    else:
        raise ValueError(f"We does not support {args.mode} mode")

    if args.longbench_type == "longbench-e":
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
    elif args.longbench_type == "longbench":
        data = load_dataset('THUDM/LongBench', dataset, split='test')
    else:
        raise ValueError

    out_path = os.path.join(save_path, f"{dataset}.jsonl")

    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    data_all = [data_sample for data_sample in data]

    # Load Model & Tokenizer
    logging.info(f'Load Model & Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map='auto', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.float16)
    model.eval()

    # Generation
    generate_longbench(data=data_all, max_length=max_length, max_gen=max_gen, prompt_format=prompt_format, 
                       dataset=dataset, model_name=args.model, model=model, tokenizer=tokenizer,
                       out_path=out_path, args=args)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Arguments
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    # KV Compression
    parser.add_argument("--mode", type=str, default="fastkv", choices=["fullkv", "fastkv", "snapkv", "gemfilter", "adakv", "headkv"])
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--max_capacity_prompt", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", type=str, default="avgpool")
    
    # FastKV
    parser.add_argument("--tsp_idx", type=int, default=15)
    parser.add_argument("--tsp_len", type=int, default=2048)

    # GemFilter
    parser.add_argument("--filter_idx", type=int, default=13)

    # AdaKV
    parser.add_argument("--skip", type=int, default=-1)
    parser.add_argument('--floor_alpha', type=float, default=0.2)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--pyram', action='store_true')
    parser.add_argument('--pyram_beta', default=20,type=int)
    parser.add_argument('--gqa_support', action='store_true')

    # HeadKV
    parser.add_argument("--method", type=str, default='ReasonKV', choices=['ReasonKV'])
    parser.add_argument("--head_choice", type=str, default='reason', choices=['copy', 'reason'])
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--temp', type=float, default=1.0)
    
    # Evaluation
    parser.add_argument('--dataset', type=str, default='qasper', help="Dataset to evaluate on")
    parser.add_argument('--longbench_type', type=str, default='longbench', choices=['longbench', 'longbench-e'])

    args = parser.parse_args()

    main(args)
