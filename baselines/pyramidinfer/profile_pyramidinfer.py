import os
import json
import random
import argparse
from tkinter import W

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.fastkv.utils import compress_fastkv
from baselines.gemfilter.utils import set_topk, gemfilter_generate_selection
from baselines.pyramidinfer.utils import load_pyramid_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--method", type=str,  default="pyramidinfer")
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument('--head_path', type=str, default='./data/heads_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json', help='Path to head score (HeadKV)')
    parser.add_argument('--head_beta', type=float, default=1.01, help='hyper-parameter used on HeadKV')
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")

    # PyramidInfer
    parser.add_argument("--pyramidinfer_config", type=str, default="baselines/pyramidinfer_configs/llama31_8b_62.5%.json")

    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from baselines.monkeypatch import replace_llama,replace_mistral
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
    

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation="eager"
    )
        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        
    model.eval()
    
    
    pyramidinfer_config = json.load(open(args.pyramidinfer_config))

    
    for seqlen in [1000, 2000, 3000, 4000]:
        for dr in [0.01]:
            pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"] = dr
            model = load_pyramid_config(model, pyramidinfer_config)

            input_id = torch.ones((1,seqlen), dtype=torch.int64).to(model.device)
            attn_mask = torch.ones((1,seqlen), dtype=torch.int64).to(model.device)

            outputs = model(input_id, attn_mask)

            kvlen_sum = 0
            for i in range(len(model.model.layers)):

                temp_kvlen = outputs.past_key_values[i][0].shape[-2]
                #print(f"Layer {i}, kvlen={temp_kvlen}")
                kvlen_sum += temp_kvlen

            retain_rate = (kvlen_sum/len(model.model.layers)/seqlen)*100

            # print(f"prefill_decay_ratio={pyramidinfer_config["prefill_stage"]["prefill_decay_ratio"]}, retain_rate={retain_rate:.2f}")
            print(f"seqlen={seqlen}, prefill_decay_ratio={dr}, retain_rate={retain_rate:.2f}%")

