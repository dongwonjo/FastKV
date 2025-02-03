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

def average_excluding_min_max(numbers):
    if len(numbers) <= 2:
        raise ValueError("The list must contain more than two elements.")
    
    numbers_excluding_min_max = numbers.copy()
    numbers_excluding_min_max.remove(min(numbers))
    numbers_excluding_min_max.remove(max(numbers))

    return sum(numbers_excluding_min_max) / len(numbers_excluding_min_max)


def main(args):
    set_seed(args.seed)

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
        from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection_prefill, set_topk
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

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map='auto', trust_remote_code=True)
    tokenizer.padding_side  = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', attn_implementation='flash_attention_2', torch_dtype=torch.float16)
    model.eval()
    

    # load compress args
    if args.mode == 'fastkv':
        from baseline.fastkv.fastkv_utils import compress
        compress(model, args)
    elif args.mode == 'snapkv':
        from baseline.snapkv.snapkv_utils import compress
        compress(model, args)
    elif args.mode == 'adakv':
        from baseline.adakv.adaptive_snapkv.snapkv_utils import compress
        compress(model, args)
    elif args.mode == 'headkv':
        from baseline.headkv.headkv.snapkv_utils import compress
        compress(model, args)
        
    # Input Sequence
    input_id = torch.ones((1,args.seqlen), dtype=torch.int64).to(model.device)
    attn_mask = torch.ones((1,args.seqlen), dtype=torch.int64).to(model.device)
    context_length = input_id.shape[-1]


    # warmup
    if args.num_warmups > 0:
        for i in range(args.num_warmups):

            total_time = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Prefill
            if args.mode in ['fullkv', 'fastkv', 'snapkv', 'adakv', 'headkv']:
                with torch.no_grad():
                    start.record()
                    outputs = model(input_id, attn_mask)
                    end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                del outputs
            elif args.mode in ['gemfilter']:
                set_topk(model, args.max_capacity_prompt, mode='gemfilter')
                with torch.no_grad():
                    start.record()
                    pred_token_idx, past_key_values = gemfilter_generate_selection_prefill(
                        input_id, attn_mask, model, tokenizer, select_layer_idx=args.filter_idx)
                    end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                del pred_token_idx
                del past_key_values
            else:
                raise ValueError(f"We does not support {args.mode} mode")

            utils.cleanup_memory()

    result_list = []
    for i in range(args.num_runs):        

        total_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Prefill
        if args.mode in ['fullkv', 'fastkv', 'snapkv', 'adakv', 'headkv']:
            with torch.no_grad():
                start.record()
                outputs = model(input_id, attn_mask)
                end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
            del outputs
        elif args.mode in ['gemfilter']:
            set_topk(model, args.max_capacity_prompt, mode='gemfilter')
            with torch.no_grad():
                start.record()
                pred_token_idx, past_key_values = gemfilter_generate_selection_prefill(
                    input_id, attn_mask, model, tokenizer, select_layer_idx=args.filter_idx)
                end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
            del pred_token_idx
            del past_key_values
        else:
            raise ValueError(f"We does not support {args.mode} mode")
        
        result_list.append(total_time)
        utils.cleanup_memory()

    mean_ttft = average_excluding_min_max(result_list)

    print(f"\nMode: {args.mode}")
    print(f"Context Length = {context_length}")
    print(f"TTFT: {(mean_ttft):.5f} msec")
    print(f"Number of Warmup Runs: {args.num_warmups}, Number of Runs: {args.num_runs}, Min & Max values excluded")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1000**2 / 1000:.2f} GB\n")

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

    # Benchmark Option
    parser.add_argument("--seqlen", type=int, default=131072, help="")
    parser.add_argument("--num_warmups", type=int, default=2, help="")
    parser.add_argument("--num_runs", type=int, default=10, help="num_runs must be larger than 2")


    args = parser.parse_args()

    main(args)
    utils.cleanup_memory()
