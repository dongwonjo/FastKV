import os
import json
import random
import argparse
import logging

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.monkeypatch import set_model
from baselines.gemfilter.utils import gemfilter_generate_selection_prefill


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def main(model, args):
    # Input Sequence
    input_id = torch.ones((args.eval_batch_size, args.context_length), dtype=torch.int64).to(model.device)
    attn_mask = torch.ones((args.eval_batch_size, args.context_length), dtype=torch.int64).to(model.device)

    set_model(model, args)
    
    # warmup
    if args.num_warmups > 0:
        for i in range(args.num_warmups):

            total_time = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Generation
            if args.method in ['fullkv', 'fastkv', 'snapkv', 'h2o', 'streamingllm']:
                with torch.no_grad():
                    # Prefill
                    start.record()
                    outputs = model(input_id, attn_mask)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)
                    
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids = [pred_token_idx.item()]

                    for _ in range(args.genlen-1):
                        start.record()
                        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
                        end.record()
                        torch.cuda.synchronize()
                        total_time += start.elapsed_time(end)
                        past_key_values = outputs.past_key_values
                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        generated_ids.append(pred_token_idx.item())

                generation_length = len(generated_ids)
                throughput = (args.genlen-1) / (total_time / 1000)
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                del outputs
                del past_key_values
                
            elif args.method in ['gemfilter']:

                with torch.no_grad():
                    # Prefill
                    start.record()
                    pred_token_idx, past_key_values = gemfilter_generate_selection_prefill(input_id, 
                        attn_mask, model, tokenizer, select_layer_idx=args.filter_idx)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)
                    
                    generated_ids = [pred_token_idx.item()]
                    
                    for _ in range(args.genlen-1):
                        start.record()
                        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
                        end.record()
                        torch.cuda.synchronize()
                        total_time += start.elapsed_time(end)
                        past_key_values = outputs.past_key_values
                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        generated_ids.append(pred_token_idx.item())

                generation_length = len(generated_ids)
                throughput = (args.genlen-1) / (total_time / 1000)
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                del outputs
                del pred_token_idx
                del past_key_values

            elif args.method in ['pyramidinfer']:

                start.record()
                output = model.generate(
                    input_ids=input_id,
                    attention_mask=attn_mask,
                    max_new_tokens=args.genlen,
                    min_new_tokens=args.genlen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                del output

            else:
                raise ValueError(f"We does not support {args.method} mode")
            cleanup_memory()

    latency_list = []
    throughput_list = []
    
    for i in range(args.num_runs):
        total_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Generation
        if args.method in ['fullkv', 'fastkv', 'snapkv', 'h2o', 'streamingllm']:
            with torch.no_grad():
                # Prefill
                
                start.record()
                outputs = model(input_id, attn_mask)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids = [pred_token_idx.item()]

                for _ in range(args.genlen-1):
                    start.record()
                    outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids.append(pred_token_idx.item())
    
            generation_length = len(generated_ids)
            throughput = (args.genlen-1) / (total_time / 1000)
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            del outputs
            del past_key_values

        elif args.method in ['gemfilter']:

            with torch.no_grad():
                # Prefill
                start.record()
                pred_token_idx, past_key_values = gemfilter_generate_selection_prefill(input_id, 
                    attn_mask, model, tokenizer, select_layer_idx=args.filter_idx)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                
                generated_ids = [pred_token_idx.item()]
                
                for _ in range(args.genlen-1):
                    start.record()
                    outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids.append(pred_token_idx.item())

            generation_length = len(generated_ids)
            throughput = (args.genlen-1) / (total_time / 1000)
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            del outputs
            del pred_token_idx
            del past_key_values

        elif args.method in ['pyramidinfer']:

            start.record()
            output = model.generate(
                input_ids=input_id,
                attention_mask=attn_mask,
                max_new_tokens=args.genlen,
                min_new_tokens=args.genlen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
            
            generation_length = output.shape[-1] - args.context_length
            throughput = (args.genlen-1) / (total_time / 1000)

            del output
        else:
            raise ValueError(f"We does not support {args.method} mode")
        
        latency_list.append(total_time/1000)
        throughput_list.append(throughput)
        cleanup_memory()
        
    mean_latency = sum(latency_list) / len(latency_list) if latency_list else 0
    mean_throughput = sum(throughput_list) / len(throughput_list) if throughput_list else 0

    print(f"\nMethod: {args.method}")
    print(f"Context Length = {args.context_length}")
    if args.method == "fastkv":
        print(f"TSP Rate = {args.tsp_rate} | Retention Rate = {args.retain_rate}")
    if args.method != "fullkv":
        print(f"Retention Rate = {args.retain_rate}")
    print(f"Generation Length = {generation_length}")
    print(f"Avg E2E Latency = {(mean_latency):.2f} seconds")
    print(f"Throughput: {(mean_throughput):.2f} tokens/sec")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1000**2 / 1000:.2f} GB\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Base settings
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    
    # Benchmark settings
    parser.add_argument("--genlen", type=int, default=128, help="")
    parser.add_argument("--num_warmups", type=int, default=1, help="")
    parser.add_argument("--num_runs", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
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
        
    context_lengths = [8192, 32768, 131072]
    
    for context_length in context_lengths:
        args.context_length = context_length
        print(f"E2E latency benchmark ({args.method}) | Context length={context_length} | Generation length={args.genlen}")
        main(model, args)
