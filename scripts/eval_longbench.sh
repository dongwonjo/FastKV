method=fastkv
eviction_mode=proportional
tsp_idx=15
tsp_rate=0.2
retain_rate=0.1
attn_implementation=flash_attention_2
model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
save_dir="outputs/results_longbench"

CUDA_VISIBLE_DEVICES=0 python -m eval.run_longbench \
    --method ${method} \
    --model_path ${model_path} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --eviction_mode ${eviction_mode} \
    --tsp_rate ${tsp_rate} \
    --tsp_idx ${tsp_idx} \
    --retain_rate ${retain_rate}

CUDA_VISIBLE_DEVICES=0 python -m eval.eval_longbench \
    --results_dir ${save_dir}
