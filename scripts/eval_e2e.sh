model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
method="fastkv"

CUDA_VISIBLE_DEVICES=0 python -m benchmark.e2e \
    --method $method \
    --model_path $model_path \
    --genlen 256 \
    --tsp_idx 15 \
    --tsp_rate 0.2 \
    --retain_rate 0.1 \
    --eviction_mode proportional \
    --num_warmups 1 \
    --num_runs 1