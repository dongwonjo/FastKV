model_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
methods="fastkv fullkv streamingllm gemfilter h2o"
device=2


for method in $methods
do
    CUDA_VISIBLE_DEVICES=${device} python -m benchmark.prefill \
        --method $method \
        --model_path $model_path \
        --tsp_idx 15 \
        --filter_idx 13 \
        --tsp_rate 0.2 \
        --retain_rate 0.1 \
        --eviction_mode proportional \
        --num_warmups 1 \
        --num_runs 5 \
        --save_dir outputs/prefill/llama   
done

CUDA_VISIBLE_DEVICES=${device} python -m benchmark.prefill \
        --method gemfilter \
        --model_path $model_path \
        --tsp_idx 15 \
        --filter_idx 15 \
        --tsp_rate 0.2 \
        --retain_rate 0.1 \
        --eviction_mode proportional \
        --num_warmups 1 \
        --num_runs 5 \
        --save_dir outputs/prefill/llama


model_path="mistralai/Ministral-8B-Instruct-2410"
methods="fastkv fullkv streamingllm gemfilter h2o"
device=3


for method in $methods
do
    CUDA_VISIBLE_DEVICES=${device} python -m benchmark.prefill \
        --method $method \
        --model_path $model_path \
        --tsp_idx 17 \
        --filter_idx 17 \
        --tsp_rate 0.2 \
        --retain_rate 0.1 \
        --eviction_mode proportional \
        --num_warmups 1 \
        --num_runs 5 \
        --save_dir outputs/prefill/ministral
done


model_path="mistralai/Mistral-Nemo-Instruct-2407"
methods="fastkv fullkv streamingllm gemfilter h2o"
device=3


for method in $methods
do
    CUDA_VISIBLE_DEVICES=${device} python -m benchmark.prefill \
        --method $method \
        --model_path $model_path \
        --tsp_idx 19 \
        --filter_idx 19 \
        --tsp_rate 0.2 \
        --retain_rate 0.1 \
        --eviction_mode proportional \
        --num_warmups 1 \
        --num_runs 5 \
        --save_dir outputs/prefill/nemo
done
