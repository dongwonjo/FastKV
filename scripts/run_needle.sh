# This is example for Needle-in-a-Haystack script.

# Model List
# nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# mistralai/Mistral-Nemo-Instruct-2407

# Config
model="nvidia/Llama-3.1-8B-UltraLong-1M-Instruct"
device=0
max_prompt=2048

# FastKV
path="fastkv-$max_prompt"
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.main \
--model $model \
--mode fastkv \
--save_path $path \
--max_capacity_prompt $max_prompt \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--tsp_idx 15 \
--tsp_len 2048
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.visualize \
--model $model \
--eval_path $path

# FullKV
path="fullkv"
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.main \
--model $model \
--mode fullkv \
--save_path $path
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.visualize \
--model $model \
--eval_path $path

# SnapKV
path="snapkv-$max_prompt"
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.main \
--model $model \
--mode snapkv \
--save_path $path \
--max_capacity_prompt $max_prompt \
--pooling avgpool \
--kernel_size 7 \
--window_size 8
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.visualize \
--model $model \
--eval_path $path

# GemFilter
path="gemfilter-$max_prompt"
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.main \
--model $model \
--mode gemfilter \
--save_path $path \
--max_capacity_prompt $max_prompt \
--filter_idx 13
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.visualize \
--model $model \
--eval_path $path

# AdaKV
path="adakv-$max_prompt"
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.main \
--model $model \
--mode adakv \
--save_path $path \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--floor_alpha 0.2  \
--max_capacity_prompt $max_prompt
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.visualize \
--model $model \
--eval_path $path

# HeadKV
path="headkv-$max_prompt"
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.main \
--model $model \
--mode headkv \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--floor_alpha 0.2 \
--method ReasonKV \
--head_choice reason \
--normalize \
--beta 1.2 \
--temp 1.0 \
--save_path $path \
--max_capacity_prompt $max_prompt
CUDA_VISIBLE_DEVICES=$device python -m eval.needle.visualize \
--model $model \
--eval_path $path