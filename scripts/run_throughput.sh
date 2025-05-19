# This is example for throughput scripts
# 8192 / 16384 / 32768 / 65536 / 131072

model="nvidia/Llama-3.1-8B-UltraLong-1M-Instruct"
device=0
seqlen=131072
genlen=128
max_prompt=512

CUDA_VISIBLE_DEVICES=$device python -m benchmark.throughput \
--model $model \
--mode fastkv \
--max_capacity_prompt $max_prompt \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--tsp_idx 15 \
--tsp_len 2048 \
--seqlen $seqlen \
--genlen $genlen \
--num_warmups 2 \
--num_runs 10

CUDA_VISIBLE_DEVICES=$device python -m benchmark.throughput \
--model $model \
--mode fullkv \
--seqlen $seqlen \
--genlen $genlen \
--num_warmups 2 \
--num_runs 10

CUDA_VISIBLE_DEVICES=$device python -m benchmark.throughput \
--model $model \
--mode gemfilter \
--max_capacity_prompt $max_prompt \
--filter_idx 13 \
--seqlen $seqlen \
--genlen $genlen \
--num_warmups 2 \
--num_runs 10

CUDA_VISIBLE_DEVICES=$device python -m benchmark.throughput \
--model $model \
--mode snapkv \
--max_capacity_prompt $max_prompt \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--seqlen $seqlen \
--genlen $genlen \
--num_warmups 2 \
--num_runs 10

CUDA_VISIBLE_DEVICES=$device python -m benchmark.throughput \
--model $model \
--mode adakv \
--max_capacity_prompt $max_prompt \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--floor_alpha 0.2 \
--seqlen $seqlen \
--genlen $genlen \
--num_warmups 2 \
--num_runs 10

CUDA_VISIBLE_DEVICES=$device python -m benchmark.throughput \
--model $model \
--mode headkv \
--max_capacity_prompt $max_prompt \
--pooling avgpool \
--kernel_size 7 \
--window_size 8 \
--floor_alpha 0.2 \
--method ReasonKV \
--head_choice reason \
--normalize \
--beta 1.2 \
--temp 1.0 \
--seqlen $seqlen \
--genlen $genlen \
--num_warmups 2 \
--num_runs 10