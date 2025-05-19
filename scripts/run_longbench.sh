# This is example for LongBench script.

# Dataset List
# "narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"

# Model List
# nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
# meta-llama/Meta-Llama-3.1-8B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# mistralai/Mistral-Nemo-Instruct-2407

# Config
dataset_list="qasper"
model="nvidia/Llama-3.1-8B-UltraLong-1M-Instruct"
device=0
max_prompt=2048

# FastKV
path="fastkv-$max_prompt"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode fastkv \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt $max_prompt \
    --pooling avgpool \
    --kernel_size 7 \
    --window_size 8 \
    --tsp_idx 15 \
    --tsp_len 2048
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done

# FullKV
path="fullkv"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode fullkv \
    --save_path $path \
    --dataset $dataset
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done

# SnapKV
path="snapkv-$max_prompt"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode snapkv \
    --pooling avgpool \
    --kernel_size 7 \
    --window_size 8 \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt $max_prompt
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done

# GemFilter
path="gemfilter-$max_prompt"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode gemfilter \
    --save_path $path \
    --dataset $dataset \
    --max_capacity_prompt $max_prompt \
    --filter_idx 13
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done

# AdaKV
path="adakv-$max_prompt"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
    --model $model \
    --mode adakv \
    --save_path $path \
    --dataset $dataset \
    --pooling avgpool \
    --kernel_size 7 \
    --window_size 8 \
    --floor_alpha 0.2 \
    --max_capacity_prompt $max_prompt
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path 
done

# HeadKV
path="headkv-$max_prompt"
for dataset in $dataset_list
do
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.main \
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
    --dataset $dataset \
    --max_capacity_prompt $max_prompt
    CUDA_VISIBLE_DEVICES=$device python -m eval.longbench.evaluate \
    --model $model \
    --eval_path $path
done