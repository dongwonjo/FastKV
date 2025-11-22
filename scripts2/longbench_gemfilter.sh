device=2
methods="gemfilter"
eviction_mode=proportional
tsp_idx=19
tsp_rate=0.2
retain_rates="0.1 0.2"
attn_implementation=flash_attention_2
model_path="mistralai/Mistral-Nemo-Instruct-2407"
save_dir="outputs/results_longbench/nemo_gemfilter"

for method in $methods
do
    for retain_rate in $retain_rates
    do
        CUDA_VISIBLE_DEVICES=${device} python -m eval.run_longbench \
            --method ${method} \
            --model_path ${model_path} \
            --attn_implementation ${attn_implementation} \
            --save_dir ${save_dir} \
            --eviction_mode ${eviction_mode} \
            --tsp_rate ${tsp_rate} \
            --tsp_idx ${tsp_idx} \
            --filter_idx ${tsp_idx} \
            --retain_rate ${retain_rate}
    done
done
