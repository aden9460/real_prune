#!/bin/bash

set -e

# 捕获 kill 或 Ctrl+C，终止所有后台进程并退出

trap "echo '收到终止信号，正在退出...'; kill 0; exit 1" SIGINT SIGTERM
first_seed=1

prev_fid_pid=""
total_seed=(0 1 2 3 4 5 6 7 8 9)
result_txt="rebuttal1_seed_20%_train1epoch.log"
> $result_txt  # 清空输出文件

for seed in "${total_seed[@]}"; do
    
    # 1. 剪枝
    specific_layer=256
    maxlayer=16
    sparsity=0.2
    num_samples=150
    model_name="d${maxlayer}_${sparsity}sparsity_${num_samples}i_${specific_layer}eva_scale_randomseed_${seed}seed_temporary.pth" 
    ##刚才出错了 记得后续改掉这里的if
    if [ "$first_seed" -eq 1 ]; then
        echo "跳过第一次剪枝 seed=$seed"
        first_seed=0
    else
    CUDA_VISIBLE_DEVICES=2 python -u /home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/prune_v2.py \
        --minlayer 0 \
        --maxlayer $maxlayer \
        --num_samples $num_samples \
        --percdamp 1e-3 \
        --skip_evaluate \
        --prune_method slimgpt \
        --sparsity $sparsity \
        --specific_layer $specific_layer \
        --model_name $model_name \
        --seed $seed
    

    # 2. 训练
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun  \
        --nnodes=1 \
        --nproc_per_node=6 \
        --node_rank=0 \
        /home/wangzefang/edgevar/EdgeVAR/VAR/train.py \
        --depth=$maxlayer --bs=390 --ep=1 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=$sparsity \
        --local_out_dir_path="/home/wangzefang/edgevar/EdgeVAR/VAR/traind_model/${model_name}_1epoch" --data_path="/home/wangzefang/Datasets/ImageNet-1K" \
        --var_path="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/${model_name}" \
        --vae_path='/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/model_zoo/vae_ch160v4096z32.pth'
    fi
    # 3. 等待上一次FID测试完成（第一次不用等）
    if [ -n "$prev_fid_pid" ]; then
        wait $prev_fid_pid
    fi
    
    # 4. FID测试（放到后台，和下次剪枝训练并行）
    output_name=$model_name
    (
        CUDA_VISIBLE_DEVICES=1 python -u /home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/FID_test.py --depth $maxlayer --sparsity $sparsity \
            --var_model="/home/wangzefang/edgevar/EdgeVAR/VAR/traind_model/${model_name}_1epoch/ar-ckpt-best.pth" \
            --output_name=$output_name

        echo "seed: $seed" >> $result_txt
        echo "----- fidelity -----" >> $result_txt

        CUDA_VISIBLE_DEVICES=1 fidelity \
            --input1 "/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/${output_name}/" \
            --input2 "/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/virtual_images" \
            --fid \
            --kid \
            --isc \
            --gpu 0 >> $result_txt 2>&1

        echo "----- fidelity -----" >> $result_txt
        echo "----- pytorch_fid -----" >> $result_txt
        python -m pytorch_fid "/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/VIRTUAL_imagenet256_labeled.npz" "/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/${output_name}/" >> $result_txt 2>&1
        echo "----------------------------------------------" >> $result_txt
    ) &
    prev_fid_pid=$!
done

# 最后一次FID测试要等它结束
if [ -n "$prev_fid_pid" ]; then
    wait $prev_fid_pid
fi