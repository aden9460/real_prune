#!/bin/bash

set -e

# 捕获 kill 或 Ctrl+C，终止所有后台进程并退出

trap "echo '收到终止信号，正在退出...'; kill 0; exit 1" SIGINT SIGTERM

specific_layer="9 9 9 8 8 8 7 7 7 6 6 6 5 5 4 4 3 3 2 2 1 1 0 0 "  #test1

maxlayer=24
sparsity=0.2
num_samples=150
method="slimgpt"
result_txt="rebuttal1_d16_${sparsity}sparsity_train1epoch_${method}.log"
> $result_txt  # 清空输出文件
model_name="d${maxlayer}_${sparsity}sparsity_${num_samples}i_${specific_layer}eva_scale_${method}_multiscale.pth"

CUDA_VISIBLE_DEVICES=2 python -u /home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/prune_v2.py \
    --minlayer 0 \
    --maxlayer $maxlayer \
    --num_samples $num_samples \
    --percdamp 1e-3 \
    --skip_evaluate \
    --prune_method $method \
    --sparsity $sparsity \
    --specific_layer $specific_layer \
    --model_name $model_name 
    # --seed $seed

# # 2. 训练
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun  \
#     --nnodes=1 \
#     --nproc_per_node=7 \
#     --node_rank=0 \
#     /home/wangzefang/edgevar/EdgeVAR/VAR/train.py \
#     --depth=$maxlayer --bs=480 --ep=1 --fp16=1 --alng=1e-3 --wpe=0.1 --sparsity=$sparsity \
#     --local_out_dir_path="/home/wangzefang/edgevar/EdgeVAR/VAR/traind_model/${model_name}_1epoch" --data_path="/home/wangzefang/Datasets/ImageNet-1K" \
#     --var_path="/home/wangzefang/edgevar/EdgeVAR/slimgpt_pub/output/sparsity_model/${model_name}" \
#     --vae_path='/home/wangzefang/Project/distilled_decoding/VAR/model_zoo/original_VAR/model_zoo/vae_ch160v4096z32.pth'
# # 3. 等待上一次FID测试完成（第一次不用等）


# 4. FID测试（放到后台，和下次剪枝训练并行）
output_name=$model_name

CUDA_VISIBLE_DEVICES=0 python -u /home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/FID_test.py --depth $maxlayer --sparsity $sparsity \
    --var_model="/home/wangzefang/edgevar/EdgeVAR/VAR/traind_model/${model_name}_1epoch/ar-ckpt-best.pth" \
    --output_name=$output_name

echo "seed: $seed" >> $result_txt
echo "----- fidelity -----" >> $result_txt

CUDA_VISIBLE_DEVICES=0 fidelity \
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


