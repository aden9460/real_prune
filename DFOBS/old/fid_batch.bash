#!/bin/bash
set -euo

echo "==> Training..."


#!/bin/sh 
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate slim
cd /wanghuan/data/wangzefang/slim_VAR_copy/slimgpt_pub

patch_nums_list=(1 2 3 4 5 6 8 10 13 16)
result_txt="fid_results.txt"
> $result_txt  # 清空输出文件

for patch_num in "${patch_nums_list[@]}"; do
    specific_layer=$((patch_num * patch_num))
    model_name="d24_0.2_${specific_layer}_input.pth"
    output_name="new_d24_0.2_${specific_layer}_input"

    # 1. 执行 prune.py
    CUDA_VISIBLE_DEVICES=0 python -u /wanghuan/data/wangzefang/slim_VAR_copy/slimgpt_pub/prune.py \
        --minlayer 0 \
        --maxlayer 24 \
        --num_samples 50 \
        --seqlen 256 \
        --percdamp 1e-2 \
        --min_sparsity 0.0625 \
        --max_sparsity 0.3 \
        --skip_evaluate \
        --prune_method slimgpt \
        --sparsity 0.2 \
        --specific_layer $specific_layer \
        --model_name $model_name

    # 2. 切换目录并执行 FID_test.py

    python /wanghuan/data/wangzefang/VAR/FID_test.py --depth 24 --sparsity 0.2 --data_path="/datasets/liying/datasets/imagenet" \
        --var_model="/wanghuan/data/wangzefang/slim_VAR_copy/slimgpt_pub/sparsity_model/${model_name}" \
        --output_name="${output_name}"

    # 3. 执行 torch-fidelity，并将输出追加到结果文件
    echo "Patch_num: $patch_num, Specific_layer: $specific_layer" >> $result_txt

    fidelity \
        --input1 "/wanghuan/data/wangzefang/VAR/FID_test/image/${output_name}/" \
        --input2 "/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/virtual_images/" \
        --fid \
        --kid \
        --isc \
        --gpu 0 \ >> $result_txt 2>&1

    echo "----- pytorch_fid -----" >> $result_txt
    python -m pytorch_fid "/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/VIRTUAL_imagenet256_labeled.npz" "/wanghuan/data/wangzefang/VAR/FID_test/image/${output_name}" >> $result_txt 2>&1

    echo "----------------------------------------------" >> $result_txt

done


fidelity --input1 "/wanghuan/data/wangzefang/VAR/FID_test/image/new_d24_0.2_1_input/"  --input2 "/wanghuan/data/wangzefang/slim_VAR_copy/VAR/FID_test/virtual_images/" --fid --kid --isc --gpu 0

