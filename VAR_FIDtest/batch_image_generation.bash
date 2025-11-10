#!/bin/bash

# VAR 剪枝方法对比 - 批量图片生成脚本
# 模仿 FID_test.bash 的结构

echo "开始批量生成对比图片..."

# 基础参数
DEPTH=16
NUM_SAMPLES=50

# 所有要测试的模型
SPARSITIES=(0.1 0.2 0.3 0.4)
SCALE_STRATEGIES=("natural" "equal" "sqrt")

# 创建输出目录
mkdir -p ./output/comparison_images

echo "第一阶段：生成 Chain KFAC 图片..."

# 生成 Chain KFAC 图片
for sparsity in "${SPARSITIES[@]}"; do
    for strategy in "${SCALE_STRATEGIES[@]}"; do
        model_name="var_d${DEPTH}_chain_kfac_s${sparsity}_${strategy}_n${NUM_SAMPLES}.pth"
        var_model="./batch_models/chain_kfac/${model_name}"
        output_name="chain_kfac_d${DEPTH}_s${sparsity}_${strategy}_n${NUM_SAMPLES}"

        if [ -f "${var_model}" ]; then
            echo "生成图片: ${output_name}"
            CUDA_VISIBLE_DEVICES=0 python generate_single_image.py \
                --depth ${DEPTH} \
                --sparsity ${sparsity} \
                --var_model="${var_model}" \
                --output_name="${output_name}" \
                --method_label="Chain-KFAC ${strategy}"
        else
            echo "⚠️  模型文件不存在: ${var_model}"
        fi
    done
done

echo "第二阶段：生成 SlimGPT 图片..."

# 生成 SlimGPT 图片
for sparsity in "${SPARSITIES[@]}"; do
    model_name="var_d${DEPTH}_slimgpt_s${sparsity}_n${NUM_SAMPLES}.pth"
    var_model="./batch_models/slimgpt/${model_name}"
    output_name="slimgpt_d${DEPTH}_s${sparsity}_n${NUM_SAMPLES}"

    if [ -f "${var_model}" ]; then
        echo "生成图片: ${output_name}"
        CUDA_VISIBLE_DEVICES=0 python generate_single_image.py \
            --depth ${DEPTH} \
            --sparsity ${sparsity} \
            --var_model="${var_model}" \
            --output_name="${output_name}" \
            --method_label="SlimGPT"
    else
        echo "⚠️  模型文件不存在: ${var_model}"
    fi
done

echo "第三阶段：拼接对比图..."
python create_comparison_grid.py

echo "✅ 批量图片生成完成！"
echo "查看结果: ./output/comparison_images/"