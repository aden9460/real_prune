#!/bin/bash

# 批量训练脚本 - 生成所有剪枝模型
# Chain KFAC: 4 sparsity × 3 strategies = 12 models
# SlimGPT: 4 sparsity = 4 models
# 总共16个模型

echo "开始批量训练所有剪枝模型..."

# 参数设置
SPARSITIES=(0.1 0.2 0.3 0.4)
SCALE_STRATEGIES=("natural" "equal" "sqrt")
MAX_LAYER=16
NUM_SAMPLES=50

# 创建输出目录 - 修改到VAR_FIDtest目录
mkdir -p /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac
mkdir -p /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt

echo "第一阶段：训练 Chain KFAC 模型..."

# 训练 Chain KFAC 模型
for sparsity in "${SPARSITIES[@]}"; do
    for strategy in "${SCALE_STRATEGIES[@]}"; do
        echo "训练 Chain KFAC: sparsity=${sparsity}, strategy=${strategy}"

        MODEL_NAME="var_d${MAX_LAYER}_chain_kfac_s${sparsity}_${strategy}_n${NUM_SAMPLES}.pth"

        python prune_chain.py \
            --sparsity ${sparsity} \
            --scale_weight_strategy ${strategy} \
            --prune_method chain_kfac \
            --maxlayer ${MAX_LAYER} \
            --num_samples ${NUM_SAMPLES} \
            --save_pruned_weights \
            --save_dir /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac \
            --model_name "${MODEL_NAME}"

        if [ $? -eq 0 ]; then
            echo "✅ 完成: ${MODEL_NAME}"
        else
            echo "❌ 失败: ${MODEL_NAME}"
        fi
        echo "----------------------------------------"
    done
done

echo "第二阶段：训练 SlimGPT 模型..."

# 训练 SlimGPT 模型
for sparsity in "${SPARSITIES[@]}"; do
    echo "训练 SlimGPT: sparsity=${sparsity}"

    MODEL_NAME="var_d${MAX_LAYER}_slimgpt_s${sparsity}_n${NUM_SAMPLES}.pth"

    python prune_v6.py \
        --sparsity ${sparsity} \
        --prune_method slimgpt \
        --maxlayer ${MAX_LAYER} \
        --num_samples ${NUM_SAMPLES} \
        --save_pruned_weights \
        --save_dir /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt \
        --model_name "${MODEL_NAME}"

    if [ $? -eq 0 ]; then
        echo "✅ 完成: ${MODEL_NAME}"
    else
        echo "❌ 失败: ${MODEL_NAME}"
    fi
    echo "----------------------------------------"
done

echo "批量训练完成！"
echo "Chain KFAC 模型: 12个"
echo "SlimGPT 模型: 4个"
echo "总计: 16个模型"