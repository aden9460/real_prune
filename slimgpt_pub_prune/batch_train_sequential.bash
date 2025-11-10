#!/bin/bash

# 批量训练脚本 - 顺序执行 (Sequential Execution)
# Chain KFAC: 4 sparsity × 3 strategies = 12 models
# SlimGPT: 4 sparsity = 4 models
# 总共16个模型

echo "开始批量训练所有剪枝模型 (Sequential Mode)..."

# 参数设置
SPARSITIES=(0.1 0.2 0.3 0.4)
SCALE_STRATEGIES=("natural" "equal" "sqrt")
MAX_LAYER=16
NUM_SAMPLES=50

# 创建输出目录
mkdir -p /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac
mkdir -p /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt

# 计数器
total_models=16
current_model=0

echo "第一阶段：训练 Chain KFAC 模型..."

# 训练 Chain KFAC 模型
for sparsity in "${SPARSITIES[@]}"; do
    for strategy in "${SCALE_STRATEGIES[@]}"; do
        current_model=$((current_model + 1))
        echo "=========================================="
        echo "进度: ${current_model}/${total_models}"
        echo "训练 Chain KFAC: sparsity=${sparsity}, strategy=${strategy}"
        echo "=========================================="

        MODEL_NAME="var_d${MAX_LAYER}_chain_kfac_s${sparsity}_${strategy}_n${NUM_SAMPLES}.pth"

        # 清理GPU缓存
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

        # 顺序执行训练
        CUDA_VISIBLE_DEVICES=0 python prune_chain.py \
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

        # 清理GPU缓存
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
        sleep 2
        echo "----------------------------------------"
    done
done

echo "第二阶段：训练 SlimGPT 模型..."

# 训练 SlimGPT 模型
for sparsity in "${SPARSITIES[@]}"; do
    current_model=$((current_model + 1))
    echo "=========================================="
    echo "进度: ${current_model}/${total_models}"
    echo "训练 SlimGPT: sparsity=${sparsity}"
    echo "=========================================="

    MODEL_NAME="var_d${MAX_LAYER}_slimgpt_s${sparsity}_n${NUM_SAMPLES}.pth"

    # 清理GPU缓存
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

    # 顺序执行训练
    CUDA_VISIBLE_DEVICES=0 python prune_v6.py \
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

    # 清理GPU缓存
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 2
    echo "----------------------------------------"
done

echo "=========================================="
echo "批量训练完成！"
echo "Chain KFAC 模型: 12个"
echo "SlimGPT 模型: 4个"
echo "总计: 16个模型"
echo "=========================================="

# 列出所有生成的模型
echo "生成的模型文件:"
ls -lh /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac/*.pth 2>/dev/null | wc -l | xargs echo "Chain KFAC:"
ls -lh /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt/*.pth 2>/dev/null | wc -l | xargs echo "SlimGPT:"
