#!/bin/bash
# VAR OBA Pruning 运行脚本
# 测试OBA剪枝VAR模型

echo "🚀 Starting VAR OBA Pruning Test..."

# 基本参数
MODEL_DEPTH=16
TARGET_SPARSITY=0.4
NUM_SAMPLES=64  # 减少样本用于测试
BATCH_SIZE=4    # 减少批次大小

# OBA参数（均衡配置）
DELTA=1.0
UPWARD_DELTA=1.0
DOWNWARD_DELTA=1.0
PARALLEL_DELTA=1.0

# 路径
OUTPUT_PATH="./var_d${MODEL_DEPTH}_oba_pruned_${TARGET_SPARSITY}sparsity.pth"

echo "📋 Configuration:"
echo "  Model: VAR-d${MODEL_DEPTH}"
echo "  Target sparsity: ${TARGET_SPARSITY}"
echo "  Calibration samples: ${NUM_SAMPLES}"
echo "  OBA weights: delta=${DELTA}, upward=${UPWARD_DELTA}, downward=${DOWNWARD_DELTA}, parallel=${PARALLEL_DELTA}"

# 运行剪枝
python var_prune_oba_v2.py \
    --model_depth ${MODEL_DEPTH} \
    --target_sparsity ${TARGET_SPARSITY} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --delta ${DELTA} \
    --upward_delta ${UPWARD_DELTA} \
    --downward_delta ${DOWNWARD_DELTA} \
    --parallel_delta ${PARALLEL_DELTA} \
    --normalizer max \
    --iters_per_step 50 \
    --output_path ${OUTPUT_PATH} \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✅ OBA Pruning completed successfully!"
    echo "📊 Pruned model saved to: ${OUTPUT_PATH}"
    echo ""
    echo "🔍 Model info:"
    python -c "
import torch
data = torch.load('${OUTPUT_PATH}', map_location='cpu')
print(f'  Original params: {data[\"original_params\"]:,}')
print(f'  Pruned params: {data[\"pruned_params\"]:,}')
print(f'  Achieved sparsity: {data[\"achieved_sparsity\"]:.1%}')
"
else
    echo "❌ OBA Pruning failed!"
    exit 1
fi

echo ""
echo "🎯 Next steps:"
echo "  1. Test the pruned model quality (FID evaluation)"
echo "  2. Compare with SlimGPT baseline"
echo "  3. Try different OBA parameter combinations"