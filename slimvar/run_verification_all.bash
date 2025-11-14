#!/bin/bash
#
# Scale_mul聚类假设全面验证脚本
# Phase 2: 多层验证
#
# 验证所有高方差层和代表性低方差层

set -e

echo "======================================"
echo " Scale_mul Clustering Verification"
echo " Phase 2: Comprehensive Validation"
echo "======================================"
echo ""

# 配置
MODEL_DEPTH=16
NUM_SAMPLES=100  # 全面验证用更多样本
THRESHOLD_FACTOR=0.3

# 验证的层 (高方差层 + 低方差层)
LAYERS=(5 7 11 12 0 2)  # 前4个是高方差，后2个是低方差

echo "Configuration:"
echo "  Model: VAR-d${MODEL_DEPTH}"
echo "  Layers to verify: ${LAYERS[@]}"
echo "  Num Samples: ${NUM_SAMPLES}"
echo "  Threshold Factor: ${THRESHOLD_FACTOR}"
echo ""

# 逐层验证
for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "======================================"
    echo " Verifying Layer ${LAYER}"
    echo "======================================"

    OUTPUT_DIR="./verification_results_d16_layer${LAYER}"

    python verify_scale_mul_clustering.py \
        --model_depth ${MODEL_DEPTH} \
        --layer_idx ${LAYER} \
        --num_samples ${NUM_SAMPLES} \
        --threshold_factor ${THRESHOLD_FACTOR} \
        --output_dir ${OUTPUT_DIR}

    echo ""
done

echo ""
echo "======================================"
echo " All Verifications Complete!"
echo "======================================"
echo ""
echo "Results summary:"
for LAYER in "${LAYERS[@]}"; do
    OUTPUT_DIR="./verification_results_d16_layer${LAYER}"
    if [ -f "${OUTPUT_DIR}/verification_report.json" ]; then
        RESULT=$(python3 -c "import json; r=json.load(open('${OUTPUT_DIR}/verification_report.json')); print(f\"Layer {r['layer_idx']:2d}: {r['overall']['result']:10s} ({r['overall']['confidence']})\")")
        echo "  ${RESULT}"
    fi
done
echo ""
echo "Next steps:"
echo "  1. Compare high-variance vs low-variance layer results"
echo "  2. Generate comprehensive report"
echo "  3. Decide on pruning strategy based on findings"
echo ""
