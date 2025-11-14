#!/bin/bash
#
# Scale_mul聚类假设快速验证脚本
# Phase 1: 单层快速验证
#
# 推荐验证高方差层:
# - Layer 5: std=5.24 (最高)
# - Layer 7: std=4.85
# - Layer 11: std=4.45
# - Layer 12: std=3.93

set -e

echo "=================================="
echo " Scale_mul Clustering Verification"
echo " Phase 1: Quick Validation"
echo "=================================="
echo ""

# 配置
MODEL_DEPTH=16
LAYER_IDX=7  # 高方差层，推荐先验证这一层
NUM_SAMPLES=50  # 快速验证用50个样本
THRESHOLD_FACTOR=0.3  # 聚类阈值系数

# 输出目录
OUTPUT_DIR="./verification_results_d16_layer${LAYER_IDX}"

echo "Configuration:"
echo "  Model: VAR-d${MODEL_DEPTH}"
echo "  Target Layer: ${LAYER_IDX}"
echo "  Num Samples: ${NUM_SAMPLES}"
echo "  Threshold Factor: ${THRESHOLD_FACTOR}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""

# 运行验证
echo "Starting verification..."
echo ""

python verify_scale_mul_clustering.py \
    --model_depth ${MODEL_DEPTH} \
    --layer_idx ${LAYER_IDX} \
    --num_samples ${NUM_SAMPLES} \
    --threshold_factor ${THRESHOLD_FACTOR} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "=================================="
echo " Verification Complete!"
echo "=================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Key files:"
echo "  - verification_report.json    : Summary results"
echo "  - method1_scatter_layer*.png  : Scale diff vs similarity"
echo "  - method3_heatmap_layer*.png  : Similarity matrix with clusters"
echo "  - clusters.json               : Identified clusters"
echo ""
echo "Next steps:"
echo "  1. Check verification_report.json for overall result"
echo "  2. View visualizations to understand the relationship"
echo "  3. If PASS, proceed with clustering pruning strategy"
echo "  4. If FAIL, consider alternative strategies (variance-based, etc.)"
echo ""
