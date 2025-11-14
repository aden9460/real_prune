#!/bin/bash
# 运行scale_mul分析脚本

echo "================================"
echo "  VAR Scale_Mul Analysis"
echo "================================"
echo ""

# 默认参数
MODEL_DEPTH=16
OUTPUT_DIR="./scale_mul_analysis_d16_0.4_0epoch_qfix"

# 运行分析
python analyze_scale_mul.py \
    --var_ckpt /home/project/real_prune/slimvar/pruned_models/qscaluefix_var_d16_0.4_2000sample_basic.pth \
    --model_depth ${MODEL_DEPTH} \
    --output_dir ${OUTPUT_DIR} \
    --sparsity 0.4 \

echo ""
echo "================================"
echo "Analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Generated files:"
echo "  - scale_mul_analysis.json      # 统计数据"
echo "  - pruning_strategy.json        # 推荐的剪枝策略"
echo "  - scale_mul_matrix.npy         # 原始数据矩阵"
echo "  - scale_mul_heatmap.png        # 热力图"
echo "  - scale_mul_boxplot.png        # 箱线图"
echo "  - scale_mul_classification.png # 分类柱状图"
echo "  - scale_mul_trends.png         # 趋势折线图"
echo "  - scale_mul_histogram.png      # 全局分布直方图"
echo "================================"
