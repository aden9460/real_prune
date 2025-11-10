#!/bin/bash

################################################################################
# VAR-d16 FastOBA Attention Pruning - Ablation Study
#
# This script runs all ablation experiments to evaluate:
# 1. Hessian computation mode (slimgpt vs block_diagonal)
# 2. Head importance evaluation (block_mean vs slimgpt_mean)
# 3. Pruning mode (head_dims vs num_heads vs both)
# 4. Weight compensation (with vs without)
# 5. Per-stage analysis for scale-specific understanding
#
# Model: VAR-d16
# Dataset: ImageNet (or subset)
################################################################################

# Configuration
MODEL_PATH="/path/to/var_d16.pth"  # TODO: Update this path
DATA_DIR="./data/imagenet"          # TODO: Update this path
OUTPUT_BASE="./results/ablation_var_d16"
NUM_SAMPLES=128
LAYER_IDX=0  # Analyze first layer (can be changed)

# Create output directory
mkdir -p ${OUTPUT_BASE}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "VAR-d16 FastOBA Attention Pruning Ablation Study"
echo "=============================================="
echo ""
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_BASE}"
echo "Samples: ${NUM_SAMPLES}"
echo "Layer: ${LAYER_IDX}"
echo ""

################################################################################
# Experiment 1: Hessian Mode Comparison
################################################################################

echo -e "${GREEN}[Experiment 1/5] Hessian Mode Comparison${NC}"
echo "Comparing SlimGPT (H=XX^T) vs Block-Diagonal (FastOBA) Hessian"
echo ""

# 1.1 SlimGPT mode (baseline)
echo "  Running SlimGPT mode..."
python sobs/var_per_stage_analysis.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --layer_idx ${LAYER_IDX} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_BASE}/exp1_hessian_mode/slimgpt \
    --hessian_mode slimgpt \
    --head_importance_mode slimgpt_mean \
    --sparsity 0.4

# 1.2 Block-diagonal mode (FastOBA)
echo "  Running Block-Diagonal mode..."
python sobs/var_per_stage_analysis.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --layer_idx ${LAYER_IDX} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_BASE}/exp1_hessian_mode/block_diagonal \
    --hessian_mode block_diagonal \
    --head_importance_mode block_mean \
    --sparsity 0.4

echo -e "${GREEN}Experiment 1 Complete${NC}"
echo ""

################################################################################
# Experiment 2: Head Importance Mode Comparison
################################################################################

echo -e "${GREEN}[Experiment 2/5] Head Importance Mode Comparison${NC}"
echo "Comparing block_mean vs slimgpt_mean for head evaluation"
echo ""

# 2.1 block_mean mode
echo "  Running block_mean mode..."
python sobs/var_per_stage_analysis.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --layer_idx ${LAYER_IDX} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_BASE}/exp2_head_importance/block_mean \
    --hessian_mode block_diagonal \
    --head_importance_mode block_mean \
    --sparsity 0.4

# 2.2 slimgpt_mean mode
echo "  Running slimgpt_mean mode..."
python sobs/var_per_stage_analysis.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --layer_idx ${LAYER_IDX} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_BASE}/exp2_head_importance/slimgpt_mean \
    --hessian_mode block_diagonal \
    --head_importance_mode slimgpt_mean \
    --sparsity 0.4

echo -e "${GREEN}Experiment 2 Complete${NC}"
echo ""

################################################################################
# Experiment 3: Sparsity Sweep
################################################################################

echo -e "${GREEN}[Experiment 3/5] Sparsity Sweep${NC}"
echo "Testing different sparsity levels: 0.2, 0.3, 0.4, 0.5, 0.6"
echo ""

for SPARSITY in 0.2 0.3 0.4 0.5 0.6; do
    echo "  Running sparsity=${SPARSITY}..."
    python sobs/var_per_stage_analysis.py \
        --model_path ${MODEL_PATH} \
        --data_dir ${DATA_DIR} \
        --layer_idx ${LAYER_IDX} \
        --num_samples ${NUM_SAMPLES} \
        --output_dir ${OUTPUT_BASE}/exp3_sparsity_sweep/sparsity_${SPARSITY} \
        --hessian_mode block_diagonal \
        --head_importance_mode block_mean \
        --sparsity ${SPARSITY}
done

echo -e "${GREEN}Experiment 3 Complete${NC}"
echo ""

################################################################################
# Experiment 4: Layer-wise Analysis
################################################################################

echo -e "${GREEN}[Experiment 4/5] Layer-wise Analysis${NC}"
echo "Analyzing different layers: 0, 4, 8, 12, 15"
echo ""

for LAYER in 0 4 8 12 15; do
    echo "  Running layer=${LAYER}..."
    python sobs/var_per_stage_analysis.py \
        --model_path ${MODEL_PATH} \
        --data_dir ${DATA_DIR} \
        --layer_idx ${LAYER} \
        --num_samples ${NUM_SAMPLES} \
        --output_dir ${OUTPUT_BASE}/exp4_layer_wise/layer_${LAYER} \
        --hessian_mode block_diagonal \
        --head_importance_mode block_mean \
        --sparsity 0.4
done

echo -e "${GREEN}Experiment 4 Complete${NC}"
echo ""

################################################################################
# Experiment 5: Per-Stage Detailed Analysis
################################################################################

echo -e "${GREEN}[Experiment 5/5] Per-Stage Detailed Analysis${NC}"
echo "Running comprehensive per-stage analysis with full statistics"
echo ""

# Run with higher sample count for more reliable statistics
python sobs/var_per_stage_analysis.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --layer_idx 0 \
    --num_samples 256 \
    --output_dir ${OUTPUT_BASE}/exp5_per_stage_detailed \
    --hessian_mode block_diagonal \
    --head_importance_mode block_mean \
    --sparsity 0.4

echo -e "${GREEN}Experiment 5 Complete${NC}"
echo ""

################################################################################
# Summary and Comparison
################################################################################

echo "=============================================="
echo -e "${YELLOW}All Ablation Experiments Complete!${NC}"
echo "=============================================="
echo ""
echo "Results saved to: ${OUTPUT_BASE}/"
echo ""
echo "To analyze results, use:"
echo "  python sobs/analyze_ablation_results.py --results_dir ${OUTPUT_BASE}"
echo ""
echo "Key files to check:"
echo "  - results.json: Raw numerical results"
echo "  - head_importance_by_stage.png: Heatmap visualization"
echo "  - stage_head_ranking_correlation.png: Correlation matrix"
echo "  - scale_specific_heads.json: Scale-specific findings"
echo ""

################################################################################
# Optional: Generate Comparison Report
################################################################################

echo -e "${YELLOW}Generating comparison report...${NC}"

cat > ${OUTPUT_BASE}/README.md << EOF
# VAR-d16 FastOBA Attention Pruning - Ablation Study Results

Generated: $(date)

## Experiments Conducted

### Experiment 1: Hessian Mode Comparison
- **Purpose**: Compare first-order (H=XX^T) vs second-order (FastOBA) Hessian
- **Configurations**:
  - SlimGPT: \`hessian_mode=slimgpt\`, \`head_importance_mode=slimgpt_mean\`
  - Block-Diagonal: \`hessian_mode=block_diagonal\`, \`head_importance_mode=block_mean\`
- **Output**: \`exp1_hessian_mode/\`

### Experiment 2: Head Importance Mode Comparison
- **Purpose**: Compare head evaluation methods
- **Configurations**:
  - block_mean: Average of block-diagonal H elements
  - slimgpt_mean: Average of per-channel OBS importance
- **Output**: \`exp2_head_importance/\`

### Experiment 3: Sparsity Sweep
- **Purpose**: Understand behavior at different pruning ratios
- **Sparsity levels**: 0.2, 0.3, 0.4, 0.5, 0.6
- **Output**: \`exp3_sparsity_sweep/\`

### Experiment 4: Layer-wise Analysis
- **Purpose**: Compare pruning behavior across model depth
- **Layers**: 0, 4, 8, 12, 15 (evenly spaced through VAR-d16)
- **Output**: \`exp4_layer_wise/\`

### Experiment 5: Per-Stage Detailed Analysis
- **Purpose**: Deep dive into scale-specific pruning preferences
- **Sample size**: 256 (2x standard)
- **Output**: \`exp5_per_stage_detailed/\`

## Key Metrics

### Head Ranking Consistency (Kendall's τ)
- **High consistency** (τ > 0.7): Scales agree on head importance
- **Low consistency** (τ < 0.3): Scales have different preferences

### Dimension Selection Overlap (Jaccard Similarity)
- **High overlap** (J > 0.6): Stable pruning decisions
- **Low overlap** (J < 0.3): Scale-dependent dimension selection

### Scale-Specific Heads
- Heads ranked high in one scale but low in others
- Indicates scale-specialized attention patterns

## Directory Structure

\`\`\`
${OUTPUT_BASE}/
├── exp1_hessian_mode/
│   ├── slimgpt/
│   │   ├── results.json
│   │   ├── head_importance_by_stage.png
│   │   └── stage_head_ranking_correlation.png
│   └── block_diagonal/
│       └── ...
├── exp2_head_importance/
│   ├── block_mean/
│   └── slimgpt_mean/
├── exp3_sparsity_sweep/
│   ├── sparsity_0.1/
│   ├── sparsity_0.2/
│   ├── sparsity_0.3/
│   ├── sparsity_0.4/
│   └── sparsity_0.5/
├── exp4_layer_wise/
│   ├── layer_0/
│   ├── layer_4/
│   ├── layer_8/
│   ├── layer_12/
│   └── layer_15/
└── exp5_per_stage_detailed/
    ├── results.json
    ├── scale_specific_heads.json
    └── visualizations/
\`\`\`

## Analysis Commands

\`\`\`bash
# Compare hessian modes
python sobs/compare_experiments.py \\
    --exp1 exp1_hessian_mode/slimgpt \\
    --exp2 exp1_hessian_mode/block_diagonal \\
    --output comparison_hessian_mode.pdf

# Analyze sparsity trends
python sobs/plot_sparsity_trends.py \\
    --input exp3_sparsity_sweep \\
    --output sparsity_trends.pdf

# Layer-wise heatmap
python sobs/plot_layer_heatmap.py \\
    --input exp4_layer_wise \\
    --output layer_heatmap.pdf
\`\`\`

## Expected Findings

### Hessian Mode
- Block-diagonal should show **higher consistency** across scales
- Better capture of attention structure (softmax Jacobian)

### Head Importance Mode
- block_mean: More suitable for head-wise pruning
- slimgpt_mean: Simpler, but may miss head-internal correlations

### Sparsity
- Low sparsity (0.1-0.2): All methods perform similarly
- High sparsity (0.4-0.5): Block-diagonal advantage becomes clear

### Layer Depth
- Early layers: More scale-specific heads (different scales need different features)
- Late layers: More consistent across scales (semantic features)

## Configuration Summary

- Model: VAR-d16 (16 layers, 16 heads, 1024 embed_dim, 64 head_dim)
- Dataset: ImageNet
- Samples: ${NUM_SAMPLES} (256 for exp5)
- Target Sparsity: 0.4 (40% pruning)
- FastOBA Order: 2 (Hessian)
- FastOBA Delta: 1.0

EOF

echo "Report generated: ${OUTPUT_BASE}/README.md"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}    Ablation Study Script Complete!        ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
