#!/bin/bash
#
# PyTorch Quantization Script for VAR Model
#
# Usage:
#   bash run_gptq_quantization.bash [int4|int8] [depth] [sparsity] [method] [model_path] [data_path]
#
# Examples:
#   bash run_gptq_quantization.bash int4                    # Use linear quantization (default)
#   bash run_gptq_quantization.bash int8 16 0.4 linear       # Explicit linear
#   bash run_gptq_quantization.bash int4 16 0.4 gptq         # Try GPTQ (will fallback if not available)
#   bash run_gptq_quantization.bash int4 16 0.4 linear /path/to/model.pth
#

set -e  # Exit on error

# ====================== Configuration ======================

# Parse arguments
WEIGHT_DTYPE=${1:-"int8"}  # Default: int8
MODEL_DEPTH=${2:-16}       # Default: 16
SPARSITY=${3:-0.4}         # Default: 0.4
METHOD=${4:-"linear"}      # Default: linear (can be "gptq" or "linear")
VAR_MODEL=${5:-"/home/project/daily/AR/model_zoo/d16_0.4_distill.pth"}
DATA_PATH=${6:-"/home/project/ImageNet-1K"}

# Validate weight dtype
if [[ "$WEIGHT_DTYPE" != "int4" && "$WEIGHT_DTYPE" != "int8" ]]; then
    echo "Error: WEIGHT_DTYPE must be 'int4' or 'int8'"
    echo "Usage: bash $0 [int4|int8] [depth] [sparsity] [method] [model_path] [data_path]"
    exit 1
fi

# Validate method
if [[ "$METHOD" != "gptq" && "$METHOD" != "linear" ]]; then
    echo "Error: METHOD must be 'gptq' or 'linear'"
    echo "Usage: bash $0 [int4|int8] [depth] [sparsity] [method] [model_path] [data_path]"
    exit 1
fi

# Validate model depth
if [[ ! "$MODEL_DEPTH" =~ ^(12|16|20|24|30)$ ]]; then
    echo "Error: MODEL_DEPTH must be one of: 12, 16, 20, 24, 30"
    exit 1
fi

# Check if model exists
if [[ ! -f "$VAR_MODEL" ]]; then
    echo "Error: VAR model not found: $VAR_MODEL"
    echo "Please specify the correct model path"
    exit 1
fi

# ====================== Display Configuration ======================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         PyTorch Quantization for VAR Model                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  • Quantization type: ${WEIGHT_DTYPE}"
echo "  • Quantization method: ${METHOD}"
if [[ "$METHOD" == "gptq" ]]; then
    echo "    (Note: GPTQ will fallback to Linear if not available)"
fi
echo "  • Model depth:       ${MODEL_DEPTH}"
echo "  • Sparsity ratio:    ${SPARSITY}"
echo "  • Model path:        ${VAR_MODEL}"
if [[ "$METHOD" == "gptq" ]]; then
    if [[ -n "$DATA_PATH" && -d "$DATA_PATH" ]]; then
        echo "  • Data path:         ${DATA_PATH}"
    else
        echo "  • Data path:         Not found (will use synthetic)"
    fi
fi
echo ""

# ====================== Update Script Configuration ======================

# Create temporary script with updated configuration
TEMP_SCRIPT="./quantize_gptq_pytorch_temp.py"
cp quantize_gptq_pytorch.py "$TEMP_SCRIPT"

# Update configuration in temporary script
sed -i "s/WEIGHT_DTYPE = \"int4\"/WEIGHT_DTYPE = \"${WEIGHT_DTYPE}\"/" "$TEMP_SCRIPT"

echo "✓ Script configuration updated"
echo ""

# ====================== Run GPTQ Quantization ======================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Starting GPTQ Quantization...                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Build command
CMD="python $TEMP_SCRIPT --depth $MODEL_DEPTH --sparsity $SPARSITY --var_model $VAR_MODEL --data_path $DATA_PATH"

# Add use_gptq flag if method is gptq
if [[ "$METHOD" == "gptq" ]]; then
    CMD="$CMD --use_gptq"
fi

# Run quantization
echo "Running: $CMD"
echo ""
$CMD

# Cleanup
rm -f "$TEMP_SCRIPT"

# ====================== Summary ======================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         GPTQ Quantization Completed!                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Find the output model
OUTPUT_DIR="./pytorch_quantized_models"
OUTPUT_MODEL=$(find "$OUTPUT_DIR" -name "var_d${MODEL_DEPTH}*${WEIGHT_DTYPE}.pth" -type f | head -1)

if [[ -f "$OUTPUT_MODEL" ]]; then
    MODEL_SIZE=$(du -h "$OUTPUT_MODEL" | cut -f1)
    echo "✓ Quantized model saved:"
    echo "    Path: $OUTPUT_MODEL"
    echo "    Size: $MODEL_SIZE"
    echo ""
    echo "Next steps:"
    echo "  1. Test quantized model on Linux:"
    echo "       checkpoint = torch.load('$OUTPUT_MODEL')"
    echo "       model.load_state_dict(checkpoint['model_state_dict'])"
    echo ""
    echo "  2. Convert to CoreML:"
    echo "       python transfer.py --depth $MODEL_DEPTH --sparsity $SPARSITY --var_model $OUTPUT_MODEL"
    echo ""
else
    echo "⚠ Output model not found in $OUTPUT_DIR"
    echo "Please check the logs above for errors"
fi
