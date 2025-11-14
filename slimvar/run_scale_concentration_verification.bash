#!/bin/bash
# 运行Scale_mul与Attention集中度验证实验 (Verification 1)
# 验证scale_mul与attention concentration的关系

echo "================================"
echo "  Verification 1: Scale-Concentration"
echo "================================"
echo ""

# 配置
MODEL_DEPTH=16
NUM_SAMPLES=50
LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"  # 代表性层：低方差早期(0,2), 中方差过渡(5), 高方差峰值(7,11,12)

# 使用真实图片（NEW）
USE_IMAGES=true
IMAGENET_DIR="/home/project/ImageNet-1K"

# 模型路径
VAE_CKPT="/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth"
VAR_CKPT="/home/project/daily/AR/model_zoo/var_d${MODEL_DEPTH}.pth"

# 检查模型是否存在
if [ ! -f "$VAE_CKPT" ]; then
    echo "❌ Error: VAE checkpoint not found: $VAE_CKPT"
    exit 1
fi

if [ ! -f "$VAR_CKPT" ]; then
    echo "❌ Error: VAR checkpoint not found: $VAR_CKPT"
    exit 1
fi

echo "Model: VAR-d${MODEL_DEPTH}"
echo "Calibration samples: ${NUM_SAMPLES}"
echo "Test layers: ${LAYERS}"
echo "Using real images: ${USE_IMAGES}"
if [ "$USE_IMAGES" = true ]; then
    echo "ImageNet directory: ${IMAGENET_DIR}"
fi
echo ""

# 切换到slimvar目录
cd /home/project/real_prune/slimvar

# 运行验证
echo "Running Verification 1..."
echo ""

# 构建参数
PYTHON_ARGS="--model_depth ${MODEL_DEPTH} --num_samples ${NUM_SAMPLES} --layers ${LAYERS} --vae_ckpt ${VAE_CKPT} --var_ckpt ${VAR_CKPT}"

# 添加真实图片参数
if [ "$USE_IMAGES" = true ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --use_images --imagenet_dir ${IMAGENET_DIR}"
fi

python verify_scale_concentration.py ${PYTHON_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ Verification 1 completed successfully!"
    echo "================================"
    echo ""
    echo "Next steps:"
    echo "  1. Check results in: scale_concentration_verification/"
    echo "  2. Review visualizations for each layer"
    echo "  3. If ≥4/6 layers pass, proceed to Verification 2"
    echo ""
else
    echo ""
    echo "================================"
    echo "❌ Verification 1 failed with error code: $EXIT_CODE"
    echo "================================"
    echo ""
fi

exit $EXIT_CODE
