#!/bin/bash
# Basic VAR Pruning Test Script

echo "======================================"
echo "VAR Basic Pruning Test"
echo "======================================"

cd /home/project/real_prune/slimvar

# Ensure directories exist
mkdir -p pruned_models

# Basic pruning with 20% sparsity
CUDA_VISIBLE_DEVICES=4 python model_slimming_basic.py \
    --model_depth 16 \
    --num_samples 2000 \
    --sparsity 0.2 \
    --minlayer 0 \
    --maxlayer 16 \
    --percdamp 0.01 \
    --save_dir ./pruned_models \
    --model_name qscaluefix_var_d16_0.2_2000sample_basic.pth \
    --seed 0 \
    --use_images \
    --imagenet_dir "/home/project/ImageNet-1K" \

echo ""
echo "======================================"
echo "Pruning Complete!"
echo "======================================"
echo "Check pruned model at: ./pruned_models/var_d16_s20_basic.pth"
