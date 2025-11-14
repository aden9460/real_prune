#!/bin/bash
# Basic VAR Pruning Test Script

echo "======================================"
echo "VAR Basic Pruning Test"
echo "======================================"

cd /home/project/real_prune/slimvar

# Ensure directories exist
# mkdir -p pruned_models

# Basic pruning with 20% sparsity
# CUDA_VISIBLE_DEVICES=0 python model_slimming_basic_v1.py \
#     --model_depth 16 \
#     --num_samples 1000 \
#     --sparsity 0.2 \
#     --minlayer 0 \
#     --maxlayer 16 \
#     --percdamp 0.01 \
#     --save_dir ./pruned_models \
#     --model_name qscaluefix_var_d16_0.2_1000sample_mag.pth \
#     --seed 0 \
#     --use_images \
#     --imagenet_dir "/home/project/ImageNet-1K" \
#     --prune_method magnitude


python model_slimming_basic_v1.py \
    --model_depth 16 \
    --sparsity 0.4 \
    --prune_method taylor \
    --taylor_type param_second \
    --num_taylor_samples 100 \
    --num_samples 1000 \
    --save_dir ./pruned_models \
    --model_name var_d16_0.4_llm-pruner.pth \
    --seed 0 \
    --use_images \
    --imagenet_dir "/home/project/ImageNet-1K" \
    # --use_selected_heads \
    # --selected_heads_json head_sharpness_images/pruning_plan_40pct.json
    # --non_uniform \
    # --non_uniform_strategy "log_increase" \


#  ['0.200', '0.280', '0.327', '0.360', '0.386', '0.407', '0.425', '0.440', '0.454', '0.466', '0.477', '0.487', '0.496', '0.505', '0.513', '0.520']
#  ['0.520', '0.513', '0.505', '0.496', '0.487', '0.477', '0.466', '0.454', '0.440', '0.425', '0.407', '0.386', '0.360', '0.327', '0.280', '0.200']
#  ['0.480', '0.469', '0.457', '0.444', '0.430', '0.415', '0.399', '0.380', '0.360', '0.337', '0.310', '0.279', '0.240', '0.190', '0.120', '0.000']
#  ['0.000', '0.120', '0.190', '0.240', '0.279', '0.310', '0.337', '0.360', '0.380', '0.399', '0.415', '0.430', '0.444', '0.457', '0.469', '0.480']