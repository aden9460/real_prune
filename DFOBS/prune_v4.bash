#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 执行 Python 脚本
python3 /home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/prune_v4.py \
    --minlayer 0 \
    --maxlayer 16 \
    --num_samples 15 \
    --percdamp 1e-2 \
    --prune_method slimgpt \
    --sparsity 0.2 \
    --specific_layer 256 \
    --model_name prune_d16_0.2sparsity_150i_256eva_scale.pth
