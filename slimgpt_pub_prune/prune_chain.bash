#!/bin/bash

# 设置参数变量
SPARSITY=0.4
SCALE_STRATEGY="natural"
PRUNE_METHOD="chain_kfac"
MAX_LAYER=16
NUM_SAMPLES=150

# 生成模型文件名
MODEL_NAME="var_d${MAX_LAYER}_${PRUNE_METHOD}_s${SPARSITY}_${SCALE_STRATEGY}_n${NUM_SAMPLES}.pth"

python prune_chain.py \
     --sparsity ${SPARSITY} \
     --scale_weight_strategy ${SCALE_STRATEGY} \
     --prune_method ${PRUNE_METHOD} \
     --maxlayer ${MAX_LAYER} \
     --num_samples ${NUM_SAMPLES} \
     --save_pruned_weights \
     --save_dir ./pruned_models \
     --model_name "${MODEL_NAME}"