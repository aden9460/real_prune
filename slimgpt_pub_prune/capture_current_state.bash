#!/bin/bash

# 实时输出捕获脚本
# 保存完整的训练输出

LOG_DIR="/home/project/real_prune/VAR_FIDtest/training_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DETAILED_LOG="${LOG_DIR}/detailed_training_${TIMESTAMP}.log"

echo "========================================" | tee "${DETAILED_LOG}"
echo "详细训练日志" | tee -a "${DETAILED_LOG}"
echo "开始时间: $(date)" | tee -a "${DETAILED_LOG}"
echo "日志文件: ${DETAILED_LOG}" | tee -a "${DETAILED_LOG}"
echo "========================================" | tee -a "${DETAILED_LOG}"
echo "" | tee -a "${DETAILED_LOG}"

# 记录当前训练进度快照
echo "当前训练快照 ($(date)):" | tee -a "${DETAILED_LOG}"
echo "----------------------------------------" | tee -a "${DETAILED_LOG}"

# 检查已完成的模型
echo "已完成的模型:" | tee -a "${DETAILED_LOG}"
ls -lh /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac/*.pth 2>/dev/null | tee -a "${DETAILED_LOG}"
ls -lh /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt/*.pth 2>/dev/null | tee -a "${DETAILED_LOG}"

CHAIN_COUNT=$(ls /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac/*.pth 2>/dev/null | wc -l)
SLIM_COUNT=$(ls /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt/*.pth 2>/dev/null | wc -l)
TOTAL=$((CHAIN_COUNT + SLIM_COUNT))

echo "" | tee -a "${DETAILED_LOG}"
echo "统计: Chain KFAC=${CHAIN_COUNT}, SlimGPT=${SLIM_COUNT}, 总计=${TOTAL}/16" | tee -a "${DETAILED_LOG}"
echo "" | tee -a "${DETAILED_LOG}"

# GPU状态
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 状态:" | tee -a "${DETAILED_LOG}"
    nvidia-smi | tee -a "${DETAILED_LOG}"
    echo "" | tee -a "${DETAILED_LOG}"
fi

# 正在运行的进程
echo "正在运行的训练进程:" | tee -a "${DETAILED_LOG}"
ps aux | grep -E "(prune_chain|prune_v6|batch_train)" | grep -v grep | tee -a "${DETAILED_LOG}"

echo "" | tee -a "${DETAILED_LOG}"
echo "========================================" | tee -a "${DETAILED_LOG}"
echo "日志已保存到: ${DETAILED_LOG}"
echo "========================================" | tee -a "${DETAILED_LOG}"
