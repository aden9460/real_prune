#!/bin/bash

# 训练监控脚本 - 持续保存训练输出到日志文件
# 每30秒检查一次训练进度并保存到日志

LOG_DIR="/home/project/real_prune/VAR_FIDtest/training_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_training_${TIMESTAMP}.log"

echo "========================================" | tee -a "${LOG_FILE}"
echo "训练监控开始时间: $(date)" | tee -a "${LOG_FILE}"
echo "日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 查找正在运行的训练进程
TRAIN_PID=$(ps aux | grep "batch_train_sequential.bash" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "${TRAIN_PID}" ]; then
    echo "错误: 未找到运行中的训练进程" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "找到训练进程 PID: ${TRAIN_PID}" | tee -a "${LOG_FILE}"
echo "开始监控..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 持续监控直到进程结束
COUNTER=0
while kill -0 ${TRAIN_PID} 2>/dev/null; do
    COUNTER=$((COUNTER + 1))

    echo "========================================" >> "${LOG_FILE}"
    echo "检查点 #${COUNTER} - $(date)" >> "${LOG_FILE}"
    echo "========================================" >> "${LOG_FILE}"

    # 检查已完成的模型
    CHAIN_KFAC_COUNT=$(ls /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac/*.pth 2>/dev/null | wc -l)
    SLIMGPT_COUNT=$(ls /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt/*.pth 2>/dev/null | wc -l)
    TOTAL_COUNT=$((CHAIN_KFAC_COUNT + SLIMGPT_COUNT))

    echo "已完成模型数: ${TOTAL_COUNT}/16 (Chain KFAC: ${CHAIN_KFAC_COUNT}, SlimGPT: ${SLIMGPT_COUNT})" | tee -a "${LOG_FILE}"

    # 检查当前GPU使用情况
    if command -v nvidia-smi &> /dev/null; then
        echo "" >> "${LOG_FILE}"
        echo "GPU 状态:" >> "${LOG_FILE}"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "${LOG_FILE}" 2>&1
    fi

    echo "" >> "${LOG_FILE}"

    # 列出已生成的模型
    if [ ${TOTAL_COUNT} -gt 0 ]; then
        echo "已生成的模型文件:" >> "${LOG_FILE}"
        ls -lh /home/project/real_prune/VAR_FIDtest/batch_models/chain_kfac/*.pth 2>/dev/null >> "${LOG_FILE}"
        ls -lh /home/project/real_prune/VAR_FIDtest/batch_models/slimgpt/*.pth 2>/dev/null >> "${LOG_FILE}"
    fi

    echo "" >> "${LOG_FILE}"

    # 等待30秒
    sleep 30
done

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "训练进程已完成: $(date)" | tee -a "${LOG_FILE}"
echo "最终统计:" | tee -a "${LOG_FILE}"
echo "  Chain KFAC 模型: ${CHAIN_KFAC_COUNT}" | tee -a "${LOG_FILE}"
echo "  SlimGPT 模型: ${SLIMGPT_COUNT}" | tee -a "${LOG_FILE}"
echo "  总计: ${TOTAL_COUNT}/16" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"
