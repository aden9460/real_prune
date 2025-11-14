#!/bin/bash

# VAR D20 模型下载脚本
# URL: https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth

URL="https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth"
OUTPUT_FILE="/home/project/daily/AR/model_zoo/var_d20.pth"

echo "开始下载 VAR D20 模型..."
echo "URL: $URL"
echo "输出文件: $OUTPUT_FILE"
echo "===================="

# 检查wget是否可用
# if command -v wget &> /dev/null; then
#     echo "使用 wget 下载..."
#     wget --progress=bar --show-progress --continue "$URL" -O "$OUTPUT_FILE"
if command -v curl &> /dev/null; then
    echo "使用 curl 下载..."
    curl -L --progress-bar --continue-at - "$URL" -o "$OUTPUT_FILE"
else
    echo "错误: 未找到 wget 或 curl 命令"
    echo "请安装其中一个: sudo apt install wget 或 sudo apt install curl"
    exit 1
fi

# 检查下载是否成功
if [ $? -eq 0 ]; then
    echo "===================="
    echo "下载完成!"
    echo "文件大小: $(ls -lh $OUTPUT_FILE | awk '{print $5}')"
    echo "文件位置: $(pwd)/$OUTPUT_FILE"
else
    echo "===================="
    echo "下载失败!"
    exit 1
FID