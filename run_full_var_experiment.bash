#!/bin/bash

# 完整的 VAR 剪枝对比实验流程
# 1. 在 slimgpt_pub_prune 中训练模型
# 2. 在 VAR_FIDtest 中生成对比图片

echo "=========================================="
echo "VAR 剪枝方法完整对比实验"
echo "=========================================="

# 设置路径
PRUNE_DIR="/home/project/real_prune/slimgpt_pub_prune"
TEST_DIR="/home/project/real_prune/VAR_FIDtest"

echo "实验目录:"
echo "  剪枝训练目录: ${PRUNE_DIR}"
echo "  图片生成目录: ${TEST_DIR}"
echo ""

# 第一步：训练所有模型
echo "第一步：开始模型训练..."
cd "${PRUNE_DIR}"

# 确保脚本可执行
chmod +x batch_train.bash

# 执行批量训练
./batch_train.bash

if [ $? -ne 0 ]; then
    echo "❌ 模型训练失败，退出"
    exit 1
fi

echo "✅ 模型训练完成"
echo ""

# 检查生成的模型
echo "检查生成的模型文件..."
chain_count=$(find "${TEST_DIR}/batch_models/chain_kfac" -name "*.pth" 2>/dev/null | wc -l)
slim_count=$(find "${TEST_DIR}/batch_models/slimgpt" -name "*.pth" 2>/dev/null | wc -l)

echo "Chain KFAC 模型: ${chain_count} 个"
echo "SlimGPT 模型: ${slim_count} 个"
echo "总计: $((chain_count + slim_count)) 个模型"

if [ $((chain_count + slim_count)) -eq 0 ]; then
    echo "❌ 没有找到训练好的模型文件"
    exit 1
fi

echo ""

# 第二步：生成对比图片
echo "第二步：生成对比图片..."
cd "${TEST_DIR}"

# 确保脚本可执行
chmod +x batch_image_generation.bash

# 执行图片生成
./batch_image_generation.bash

if [ $? -ne 0 ]; then
    echo "❌ 图片生成失败"
    exit 1
fi

echo "✅ 图片生成完成"
echo ""

# 第三步：显示结果
echo "第三步：实验结果总结..."

# 统计结果
image_count=$(find "${TEST_DIR}/output/comparison_images" -name "*.png" -not -name "*comparison*" 2>/dev/null | wc -l)
comparison_count=$(find "${TEST_DIR}/output/comparison_images" -name "*comparison*.png" 2>/dev/null | wc -l)

echo "生成的单张图片: ${image_count} 张"
echo "生成的对比拼接图: ${comparison_count} 张"
echo ""

echo "=========================================="
echo "实验完成！"
echo "=========================================="

echo "📁 生成的文件:"
echo ""

echo "🔧 训练好的模型:"
if [ ${chain_count} -gt 0 ]; then
    echo "  Chain KFAC 模型: ${TEST_DIR}/batch_models/chain_kfac/"
    ls -la "${TEST_DIR}/batch_models/chain_kfac/" | head -5
    if [ ${chain_count} -gt 5 ]; then
        echo "  ... 以及其他 $((chain_count - 5)) 个模型"
    fi
fi

if [ ${slim_count} -gt 0 ]; then
    echo "  SlimGPT 模型: ${TEST_DIR}/batch_models/slimgpt/"
    ls -la "${TEST_DIR}/batch_models/slimgpt/"
fi

echo ""

echo "🖼️  生成的对比图片:"
echo "  单张图片: ${TEST_DIR}/output/comparison_images/"
echo "  主要对比图:"

if [ -f "${TEST_DIR}/output/comparison_images/all_methods_comparison.png" ]; then
    echo "    ✅ 完整对比图: all_methods_comparison.png"
else
    echo "    ❌ 完整对比图未生成"
fi

if [ -f "${TEST_DIR}/output/comparison_images/chain_kfac_comparison.png" ]; then
    echo "    ✅ Chain KFAC对比: chain_kfac_comparison.png"
else
    echo "    ❌ Chain KFAC对比图未生成"
fi

if [ -f "${TEST_DIR}/output/comparison_images/slimgpt_comparison.png" ]; then
    echo "    ✅ SlimGPT对比: slimgpt_comparison.png"
else
    echo "    ❌ SlimGPT对比图未生成"
fi

echo ""

echo "📋 查看结果:"
echo "  cd ${TEST_DIR}/output/comparison_images"
echo "  # 查看完整对比图"
echo "  xdg-open all_methods_comparison.png"
echo ""

echo "🎯 实验设置总结:"
echo "  - 模型: VAR-16"
echo "  - Sparsity: 0.1, 0.2, 0.3, 0.4"
echo "  - Chain KFAC Strategies: natural, equal, sqrt"
echo "  - 校准样本: 150"
echo "  - 测试类别: 980 (volcano)"
echo ""

echo "实验完成时间: $(date)"