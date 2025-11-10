#!/bin/bash

# 完整的实验流程脚本
# 1. 批量训练所有模型
# 2. 生成对比图片
# 3. 创建结果报告

echo "=========================================="
echo "VAR 剪枝方法对比实验 - 完整流程"
echo "=========================================="

# 设置实验参数
export CUDA_VISIBLE_DEVICES=0  # 使用GPU 0

echo "实验配置:"
echo "- Chain KFAC: 4 sparsity × 3 strategies = 12 models"
echo "- SlimGPT: 4 sparsity = 4 models"
echo "- 总计: 16 models, 16 images"
echo ""

# 第一步：批量训练
echo "第一步：开始批量训练..."
chmod +x batch_train.bash
./batch_train.bash

if [ $? -ne 0 ]; then
    echo "❌ 批量训练失败，退出"
    exit 1
fi

echo "✅ 批量训练完成"
echo ""

# 检查生成的模型数量
echo "检查生成的模型..."
chain_models=$(find ./batch_models/chain_kfac -name "*.pth" | wc -l)
slim_models=$(find ./batch_models/slimgpt -name "*.pth" | wc -l)

echo "Chain KFAC 模型: ${chain_models} 个"
echo "SlimGPT 模型: ${slim_models} 个"
echo "总计: $((chain_models + slim_models)) 个模型"
echo ""

# 第二步：生成对比图片
echo "第二步：生成对比图片..."
python generate_comparison.py

if [ $? -ne 0 ]; then
    echo "❌ 图片生成失败"
    exit 1
fi

echo "✅ 图片生成完成"
echo ""

# 第三步：创建结果总结
echo "第三步：创建结果总结..."

cat > experiment_report.md << 'EOF'
# VAR 剪枝方法对比实验报告

## 实验配置
- **模型**: VAR-16 (depth=16)
- **校准样本**: 150
- **测试图片类别**: 980 (volcano)

## 实验方法

### 1. Chain KFAC (12个模型)
- **Sparsity**: 0.1, 0.2, 0.3, 0.4
- **Scale Strategies**: natural, equal, sqrt
- **特点**: 链式剪枝 + KFAC-OBS + 多尺度处理

### 2. SlimGPT (4个模型)
- **Sparsity**: 0.1, 0.2, 0.3, 0.4
- **特点**: 传统的输入协方差剪枝

## 生成的文件

### 模型文件
- `batch_models/chain_kfac/`: 12个Chain KFAC模型
- `batch_models/slimgpt/`: 4个SlimGPT模型

### 对比图片
- `comparison_images/all_models_comparison.png`: 所有模型拼接图
- `comparison_images/organized_comparison.png`: 按方法组织的布局图
- `comparison_images/*.png`: 各个模型的单独生成图

## 分析要点

1. **Scale Strategy比较**:
   - Natural: 符合VAR实际token分布
   - Equal: 所有尺度平等权重
   - Sqrt: 平衡大小尺度

2. **Sparsity影响**: 观察不同剪枝率对图像质量的影响

3. **方法对比**: Chain KFAC vs SlimGPT 在不同剪枝率下的表现

## 使用说明

查看对比图片:
```bash
# 查看所有模型对比
xdg-open comparison_images/all_models_comparison.png

# 查看组织化布局
xdg-open comparison_images/organized_comparison.png
```
EOF

echo "✅ 实验报告创建完成: experiment_report.md"
echo ""

# 显示结果
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "生成的文件:"
echo "📁 Models:"
ls -la batch_models/*/
echo ""
echo "🖼️  Images:"
ls -la comparison_images/
echo ""
echo "📋 Report: experiment_report.md"
echo ""
echo "查看对比图片:"
echo "comparison_images/all_models_comparison.png"
echo "comparison_images/organized_comparison.png"