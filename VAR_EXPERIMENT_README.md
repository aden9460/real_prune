# VAR 剪枝方法对比实验

## 🎯 实验目标
对比 Chain KFAC 和 SlimGPT 两种剪枝方法在不同参数下的表现

## 📊 实验设置

### 模型配置
- **模型**: VAR-16 (depth=16)
- **校准样本**: 150
- **测试图片类别**: 980 (volcano)

### 剪枝方法对比
1. **Chain KFAC** (12个模型)
   - Sparsity: 0.1, 0.2, 0.3, 0.4
   - Scale Strategies: natural, equal, sqrt
   - 特点: 链式剪枝 + KFAC-OBS + 多尺度处理

2. **SlimGPT** (4个模型)
   - Sparsity: 0.1, 0.2, 0.3, 0.4
   - 特点: 传统输入协方差剪枝

### 总计: 16个模型，16张对比图

## 🚀 快速开始

### 方法1: 一键运行完整实验
```bash
# 执行完整的训练+生成流程
cd /home/project/real_prune
chmod +x run_full_var_experiment.bash
./run_full_var_experiment.bash
```

### 方法2: 分步执行

#### 第一步：训练所有模型
```bash
cd /home/project/real_prune/slimgpt_pub_prune
chmod +x batch_train.bash
./batch_train.bash
```

#### 第二步：生成对比图片
```bash
cd /home/project/real_prune/VAR_FIDtest
chmod +x batch_image_generation.bash
./batch_image_generation.bash
```

## 📁 文件结构

```
/home/project/real_prune/
├── run_full_var_experiment.bash     # 🚀 完整实验流程
├── slimgpt_pub_prune/
│   ├── batch_train.bash             # 批量训练脚本
│   ├── prune_chain.py               # Chain KFAC剪枝实现
│   └── prune_v6.py                  # SlimGPT剪枝实现
└── VAR_FIDtest/
    ├── batch_image_generation.bash  # 批量图片生成
    ├── generate_single_image.py     # 单张图片生成
    ├── create_comparison_grid.py    # 图片拼接脚本
    ├── batch_models/                # 训练好的模型
    │   ├── chain_kfac/             # Chain KFAC模型 (12个)
    │   └── slimgpt/                # SlimGPT模型 (4个)
    └── output/comparison_images/    # 生成的对比图片
        ├── all_methods_comparison.png      # 完整对比图
        ├── chain_kfac_comparison.png       # Chain KFAC对比
        ├── slimgpt_comparison.png          # SlimGPT对比
        └── *.png                          # 各个模型单张图
```

## 🖼️ 结果查看

### 主要对比图
- **完整对比**: `VAR_FIDtest/output/comparison_images/all_methods_comparison.png`
- **Chain KFAC**: `VAR_FIDtest/output/comparison_images/chain_kfac_comparison.png`
- **SlimGPT**: `VAR_FIDtest/output/comparison_images/slimgpt_comparison.png`

### 图片布局说明
- **4列布局**: 对应 4种 sparsity (0.1, 0.2, 0.3, 0.4)
- **Chain KFAC**: 3行对应 3种策略 (natural, equal, sqrt)
- **SlimGPT**: 1行对应传统方法

## 📋 分析要点

1. **横向对比**: 同一方法下不同sparsity的效果变化
2. **纵向对比**: 不同方法在相同sparsity下的表现差异
3. **策略对比**: Chain KFAC三种scale策略的视觉效果差异
4. **方法对比**: Chain KFAC vs SlimGPT的整体性能差异

## 🔧 自定义参数

### 修改实验参数
编辑 `/home/project/real_prune/slimgpt_pub_prune/batch_train.bash`:
```bash
# 可修改的参数
SPARSITIES=(0.1 0.2 0.3 0.4)              # 剪枝率
SCALE_STRATEGIES=("natural" "equal" "sqrt") # Chain KFAC策略
MAX_LAYER=16                                # 模型深度
NUM_SAMPLES=150                             # 校准样本数
```

### 修改测试类别
编辑 `/home/project/real_prune/VAR_FIDtest/batch_image_generation.bash`:
```bash
# 修改class_label参数 (默认980=volcano)
--class_label 980
```

## ⚠️ 注意事项

1. **路径兼容性**: 模型在 `slimgpt_pub_prune` 训练，在 `VAR_FIDtest` 生成图片
2. **GPU内存**: 建议使用 `CUDA_VISIBLE_DEVICES=0` 限制GPU使用
3. **时间消耗**: 完整实验约需 1-2 小时（取决于硬件）
4. **存储空间**: 约需 5-10GB 存储空间（模型+图片）

## 🐛 常见问题

### 1. 模型文件不存在
```bash
# 检查模型是否训练成功
ls -la /home/project/real_prune/VAR_FIDtest/batch_models/*/
```

### 2. CUDA内存不足
```bash
# 减少batch size或使用CPU
export CUDA_VISIBLE_DEVICES=0
```

### 3. 图片生成失败
```bash
# 检查VAE权重路径
ls -la /home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth
```

## 📈 实验报告模板

实验完成后会自动生成:
- `VAR_FIDtest/output/comparison_images/layout_description.md`

包含详细的实验设置、结果分析和布局说明。