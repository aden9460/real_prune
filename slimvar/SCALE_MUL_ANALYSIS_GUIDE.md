# Scale_Mul分析工具使用指南

## 快速开始

### 运行分析

```bash
cd /home/project/real_prune/slimvar

# 方式1: 使用bash脚本（推荐）
bash run_scale_analysis.bash

# 方式2: 直接运行Python脚本
python analyze_scale_mul.py --model_depth 16 --output_dir ./scale_mul_analysis_d16
```

### 输出文件

运行后会在`scale_mul_analysis_d16/`目录下生成：

#### 数据文件
- `scale_mul_analysis.json` - 完整统计数据（均值、方差、分类等）
- `pruning_strategy.json` - 基于scale_mul推荐的剪枝策略
- `scale_mul_matrix.npy` - 原始scale_mul矩阵 (num_layers × num_heads)

#### 可视化图片
1. `scale_mul_heatmap.png` - **热力图**: 所有层所有head的scale_mul（对数scale）
2. `scale_mul_boxplot.png` - **箱线图**: 每层的scale_mul分布范围
3. `scale_mul_classification.png` - **分类图**: 高/中/低scale head的比例
4. `scale_mul_trends.png` - **趋势图**: 均值、方差等统计量的层级变化
5. `scale_mul_histogram.png` - **直方图**: 全局scale_mul分布

---

## 分析内容

### 1. 全局统计

- 所有heads的scale_mul均值、方差、范围
- High scale (>50) / Mid scale (5-50) / Low scale (<5) 的比例
- 总体分布特征

### 2. 层级分析

每层的统计信息：
- 均值、中位数、标准差、方差
- 最小值、最大值、四分位数
- 高/中/低scale head的数量和比例
- Head分化程度（方差）

### 3. 剪枝策略推荐

基于scale_mul分布自动推荐：

#### 逐层剪枝率
根据每层的特征决定剪枝率：
- **高专家比例层** (>40% high-scale heads) → 保守剪枝 (0.15)
- **高通才比例层** (>30% low-scale heads) → 激进剪枝 (0.35)
- **高方差层** (variance >20) → 可剪枝冗余heads (0.30)
- **低方差层** (variance <10) → 难区分，保守剪枝 (0.20)

#### 三尺度组策略
将16层分为3组，对应VAR的3个尺度阶段：
- **Early Group** (Layers 0-5): 处理早期尺度 (scale 0-3)
- **Mid Group** (Layers 6-11): 处理中期尺度 (scale 3-7)
- **Late Group** (Layers 12-15): 处理后期尺度 (scale 7-10)

每组给出：
- 平均方差（head分化程度）
- 专家型head比例
- 推荐剪枝率

---

## 使用场景

### 场景1: 理解VAR学到的注意力模式

```bash
# 运行分析
bash run_scale_analysis.bash

# 查看热力图 - 观察哪些层哪些head的scale_mul高
# scale_mul高 → 注意力集中 → 精确定位
# scale_mul低 → 注意力分散 → 全局整合
```

**问题**:
- 早期层vs后期层的scale_mul有何不同？
- 不同head之间的scale_mul差异大吗？
- 哪些层的head功能分化更明显？

### 场景2: 数据驱动的剪枝率设计

```bash
# 运行分析
python analyze_scale_mul.py --model_depth 16

# 读取推荐策略
cat scale_mul_analysis_d16/pruning_strategy.json

# 在prune_v6.py中使用推荐的剪枝率
# --sparsity参数改为逐层sparsity列表
```

**使用策略文件**:
```python
import json

# 加载推荐策略
with open('scale_mul_analysis_d16/pruning_strategy.json', 'r') as f:
    strategy = json.load(f)

# 获取逐层剪枝率
layer_sparsities = [s['sparsity'] for s in strategy['per_layer_sparsity']]

# 应用到prune_v6.py
args.sparsity = layer_sparsities  # 传入列表而非单一值
```

### 场景3: 三尺度渐进剪枝

```python
# 读取三尺度组策略
strategy = json.load(open('scale_mul_analysis_d16/pruning_strategy.json'))

for group in strategy['scale_groups']:
    print(f"{group['group']} group:")
    print(f"  Layers: {group['layers']}")
    print(f"  Recommended sparsity: {group['recommended_sparsity']:.2%}")

# 实现渐进剪枝
# Stage 1: 剪枝 Early Group (layers 0-5) 使用推荐sparsity
# Stage 2: 剪枝 Mid Group (layers 6-11)
# Stage 3: 剪枝 Late Group (layers 12-15)
```

---

## 理解输出

### 示例：分析报告解读

```
📊 Global Statistics:
  Total Heads: 256              # 16层 × 16头
  Mean Scale_Mul: 18.5          # 全局平均scale
  Range: [1.2, 95.3]            # 最小到最大

🎯 Head Classification:
  High Scale (>50):  45 heads (17.6%)   # 专家型heads
  Mid Scale (5-50):  180 heads (70.3%)  # 平衡型heads
  Low Scale (<5):    31 heads (12.1%)   # 通才型heads

📈 Layer-wise Trends:
  High Variance Layers (>20): [3, 7, 11, 14]  # 这些层head分化明显
  Low Variance Layers (<10):  [0, 1, 15]      # 这些层head功能相似
```

**解读**:
- 17.6%的heads是专家型（精确定位）→ 需要保留
- 12.1%的heads是通才型（全局整合）→ 可能冗余，可以剪枝
- Layers 3, 7, 11, 14的head分化明显 → 可以剪掉低scale的heads
- Layers 0, 1, 15的head功能相似 → 难以区分重要性，保守剪枝

---

## 高级用法

### 对比不同模型深度

```bash
# 分析d16
python analyze_scale_mul.py --model_depth 16 --output_dir analysis_d16

# 分析d20
python analyze_scale_mul.py --model_depth 20 --output_dir analysis_d20

# 分析d24
python analyze_scale_mul.py --model_depth 24 --output_dir analysis_d24

# 对比：更深的模型scale_mul分布有何不同？
```

### 自定义剪枝策略

修改`recommend_pruning_strategy()`函数：

```python
# 在analyze_scale_mul.py中修改决策逻辑

if high_ratio > 0.4:
    sparsity = base_sparsity * 0.5  # 改为更保守
elif low_ratio > 0.5:  # 改为更激进的阈值
    sparsity = base_sparsity * 1.8
# ...
```

### 加载已有的scale_mul矩阵

```python
import numpy as np

# 加载保存的矩阵
scale_mul = np.load('scale_mul_analysis_d16/scale_mul_matrix.npy')

# 手动分析特定层
layer_5_scales = scale_mul[5]
print(f"Layer 5 scale_mul: {layer_5_scales}")
print(f"High-scale heads: {(layer_5_scales > 50).sum()}")
```

---

## 常见问题

### Q1: 为什么使用log scale显示？

A: scale_mul的范围很大（1-100），对数scale可以更清晰地显示差异。

### Q2: High/Mid/Low scale的阈值是如何确定的？

A:
- Low (<5): 接近标准scale (1/√64 ≈ 0.125的倒数≈8)，注意力相对分散
- Mid (5-50): 中等锐化
- High (>50): 强烈锐化，接近one-hot

可以根据实际分布调整。

### Q3: 推荐的剪枝率是否一定准确？

A: 这只是基于scale_mul分布的启发式策略，建议：
1. 先运行分析了解分布
2. 小规模实验验证推荐的剪枝率
3. 根据FID结果微调

### Q4: 如何判断哪个策略更好？

A: 对比实验：
1. 均匀剪枝（baseline）
2. Scale_mul引导的逐层剪枝
3. 三尺度组渐进剪枝

比较FID、IS、参数量、推理速度。

---

## 下一步

1. ✅ 运行分析，理解scale_mul分布
2. ✅ 查看可视化，识别模式
3. ✅ 读取推荐策略
4. ⬜ 在prune_v6.py中实现scale_mul引导剪枝
5. ⬜ 对比实验验证效果

---

## 相关文档

- `PRUNE_V6_DEEP_ANALYSIS.md` - Prune_v6详细技术分析
- `VAR_VS_LLM_PRUNING_INNOVATION_ANALYSIS.md` - VAR剪枝创新点分析
- `pruning_strategy.json` - 生成的剪枝策略配置
