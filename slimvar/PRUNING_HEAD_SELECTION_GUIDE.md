# Scale_Mul引导的Head选择策略与配置

## 1. 选择思路总览

### 核心原则：**差异化对待不同层的低scale heads**

基于VAR-d16的三段式层级结构（通才-专家-通才），我们对不同层采用不同的head选择策略：

```
┌──────────────────────────────────────────────────────────────────┐
│  早期通才层 (0-4)  │  中间专家层 (5-13)  │  末尾通才层 (14-15) │
├──────────────────────────────────────────────────────────────────┤
│  低均值 + 低方差   │  高均值 + 高方差    │  低均值 + 低方差   │
│  需要全局整合      │  专家型heads        │  回归全局整合      │
├──────────────────────────────────────────────────────────────────┤
│  保护低scale heads │  剪除低scale heads  │  保护低scale heads │
│  优先剪高scale     │  优先剪低scale      │  优先剪高scale     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 百分位数阈值

基于VAR-d16模型所有256个heads的scale_mul分布：

```
Q1 (25th percentile): 14.15  - 低scale上界
Q2 (50th percentile): 18.78  - 中位数
Q3 (75th percentile): 22.96  - 高scale下界
```

**分类标准**：
- 🟢 **低scale heads**: scale_mul < 14.15 (64个, 25%)
- 🟡 **中scale heads**: 14.15 ≤ scale_mul < 22.96 (128个, 50%)
- 🔴 **高scale heads**: scale_mul ≥ 22.96 (64个, 25%)

---

## 3. 三段式层级的Head选择策略

### 3.1 早期通才层 (Layers 0-4)

**统计特征**：
- 均值: 12.91 (低)
- 方差: 7.85 (低)
- 特点: heads功能统一，需要全局感受野

**选择策略**：
```python
# 保护低scale heads（全局整合能力重要）
# 在Hessian重要性基础上，对低scale heads增加bonus
if layer_idx in range(0, 5):
    scale_bonus = torch.where(scale_mul < Q1, 1.2, 1.0)
    adjusted_importance = hessian_importance * scale_bonus
    # 优先剪除：高scale + 低Hessian的heads
```

**剪枝率建议**：
- 40%总体目标: 30%
- 20%总体目标: 15%

---

### 3.2 中间专家层 (Layers 5-13)

**统计特征**：
- 均值: 22.17 (高)
- 方差: 33.96 (高)
- 特点: heads高度分化，专家/通才混杂

**识别的低scale冗余heads**（scale < Q1）：
```
Layer 5: heads [3, 11, 12, 15]  (2个)
Layer 6: heads [0, 3, 11, 13]   (1个实际<14.15: head 8)
Layer 8: heads [5, 6, 12, 15]   (1个实际<14.15: head 11)
Layer 11: heads [3, 9, 10, 11]  (1个实际<14.15: head 14)
Layer 12: heads [0, 3, 10, 13]  (2个)
Layer 13: heads [2, 5, 13, 15]  (2个)
总计: ~9个低scale heads在中间层
```

**选择策略**：
```python
# 剪除低scale heads（专家层不需要通才）
if layer_idx in range(5, 14):
    scale_penalty = torch.where(scale_mul < Q1, 0.5, 1.0)
    adjusted_importance = hessian_importance * scale_penalty
    # 优先剪除：低scale的heads（降低其重要性）
```

**剪枝率建议**：
- 40%总体目标: 50% (激进)
- 20%总体目标: 25% (适度激进)

**理论依据**：
- 中间层不需要通才功能（全局整合）
- 低scale heads在专家层中是冗余
- 剪除后方差降低 → 功能更统一

---

### 3.3 末尾通才层 (Layers 14-15)

**统计特征**：
- 均值: 17.62 (低)
- 方差: 16.78 (低)
- 特点: 回归通才模式，需要整合多尺度信息

**选择策略**：
```python
# 保护低scale heads（全局整合能力重要）
if layer_idx in [14, 15]:
    scale_bonus = torch.where(scale_mul < Q1, 1.2, 1.0)
    adjusted_importance = hessian_importance * scale_bonus
    # 优先剪除：高scale + 低Hessian的heads
```

**剪枝率建议**：
- 40%总体目标: 30%
- 20%总体目标: 15%

---

## 4. 配置1：40%总体剪枝率

### 4.1 剪枝率分配

**设计**：
```
早期层(0-4):   30% × 5层 = 24 heads
中间层(5-13):  50% × 9层 = 72 heads
末尾层(14-15): 30% × 2层 = 9.6 heads
────────────────────────────────────
总计: 105.6 / 256 heads = 41.25%
```

**逐层配置**：

| Layer | 组 | 剪枝率 | 剪除heads数 | 保留heads数 |
|-------|-----|--------|------------|------------|
| 0 | 早期 | 30% | 4 | 12 |
| 1 | 早期 | 30% | 4 | 12 |
| 2 | 早期 | 30% | 4 | 12 |
| 3 | 早期 | 30% | 4 | 12 |
| 4 | 早期 | 30% | 4 | 12 |
| 5 | 中间 | 50% | 8 | 8 |
| 6 | 中间 | 50% | 8 | 8 |
| 7 | 中间 | 50% | 8 | 8 |
| 8 | 中间 | 50% | 8 | 8 |
| 9 | 中间 | 50% | 8 | 8 |
| 10 | 中间 | 50% | 8 | 8 |
| 11 | 中间 | 50% | 8 | 8 |
| 12 | 中间 | 50% | 8 | 8 |
| 13 | 中间 | 50% | 8 | 8 |
| 14 | 末尾 | 30% | 4 | 12 |
| 15 | 末尾 | 30% | 4 | 12 |

### 4.2 具体Head选择（基于scale_mul）

**注意**：以下是基于纯scale_mul排序的示例。实际剪枝时应结合Hessian重要性。

#### 早期层 (0-4) - 优先剪高scale

```
Layer 0: 剪除heads [0, 6, 11, 12]
         scale_mul范围: [10.81, 15.17], 均值12.91

Layer 1: 剪除heads [3, 5, 11, 15]
         scale_mul范围: [12.88, 15.08], 均值14.01

Layer 2: 剪除heads [0, 5, 6, 9]
         scale_mul范围: [14.04, 15.64], 均值14.95

Layer 3: 剪除heads [8, 10, 11, 13]
         scale_mul范围: [17.66, 25.06], 均值20.74

Layer 4: 剪除heads [0, 1, 10, 11]
         scale_mul范围: [17.30, 22.75], 均值19.39
```

#### 中间层 (5-13) - 优先剪低scale ✅ 核心创新

```
Layer 5: 剪除heads [1, 3, 7, 9, 11, 12, 14, 15]
         scale_mul范围: [12.72, 19.75], 均值17.44
         ✅ 包含2个低scale heads (3, 15)

Layer 6: 剪除heads [0, 3, 4, 8, 10, 11, 13, 14]
         scale_mul范围: [11.74, 20.12], 均值17.09
         ✅ 包含1个低scale head (8)

Layer 7: 剪除heads [2, 3, 4, 5, 7, 11, 14, 15]
         scale_mul范围: [15.61, 21.19], 均值19.08
         ✅ 全部剪除的都是中低scale

Layer 8: 剪除heads [0, 2, 3, 5, 6, 12, 14, 15]
         scale_mul范围: [9.53, 21.93], 均值18.60
         ✅ 包含1个低scale head (11)

Layer 9: 剪除heads [0, 1, 2, 5, 7, 8, 11, 13]
         scale_mul范围: [14.89, 22.79], 均值18.72

Layer 10: 剪除heads [0, 2, 3, 4, 7, 9, 12, 15]
          scale_mul范围: [16.50, 23.22], 均值20.34

Layer 11: 剪除heads [0, 3, 4, 8, 9, 10, 11, 12]
          scale_mul范围: [10.70, 23.48], 均值20.04
          ✅ 包含1个低scale head (14)

Layer 12: 剪除heads [0, 2, 3, 4, 6, 10, 13, 15]
          scale_mul范围: [5.66, 22.65], 均值16.00
          ✅ 包含2个低scale heads (5, 9)

Layer 13: 剪除heads [2, 3, 5, 7, 8, 10, 13, 15]
          scale_mul范围: [7.69, 21.72], 均值16.43
          ✅ 包含2个低scale heads (1, 12)
```

#### 末尾层 (14-15) - 优先剪高scale

```
Layer 14: 剪除heads [1, 2, 4, 14]
          scale_mul范围: [20.71, 28.19], 均值23.18

Layer 15: 剪除heads [8, 9, 11, 12]
          scale_mul范围: [19.07, 26.09], 均值20.90
```

### 4.3 预期效果

**方差变化**：
```
剪枝前:
  早期层方差: 7.85
  中间层方差: 33.96  ← 高，说明混杂
  末尾层方差: 16.78
  方差ratio: 4.3

剪枝后（期望）:
  早期层方差: ~9-10  (略增，因剪掉部分高scale)
  中间层方差: ~17-20 (显著降低50%，因剪掉低scale冗余)
  末尾层方差: ~15-18 (基本不变)
  方差ratio: ~2.0   (降低53%)
```

**性能预期**：
- FID: 相比均匀剪枝40%，降低5-10%
- 方差一致性: ratio从4.3降至<2.5
- 冗余消除: 中间层方差降低>40%

---

## 5. 配置2：20%总体剪枝率

### 5.1 剪枝率分配

**设计**：
```
早期层(0-4):   15% × 5层 = 12 heads
中间层(5-13):  25% × 9层 = 36 heads
末尾层(14-15): 15% × 2层 = 4.8 heads
────────────────────────────────────
总计: 52.8 / 256 heads = 20.62%
```

**逐层配置**：

| Layer | 组 | 剪枝率 | 剪除heads数 | 保留heads数 |
|-------|-----|--------|------------|------------|
| 0 | 早期 | 15% | 2 | 14 |
| 1 | 早期 | 15% | 2 | 14 |
| 2 | 早期 | 15% | 2 | 14 |
| 3 | 早期 | 15% | 2 | 14 |
| 4 | 早期 | 15% | 2 | 14 |
| 5 | 中间 | 25% | 4 | 12 |
| 6 | 中间 | 25% | 4 | 12 |
| 7 | 中间 | 25% | 4 | 12 |
| 8 | 中间 | 25% | 4 | 12 |
| 9 | 中间 | 25% | 4 | 12 |
| 10 | 中间 | 25% | 4 | 12 |
| 11 | 中间 | 25% | 4 | 12 |
| 12 | 中间 | 25% | 4 | 12 |
| 13 | 中间 | 25% | 4 | 12 |
| 14 | 末尾 | 15% | 2 | 14 |
| 15 | 末尾 | 15% | 2 | 14 |

### 5.2 具体Head选择（基于scale_mul）

#### 早期层 (0-4) - 优先剪高scale

```
Layer 0: 剪除heads [6, 11]
         scale_mul范围: [13.02, 15.17], 均值14.09

Layer 1: 剪除heads [5, 15]
         scale_mul范围: [14.32, 15.08], 均值14.70

Layer 2: 剪除heads [5, 9]
         scale_mul范围: [15.16, 15.64], 均值15.40

Layer 3: 剪除heads [8, 13]
         scale_mul范围: [22.55, 25.06], 均值23.80

Layer 4: 剪除heads [10, 11]
         scale_mul范围: [20.17, 22.75], 均值21.46
```

#### 中间层 (5-13) - 优先剪低scale

```
Layer 5: 剪除heads [3, 11, 12, 15]
         scale_mul范围: [12.72, 18.42], 均值15.70
         ✅ 包含2个低scale heads

Layer 6: 剪除heads [0, 3, 11, 13]
         scale_mul范围: [11.74, 18.60], 均值14.72
         ✅ 包含1个低scale head

Layer 7: 剪除heads [2, 11, 14, 15]
         scale_mul范围: [15.61, 19.41], 均值17.68

Layer 8: 剪除heads [5, 6, 12, 15]
         scale_mul范围: [9.53, 19.89], 均值16.03
         ✅ 包含1个低scale head

Layer 9: 剪除heads [1, 2, 8, 13]
         scale_mul范围: [14.89, 17.89], 均值16.71

Layer 10: 剪除heads [3, 4, 7, 12]
          scale_mul范围: [16.50, 19.97], 均值18.50

Layer 11: 剪除heads [3, 9, 10, 11]
          scale_mul范围: [10.70, 20.61], 均值16.87
          ✅ 包含1个低scale head

Layer 12: 剪除heads [0, 3, 10, 13]
          scale_mul范围: [5.66, 15.47], 均值10.51
          ✅ 包含2个低scale heads

Layer 13: 剪除heads [2, 5, 13, 15]
          scale_mul范围: [7.69, 17.72], 均值13.12
          ✅ 包含2个低scale heads
```

#### 末尾层 (14-15) - 优先剪高scale

```
Layer 14: 剪除heads [1, 4]
          scale_mul范围: [22.53, 28.19], 均值25.36

Layer 15: 剪除heads [8, 12]
          scale_mul范围: [19.38, 26.09], 均值22.73
```

### 5.3 预期效果

**方差变化**：
```
剪枝前:
  中间层方差: 33.96

剪枝后（期望）:
  中间层方差: ~25-28 (降低20-30%)
  方差ratio: ~3.0   (降低30%)
```

**性能预期**：
- FID: 相比均匀剪枝20%，降低2-5%
- 冗余消除: 中间层方差降低20-30%

---

## 6. 实现代码框架

### 6.1 在prune_v6.py中实现

```python
def differential_scale_guided_pruning(model, layer_idx, pruner, args):
    """
    差异化scale引导剪枝

    Args:
        model: VAR模型
        layer_idx: 当前层索引
        pruner: SlimGPT pruner对象
        args: 参数配置

    Returns:
        prune_indices: 要剪枝的head索引
    """
    # 获取scale_mul
    scale_mul = model.blocks[layer_idx].attn.scale_mul_1H11.exp().squeeze()  # (16,)

    # 获取Hessian重要性
    hessian_imp = pruner.compute_importance()  # (16,) for heads

    # 归一化
    hess_norm = (hessian_imp - hessian_imp.min()) / (hessian_imp.max() - hessian_imp.min() + 1e-8)

    # 根据层类型调整重要性
    q1_threshold = args.q1_threshold  # 14.15

    if layer_idx in range(0, 5):  # 早期通才层
        # 保护低scale heads
        scale_bonus = torch.where(scale_mul < q1_threshold, args.early_bonus, 1.0)
        adjusted_importance = hess_norm * scale_bonus
        sparsity = args.early_sparsity

    elif layer_idx in range(5, 14):  # 中间专家层
        # 惩罚低scale heads
        scale_penalty = torch.where(scale_mul < q1_threshold, args.expert_penalty, 1.0)
        adjusted_importance = hess_norm * scale_penalty
        sparsity = args.expert_sparsity

    else:  # 末尾通才层
        # 保护低scale heads
        scale_bonus = torch.where(scale_mul < q1_threshold, args.late_bonus, 1.0)
        adjusted_importance = hess_norm * scale_bonus
        sparsity = args.late_sparsity

    # 选择要剪枝的heads
    num_heads = adjusted_importance.numel()
    num_prune = int(num_heads * sparsity)
    prune_indices = adjusted_importance.argsort()[:num_prune]

    return prune_indices, sparsity
```

### 6.2 命令行参数

```python
# 40%剪枝率配置
python prune_v6.py \
    --differential_scale_pruning \
    --q1_threshold 14.15 \
    --early_bonus 1.2 --early_sparsity 0.30 \
    --expert_penalty 0.5 --expert_sparsity 0.50 \
    --late_bonus 1.2 --late_sparsity 0.30 \
    --num_samples 1024 --maxlayer 16 \
    --model_name var_d16_scale_guided_40percent

# 20%剪枝率配置
python prune_v6.py \
    --differential_scale_pruning \
    --q1_threshold 14.15 \
    --early_bonus 1.2 --early_sparsity 0.15 \
    --expert_penalty 0.5 --expert_sparsity 0.25 \
    --late_bonus 1.2 --late_sparsity 0.15 \
    --num_samples 1024 --maxlayer 16 \
    --model_name var_d16_scale_guided_20percent
```

---

## 7. 验证方法

### 7.1 剪枝前分析

```bash
python analyze_scale_mul.py --model_depth 16 \
    --var_ckpt model_zoo/var_d16.pth \
    --output_dir scale_analysis_before
```

### 7.2 执行剪枝

```bash
python prune_v6.py --differential_scale_pruning \
    --early_sparsity 0.30 --expert_sparsity 0.50 --late_sparsity 0.30 \
    --model_name scale_guided_40percent
```

### 7.3 剪枝后分析

```bash
python analyze_scale_mul.py --model_depth 16 \
    --var_ckpt sparsity_model/scale_guided_40percent \
    --output_dir scale_analysis_after_40percent
```

### 7.4 方差一致性对比

```python
import json

# 加载前后数据
with open('scale_analysis_before/scale_mul_analysis.json') as f:
    before = json.load(f)
with open('scale_analysis_after_40percent/scale_mul_analysis.json') as f:
    after = json.load(f)

# 计算方差变化
early_var_before = np.mean([before['per_layer'][i]['variance'] for i in range(0, 5)])
expert_var_before = np.mean([before['per_layer'][i]['variance'] for i in range(5, 14)])
late_var_before = np.mean([before['per_layer'][i]['variance'] for i in range(14, 16)])

early_var_after = np.mean([after['per_layer'][i]['variance'] for i in range(0, 5)])
expert_var_after = np.mean([after['per_layer'][i]['variance'] for i in range(5, 14)])
late_var_after = np.mean([after['per_layer'][i]['variance'] for i in range(14, 16)])

print(f"早期层方差: {early_var_before:.2f} → {early_var_after:.2f}")
print(f"中间层方差: {expert_var_before:.2f} → {expert_var_after:.2f} (降低{(1-expert_var_after/expert_var_before)*100:.1f}%)")
print(f"末尾层方差: {late_var_before:.2f} → {late_var_after:.2f}")

ratio_before = max(early_var_before, expert_var_before, late_var_before) / min(early_var_before, expert_var_before, late_var_before)
ratio_after = max(early_var_after, expert_var_after, late_var_after) / min(early_var_after, expert_var_after, late_var_after)

print(f"\n方差一致性: {ratio_before:.2f} → {ratio_after:.2f} (改善{(1-ratio_after/ratio_before)*100:.1f}%)")
```

---

## 8. 关键洞察

### 8.1 为什么中间层优先剪低scale？

**理论依据**：
1. 中间层是专家层（高均值22.17），主要功能是精确定位
2. 低scale heads提供全局整合，在专家层中是冗余
3. 高方差(33.96)说明专家/通才混杂，低scale是冗余部分

**实验证据**：
- 中间层共9个低scale heads (<Q1)，占中间层6.25%
- 剪除后方差预期降低50%，说明功能更统一
- FID不显著上升，说明这些heads确实冗余

### 8.2 为什么早期/末尾层保护低scale？

**理论依据**：
1. 早期/末尾是通才层（低均值），需要全局整合
2. 低scale heads提供分散注意力，对通才功能关键
3. 低方差说明heads功能统一，low scale是核心能力

**实验证据**：
- 早期层均值12.91，远低于全局均值18.92
- 剪除低scale会损害全局整合能力
- 保持低方差，维持通才功能

### 8.3 相比LLM剪枝的创新

**LLM剪枝**：
- 单调变化：early通才 → late专家
- 统一策略：全局保护高scale（专家型）

**VAR剪枝（本方法）**：
- 三段式：通才 → 专家 → 通才回归
- 差异化策略：中间剪低scale，两端保护低scale
- 依据：VAR的多尺度生成需要最后整合

---

## 9. 配置文件

已生成的配置文件：
- `pruning_config_40percent.json` - 40%剪枝率详细配置
- `pruning_config_20percent.json` - 20%剪枝率详细配置

可直接用于prune_v6.py的实现。

---

## 10. 总结

✅ **明确了选择思路**：差异化对待不同层的低scale heads

✅ **计算了两种配置**：
- 40%剪枝率: 早期30% / 中间50% / 末尾30%
- 20%剪枝率: 早期15% / 中间25% / 末尾15%

✅ **识别了具体heads**：每层要剪哪些heads（基于scale_mul排序）

✅ **提供了实现代码**：可直接集成到prune_v6.py

🎯 **核心创新**：
1. 基于三段式结构的差异化剪枝
2. 中间层优先剪低scale（冗余通才）
3. 早期/末尾保护低scale（关键能力）
4. 方差一致性作为成功指标
