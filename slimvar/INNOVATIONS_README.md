# VAR 剪枝创新方案 - 快速指南

> 本文档是 `VAR_PRUNING_CALIBRATION_GUIDE.md` 第8章创新方案的快速参考

---

## 🎯 创新方案概览

我们在基础VAR剪枝的基础上，提出了三个创新方案，通过参数控制可以灵活启用：

### 1. 分尺度重要性分析
**核心思想**：分析每个尺度下head/通道的重要性，基于跨尺度平均重要性进行全局剪枝。

**优势**：
- 更准确的全局剪枝策略
- 避免局部最优
- 可视化发现规律

**适用场景**：研究导向，发现VAR的重要性规律

### 2. QKV/FC1 补偿剪枝
**核心思想**：在剪枝proj/fc2时，将被剪通道的信息补偿到保留通道。

**优势**：
- 减少剪枝误差
- 提升剪枝后的生成质量 (预期10-20%)
- 计算开销小

**适用场景**：提升剪枝质量，推荐使用

### 3. 渐进式剪枝（实用版）
**核心思想**：分多个阶段逐步剪枝，使用轻量级指标评估，避免频繁FID评估。

**优势**：
- 更稳定的剪枝过程
- 避免一次性剪枝过度
- 每阶段重新评估重要性

**适用场景**：追求稳定性，实验探索

---

## 🚀 快速开始

### 最简单的使用（基础剪枝）

```bash
cd /home/project/real_prune/slimvar

python model_slimming_var.py \
    --num_samples 256 \
    --sparsity 0.2 \
    --minlayer 0 \
    --maxlayer 16
```

### 推荐组合（QKV补偿 + 渐进式剪枝）

```bash
python model_slimming_var.py \
    --num_samples 256 \
    --sparsity 0.3 \
    --enable_qkv_compensation \
    --compensation_method cosine_similarity \
    --progressive_stages 3 \
    --stage_sparsity "0.1,0.2,0.3" \
    --lightweight_eval_metric reconstruction_loss
```

### 完整功能（研究用途）

```bash
python model_slimming_var.py \
    --num_samples 256 \
    --sparsity 0.3 \
    --enable_scale_analysis \
    --scale_importance_output ./analysis/importance.npz \
    --enable_qkv_compensation \
    --compensation_method optimal_alpha \
    --progressive_stages 3 \
    --save_samples_per_stage
```

---

## 📊 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_samples` | 256 | 校准样本数 |
| `--sparsity` | 0.2 | 目标稀疏度 |
| `--minlayer` | 0 | 起始剪枝层 |
| `--maxlayer` | 16 | 结束剪枝层 |
| `--percdamp` | 0.01 | Hessian阻尼系数 |

### 创新功能开关

#### 分尺度重要性分析
```bash
--enable_scale_analysis             # 启用分尺度分析
--scale_importance_output <path>    # 保存分析结果路径
```

#### QKV/FC1补偿
```bash
--enable_qkv_compensation                    # 启用补偿剪枝
--compensation_method cosine_similarity      # 补偿方法（推荐）
--compensation_method optimal_alpha          # 更准确，但更慢
```

#### 渐进式剪枝
```bash
--progressive_stages 3                       # 阶段数（0=关闭）
--stage_sparsity "0.1,0.2,0.3"               # 各阶段稀疏度
--lightweight_eval_metric reconstruction_loss # 轻量级评估指标
--save_samples_per_stage                     # 每阶段保存样本
--enable_early_stop                          # 启用早停
--early_stop_threshold 1.5                   # 早停阈值
```

---

## 📈 预期效果

| 方案 | FID提升 | 额外时间 | 推荐度 |
|------|---------|---------|--------|
| 基础剪枝 | baseline | - | ⭐⭐⭐ |
| + QKV补偿 | +10-20% | +10% | ⭐⭐⭐⭐⭐ |
| + 渐进式剪枝 | +5-15% | +50% | ⭐⭐⭐⭐ |
| + 分尺度分析 | +5-10% | +20% | ⭐⭐⭐（研究向） |
| 全部组合 | +20-40% | +80% | ⭐⭐⭐⭐ |

---

## 🛠️ 实现状态

### ✅ 已完成
- [x] 完整文档（第8章）
- [x] 理论分析和代码示例
- [x] 参数设计

### 🚧 待实现（按优先级）
1. **基础剪枝模块** (`model_slimming_basic.py`)
   - 全尺度收集激活值
   - SlimGPT评估重要性
   - Torch-Pruning基础剪枝

2. **QKV补偿模块** (`pruning_innovations.py`)
   - `compensate_qkv_cosine_similarity()`
   - `compensate_qkv_optimal()`
   - `compensate_fc1()`

3. **分尺度分析模块** (`pruning_innovations.py`)
   - `collect_activations_by_scale()`
   - `evaluate_head_importance_by_scale()`
   - `analyze_importance_patterns()`

4. **渐进式剪枝模块** (`pruning_innovations.py`)
   - `progressive_pruning_pipeline()`
   - `lightweight_evaluation()`

5. **集成主脚本** (`model_slimming_var.py`)
   - 参数解析
   - 功能调度
   - 完整流程

---

## 📝 实验建议

### 第一周：基础功能验证
```bash
# Day 1-2: 基础剪枝
python model_slimming_var.py --sparsity 0.2

# Day 3-4: QKV补偿效果
python model_slimming_var.py --sparsity 0.2 --enable_qkv_compensation

# Day 5-7: 对比FID，验证提升
```

### 第二周：高级功能探索
```bash
# Day 1-3: 分尺度分析
python model_slimming_var.py --sparsity 0.2 --enable_scale_analysis

# Day 4-7: 渐进式剪枝
python model_slimming_var.py --sparsity 0.3 --progressive_stages 3
```

### 第三周：完整方案验证
```bash
# 全部功能组合测试
python model_slimming_var.py \
    --sparsity 0.3 \
    --enable_scale_analysis \
    --enable_qkv_compensation \
    --progressive_stages 3

# 全面评估：FID、参数量、推理速度、视觉质量
```

---

## 💡 重要说明

### 关于VAR架构的澄清
- ✅ VAR的Transformer层**是共享的**（所有10个尺度共享16层）
- ❌ **不能**为不同尺度设置不同的剪枝比例
- ✅ **可以**基于跨尺度平均重要性进行全局剪枝

### 关于FID评估
- ⚠️ FID评估耗时约1小时
- 建议在渐进式剪枝中使用轻量级指标
- 只在最后进行一次完整FID评估

### 关于补偿方法
- `cosine_similarity`：推荐日常使用，快速且效果不错
- `optimal_alpha`：追求最优效果时使用，需要更多计算

---

## 📚 文档链接

- **完整指南**：`VAR_PRUNING_CALIBRATION_GUIDE.md`（第8章）
- **基础教程**：`VAR_PRUNING_CALIBRATION_GUIDE.md`（第1-7章）
- **参考实现**：`tp_prune_reference.py`

---

## 🤝 下一步

1. **阅读完整文档**第8章，理解每个方案的原理
2. **选择要实现的功能**（建议从QKV补偿开始）
3. **运行基础剪枝**，建立baseline
4. **逐步添加创新功能**，对比效果

---

**文档版本**: v1.0
**创建日期**: 2025-01
**状态**: 方案设计完成，代码实现进行中

有任何问题或建议，欢迎讨论！
