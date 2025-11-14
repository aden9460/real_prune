# Scale_Mul引导的VAR剪枝完整指南

本目录包含基于scale_mul分析的VAR模型差异化剪枝策略的完整实现。

---

## 📂 文档结构

```
slimvar/
├── README_SCALE_GUIDED_PRUNING.md           # 本文档（总览）
├── SCALE_MUL_DISTRIBUTION_INSIGHTS.md       # Scale_mul分布分析与理论
├── PRUNING_HEAD_SELECTION_GUIDE.md          # Head选择策略与具体配置
├── analyze_scale_mul.py                     # Scale_mul分析工具
├── run_scale_analysis.bash                  # 快速运行脚本
└── scale_mul_analysis_d16/                  # 分析结果（需生成）
    ├── scale_mul_analysis.json
    ├── pruning_strategy.json
    ├── scale_mul_matrix.npy
    └── *.png (可视化图片)
```

---

## 🎯 核心创新点

### 1. 三段式层级结构发现

**数据验证**：
```
早期通才层 (0-4):   低均值(12.91) + 低方差(7.85)  → 全局整合
中间专家层 (5-13):  高均值(22.17) + 高方差(33.96) → 专家型功能
末尾通才层 (14-15): 低均值(17.62) + 低方差(16.78) → 回归通才
```

**与LLM的区别**：
- LLM: 单调变化（通才 → 专家）
- VAR: 三段式（通才 → 专家 → 通才回归）

### 2. 差异化Head选择策略

**核心思想**：不同层对低scale heads的态度不同

```
┌──────────────────────────────────────────────────────┐
│  早期(0-4)   │  中间(5-13)  │  末尾(14-15)  │
├──────────────────────────────────────────────────────┤
│  保护低scale │  剪除低scale │  保护低scale  │
│  优先剪高    │  优先剪低    │  优先剪高     │
└──────────────────────────────────────────────────────┘
```

### 3. 方差一致性评估指标

**传统评估**：只看FID/IS（外部效果）

**创新评估**：方差一致性（内部冗余消除）
```
成功标志：
✅ 中间层方差降低 > 40%
✅ 三组方差ratio从4.3降至<2.5
✅ FID不显著上升
```

### 4. 百分位数阈值

**数据驱动**：
```
Q1 (25%): 14.15 - 低scale上界
Q2 (50%): 18.78 - 中位数
Q3 (75%): 22.96 - 高scale下界
```

优于固定阈值(5, 50)，更科学、可迁移。

---

## 🚀 快速开始

### Step 1: 分析scale_mul分布

```bash
cd /home/project/real_prune/slimvar
bash run_scale_analysis.bash
```

**生成**：
- `scale_mul_analysis_d16/` 目录
- 5张可视化图片
- JSON配置文件

### Step 2: 查看分析结果

```bash
# 查看统计数据
cat scale_mul_analysis_d16/scale_mul_analysis.json | jq .

# 查看推荐策略
cat scale_mul_analysis_d16/pruning_strategy.json | jq .
```

### Step 3: 执行差异化剪枝

```bash
# 40%剪枝率
python prune_v6.py \
    --differential_scale_pruning \
    --q1_threshold 14.15 \
    --early_bonus 1.2 --early_sparsity 0.30 \
    --expert_penalty 0.5 --expert_sparsity 0.50 \
    --late_bonus 1.2 --late_sparsity 0.30 \
    --num_samples 1024 --maxlayer 16 \
    --model_name scale_guided_40percent

# 20%剪枝率
python prune_v6.py \
    --differential_scale_pruning \
    --q1_threshold 14.15 \
    --early_bonus 1.2 --early_sparsity 0.15 \
    --expert_penalty 0.5 --expert_sparsity 0.25 \
    --late_bonus 1.2 --late_sparsity 0.15 \
    --num_samples 1024 --maxlayer 16 \
    --model_name scale_guided_20percent
```

### Step 4: 验证方差一致性

```bash
# 剪枝后分析
python analyze_scale_mul.py --model_depth 16 \
    --var_ckpt sparsity_model/scale_guided_40percent \
    --output_dir scale_analysis_after_40percent

# 对比前后
python compare_variance_consistency.py \
    --before scale_analysis_d16 \
    --after scale_analysis_after_40percent
```

---

## 📊 实验设计：6组消融实验

### 维度1：Head选择方法
- **A方法**：Scale_mul guided（差异化策略）
- **B方法**：Pure Hessian（SlimGPT原始）

### 维度2：剪枝率分配
- **策略1**：均匀剪枝率（所有层相同）
- **策略2**：指数缩小剪枝率
- **策略3**：Scale引导剪枝率（两边小中间大）

### 6组配置

| 实验 | Head选择 | 剪枝率策略 | 命令 |
|------|---------|-----------|------|
| Exp1 | Scale | 均匀 | `--differential_scale_pruning --sparsity 0.25` |
| Exp2 | Scale | 指数 | `--differential_scale_pruning --non_uniform --log_increase` |
| Exp3 | **Scale** | **差异化** | `--differential_scale_pruning --early_sp 0.2 --expert_sp 0.35 --late_sp 0.2` |
| Exp4 | Hessian | 均匀 | `--sparsity 0.25` (baseline) |
| Exp5 | Hessian | 指数 | `--non_uniform --log_increase` |
| Exp6 | Hessian | 差异化 | `--layerwise_sparsity 0.2,...,0.35,...,0.2` |

### 预期结果

**最优组合**：Exp3（Scale + 差异化剪枝率）

**预期排名**（FID从低到高）：
```
Exp3 < Exp1 < Exp6 < Exp2 < Exp5 < Exp4
```

**方差一致性排名**（ratio从低到高）：
```
Exp3 < Exp6 < Exp1 < Exp2 < Exp5 < Exp4
```

---

## 📈 预期性能提升

### 40%剪枝率配置

**相比均匀剪枝40% (baseline)**：

| 指标 | Baseline | Scale引导 | 改善 |
|------|---------|----------|------|
| FID | 2.50 | 2.40 | ↓4% |
| 方差ratio | 4.3 | <2.5 | ↓42% |
| 专家层方差 | 33.96 | ~17 | ↓50% |
| 参数量 | 60% | 60% | - |

### 20%剪枝率配置

**相比均匀剪枝20% (baseline)**：

| 指标 | Baseline | Scale引导 | 改善 |
|------|---------|----------|------|
| FID | 2.30 | 2.25 | ↓2% |
| 方差ratio | 4.3 | ~3.0 | ↓30% |
| 专家层方差 | 33.96 | ~25 | ↓26% |
| 参数量 | 80% | 80% | - |

---

## 📖 详细文档索引

### 理论与分析

**[SCALE_MUL_DISTRIBUTION_INSIGHTS.md](./SCALE_MUL_DISTRIBUTION_INSIGHTS.md)**
- 第1节：全局分布特征（百分位数阈值）
- 第2节：三段式层级结构（数据验证）
- 第3节：差异化剪枝策略（理论基础）
- 第4节：实验设计
- 第5节：方差一致性分析
- 第6节：关键洞察与创新点

### 实施指南

**[PRUNING_HEAD_SELECTION_GUIDE.md](./PRUNING_HEAD_SELECTION_GUIDE.md)**
- 第1节：选择思路总览
- 第2节：百分位数阈值
- 第3节：三段式层级策略
- 第4节：40%剪枝率配置（逐层详细）
- 第5节：20%剪枝率配置（逐层详细）
- 第6节：实现代码框架
- 第7节：验证方法

---

## 🔧 实现清单

### 已完成 ✅

- [x] scale_mul分析工具 (`analyze_scale_mul.py`)
- [x] 三段式层级结构验证
- [x] 40%和20%剪枝率配置计算
- [x] 具体head选择列表
- [x] 完整文档

### 待实现 ⬜

- [ ] 修改`prune_v6.py`，添加`--differential_scale_pruning`参数
- [ ] 实现`differential_scale_guided_pruning()`函数
- [ ] 创建`compare_variance_consistency.py`脚本
- [ ] 运行6组消融实验
- [ ] 剪枝后方差一致性验证

---

## 🎓 理论意义

### 对VAR剪枝的贡献

1. **发现VAR的三段式结构**
   - 首次明确VAR的层级不是单调的
   - 解释了为什么末尾层需要回归通才

2. **提出差异化剪枝策略**
   - 不同层对同一特征（低scale）的态度不同
   - 打破"全局统一策略"的假设

3. **引入结构评估指标**
   - 方差一致性作为冗余消除的证据
   - 不仅看性能，还看内部结构改善

4. **数据驱动的阈值设计**
   - 百分位数替代固定阈值
   - 可迁移到其他模型深度

### 与LLM剪枝的区别

| 特点 | LLM剪枝 | VAR剪枝（本方法）|
|------|---------|----------------|
| 层级结构 | 单调变化 | 三段式 |
| 剪枝策略 | 统一策略 | 差异化策略 |
| 评估指标 | 性能维度 | 性能+结构 |
| 先验信息 | 无scale_mul | 利用scale_mul |

---

## 📬 问题反馈

如有问题，请查看：
1. `SCALE_MUL_ANALYSIS_GUIDE.md` - 分析工具使用指南
2. `PRUNE_V6_DEEP_ANALYSIS.md` - prune_v6技术细节
3. `VAR_VS_LLM_PRUNING_INNOVATION_ANALYSIS.md` - LLM对比分析

---

## 📝 引用

如果使用本方法，请引用：

```bibtex
@article{var_differential_pruning,
  title={Differential Scale-Guided Pruning for Visual Autoregressive Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

**最后更新**：2025-11-10

**核心贡献**：
- 三段式层级结构（通才→专家→通才）
- 差异化scale引导剪枝策略
- 方差一致性评估指标
- 数据驱动的阈值选择
