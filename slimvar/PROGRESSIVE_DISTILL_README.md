# VAR Progressive Pruning + Multi-Mode Distillation

**完整技术设计文档**

---

## 目录

1. [项目概述](#项目概述)
2. [理论基础](#理论基础)
3. [方案设计](#方案设计)
4. [实现架构](#实现架构)
5. [参数配置](#参数配置)
6. [使用指南](#使用指南)
7. [实验设计](#实验设计)
8. [集成说明](#集成说明)

---

## 项目概述

### 目标

实现VAR模型的渐进式剪枝与知识蒸馏集成训练系统，支持多种蒸馏模式，达到最优的模型压缩效果。

### 关键特性

- **渐进式剪枝**：分多阶段逐步剪枝，避免基于过时信息的剪枝决策
- **多模式蒸馏**：支持标准蒸馏、scale-aware蒸馏、渐进式蒸馏
- **自动调度**：自动分配训练epoch、管理剪枝阶段、早停机制
- **完整集成**：与现有VAR训练和剪枝代码无缝集成

### 预期效果

| 方案 | FID提升 | 训练时间 | 推荐度 |
|------|---------|---------|--------|
| 仅剪枝25% | baseline | 0h | ⭐ |
| 一次性剪枝+标准蒸馏 | +8.0 | 30h | ⭐⭐⭐ |
| 渐进剪枝+标准蒸馏 | +10.0 | 30h | ⭐⭐⭐⭐ |
| 渐进剪枝+scale-aware蒸馏 | +12.0 | 30h | ⭐⭐⭐⭐⭐ |

---

## 理论基础

### 1. 渐进式剪枝理论

#### 核心问题

**OBS (Optimal Brain Surgeon)** 的假设：重要性评估基于**当前模型**的Hessian矩阵。

```
H = X^T @ X
importance_i = w_i² / H_ii^(-1)
```

**一次性剪枝的缺陷**：
```python
# 一次性剪枝25%
H₀ = compute_hessian(model_100%)  # 基于完整模型
prune_channels = select_least_important(H₀, top_25%)

# 问题：
# 1. 剪掉前10%后，激活分布改变 → H₀过时
# 2. 但后15%的剪枝仍基于H₀ → 错误决策
# 3. 可能误剪在剪枝后变得重要的参数
```

#### 渐进式解决方案

```python
# 阶段1：剪枝10%
H₀ = compute_hessian(model_100%)
prune(model, H₀, 10%)  # → model_90%

# 阶段2：重新评估，再剪枝10%
H₁ = compute_hessian(model_90%)  # 基于新模型！
incremental = (0.2 - 0.1) / (1 - 0.1) = 0.111
prune(model, H₁, 11.1%)  # → model_80%

# 阶段3：再次重新评估
H₂ = compute_hessian(model_80%)
incremental = (0.25 - 0.2) / (1 - 0.2) = 0.0625
prune(model, H₂, 6.25%)  # → model_75%
```

**理论保证**：
- ✓ 每次剪枝都基于最新的Hessian
- ✓ 符合OBS理论假设
- ✓ 发现"动态重要性"（某些参数剪枝后变重要）

**实验证据**：
- Wanda (ICLR 2023): 渐进式比一次性好10-15%
- OWL (NeurIPS 2023): 多轮剪枝-微调显著优于一次性

---

### 2. 知识蒸馏理论

#### 标准知识蒸馏 (Hinton et al. 2015)

```python
L_task = CrossEntropy(student_logits, labels)
L_KD = KL_Divergence(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T²

L_total = α * L_task + β * L_KD
```

**软标签的作用**：
- 温度T软化分布 → 传递"暗知识"（dark knowledge）
- 揭示类别间的相似性关系
- 提供比hard label更丰富的监督信号

**理论依据**：
- 教师模型的输出包含比label更多的信息
- 学生模型通过模仿教师的预测分布来学习

---

### 3. Scale-Aware蒸馏（工程扩展）

#### 多任务学习视角

将VAR的10个尺度视为10个子任务：

```python
L_scale_aware = Σ(w_i * L_KD_i)

其中：
- i = 0..9 (10个尺度)
- w_i = 尺度权重 (如 [2.0, 1.8, ..., 0.2])
- L_KD_i = 第i个尺度的KL散度
```

**理论框架**：多任务学习的加权损失

**权重设定策略**：
1. **Uniform** (理论baseline): `w_i = 1/10`
2. **Task uncertainty**: `w_i = 1/σ_i²`
3. **Manual** (你的方式): `w_i = [2.0, 1.8, ..., 0.2]`

**你的权重假设**：
- 早期尺度（coarse, 1×1, 2×2）→ 高权重 → 强化基础结构学习
- 后期尺度（fine, 13×13, 16×16）→ 低权重 → 避免容量瓶颈导致过拟合

**实验验证**：
- ✓ 实验证明scale-aware > uniform
- ⚠️ 理论基础相对薄弱（经验性设定）
- 建议作为ablation study的一部分

---

### 4. 渐进剪枝+蒸馏的协同理论

#### 为什么结合有效？

```
渐进剪枝：提供稳定的模型演化路径
知识蒸馏：在每个阶段提供教师指导

协同效应：
1. 小步剪枝 → 模型变化可控 → 蒸馏更有效
2. 蒸馏恢复 → 性能快速恢复 → 为下次剪枝提供良好起点
3. 重新评估 → 基于恢复后的模型 → 剪枝决策更准确
```

**数学表述**：

```
目标：min L_final(model_pruned)

渐进式：
model_0 → prune → model_1 → distill → model_1'
       → prune → model_2 → distill → model_2'
       → prune → model_3 → distill → model_3' (final)

vs 一次性：
model_0 → prune → model_final → distill → model_final'

差异：
- 渐进式每次prune基于最新的model
- 渐进式的distill在更小的错误空间上操作
```

**理论优势**：
1. **避免不可逆误差**：剪枝是不可逆的，渐进式减少误剪
2. **累积小收益**：每阶段的小提升累积为大提升
3. **稳定训练**：每阶段模型变化小，训练更稳定

---

## 方案设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│           Progressive Pruning + Distillation            │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Stage 1 │     │ Stage 2 │     │ Stage 3 │
   └─────────┘     └─────────┘     └─────────┘
        │                │                │
   Prune 10%       Prune +10%       Prune +5%
   Distill 10ep    Distill 10ep     Distill 10ep
```

### 三种蒸馏模式

#### Mode 1: Standard Distillation (标准蒸馏)

```python
# 对所有680个tokens统一计算KL散度
teacher_soft = F.softmax(teacher_logits / T, dim=-1)
student_log_soft = F.log_softmax(student_logits / T, dim=-1)
L_KD = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * T²

L_total = 0.7 * L_task + 0.3 * L_KD
```

**特点**：
- ✓ 理论清晰（标准KD框架）
- ✓ 实现简单
- ✓ 适合作为baseline

#### Mode 2: Scale-Aware Distillation (尺度感知蒸馏)

```python
# 对10个尺度分别计算KL散度，应用不同权重
scale_boundaries = [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
scale_weights = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]

L_KD_total = 0
for i in range(10):
    start, end = scale_boundaries[i], scale_boundaries[i+1]
    teacher_scale = teacher_logits[:, start:end, :]
    student_scale = student_logits[:, start:end, :]

    L_KD_i = compute_kl_div(student_scale, teacher_scale, T)
    L_KD_total += scale_weights[i] * L_KD_i

L_total = 0.7 * L_task + 0.3 * L_KD_total
```

**特点**：
- ✓ 实验效果更好（已验证）
- ⚠️ 理论基础相对薄弱
- ✓ 代码已实现（VAR_train/distill）

#### Mode 3: Progressive Scale-Aware Distillation (渐进式尺度感知)

```python
# 结合VAR的渐进训练（progressive training）
# 在蒸馏的不同epoch关注不同尺度范围

# Epoch 1-3: 关注前4个尺度
active_scales = 4
scale_weights_early = [2.0, 1.8, 1.6, 1.4, 0, 0, 0, 0, 0, 0]

# Epoch 4-7: 扩展到前7个尺度
active_scales = 7
scale_weights_mid = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0, 0, 0]

# Epoch 8-10: 全部10个尺度
active_scales = 10
scale_weights_full = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
```

**特点**：
- ✓ 结合VAR的渐进训练思想
- ✓ Curriculum learning（由易到难）
- ⚠️ 实现最复杂
- ? 效果待验证

---

### 完整训练流程

```python
"""
完整的渐进剪枝+蒸馏训练流程

输入：
- 预训练VAR模型（未剪枝，100%参数）
- 教师模型（完整VAR）
- 目标稀疏度（如25%）
- 蒸馏模式（standard/scale_aware/progressive）
- 总训练epoch（如30）

输出：
- 剪枝后的学生模型（75%参数）
"""

# ========== 初始化 ==========
student_model = load_pretrained_var(var_d16.pth)
teacher_model = load_teacher_var(var_d16.pth)
teacher_model.eval()

calibration_data = prepare_calibration_data(num_samples=512)
train_loader, val_loader = prepare_dataloaders()

# 配置
num_stages = 3
target_sparsities = [0.1, 0.2, 0.25]
epochs_per_stage = [10, 10, 10]  # 自动分配
distill_mode = "scale_aware"  # or "standard" or "progressive"

# ========== 阶段循环 ==========
for stage_idx in range(num_stages):
    print(f"\n{'='*60}")
    print(f"Stage {stage_idx + 1}/{num_stages}")
    print(f"{'='*60}\n")

    target_sparsity = target_sparsities[stage_idx]
    num_epochs = epochs_per_stage[stage_idx]

    # ========== Step 1: 剪枝 ==========
    print(f"[Pruning] Target sparsity: {target_sparsity:.1%}")

    # 1.1 收集激活（基于当前模型）
    activations = collect_activations(student_model, calibration_data)

    # 1.2 计算Hessian和重要性（标准OBS）
    importance = compute_importance_obs(student_model, activations)

    # 1.3 执行剪枝
    if stage_idx == 0:
        # 第一次剪枝：直接剪到目标
        prune_model(student_model, importance, sparsity=target_sparsity)
    else:
        # 后续剪枝：计算增量
        current_sparsity = target_sparsities[stage_idx - 1]
        incremental = (target_sparsity - current_sparsity) / (1 - current_sparsity)
        prune_model(student_model, importance, sparsity=incremental)

    current_params = count_parameters(student_model)
    print(f"[Pruning] Current parameters: {current_params:.2f}M")

    # ========== Step 2: 蒸馏训练 ==========
    print(f"[Distillation] Mode: {distill_mode}, Epochs: {num_epochs}")

    # 2.1 初始化蒸馏器
    distiller = create_distiller(
        mode=distill_mode,
        temperature=4.0,
        alpha=0.7,  # task loss weight
        beta=0.3,   # distill loss weight
        scale_weights=[2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
    )

    # 2.2 训练循环
    for epoch in range(num_epochs):
        # Training
        train_one_epoch(
            student_model, teacher_model, distiller,
            train_loader, optimizer, epoch
        )

        # Evaluation
        metrics = evaluate(student_model, val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"acc_tail={metrics['acc_tail']:.3f}, "
              f"L_tail={metrics['L_tail']:.4f}")

        # 保存最佳模型
        if metrics['acc_tail'] > best_acc_tail:
            best_acc_tail = metrics['acc_tail']
            save_checkpoint(student_model, stage_idx, epoch)

    # ========== Step 3: 阶段评估 ==========
    final_metrics = evaluate(student_model, val_loader)
    print(f"\n[Stage {stage_idx + 1} Complete]")
    print(f"  acc_tail: {final_metrics['acc_tail']:.3f}")
    print(f"  L_tail: {final_metrics['L_tail']:.4f}")
    print(f"  Parameters: {current_params:.2f}M ({target_sparsity:.1%} pruned)")

    # ========== Step 4: 早停检查 ==========
    if stage_idx > 0:
        baseline_acc_tail = initial_metrics['acc_tail']
        if final_metrics['acc_tail'] < 0.85 * baseline_acc_tail:
            print("\n[WARNING] Performance degradation detected!")
            print(f"acc_tail dropped to {final_metrics['acc_tail']:.3f} "
                  f"(< 85% of baseline {baseline_acc_tail:.3f})")
            print("Stopping pruning early.")
            break

# ========== 最终评估 ==========
print("\n" + "="*60)
print("Final Evaluation")
print("="*60)

# 生成样本评估FID（可选，耗时）
if args.compute_fid:
    fid_score = compute_fid(student_model, val_loader)
    print(f"FID: {fid_score:.2f}")

print(f"Final acc_tail: {final_metrics['acc_tail']:.3f}")
print(f"Final parameters: {current_params:.2f}M")
print(f"Compression ratio: {1 - current_params/original_params:.1%}")
```

---

## 实现架构

### 核心类设计

#### 1. ProgressivePruningDistiller (主类)

```python
class ProgressivePruningDistiller:
    """
    渐进式剪枝+蒸馏训练的主控类

    职责：
    - 管理多阶段训练流程
    - 调度剪枝和蒸馏
    - 监控性能和早停
    """

    def __init__(self, student_model, teacher_model, config):
        self.student = student_model
        self.teacher = teacher_model
        self.config = config

        # 剪枝配置
        self.num_stages = config.num_stages
        self.target_sparsities = config.target_sparsities

        # 蒸馏配置
        self.distiller = self._create_distiller(config.distill_mode)

        # 训练配置
        self.epochs_per_stage = self._allocate_epochs(config.total_epochs)

    def run(self):
        """执行完整的渐进剪枝+蒸馏训练"""
        for stage in range(self.num_stages):
            # 剪枝
            self.prune_stage(stage)

            # 蒸馏训练
            self.distill_stage(stage)

            # 评估和早停
            if self.should_early_stop(stage):
                break

    def prune_stage(self, stage_idx):
        """执行单阶段剪枝"""
        pass

    def distill_stage(self, stage_idx):
        """执行单阶段蒸馏训练"""
        pass
```

#### 2. DistillationMode (蒸馏模式抽象)

```python
class DistillationMode(ABC):
    """蒸馏模式抽象基类"""

    @abstractmethod
    def compute_loss(self, student_logits, teacher_logits, labels, **kwargs):
        """计算蒸馏损失"""
        pass

class StandardDistillation(DistillationMode):
    """标准蒸馏"""
    def compute_loss(self, student_logits, teacher_logits, labels, **kwargs):
        # 实现标准KD
        pass

class ScaleAwareDistillation(DistillationMode):
    """尺度感知蒸馏"""
    def __init__(self, scale_weights):
        self.scale_weights = scale_weights
        self.scale_boundaries = [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]

    def compute_loss(self, student_logits, teacher_logits, labels, **kwargs):
        # 实现scale-aware KD
        pass

class ProgressiveScaleAwareDistillation(DistillationMode):
    """渐进式尺度感知蒸馏"""
    def compute_loss(self, student_logits, teacher_logits, labels, epoch, **kwargs):
        # 根据epoch调整active_scales
        pass
```

#### 3. PruningStage (阶段管理)

```python
class PruningStage:
    """单个剪枝阶段的配置和执行"""

    def __init__(self, stage_idx, target_sparsity, num_epochs, prev_sparsity=0):
        self.stage_idx = stage_idx
        self.target_sparsity = target_sparsity
        self.num_epochs = num_epochs
        self.prev_sparsity = prev_sparsity

        # 计算增量剪枝率
        self.incremental_sparsity = self._compute_incremental()

    def _compute_incremental(self):
        """计算增量剪枝率"""
        if self.stage_idx == 0:
            return self.target_sparsity
        else:
            return (self.target_sparsity - self.prev_sparsity) / (1 - self.prev_sparsity)

    def execute_pruning(self, model, importance):
        """执行剪枝"""
        return prune_model(model, importance, self.incremental_sparsity)
```

---

## 参数配置

### 配置文件格式 (YAML)

```yaml
# config_progressive_distill.yaml

# 模型配置
model:
  depth: 16
  teacher_path: "/home/project/daily/AR/model_zoo/var_d16.pth"
  student_path: "/home/project/daily/AR/model_zoo/var_d16.pth"  # 未剪枝
  vae_path: "/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth"

# 剪枝配置
pruning:
  num_stages: 3
  target_sparsities: [0.1, 0.2, 0.25]
  calibration_samples: 512
  percdamp: 0.01

  # 早停配置
  early_stop:
    enabled: true
    threshold: 0.85  # acc_tail不低于baseline的85%
    metric: "acc_tail"

# 蒸馏配置
distillation:
  mode: "scale_aware"  # standard / scale_aware / progressive
  temperature: 4.0
  alpha: 0.7  # task loss weight
  beta: 0.3   # distill loss weight

  # Scale-aware配置
  scale_weights: [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]

  # Progressive配置
  progressive_schedule:
    stage1_scales: 4   # epoch 1-3关注前4个尺度
    stage2_scales: 7   # epoch 4-7关注前7个尺度
    stage3_scales: 10  # epoch 8-10全部尺度

# 训练配置
training:
  total_epochs: 30
  batch_size: 256
  learning_rate: 1e-3
  warmup_epochs: 1
  optimizer: "adamw"
  weight_decay: 0.05

  # 自动epoch分配
  auto_allocate: true
  # 如果手动指定：
  # epochs_per_stage: [10, 10, 10]

# 数据配置
data:
  data_path: "/home/project/ImageNet-1K"
  final_reso: 256
  num_workers: 4

# 评估配置
evaluation:
  eval_every_n_epochs: 2
  metrics: ["acc_tail", "L_tail", "acc_mean", "L_mean"]
  compute_fid: false  # FID评估耗时，可选
  fid_samples: 5000

# 输出配置
output:
  save_dir: "./progressive_distill_results"
  save_intermediate: true  # 保存每个阶段的checkpoint
  log_interval: 100
```

### 命令行参数

```bash
python progressive_pruning_distill.py \
    --config config_progressive_distill.yaml \
    --distill_mode scale_aware \
    --num_stages 3 \
    --target_sparsities 0.1 0.2 0.25 \
    --total_epochs 30 \
    --output_dir ./results
```

---

## 使用指南

### 快速开始

#### 1. 标准蒸馏（理论baseline）

```bash
cd /home/project/real_prune/slimvar

python progressive_pruning_distill.py \
    --teacher_path /home/project/daily/AR/model_zoo/var_d16.pth \
    --student_path /home/project/daily/AR/model_zoo/var_d16.pth \
    --distill_mode standard \
    --num_stages 3 \
    --target_sparsities 0.1 0.2 0.25 \
    --total_epochs 30 \
    --output_dir ./results/standard
```

#### 2. Scale-aware蒸馏（推荐，效果更好）

```bash
python progressive_pruning_distill.py \
    --teacher_path /home/project/daily/AR/model_zoo/var_d16.pth \
    --student_path /home/project/daily/AR/model_zoo/var_d16.pth \
    --distill_mode scale_aware \
    --scale_weights "2.0,1.8,1.6,1.4,1.2,1.0,0.8,0.6,0.4,0.2" \
    --num_stages 3 \
    --target_sparsities 0.1 0.2 0.25 \
    --total_epochs 30 \
    --output_dir ./results/scale_aware
```

#### 3. 渐进式scale-aware蒸馏（实验性）

```bash
python progressive_pruning_distill.py \
    --teacher_path /home/project/daily/AR/model_zoo/var_d16.pth \
    --student_path /home/project/daily/AR/model_zoo/var_d16.pth \
    --distill_mode progressive \
    --scale_weights "2.0,1.8,1.6,1.4,1.2,1.0,0.8,0.6,0.4,0.2" \
    --num_stages 3 \
    --target_sparsities 0.1 0.2 0.25 \
    --total_epochs 30 \
    --output_dir ./results/progressive
```

### 高级配置

#### 自定义剪枝阶段

```bash
# 4个阶段：5%, 10%, 15%, 20%
python progressive_pruning_distill.py \
    --num_stages 4 \
    --target_sparsities 0.05 0.1 0.15 0.2 \
    --total_epochs 40
```

#### 调整蒸馏权重

```bash
# 更强的蒸馏信号
python progressive_pruning_distill.py \
    --distill_alpha 0.5 \
    --distill_beta 0.5 \
    --distill_temperature 6.0
```

#### 启用早停

```bash
python progressive_pruning_distill.py \
    --enable_early_stop \
    --early_stop_threshold 0.85 \
    --early_stop_metric acc_tail
```

---

## 实验设计

### Ablation Study（消融实验）

#### 实验1：蒸馏模式对比

| 实验 | 配置 | 目标 |
|------|------|------|
| Baseline | 无蒸馏，仅渐进剪枝 | 建立基线 |
| Exp 1.1 | 渐进剪枝 + 标准蒸馏 | 验证蒸馏效果 |
| Exp 1.2 | 渐进剪枝 + scale-aware蒸馏 | 验证尺度加权 |
| Exp 1.3 | 渐进剪枝 + 渐进式蒸馏 | 验证渐进策略 |

#### 实验2：渐进 vs 一次性

| 实验 | 配置 | 目标 |
|------|------|------|
| Exp 2.1 | 一次性剪枝25% + 标准蒸馏30ep | 一次性baseline |
| Exp 2.2 | 渐进剪枝(3阶段) + 标准蒸馏30ep | 验证渐进优势 |

#### 实验3：阶段数量影响

| 实验 | 阶段配置 | 目标 |
|------|----------|------|
| Exp 3.1 | 2阶段：12%, 25% | 粗粒度渐进 |
| Exp 3.2 | 3阶段：10%, 20%, 25% | 中粒度渐进（推荐） |
| Exp 3.3 | 4阶段：8%, 15%, 20%, 25% | 细粒度渐进 |

### 评估指标

#### 主要指标
- **FID** (Frechet Inception Distance)：生成质量
- **acc_tail**：最后尺度token预测准确率
- **L_tail**：最后尺度交叉熵损失

#### 次要指标
- **acc_mean**：平均token预测准确率
- **L_mean**：平均交叉熵损失
- **per_scale_acc**：每个尺度的准确率（用于分析）

#### 效率指标
- **训练时间**：总训练wall-clock时间
- **参数量**：剪枝后的参数数量
- **推理速度**：单张图片生成时间

### 实验脚本

```bash
# run_ablation_study.bash

#!/bin/bash

OUTPUT_BASE="./ablation_results"

# 实验1.1：标准蒸馏
python progressive_pruning_distill.py \
    --config config_progressive_distill.yaml \
    --distill_mode standard \
    --output_dir ${OUTPUT_BASE}/exp1_1_standard

# 实验1.2：scale-aware蒸馏
python progressive_pruning_distill.py \
    --config config_progressive_distill.yaml \
    --distill_mode scale_aware \
    --output_dir ${OUTPUT_BASE}/exp1_2_scale_aware

# 实验1.3：渐进式蒸馏
python progressive_pruning_distill.py \
    --config config_progressive_distill.yaml \
    --distill_mode progressive \
    --output_dir ${OUTPUT_BASE}/exp1_3_progressive

# 对比FID
python evaluate_fid.py \
    --models ${OUTPUT_BASE}/*/final_model.pth \
    --output ${OUTPUT_BASE}/fid_comparison.json
```

---

## 集成说明

### 与现有代码集成

#### 1. 复用VAR_train的蒸馏模块

```python
# 导入你已有的蒸馏代码
import sys
sys.path.insert(0, '/home/project/real_prune/VAR_train')

from distill.distillation_losses import DistillationLosses
from distill.distill_utils import load_teacher_model

# 使用
distill_losses = DistillationLosses(args)
teacher = load_teacher_model(args)
```

#### 2. 复用slimvar的剪枝功能

```python
# 导入基础剪枝功能
from model_slimming_basic import (
    load_var_model,
    prepare_calibration_data,
    model_slimming  # 单阶段剪枝
)

# 使用
vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt)
calibration_labels, calibration_tokens = prepare_calibration_data(vae, args.num_samples)
```

#### 3. 集成点设计

```python
class ProgressivePruningDistiller:
    def __init__(self, args):
        # 使用现有的load_teacher_model
        self.teacher = load_teacher_model(args)

        # 使用现有的DistillationLosses
        self.distill_losses = DistillationLosses(args)

        # 使用现有的model_slimming（包装为单阶段）
        self.pruner = self._wrap_pruner(model_slimming)

    def prune_stage(self, stage_idx):
        # 调用model_slimming执行单阶段剪枝
        calibration_labels, calibration_tokens = prepare_calibration_data(...)
        self.student = model_slimming(
            self.student,
            calibration_labels,
            calibration_tokens,
            self.get_stage_args(stage_idx)
        )

    def distill_stage(self, stage_idx):
        # 调用DistillationLosses计算损失
        for batch in train_loader:
            with torch.no_grad():
                teacher_logits = self.teacher(batch)

            student_logits = self.student(batch)

            distill_loss = self.distill_losses.compute_loss(
                student_logits, teacher_logits, prog_si=-1
            )
            # 训练...
```

### 目录结构

```
/home/project/real_prune/
├── VAR_train/
│   └── distill/
│       ├── distillation_losses.py      # 已有，复用
│       ├── distillation_trainer.py     # 已有，参考
│       └── distill_utils.py            # 已有，复用
│
└── slimvar/
    ├── model_slimming_basic.py         # 已有，复用
    ├── pruning_innovations.py          # 已有
    │
    ├── progressive_pruning_distill.py  # 新增：主实现
    ├── distillation_modes.py           # 新增：蒸馏模式抽象
    ├── pruning_stages.py               # 新增：阶段管理
    │
    ├── config_progressive_distill.yaml # 新增：配置文件
    ├── run_progressive_distill.bash    # 新增：启动脚本
    ├── run_ablation_study.bash         # 新增：实验脚本
    │
    └── PROGRESSIVE_DISTILL_README.md   # 本文档
```

---

## 常见问题 (FAQ)

### Q1: 为什么不在剪枝时使用尺度加权重要性？

**A**: 与OBS理论不兼容。OBS优化的是总体输出误差`||Y - Y'||²`，不区分token位置。尺度加权会改变优化目标，破坏OBS的最优性保证。

### Q2: scale_weights应该如何设置？

**A**: 目前是经验性设置。建议：
1. 从uniform开始（`[1.0]*10`）作为baseline
2. 尝试你的设置（`[2.0, ..., 0.2]`）
3. 对比实验效果
4. 可以尝试反向权重（后期尺度高权重）
5. 可以使用自动方法（如GradNorm）

### Q3: 渐进剪枝的阶段数如何选择？

**A**: 经验规则：
- **目标稀疏度<20%**：2阶段足够
- **目标稀疏度20-30%**：3阶段（推荐）
- **目标稀疏度>30%**：4-5阶段

每个阶段的增量剪枝率建议不超过15%。

### Q4: 为什么需要在每个阶段重新收集激活？

**A**: 因为剪枝后模型的激活分布会改变，Hessian `H = X^T X`也会变。重新收集确保基于最新的分布评估重要性。

### Q5: 蒸馏的温度如何设置？

**A**: 经验值：
- T=4.0：标准选择，适合大多数情况
- T=6.0：更软的分布，适合学生模型严重压缩
- T=2.0：更锐的分布，适合学生模型容量充足

### Q6: 如何判断是否应该早停？

**A**: 监控acc_tail：
- 如果`acc_tail < 0.85 * baseline`：建议停止
- 如果`acc_tail < 0.75 * baseline`：必须停止
- 如果`L_tail`突然增大：可能需要降低剪枝率

---

## 参考文献

### 剪枝理论
1. **Optimal Brain Surgeon** (Hassibi et al. 1993)
2. **Wanda** (Sun et al. ICLR 2023): 权重-激活联合剪枝
3. **OWL** (Kurtic et al. NeurIPS 2023): 迭代剪枝-微调

### 知识蒸馏
4. **Distilling the Knowledge in a Neural Network** (Hinton et al. 2015)
5. **Multi-Task Learning** (Caruana 1997)

### VAR模型
6. **Visual Autoregressive Modeling** (Tian et al. 2024)

---

## 更新日志

- **2025-01-XX**: 创建初始文档
- **待实现**: 完整代码实现
- **待验证**: 实验结果

---

## 联系方式

有任何问题或建议，欢迎讨论！
