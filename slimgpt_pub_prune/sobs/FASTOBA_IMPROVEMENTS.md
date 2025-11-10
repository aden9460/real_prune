# FastOBA 改进方案文档

**创建日期**: 2025-11-03
**状态**: 规划中
**优先级**: 中至低（阶段1关键修复已完成）

---

## 📋 概述

本文档记录了FastOBA剪枝方法的进一步改进方案。阶段1的关键bug修复已完成（见下方"已完成修复"），本文档专注于算法层面的优化。

---

## ✅ 已完成修复（阶段1）

### 修复1: struct_prune_head_dims 矩阵切片bug

**问题**: 第831行使用 `[:cnt]` 直接切片破坏了Cholesky矩阵的方阵结构

**位置**: `fastoba_attention_slimgpt.py:831, 853`

**修复**:
```python
# 修复前（错误）:
Hinv_chol = torch.linalg.cholesky(Hinv_head, upper=True)[:cnt]  # 变成非方阵
Hinv1 = Hinv_chol[:, :cnt]  # 维度错误

# 修复后（正确）:
Hinv_chol = torch.linalg.cholesky(Hinv_head, upper=True)  # 保持方阵
Hinv1 = Hinv_chol[:cnt, :cnt]  # 正确的子矩阵提取
```

**预期影响**: FastOBA head-dim剪枝性能提升30-50%

---

### 修复2: 改进伪损失定义

**问题**: 原始伪损失 `loss = ||attn_out||²` 没有实际意义

**修复**: 改为输出方差损失
```python
# 修复前:
loss_batch = out_batch.pow(2).sum()  # 简单L2范数

# 修复后:
mean_out = out_batch.mean(dim=-1, keepdim=True)
variance = ((out_batch - mean_out) ** 2).sum()
loss_batch = -variance  # 最大化方差 = 保持信息量
```

**理论依据**:
- 最大化输出方差 → 保持特征表示的信息量
- 方差大的维度更重要，剪枝应该保留
- 避免输出坍缩到低秩空间

**预期影响**: 整体性能提升10-20%

---

## 🔬 算法改进方案（阶段2及以后）

### 改进1: 混合Hessian方法 🟡

**优先级**: 高
**难度**: 中
**预期影响**: +10-20% 性能

#### 核心思想

结合SlimGPT的稳定性（Fisher信息矩阵 H = X^T X）和FastOBA的准确性（真实Hessian）：

```
H_hybrid = α·H_fisher + (1-α)·H_fastoba
```

其中：
- `α ∈ [0, 1]`: 权重参数，推荐 α=0.7
- `α=1`: 纯SlimGPT（最稳定，但不够准确）
- `α=0`: 纯FastOBA（最准确，但可能不稳定）
- `α=0.7`: 平衡点（70%稳定性 + 30%准确性提升）

#### 实现代码

```python
class FastOBAAttentionSlimGPT:
    def __init__(self, ..., use_hybrid=False, hybrid_alpha=0.7):
        self.use_hybrid = use_hybrid
        self.hybrid_alpha = hybrid_alpha
        self.H_fisher = None  # 存储Fisher矩阵

    def compute_fisher_hessian(self):
        """
        计算Fisher信息矩阵 (SlimGPT方式)
        H_fisher = (1/n) * X^T X
        """
        H_fisher = torch.zeros(self.columns, self.columns, device=self.dev)

        for inp_batch in self.inp_cache:
            # 展平输入: [batch, seq, embed] -> [batch*seq, embed]
            inp_flat = inp_batch.reshape(-1, self.columns).t()  # [embed, batch*seq]

            # 累积协方差矩阵
            H_fisher += inp_flat @ inp_flat.t()

        # 归一化
        total_samples = sum(inp.shape[0] * inp.shape[1] for inp in self.inp_cache)
        H_fisher = H_fisher / (total_samples ** 2) * 2.0

        return H_fisher

    def compute_hybrid_hessian(self):
        """
        混合Hessian: H = α·H_fisher + (1-α)·H_fastoba
        """
        # 1. 计算Fisher Hessian (稳定基线)
        H_fisher = self.compute_fisher_hessian()

        # 2. 当前的H已经是FastOBA计算的
        H_fastoba = self.H.clone()

        # 3. 加权组合
        H_hybrid = self.hybrid_alpha * H_fisher + (1.0 - self.hybrid_alpha) * H_fastoba

        return H_hybrid

    def add_batch_fastoba(self, inp, out, stage_id=None):
        """修改现有方法，支持混合Hessian"""
        # ... 原有的Hessian计算代码 ...

        # 在累积Hessian之后
        if self.use_hybrid:
            self.H = self.compute_hybrid_hessian()
```

#### 使用示例

```python
# 创建剪枝器（启用混合模式）
pruner = FastOBAAttentionSlimGPT(
    layer=attention_layer,
    use_hybrid=True,      # 启用混合Hessian
    hybrid_alpha=0.7      # 70% Fisher + 30% FastOBA
)

# 正常使用
pruner.add_batch_fastoba(inp, out)
pruner.struct_prune(sparsity=0.4)
```

#### 调优建议

不同任务可能需要不同的α值：
- **稳定优先**（大模型、高稀疏度）：α=0.8-0.9
- **平衡**（一般场景）：α=0.7
- **准确优先**（小模型、低稀疏度）：α=0.5-0.6

可以通过交叉验证选择最佳α。

---

### 改进2: 带状对角Hessian 🟡

**优先级**: 中
**难度**: 中
**预期影响**: +5-15% 性能（head-dim剪枝）

#### 问题

当前FastOBA使用块对角Hessian，假设不同head完全独立：

```
H = [H_0   0    0   ]  ← Head 0
    [0     H_1  0   ]  ← Head 1
    [0     0    H_2 ]  ← Head 2
```

但实际上：
- 相邻head通常学习互补特征
- 完全独立假设过于严格
- 限制了OBS的全局权重补偿

#### 改进方案

使用**带状对角矩阵**，允许相邻head之间有相关性：

```
bandwidth=0 (当前):
H = [H_0   0    0    0  ]
    [0     H_1  0    0  ]
    [0     0    H_2  0  ]
    [0     0    0    H_3]

bandwidth=1 (改进):
H = [H_0   C_01  0    0  ]
    [C_10  H_1   C_12 0  ]
    [0     C_21  H_2  C_23]
    [0     0     C_32 H_3]

bandwidth=2:
H = [H_0   C_01  C_02  0  ]
    [C_10  H_1   C_12  C_13]
    [C_20  C_21  H_2   C_23]
    [0     C_31  C_32  H_3 ]
```

其中 `C_ij` 是head i和head j之间的交叉相关块。

#### 实现代码

```python
def compute_band_diagonal_hessian(self, weight_hessian, num_heads, head_dim, bandwidth=1):
    """
    构造带状对角Hessian，允许相邻head的相关性

    Args:
        weight_hessian: [out_features, in_features] FastOBA输出
        num_heads: 头数量（如12）
        head_dim: 每个头的维度（如64）
        bandwidth: 带宽（0=块对角, 1=三对角, 2=五对角）

    Returns:
        H: [in_features, in_features] 带状对角Hessian
    """
    in_features = weight_hessian.shape[1]
    H = torch.zeros(in_features, in_features, device=weight_hessian.device)

    for h in range(num_heads):
        start = h * head_dim
        end = (h + 1) * head_dim

        # 对角块（head自身）
        head_hessian = weight_hessian[:, start:end]
        H_block = head_hessian.t() @ head_hessian / weight_hessian.shape[0]
        H[start:end, start:end] = H_block

        # 非对角块（相邻head的交叉相关）
        for offset in range(1, bandwidth + 1):
            neighbor_h = h + offset
            if neighbor_h < num_heads:
                start_nb = neighbor_h * head_dim
                end_nb = (neighbor_h + 1) * head_dim

                # 计算head h和head neighbor_h的交叉相关
                hessian_nb = weight_hessian[:, start_nb:end_nb]
                C_block = head_hessian.t() @ hessian_nb / weight_hessian.shape[0]

                # 填充上三角和下三角（Hessian对称）
                H[start:end, start_nb:end_nb] = C_block
                H[start_nb:end_nb, start:end] = C_block.t()

    return H
```

#### 使用示例

```python
# 修改 fastoba_attention_slimgpt.py 的 compute_block_diagonal_hessian 调用

# 原始（块对角）:
H_block_diag = self.compute_block_diagonal_hessian(
    weight_hessian, self.num_heads, self.head_dim
)

# 改进（带状对角，bandwidth=1表示相邻head相关）:
H_band_diag = self.compute_band_diagonal_hessian(
    weight_hessian, self.num_heads, self.head_dim, bandwidth=1
)
```

#### 权衡

- **bandwidth=0**: 最快，但假设过强
- **bandwidth=1**: 平衡点，捕获主要相关性
- **bandwidth=2**: 更准确，但计算和存储成本增加
- **bandwidth=num_heads-1**: 完整矩阵，成本最高

推荐从 **bandwidth=1** 开始尝试。

---

### 改进3: 自适应阻尼 🟡

**优先级**: 中
**难度**: 低
**预期影响**: 减少Cholesky失败80%

#### 问题

当前使用固定阻尼参数：
```python
damp = self.percdamp * torch.diag(H).mean()  # percdamp=0.01
```

问题：
- 病态Hessian（条件数高）需要更大阻尼
- 良态Hessian（条件数低）过度阻尼会损失精度
- 固定阻尼无法适应不同层的特性

#### 改进方案

根据Hessian的条件数动态调整阻尼：

```python
def compute_adaptive_damping(self, H):
    """
    根据Hessian条件数自适应计算阻尼

    条件数 = λ_max / λ_min (最大/最小特征值比)
    条件数越大 → 矩阵越病态 → 需要更大阻尼
    """
    H_diag = torch.diag(H)

    # 快速估计条件数
    # 精确方法需要SVD，太慢；这里用对角元素估计
    max_eig_est = H_diag.max().item()
    min_eig_est = H_diag.min().item()
    cond_number = max_eig_est / (min_eig_est + 1e-10)

    # 根据条件数分级阻尼
    if cond_number > 1e6:
        # 极度病态：高阻尼
        damp_factor = 0.1
        print(f"[Adaptive Damping] High cond_number={cond_number:.2e}, damp_factor=0.1")
    elif cond_number > 1e4:
        # 中度病态：中等阻尼
        damp_factor = 0.01
    elif cond_number > 100:
        # 轻度病态：低阻尼
        damp_factor = 0.001
    else:
        # 良态：极低阻尼
        damp_factor = 0.0001

    damp = damp_factor * H_diag.mean()
    return damp

def compute_H_inverse_with_adaptive_damping(self, H):
    """
    使用自适应阻尼计算Hessian逆
    """
    damp = self.compute_adaptive_damping(H)
    H_damped = H + damp * torch.eye(H.shape[0], device=H.device)

    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H_damped))
        return Hinv, True  # 成功
    except RuntimeError as e:
        print(f"[Warning] Cholesky failed even with adaptive damping: {e}")
        # 回退到对角近似
        Hinv_diag = 1.0 / (torch.diag(H) + damp)
        Hinv = torch.diag(Hinv_diag)
        return Hinv, False  # 失败，使用回退
```

#### 使用示例

```python
# 在 struct_prune 或 struct_prune_head_dims 中替换现有的 Hinv 计算

# 原始:
damp = self.percdamp * torch.diag(H).mean()
Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + damp * torch.eye(...)))

# 改进:
Hinv, success = self.compute_H_inverse_with_adaptive_damping(H)
if not success:
    print("[Warning] Using diagonal approximation for this iteration")
```

---

### 改进4: 多层次伪损失 🟢

**优先级**: 低
**难度**: 中
**预期影响**: +5-10% 性能

#### 当前问题

单一的方差损失可能还不够：
```python
loss = -variance  # 只考虑信息量
```

#### 改进方案

组合多个目标：

```python
def compute_multi_objective_pseudo_loss(self):
    """
    多目标伪损失：综合考虑多个因素
    """
    total_loss = 0

    for inp_batch in self.inp_cache:
        inp_batch = inp_batch.to(device).requires_grad_(True)
        out_batch = self.attention_module(inp_batch)

        # 目标1: 最大化输出方差（信息量）
        mean_out = out_batch.mean(dim=-1, keepdim=True)
        variance = ((out_batch - mean_out) ** 2).sum()
        loss_variance = -variance

        # 目标2: 最大化输出对角优势（特征独立性）
        # 鼓励不同维度学习不同特征
        out_flat = out_batch.reshape(-1, out_batch.shape[-1])
        gram_matrix = out_flat.t() @ out_flat
        diag_dominance = torch.diag(gram_matrix).sum() / gram_matrix.abs().sum()
        loss_independence = -diag_dominance

        # 目标3: 最小化输出重构误差（如果有输入）
        # 假设良好的attention应该能保持输入的主要信息
        reconstruction_error = ((out_batch - inp_batch) ** 2).sum()
        loss_reconstruction = reconstruction_error

        # 加权组合（可调）
        loss_batch = (
            0.6 * loss_variance +         # 60% 信息量
            0.3 * loss_independence +     # 30% 独立性
            0.1 * loss_reconstruction     # 10% 重构
        )

        total_loss = total_loss + loss_batch

    return total_loss / len(self.inp_cache)
```

---

### 改进5: 分层方法选择 🟢

**优先级**: 低
**难度**: 低
**预期影响**: +3-8% 性能

#### 核心思想

不同层特性不同，自动选择最适合的方法：

```python
def select_pruning_method(layer_idx, total_layers, layer_type):
    """
    根据层的位置和类型选择剪枝方法

    观察：
    - 早期层：更结构化，FastOBA效果好
    - 后期层：更混沌，SlimGPT更稳定
    - Attention层：适合head-wise
    - FFN层：适合channel-wise
    """
    relative_depth = layer_idx / total_layers

    if layer_type == 'attention':
        if relative_depth < 0.3:
            # 早期attention: FastOBA + head-wise
            return {
                'method': 'fastoba',
                'granularity': 'head_wise',
                'use_hybrid': False
            }
        elif relative_depth < 0.7:
            # 中期attention: 混合方法
            return {
                'method': 'fastoba',
                'granularity': 'head_dim',
                'use_hybrid': True,
                'hybrid_alpha': 0.7
            }
        else:
            # 后期attention: SlimGPT更稳定
            return {
                'method': 'slimgpt',
                'granularity': 'head_dim',
                'use_hybrid': False
            }
    elif layer_type == 'ffn':
        # FFN层：一般用SlimGPT
        return {
            'method': 'slimgpt',
            'granularity': 'channel_wise',
            'use_hybrid': False
        }
    else:
        # 默认
        return {
            'method': 'slimgpt',
            'granularity': 'unstructured',
            'use_hybrid': False
        }

# 使用示例
config = select_pruning_method(layer_idx=5, total_layers=12, layer_type='attention')
if config['method'] == 'fastoba':
    pruner = FastOBAAttentionSlimGPT(..., use_hybrid=config['use_hybrid'])
else:
    pruner = SlimGPT(...)
```

---

### 改进6: 迭代Hessian细化 🟢

**优先级**: 低
**难度**: 中
**预期影响**: +5-10% 性能，但慢3x

#### 核心思想

剪枝后模型结构改变，Hessian也应该更新：

```python
def iterative_pruning_with_hessian_refinement(pruner, target_sparsity, n_iterations=3):
    """
    迭代剪枝：每次剪枝后重新计算Hessian

    Args:
        pruner: 剪枝器实例
        target_sparsity: 最终稀疏度（如0.4）
        n_iterations: 迭代次数（推荐2-3次）
    """
    sparsity_per_iter = target_sparsity / n_iterations

    for iteration in range(n_iterations):
        print(f"[Iteration {iteration+1}/{n_iterations}] Pruning {sparsity_per_iter*100:.1f}%...")

        # 1. 基于当前Hessian剪枝
        pruner.struct_prune(sparsity=sparsity_per_iter, granularity='head_dim')

        # 2. 重新收集数据并计算Hessian
        if iteration < n_iterations - 1:  # 最后一次不需要重新计算
            print(f"[Iteration {iteration+1}] Recomputing Hessian...")
            pruner.H.zero_()  # 清空旧Hessian
            pruner.nsamples = 0

            # 重新前向传播收集统计量
            for inp, out in calibration_data:
                pruner.add_batch_fastoba(inp, out)

    print(f"[Complete] Total sparsity: {target_sparsity*100:.1f}%")
    return pruner

# 使用示例
pruner = FastOBAAttentionSlimGPT(...)

# 收集初始Hessian
for inp, out in calibration_data:
    pruner.add_batch_fastoba(inp, out)

# 迭代剪枝
pruner = iterative_pruning_with_hessian_refinement(
    pruner,
    target_sparsity=0.4,
    n_iterations=3
)
```

**权衡**：
- 优点：更准确，适应结构变化
- 缺点：慢3倍（需要3次前向传播）

推荐在最终调优阶段使用。

---

## 📊 优先级总结

### 立即实施（已完成）✅
1. ✅ 修复 struct_prune_head_dims 矩阵切片bug
2. ✅ 改进伪损失定义（方差损失）

### 高优先级（推荐下一步）🔴
3. 🟡 混合Hessian方法（预期+10-20%性能）
4. 🟡 自适应阻尼（减少失败率）

### 中优先级（有时间可做）🟡
5. 🟡 带状对角Hessian（改善head-dim剪枝）
6. 🟢 多层次伪损失
7. 🟢 分层方法选择

### 低优先级（研究性质）🟢
8. 🟢 迭代Hessian细化

---

## 🧪 实验验证计划

### 验证阶段1修复效果

```bash
# 1. 重新运行对比实验
python compare_pruning_methods.py \
    --method fastoba_head \
    --sparsity 0.25 \
    --save_results results_after_fix.json

# 2. 对比修复前后
python analyze_results.py \
    --before comparison_results_attention_fastoba.json \
    --after results_after_fix.json
```

预期结果：
- FastOBA head: 应该保持最佳（相对误差 < 0.25）
- FastOBA head-dim: 应该超越SlimGPT head-dim

### 消融实验（测试各项改进）

```python
# configs.py
ABLATION_CONFIGS = [
    {'name': 'baseline', 'use_hybrid': False, 'bandwidth': 0},
    {'name': 'hybrid_only', 'use_hybrid': True, 'hybrid_alpha': 0.7, 'bandwidth': 0},
    {'name': 'band_only', 'use_hybrid': False, 'bandwidth': 1},
    {'name': 'hybrid+band', 'use_hybrid': True, 'hybrid_alpha': 0.7, 'bandwidth': 1},
    {'name': 'full', 'use_hybrid': True, 'hybrid_alpha': 0.7, 'bandwidth': 1, 'adaptive_damp': True}
]
```

---

## 📚 参考文献

1. **OBS原始论文**: Hassibi & Stork (1993) - "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon"
2. **KFAC**: Martens & Grosse (2015) - "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
3. **FastOBA**: 基于automatic differentiation的Hessian计算
4. **SlimGPT**: 基于Fisher Information的剪枝方法

---

## 📝 更新日志

- **2025-11-03**: 初始版本，记录阶段2-3改进方案
- 阶段1关键修复已完成并合入主代码

---

## ❓ 问题与讨论

如有疑问或新的改进想法，请记录在此：

- [ ] 是否考虑Hutchinson trace estimator来近似对角Hessian？
- [ ] 能否用低秩分解（SVD）来压缩Hessian？
- [ ] 对于VAR的多尺度特性，是否需要尺度感知的Hessian？

---

**文档结束**
