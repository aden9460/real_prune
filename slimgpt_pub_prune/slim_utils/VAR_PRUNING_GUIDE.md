# VAR模型剪枝方法说明
# VAR Model Pruning: Full-KFAC-OBS + Chain Pruning + Layer Linkage

**创建日期 | Created**: 2025-11-05
**方法 | Method**: Full-KFAC-OBS + 链式剪枝 + 层级联动
**核心创新 | Core Innovation**: 使用层间输出差异构建伪损失 + 多尺度token拼接

---

## 目录 | Table of Contents

1. [概述 | Overview](#overview)
2. [核心创新：链式剪枝 | Core Innovation: Chain Pruning](#chain-pruning)
3. [数学基础 | Mathematical Foundations](#mathematical-foundations)
4. [层级联动 | Layer Linkage for Multi-Scale](#layer-linkage)
5. [完整实现流程 | Complete Implementation Workflow](#implementation)
6. [代码实现 | Code Implementation](#code)
7. [对比分析 | Comparison Analysis](#comparison)

---

## 概述 | Overview {#overview}

### VAR模型的挑战

VAR (Visual AutoRegressive) 模型采用多尺度渐进式生成，具有以下特点：

- **10个尺度层级**：1×1, 2×2, ..., 16×16
  - Scale 0: 1 token
  - Scale 1: 4 tokens
  - ...
  - Scale 9: 256 tokens
  - **Total**: 341 tokens

- **渐进式生成**：每次前向传播生成一个尺度
- **注意力机制复杂**：包含QKV投影、softmax、多头注意力等非线性操作

### 为什么使用Full-KFAC-OBS？

**F2近似的缺陷**：

```python
# F2 Surgery (不完整)
δW = -coeff @ W  # 缺少 A_inv

# Full-KFAC Surgery (完整)
δW = -G_inv @ coeff @ A_inv  # 同时使用 G_inv 和 A_inv
```

**关键区别**：
- F2在**权重补偿**阶段缺少输入特征相关性（A_inv）
- 对于attention层的O矩阵，不同head的输入高度相关
- **必须使用Full-KFAC-OBS**确保补偿准确性

### 本方法的三大创新

1. **链式剪枝（Chain Pruning）**：使用层间输出差异作为伪损失
2. **层级联动（Layer Linkage）**：拼接多尺度token计算综合Hessian
3. **尺度均衡（Scale Balancing）**：平均各尺度重要性，确保所有分辨率均衡优化

---

## 核心创新：链式剪枝 | Core Innovation: Chain Pruning {#chain-pruning}

### 问题：传统方法的局限

**传统SlimGPT方法**：

```python
H = X^T @ X  # 仅使用输入激活的协方差
```

**局限性**：
- ❌ 只包含一阶信息（输入统计）
- ❌ 不包含attention的softmax非线性
- ❌ 不包含QKV之间的交互
- ❌ 缺乏明确的优化目标

### 解决方案：基于伪损失的链式剪枝

**核心思想**：使用**剪枝后层输出与原始层输出的差异**作为损失函数

#### 伪损失定义

**来源**：`KFAC_TECHNICAL_GUIDE.md:1344`

```python
# 1. 记录原始输出（作为target）
with torch.no_grad():
    original_output = layer(input)  # [B, L, C]

# 2. 剪枝后前向传播
pruned_output = pruned_layer(input)  # [B, L, C]

# 3. 伪损失：输出差异的MSE
loss = F.mse_loss(pruned_output, original_output)

# 4. 反向传播获得梯度
loss.backward()
```

**数学表达**：

```
L_pseudo = ||f(x; W_pruned) - f(x; W_original)||²

其中：
- f(x; W): attention层的输出函数
- W_pruned: 剪枝后的权重（某些维度置零）
- W_original: 原始权重
```

#### 为什么这样更有意义？

| 特性 | SlimGPT (H = X^T @ X) | 链式剪枝 (基于伪损失) |
|------|---------------------|---------------------|
| **信息阶数** | 一阶（输入协方差） | 二阶（Hessian） |
| **包含softmax** | ❌ | ✅ 自动通过反向传播 |
| **包含QKV交互** | ❌ | ✅ 通过链式法则 |
| **优化目标** | 隐式 | 显式（最小化输出变化） |
| **理论基础** | 经验性 | Taylor展开 |

#### 自动包含复杂非线性

**关键优势**：通过PyTorch自动微分，伪损失的梯度**自动包含**：

1. **Softmax的Jacobian**：
   ```python
   # 不需要手动计算！
   ∂softmax/∂(QK^T) 自动通过 loss.backward() 计算
   ```

2. **Q/K/V的交互**：
   ```python
   # 链式法则自动传播
   ∂L/∂W_out = ∂L/∂O × ∂O/∂attn × ∂attn/∂(QKV) × ∂(QKV)/∂W_out
   ```

3. **多头注意力的耦合**：
   ```python
   # 不同head之间的相关性自动体现在梯度中
   ```

### KFAC如何使用伪损失的梯度

**KFAC的核心思想**：Fisher信息矩阵的Kronecker分解

```
Fisher矩阵：
F = E[(∇_W L) (∇_W L)^T]

对于线性层 y = Wx:
∇_W L = (∇_y L) ⊗ x = g ⊗ x

因此：
F ≈ E[g g^T] ⊗ E[x x^T] = G ⊗ A

其中：
- A = E[x x^T]: 输入激活协方差 [in_dim, in_dim]
- G = E[g g^T]: 输出梯度协方差 [out_dim, out_dim]
- g = ∇_y L:   从伪损失反向传播得到的梯度 ← 关键！
```

**传统方法 vs 链式剪枝**：

```python
# 传统SlimGPT：直接用输入构造H
H = X^T @ X
# 问题：这不是真正的Fisher矩阵，只是输入协方差

# 链式剪枝：通过伪损失获得真正的Fisher矩阵
loss = ||pruned_out - original_out||²
loss.backward()
# 从hooks获得:
A = X^T @ X          # 输入激活协方差
G = g^T @ g          # 梯度协方差 (g = ∂loss/∂y)
F = G ⊗ A            # 真正的Fisher近似
```

### 完整的链式剪枝流程

```
1. 记录原始层输出
   ↓
2. 对O矩阵执行初步剪枝（可用SlimGPT或随机）
   ↓
3. 前向传播得到剪枝后输出
   ↓
4. 计算伪损失 = MSE(pruned_output, original_output)
   ↓
5. 反向传播，通过hooks收集：
   - A矩阵：输入激活 x 的协方差
   - G矩阵：输出梯度 g 的协方差
   ↓
6. 构造Fisher矩阵：F = G ⊗ A
   ↓
7. 计算逆：F^(-1) = G^(-1) ⊗ A^(-1)
   ↓
8. 使用Full-KFAC-OBS执行精细剪枝
   ↓
9. 同步删除QKV对应维度
```

---

## 数学基础 | Mathematical Foundations {#mathematical-foundations}

### 1. 伪损失与Hessian

**目标**：最小化剪枝对模型输出的影响

```
优化目标：
min ||f(x; W_pruned) - f(x; W_original)||²

泰勒展开（二阶）：
f(x; W + δW) ≈ f(x; W) + ∇_W f · δW + (1/2) δW^T H δW

其中 H 是 Hessian 矩阵：
H = ∂²L / ∂W²
```

**对于attention层**：

```
f(x; W) = W_out @ softmax(Q K^T / √d) @ V

L_pseudo = ||f(x; W)||²

H = ∂²L_pseudo / ∂W_out²
```

### 2. KFAC近似

**Kronecker积分解**：

```
Fisher矩阵：
F = E[(∇_W L) (∇_W L)^T]

KFAC近似：
F ≈ G ⊗ A

其中：
G = E[g ⊗ g], g = ∂L/∂y  [out_dim, out_dim]
A = E[x ⊗ x], x = input   [in_dim, in_dim]
```

**逆矩阵的Kronecker性质**：

```
(G ⊗ A)^(-1) = G^(-1) ⊗ A^(-1)

优势：
- 原问题：求 [mn, mn] 矩阵的逆
- KFAC：只需求 [m,m] 和 [n,n] 矩阵的逆
- 复杂度：O((mn)³) → O(m³ + n³)
```

### 3. Full-KFAC-OBS重要性计算

**神经元重要性（输出维度）**：

```python
# 提取对角元素
A_inv_diag = diag(A^(-1))  # [in_dim]
G_inv_diag = diag(G^(-1))  # [out_dim]

# 元素重要性
importance[i, j] = W[i, j]² / (G_inv_diag[i] × A_inv_diag[j])

# 神经元重要性（沿输入维度求和）
neuron_importance[i] = Σ_j importance[i, j]
```

**直观理解**：

```
重要性 = 权重能量 / 冗余度

- 分子 W²: 权重的贡献大小
- 分母 G_inv_diag[i]: 输出神经元i的冗余度
- 分母 A_inv_diag[j]: 输入特征j的冗余度
```

### 4. Full-KFAC-OBS手术公式

**权重补偿**：

```python
# 系数矩阵
coeff = W / (G_inv_diag[:, None] @ A_inv_diag[None, :])

# 保留的神经元系数置零
coeff[keep_indices, :] = 0

# 权重更新（关键：同时使用 G_inv 和 A_inv）
δW = -G_inv @ coeff @ A_inv

# 更新后的权重
W_new = W + δW
```

**与F2的对比**：

| 方法 | 手术公式 | 包含信息 |
|------|---------|---------|
| **Full-KFAC** | `δW = -G_inv @ coeff @ A_inv` | 输出相关性 + 输入相关性 |
| **F2** | `δW = -coeff @ W` | 仅输出相关性（近似） |

---

## 层级联动 | Layer Linkage for Multi-Scale {#layer-linkage}

### 问题：VAR的多尺度挑战

**VAR的渐进式生成**：
- 每次前向传播只生成**一个尺度**的token
- 不同尺度token数量差异大（1 vs 256）
- 如何计算反映**所有尺度**的Hessian？

### 解决方案1：Token拼接（层级联动）

**核心代码**（`prune_v6.py:474`）：

```python
_cache_dict = {}  # 全局缓存

def add_batch(name):
    def hook_func(_, inp, out):
        if name not in _cache_dict:
            _cache_dict[name] = []

        # 缓存当前尺度的输入输出
        _cache_dict[name].append((
            inp[0].detach(),  # [B, L_s, C]
            out.detach()      # [B, L_s, C_out]
        ))

        # 收集10个尺度后处理
        if len(_cache_dict[name]) >= 10:
            inps = [p[0] for p in _cache_dict[name]]
            outs = [p[1] for p in _cache_dict[name]]

            # ★ 关键：沿序列维度(dim=1)拼接
            inp_cat = torch.cat(inps, dim=1)  # [B, 341, C]
            out_cat = torch.cat(outs, dim=1)  # [B, 341, C_out]

            # 使用拼接后的数据计算Hessian
            pruner_dict[name].add_batch(inp_cat, out_cat)

            _cache_dict[name] = []  # 清空缓存

    return hook_func
```

**为什么沿dim=1拼接？**

```python
# 各尺度的tensor形状
Scale 0: [batch, 1, hidden]      # 1×1 = 1 token
Scale 1: [batch, 4, hidden]      # 2×2 = 4 tokens
...
Scale 9: [batch, 256, hidden]    # 16×16 = 256 tokens

# 拼接后
inp_cat: [batch, 341, hidden]  # 1+4+...+256 = 341 tokens
```

**数学原理**：

```
单尺度Hessian：
H_s = X_s^T @ X_s  # [C, C]

拼接后Hessian：
H_all = X_cat^T @ X_cat
      = [X₀, X₁, ..., X₉]^T @ [X₀, X₁, ..., X₉]
      = Σ_s (X_s^T @ X_s)
      = Σ_s H_s

物理意义：
H_all 是所有尺度Hessian的加权和，权重 = token数量占比
```

**自动加权**：

```python
权重 w_s = n_s / N

Scale 0 (1 token):   w₀ = 1/341 = 0.3%
Scale 1 (4 tokens):  w₁ = 4/341 = 1.2%
...
Scale 9 (256 tokens): w₉ = 256/341 = 75.1%
```

### 解决方案2：尺度均衡（Scale Balancing）

**问题**：token拼接导致大尺度主导Hessian

**改进**：平均每个尺度的重要性

```python
def add_batch_scale_balanced(self, inp, out):
    """
    尺度均衡版本：确保所有尺度贡献相等
    """
    # 缓存当前尺度
    self._var_cache.append({
        'inp': inp.detach(),
        'out': out.detach(),
        'seqlen': inp.shape[1]
    })

    # 收集10个尺度后处理
    if len(self._var_cache) < 10:
        return

    # 初始化局部Hessian
    H_local = torch.zeros_like(self.H)

    # 遍历所有尺度
    for s, entry in enumerate(self._var_cache):
        x = entry['inp']
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        X_s = x.t()  # [hidden, n_tokens_s]

        seqlen_s = X_s.shape[1]

        # ★ 按序列长度归一化，使每个尺度贡献相等
        norm_factor = 1.0 / seqlen_s

        # 可选：不同尺度的权重
        gamma_s = scale_weights[s]  # 默认 gamma_s = 1.0

        H_local += gamma_s * norm_factor * (X_s @ X_s.t())

    # 与全局Hessian合并
    self.H = 0.9 * self.H + 0.1 * H_local

    # 清空缓存
    self._var_cache = []
```

**对比两种策略**：

| 版本 | 权重策略 | 各尺度贡献 | 优点 | 缺点 |
|------|---------|-----------|------|------|
| **Token拼接** | 自然加权 | 与token数成正比 | 符合实际分布 | 大尺度主导 |
| **尺度均衡** | 人工均衡 | 每个尺度相等 | 小尺度不被忽略 | 可能过度强调小尺度 |

**推荐策略**：

```python
# 方案A：纯token拼接（如果大尺度性能更关键）
use_scale_balancing = False

# 方案B：尺度均衡（如果希望所有尺度性能均衡）
use_scale_balancing = True

# 方案C：混合策略（推荐）
# 既拼接token，又进行归一化
# 权重 = sqrt(n_s) / Σ sqrt(n_s)
# 这样大尺度仍有更高权重，但不会过度主导
```

---

## 完整实现流程 | Complete Implementation Workflow {#implementation}

### 流程图

```
┌─────────────────────────────────────────────────────────┐
│ 步骤1：记录原始层输出（作为target）                      │
│ Step 1: Record Original Layer Outputs (as targets)     │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ with torch.no_grad():                    │
    │   for layer in model.blocks:             │
    │     original_output[layer] =             │
    │       layer(input).detach()              │
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤2：逐层剪枝（Chain Pruning）                         │
│ Step 2: Layer-by-Layer Pruning (Chain Pruning)         │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ for layer_idx, layer in enumerate(...): │
    │                                          │
    │   2a. 对O矩阵进行初步剪枝               │
    │   2b. 前向传播得到剪枝后输出            │
    │   2c. 计算伪损失                         │
    │   2d. 反向传播收集KFAC因子              │
    │   2e. 使用Full-KFAC-OBS精细剪枝        │
    │   2f. 同步删除QKV维度                    │
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤2a：对O矩阵进行初步剪枝                             │
│ Step 2a: Initial Pruning of O Matrix                   │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ # 可选方法：                              │
    │ # 选项1：随机剪枝                         │
    │ # 选项2：基于权重幅度                     │
    │ # 选项3：先用SlimGPT (H=X^TX)          │
    │                                          │
    │ # 这一步产生一个"候选剪枝"方案           │
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤2b-2c：前向传播 + 计算伪损失                         │
│ Step 2b-2c: Forward Pass + Compute Pseudo-Loss         │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ # 前向传播                                │
    │ pruned_output = layer(input)             │
    │                                          │
    │ # ★ 关键：伪损失定义                     │
    │ target_output = original_output[layer]   │
    │ loss = F.mse_loss(                       │
    │     pruned_output,                       │
    │     target_output                        │
    │ )                                        │
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤2d：反向传播，收集KFAC因子                           │
│ Step 2d: Backward Pass, Collect KFAC Factors           │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ # 注册hooks                               │
    │ activations = {}                         │
    │ gradients = {}                           │
    │                                          │
    │ def save_activation(name):               │
    │   def hook(module, inp, out):            │
    │     activations[name] = inp[0].data      │
    │   return hook                            │
    │                                          │
    │ def save_gradient(name):                 │
    │   def hook(module, grad_in, grad_out):  │
    │     gradients[name] = grad_out[0].data   │
    │   return hook                            │
    │                                          │
    │ # 反向传播                                │
    │ loss.backward()                          │
    │                                          │
    │ # 计算KFAC因子                            │
    │ A = activations['proj'].t() @            │
    │     activations['proj']                  │
    │ G = gradients['proj'].t() @              │
    │     gradients['proj']                    │
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤2e：使用Full-KFAC-OBS执行精细剪枝                    │
│ Step 2e: Fine-Grained Pruning with Full-KFAC-OBS       │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ # 计算逆矩阵                              │
    │ A_inv = torch.cholesky_inverse(          │
    │   torch.linalg.cholesky(A + damp*I)      │
    │ )                                        │
    │ G_inv = torch.cholesky_inverse(          │
    │   torch.linalg.cholesky(G + damp*I)      │
    │ )                                        │
    │                                          │
    │ # 计算重要性                              │
    │ importance = W² / (G_inv_diag ⊗ A_inv_diag)│
    │                                          │
    │ # 执行OBS手术                             │
    │ coeff = W / (G_inv_diag ⊗ A_inv_diag)   │
    │ coeff[keep_indices, :] = 0               │
    │ δW = -G_inv @ coeff @ A_inv              │
    │ W_new = W + δW                           │
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤2f：同步删除QKV对应维度                              │
│ Step 2f: Synchronize QKV Deletion                      │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ # O矩阵剪枝了输入维度[i1, i2, ...]      │
    │ # 则QKV的输出维度也要删除                │
    │                                          │
    │ Q_weight[pruned_dims, :] = 0             │
    │ K_weight[pruned_dims + embed_dim, :] = 0 │
    │ V_weight[pruned_dims + 2*embed_dim, :] = 0│
    └──────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 步骤3：多尺度Token处理（Layer Linkage）                  │
│ Step 3: Multi-Scale Token Handling (Layer Linkage)     │
└─────────────────────────────────────────────────────────┘
                          ↓
    ┌──────────────────────────────────────────┐
    │ # 在步骤2d收集A/G矩阵时，使用：         │
    │                                          │
    │ # 选项1：Token拼接                        │
    │ inp_cat = cat([inp₀, ..., inp₉], dim=1)  │
    │ A = inp_cat.t() @ inp_cat                │
    │                                          │
    │ # 选项2：尺度均衡                         │
    │ for s in range(10):                      │
    │   A += (1/seqlen_s) * (X_s.t() @ X_s)    │
    └──────────────────────────────────────────┘
                          ↓
                     ┌─────────┐
                     │  完成！  │
                     │  Done!  │
                     └─────────┘
```

### 关键步骤详解

#### 步骤1：记录原始输出

```python
def record_original_outputs(model, inputs):
    """
    记录模型所有层的原始输出

    Returns:
        original_outputs: dict {layer_idx: output_tensor}
    """
    original_outputs = {}

    with torch.no_grad():
        current_input = inputs
        for layer_idx, layer in enumerate(model.blocks):
            current_output = layer(current_input)
            original_outputs[layer_idx] = current_output.detach()
            current_input = current_output

    return original_outputs
```

#### 步骤2：链式剪枝

```python
def chain_prune_layer(
    layer,
    layer_idx,
    input,
    original_output,
    sparsity=0.4
):
    """
    对单层执行链式剪枝

    Args:
        layer: attention层
        layer_idx: 层索引
        input: 该层的输入
        original_output: 该层的原始输出（target）
        sparsity: 稀疏度
    """
    # 2a. 初步剪枝（可选，也可以直接用full weights）
    # initial_prune(layer, sparsity)

    # 2b. 前向传播
    pruned_output = layer(input)

    # 2c. 计算伪损失
    loss = F.mse_loss(pruned_output, original_output)

    # 2d. 收集KFAC因子
    A, G = collect_kfac_factors(layer, loss)

    # 2e. Full-KFAC-OBS剪枝
    prune_with_kfac_obs(layer, A, G, sparsity)

    # 2f. 同步QKV
    sync_prune_qkv(layer.attn, pruned_indices)
```

#### 步骤2d：收集KFAC因子（多尺度版本）

```python
def collect_kfac_factors_multiscale(
    layer,
    loss,
    use_scale_balancing=False
):
    """
    收集KFAC因子（A和G矩阵），支持多尺度

    Args:
        layer: attention层
        loss: 伪损失
        use_scale_balancing: 是否使用尺度均衡
    """
    activations_cache = []
    gradients_cache = []

    # 注册hooks
    def save_activation(module, inp, out):
        activations_cache.append(inp[0].detach())

    def save_gradient(module, grad_in, grad_out):
        gradients_cache.append(grad_out[0].detach())

    handle_fwd = layer.proj.register_forward_hook(save_activation)
    handle_bwd = layer.proj.register_backward_hook(save_gradient)

    # 反向传播
    loss.backward(retain_graph=True)

    # 移除hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # 计算A和G
    if not use_scale_balancing:
        # 选项1：直接拼接所有尺度
        activations = torch.cat(activations_cache, dim=1)  # [B, 341, C]
        gradients = torch.cat(gradients_cache, dim=1)

        # 展平
        x = activations.reshape(-1, activations.shape[-1])  # [B*341, C]
        g = gradients.reshape(-1, gradients.shape[-1])

        A = x.t() @ x / x.shape[0]
        G = g.t() @ g / g.shape[0]

    else:
        # 选项2：尺度均衡
        C = layer.proj.in_features
        A = torch.zeros(C, C, device=layer.proj.weight.device)
        G = torch.zeros(C, C, device=layer.proj.weight.device)

        for s, (act, grad) in enumerate(zip(activations_cache, gradients_cache)):
            x_s = act.reshape(-1, C)  # [B*L_s, C]
            g_s = grad.reshape(-1, C)

            seqlen_s = x_s.shape[0]
            norm_factor = 1.0 / seqlen_s

            A += norm_factor * (x_s.t() @ x_s)
            G += norm_factor * (g_s.t() @ g_s)

        # 归一化
        A = A / len(activations_cache)
        G = G / len(gradients_cache)

    return A, G
```

---

## 代码实现 | Code Implementation {#code}

### 完整VAR剪枝主函数

```python
"""
VAR模型剪枝：Full-KFAC-OBS + 链式剪枝 + 层级联动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def prune_var_model(
    var_model: nn.Module,
    calibration_data,
    sparsity: float = 0.4,
    headsize: int = 64,
    percdamp: float = 0.01,
    use_scale_balancing: bool = True,
    device: str = 'cuda'
):
    """
    VAR模型剪枝主函数

    Args:
        var_model: VAR模型
        calibration_data: 校准数据 [(images, labels), ...]
        sparsity: 稀疏度
        headsize: head维度（通常64）
        percdamp: 阻尼系数
        use_scale_balancing: 是否使用尺度均衡
        device: 设备
    """
    var_model.to(device)
    var_model.eval()

    print("="*80)
    print(f"VAR Model Pruning: Full-KFAC-OBS + Chain Pruning")
    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Head size: {headsize}")
    print(f"  Scale balancing: {use_scale_balancing}")
    print("="*80)

    # ========================================
    # 步骤1：记录所有层的原始输出
    # Step 1: Record original outputs
    # ========================================
    print("\n[Step 1] Recording original layer outputs...")
    original_outputs = {}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(calibration_data):
            images = images.to(device)
            labels = labels.to(device)

            current_input = images
            for layer_idx, layer in enumerate(var_model.blocks):
                current_output = layer(current_input)

                if layer_idx not in original_outputs:
                    original_outputs[layer_idx] = []

                original_outputs[layer_idx].append(current_output.detach())
                current_input = current_output

    print(f"  Recorded {len(calibration_data)} batches")

    # ========================================
    # 步骤2：逐层剪枝（链式剪枝）
    # Step 2: Layer-by-layer pruning (chain pruning)
    # ========================================
    current_input_batches = [img.to(device) for img, _ in calibration_data]

    for layer_idx, layer in enumerate(var_model.blocks):
        print(f"\n{'='*60}")
        print(f"[Step 2] Pruning Layer {layer_idx}/{len(var_model.blocks)}")
        print(f"{'='*60}")

        # 获取该层的原始输出（targets）
        target_outputs = original_outputs[layer_idx]

        # 2a. 前向传播（剪枝后）
        print("  [2a] Forward pass with current pruned weights...")
        pruned_outputs = []
        for inp_batch in current_input_batches:
            with torch.enable_grad():
                inp_batch = inp_batch.requires_grad_(True)
                out_batch = layer(inp_batch)
                pruned_outputs.append(out_batch)

        # 2b. 计算伪损失
        print("  [2b] Computing pseudo-loss (output difference)...")
        loss = 0
        for pruned_out, target_out in zip(pruned_outputs, target_outputs):
            loss += F.mse_loss(pruned_out, target_out)
        loss = loss / len(pruned_outputs)

        print(f"    Pseudo-loss: {loss.item():.6f}")

        # 2c. 收集KFAC因子（A和G矩阵）
        print("  [2c] Collecting KFAC factors (A and G)...")
        A, G = collect_kfac_factors_multiscale(
            layer=layer,
            inputs=current_input_batches,
            targets=target_outputs,
            use_scale_balancing=use_scale_balancing,
            device=device
        )

        print(f"    A shape: {A.shape}, G shape: {G.shape}")

        # 2d. 使用Full-KFAC-OBS剪枝
        print("  [2d] Pruning with Full-KFAC-OBS...")
        pruned_indices = prune_with_full_kfac_obs(
            layer=layer.attn.proj,
            A=A,
            G=G,
            sparsity=sparsity,
            headsize=headsize,
            percdamp=percdamp
        )

        print(f"    Pruned {len(pruned_indices)} dimensions")

        # 2e. 同步删除QKV
        print("  [2e] Synchronizing QKV pruning...")
        sync_prune_qkv(layer.attn, pruned_indices, headsize)

        # 更新current_input为下一层准备
        with torch.no_grad():
            current_input_batches = [
                layer(inp) for inp in current_input_batches
            ]

        print(f"  Layer {layer_idx} pruning complete!")

    print("\n" + "="*80)
    print("VAR Model Pruning Complete!")
    print("="*80)

    return var_model


def collect_kfac_factors_multiscale(
    layer,
    inputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    use_scale_balancing: bool,
    device: str
):
    """
    收集KFAC因子（支持多尺度VAR）

    Args:
        layer: attention层
        inputs: 输入batch列表
        targets: 目标输出列表（原始层输出）
        use_scale_balancing: 是否使用尺度均衡
        device: 设备
    """
    C = layer.attn.proj.in_features
    A = torch.zeros(C, C, device=device)
    G = torch.zeros(C, C, device=device)

    # 缓存多尺度数据
    _scale_cache = []

    # 注册hooks收集激活和梯度
    activations = []
    gradients = []

    def save_activation(module, inp, out):
        activations.append(inp[0].detach())

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = layer.attn.proj.register_forward_hook(save_activation)
    handle_bwd = layer.attn.proj.register_backward_hook(save_gradient)

    # 对每个batch进行前向+反向
    for inp_batch, target_batch in zip(inputs, targets):
        # 清空缓存
        activations.clear()
        gradients.clear()

        # 前向传播
        inp_batch = inp_batch.requires_grad_(True)
        out_batch = layer(inp_batch)

        # 计算伪损失
        loss = F.mse_loss(out_batch, target_batch)

        # 反向传播
        loss.backward(retain_graph=True)

        # 缓存当前尺度的数据
        if len(activations) > 0:
            _scale_cache.append({
                'activation': activations[0].detach(),
                'gradient': gradients[0].detach(),
                'seqlen': activations[0].shape[1]
            })

    # 移除hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # 计算A和G矩阵
    if not use_scale_balancing:
        # 选项1：直接拼接
        all_activations = torch.cat(
            [entry['activation'] for entry in _scale_cache],
            dim=1  # 拼接序列维度
        )
        all_gradients = torch.cat(
            [entry['gradient'] for entry in _scale_cache],
            dim=1
        )

        x = all_activations.reshape(-1, C)  # [B*341, C]
        g = all_gradients.reshape(-1, C)

        A = x.t() @ x / x.shape[0]
        G = g.t() @ g / g.shape[0]

    else:
        # 选项2：尺度均衡
        for entry in _scale_cache:
            x_s = entry['activation'].reshape(-1, C)
            g_s = entry['gradient'].reshape(-1, C)
            seqlen_s = entry['seqlen']

            # 归一化因子
            norm = 1.0 / seqlen_s

            A += norm * (x_s.t() @ x_s)
            G += norm * (g_s.t() @ g_s)

        # 取平均
        A = A / len(_scale_cache)
        G = G / len(_scale_cache)

    return A, G


def prune_with_full_kfac_obs(
    layer: nn.Linear,
    A: torch.Tensor,
    G: torch.Tensor,
    sparsity: float,
    headsize: int,
    percdamp: float
):
    """
    使用Full-KFAC-OBS执行剪枝

    Args:
        layer: 要剪枝的线性层（O矩阵）
        A: 输入激活协方差矩阵 [in_dim, in_dim]
        G: 输出梯度协方差矩阵 [out_dim, out_dim]
        sparsity: 稀疏度
        headsize: head维度
        percdamp: 阻尼系数

    Returns:
        pruned_indices: 被删除的维度索引列表
    """
    W = layer.weight.data.clone()
    in_features = W.shape[1]
    num_heads = in_features // headsize

    # 添加阻尼
    damp_A = percdamp * torch.mean(torch.diag(A))
    damp_G = percdamp * torch.mean(torch.diag(G))

    diag_A = torch.arange(A.shape[0], device=A.device)
    diag_G = torch.arange(G.shape[0], device=G.device)

    A[diag_A, diag_A] += damp_A
    G[diag_G, diag_G] += damp_G

    # 计算逆矩阵
    try:
        L_A = torch.linalg.cholesky(A)
        A_inv = torch.cholesky_inverse(L_A)

        L_G = torch.linalg.cholesky(G)
        G_inv = torch.cholesky_inverse(L_G)
    except:
        print("    Warning: Cholesky failed, using eigenvalue decomposition")
        eigvals_A, eigvecs_A = torch.linalg.eigh(A)
        eigvals_A = torch.clamp(eigvals_A, min=1e-6)
        A_inv = eigvecs_A @ torch.diag(1.0 / eigvals_A) @ eigvecs_A.t()

        eigvals_G, eigvecs_G = torch.linalg.eigh(G)
        eigvals_G = torch.clamp(eigvals_G, min=1e-6)
        G_inv = eigvecs_G @ torch.diag(1.0 / eigvals_G) @ eigvecs_G.t()

    # 提取对角元素
    A_inv_diag = torch.diag(A_inv)  # [in_dim]
    G_inv_diag = torch.diag(G_inv)  # [out_dim]

    # 计算重要性（按head）
    head_importances = []
    for h in range(num_heads):
        start = h * headsize
        end = start + headsize

        # 该head的权重
        W_head = W[:, start:end]
        A_inv_head = A_inv_diag[start:end]

        # 重要性：W² / (G_inv ⊗ A_inv)
        importance_head = (W_head ** 2) / (
            G_inv_diag.unsqueeze(1) @ A_inv_head.unsqueeze(0)
        )

        # Head重要性 = 所有维度重要性的和
        head_importance = importance_head.sum()
        head_importances.append(head_importance.item())

    head_importances = torch.tensor(head_importances, device=W.device)

    # 选择要删除的heads
    num_prune = int(num_heads * sparsity)
    pruned_heads = torch.argsort(head_importances)[:num_prune]

    print(f"    Head importances: {head_importances.tolist()}")
    print(f"    Pruning {num_prune}/{num_heads} heads")

    # 执行OBS手术
    pruned_indices = []
    for head_idx in pruned_heads:
        start = head_idx * headsize
        end = start + headsize
        pruned_indices.extend(range(start, end))

        # 计算误差
        Err = W[:, start:end] / A_inv_diag[start:end].unsqueeze(0)

        # 权重补偿（Full-KFAC）
        coeff = Err @ A_inv[start:end, start:end]
        W[:, start:end] -= G_inv @ coeff

        # 删除该head
        W[:, start:end] = 0

    # 更新权重
    layer.weight.data = W

    return pruned_indices


def sync_prune_qkv(attn_layer, pruned_indices, headsize):
    """
    同步删除QKV矩阵的对应维度

    Args:
        attn_layer: attention层
        pruned_indices: O矩阵被删除的输入维度
        headsize: head维度
    """
    embed_dim = attn_layer.proj.in_features

    with torch.no_grad():
        # Q
        attn_layer.mat_qkv.weight[pruned_indices, :] = 0
        if hasattr(attn_layer, 'q_bias') and attn_layer.q_bias is not None:
            attn_layer.q_bias[pruned_indices] = 0

        # K
        k_indices = [idx + embed_dim for idx in pruned_indices]
        attn_layer.mat_qkv.weight[k_indices, :] = 0

        # V
        v_indices = [idx + 2*embed_dim for idx in pruned_indices]
        attn_layer.mat_qkv.weight[v_indices, :] = 0
        if hasattr(attn_layer, 'v_bias') and attn_layer.v_bias is not None:
            attn_layer.v_bias[v_indices] = 0


if __name__ == '__main__':
    # 示例使用
    from models.var import VAR
    from torch.utils.data import DataLoader

    # 加载模型
    var_model = VAR(depth=16, embed_dim=1024, num_heads=16)

    # 准备校准数据
    # calibration_data = ...

    # 执行剪枝
    pruned_model = prune_var_model(
        var_model=var_model,
        calibration_data=calibration_data,
        sparsity=0.4,
        headsize=64,
        percdamp=0.01,
        use_scale_balancing=True,
        device='cuda'
    )

    # 保存
    torch.save(pruned_model.state_dict(), 'var_pruned.pth')
```

---

## 对比分析 | Comparison Analysis {#comparison}

### 方法对比

| 特性 | SlimGPT (H = X^T @ X) | 链式剪枝 (伪损失 + KFAC) |
|------|----------------------|-------------------------|
| **H矩阵来源** | 输入激活协方差 | Fisher信息矩阵 (G ⊗ A) |
| **信息阶数** | 一阶 | 二阶（Hessian） |
| **包含softmax** | ❌ | ✅ (自动通过反向传播) |
| **包含QKV交互** | ❌ | ✅ (链式法则) |
| **优化目标** | 隐式（保留输入方差）| 显式（最小化输出变化）|
| **理论基础** | 经验性 | Taylor展开 |
| **计算复杂度** | 低（只需前向） | 中（需要反向传播） |
| **补偿准确性** | 中等 | 高（Full-KFAC） |

### 为什么链式剪枝的H更有意义？

#### 数学视角

**SlimGPT**：
```
H = X^T @ X

这只是输入特征的协方差矩阵，它告诉我们：
- 输入特征之间的相关性
- 但不知道这些特征如何影响输出
```

**链式剪枝**：
```
F = G ⊗ A
  = E[g g^T] ⊗ E[x x^T]
  = E[(∇_y L) (∇_y L)^T] ⊗ E[x x^T]

这是真正的Fisher信息矩阵，它包含：
- A: 输入特征的协方差
- G: 输出梯度的协方差（从伪损失来）
- 梯度包含了attention内部所有非线性的影响
```

#### 物理直觉

想象你要修剪一棵树：

**SlimGPT方法**：
- 只看树干的粗细（输入统计）
- 不考虑这些树枝对整棵树的作用

**链式剪枝方法**：
- 看树干的粗细（输入）
- 也看去掉某根树枝后，整棵树会怎样变化（梯度）
- 更全面地评估每根树枝的重要性

### 实验对比（理论预期）

| 场景 | SlimGPT性能 | 链式剪枝性能 | 原因 |
|------|------------|------------|------|
| **低稀疏度 (<20%)** | 好 | 略好 | 差异不明显 |
| **中稀疏度 (20-40%)** | 中等 | 好 | 链式剪枝更准确 |
| **高稀疏度 (>40%)** | 差 | 好 | 准确补偿变得关键 |
| **Attention层** | 中等 | 好 | 链式剪枝捕获softmax |
| **FFN层** | 好 | 略好 | 线性层差异较小 |

---

## 总结 | Summary

### 核心方法

本文档介绍了VAR模型剪枝的完整方法，包含三大创新：

1. **Full-KFAC-OBS**
   - 使用完整的 `G_inv` 和 `A_inv` 进行权重补偿
   - 比F2近似更准确

2. **链式剪枝**
   - 使用层间输出差异作为伪损失：`L = ||pruned_out - original_out||²`
   - 通过反向传播自动包含softmax、QKV交互等非线性
   - 得到真正的Fisher信息矩阵

3. **层级联动 + 尺度均衡**
   - Token拼接：综合10个尺度的信息
   - 尺度均衡：平均每个尺度的重要性，确保小尺度不被忽略

### 关键公式

```
伪损失：
L = ||f(x; W_pruned) - f(x; W_original)||²

Fisher矩阵：
F ≈ G ⊗ A
G = E[g g^T], g = ∂L/∂y （从伪损失的梯度）
A = E[x x^T], x = input

逆矩阵：
F^(-1) = G^(-1) ⊗ A^(-1)

重要性：
importance = W² / (G_inv_diag ⊗ A_inv_diag)

手术：
δW = -G_inv @ coeff @ A_inv
```

### 适用场景

- ✅ VAR模型的attention层剪枝
- ✅ 多尺度生成模型
- ✅ 需要高精度权重补偿的场景
- ✅ 输入特征高度相关的网络

### 相关文档

- `KFAC_TECHNICAL_GUIDE.md` - KFAC理论和实现细节
- `OBA/fastoba_slimgpt_integration_plan.md` - OBA方法详解
- `KFAC_PRUNING_COMPREHENSIVE_GUIDE.md` - 全面的KFAC数学推导
- `prune_v6.py` - 实际代码实现

---

**创建时间**: 2025-11-05
**版本**: 1.0
**作者**: Based on KFAC-OBS and FastOBA research
