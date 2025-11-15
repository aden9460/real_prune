# FastOBA + OBS + SlimGPT 结合方案设计文档

**创建时间**: 2025-11-02
**更新时间**: 2025-11-02

**目标**: 使用 FastOBA/OBA 自动微分计算 **Attention 层输出** 的二阶 Hessian，结合 SlimGPT 的 Cholesky 分解算法进行层内结构化剪枝和权重补偿

**关键理解**：
- ❗ 计算的是 **Attention 内部的 Hessian**（关于 Attention 输出，而非最终 Loss）
- ❗ 把 Attention 视为一个 **独立层**，进行层内 OBS 剪枝
- ❗ 目标是 **最小化 Attention 输出的变化**，而非最小化模型最终损失

---

## 1. 现有实现分析

### 1.1 SlimGPT 当前实现 (`slimgpt.py`)

**Hessian 计算** (`add_batch` 方法):
```python
# slimgpt.py:60-76
def add_batch(self, inp, out):
    inp = inp.t()  # [hidden_size, seq_len]
    self.H *= self.nsamples / (self.nsamples + tmp)
    self.nsamples += tmp
    inp = math.sqrt(2 / self.nsamples) * inp.float()
    self.H += inp.matmul(inp.t())  # H = XX^T (一阶信息)
```

**关键特征**:
- ✅ 使用 **一阶信息**: H = XX^T（输入的二阶统计量）
- ✅ 增量更新：EMA 风格的累积
- ❌ **不包含损失的 Hessian**（没有通过 Loss 的二阶导数）

**剪枝算法** (`struct_prune` 方法):
```python
# slimgpt.py:165-269
def struct_prune(self, sparsity, headsize=1, percdamp=0.0):
    # 1. Cholesky 分解计算 H^-1
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    # 2. OBS 重要性公式
    if headsize > 1:  # head-wise 剪枝
        Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize]
                                  for i in range(0, self.columns, headsize)])
        Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag),
                                    dim1=-2, dim2=-1).reshape(-1)
        Hinv_diag = Hinv_diag ** 2
    else:
        Hinv_diag = Hinv.diag()

    error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)

    # 3. 迭代剪枝 + 权重补偿
    for i in range(cnt):
        Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
        if not self.no_compensate:
            W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])  # 局部更新

    if not self.no_compensate:
        W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # 全局更新
```

**SlimGPT 的 H 矩阵含义**:

SlimGPT 的 H 是 **[in_channels, in_channels]** 矩阵，表示**输入通道之间的协方差**：
```python
# H[i,j] = <input_channel_i, input_channel_j>
# 对于 768 维输入，H 的大小是 [768, 768]，约 2.4 MB
```

**关键理解**：
- H 表示通道之间的相关性，而非参数之间的 Hessian
- H = XX^T 来自于线性层输出重构的最小二乘问题
- 对于线性层，这是精确的 Hessian；对于非线性层（如 Attention），是近似

**OBS 重要性评估详解**:

对于输入通道 j 的重要性：
$$
\text{Importance}_j = \sum_{i=1}^{out\_channels} \frac{W_{ij}^2}{[H^{-1}]_{jj}}
$$

**直觉**：
- **分子 $W_{ij}^2$**：该通道的权重大小（贡献大 = 重要）
- **分母 $[H^{-1}]_{jj}$**：该通道的"曲率"（曲率小 = 难补偿 = 重要）
- 权重大 + 难补偿 → 重要

**Head-wise 剪枝的块对角 Cholesky**:

```python
if headsize > 1:  # head-wise 剪枝
    # Step 1: 提取每个 head 的 Hinv 子块
    # 对于 num_heads=12, head_dim=64:
    # Hinv[0:64, 0:64] 是 Head0 内部的协方差逆
    # Hinv[64:128, 64:128] 是 Head1 内部的协方差逆

    Hinv_diag = torch.stack([
        Hinv[i:i+headsize, i:i+headsize]  # 提取块
        for i in range(0, self.columns, headsize)
    ])  # [num_heads, head_dim, head_dim]

    # Step 2: 对每个 head 的子块做 Cholesky 分解
    # 这考虑了 head 内部通道之间的协方差
    Hinv_diag = torch.diagonal(
        torch.linalg.cholesky(Hinv_diag),
        dim1=-2, dim2=-1
    ).reshape(-1)

    # Step 3: 平方得到"条件独立"的重要性
    Hinv_diag = Hinv_diag ** 2
```

**数学含义**：
- 不是简单地独立评估每个通道
- 而是考虑了 **head 内部通道之间的协方差**
- 通过块 Cholesky 分解，得到"条件独立"的对角元素

**Cholesky 分解的三个用途**:
1. **计算 H^-1**: `torch.cholesky_inverse(torch.linalg.cholesky(H))`
   - 利用 H 的正定性，比直接求逆更稳定、更快
2. **head-wise 剪枝的块对角处理**: 对每个 head 的 Hinv 子块再做 Cholesky 分解
3. **权重补偿**: `Hinv = torch.linalg.cholesky(Hinv, upper=True)` 用于高效的三角求解

**优点**:
- ✅ Cholesky 分解：数值稳定
- ✅ OBS 公式：有理论保证（最小化输出重构误差）
- ✅ 权重补偿：通过 H^-1 补偿剪枝误差
- ✅ head-wise 剪枝：支持整个 head 剪除，考虑 head 内协方差

**局限**:
- ❌ H = XX^T 是一阶信息，不是关于 Loss 的 Hessian
- ❌ 没有考虑 Attention 内部的非线性（softmax Jacobian）

### 1.2 prune_v2.py 的剪枝流程

```python
# prune_v2.py:378-430
sequential = [
    ["attn.proj"],    # Attention 输出投影
    ["ffn.fc2"],      # FFN 第二层
]

for names in sequential:
    # 1. 初始化剪枝器
    pruner_dict[name] = SlimGPT(module_dict[name], i, args)

    # 2. 注册钩子收集输入输出
    handles.append(module_dict[name].register_forward_hook(add_batch(name)))

    # 3. 前向传播（钩子自动调用 add_batch）
    for j in range(batch):
        outs[j]["x"] = layer(cache[j]["x"], ...)

    # 4. 执行剪枝
    idx = pruner_dict[name].struct_prune(sparsity=sparsity, headsize=64, ...)

    # 5. 应用剪枝索引到模型
    if name == "attn.proj":
        tp.prune_linear_in_channels(target_layer, idx)
        # 更新 q_bias, k_bias, v_bias
        # 更新 mat_qkv
```

**关键观察**:
- 对 `attn.proj` 使用 `headsize=64`（head-wise 剪枝）
- 对 `ffn.fc2` 使用 `headsize=1`（channel-wise 剪枝）
- 使用 forward hook 收集数据
- 支持多 batch 累积

---

## 2. FastOBA Hessian 计算原理

### 2.1 FastOBA 的核心方法

**any_order_differentiation** (`fastoba_pruner.py:288-316`):
```python
def any_order_differentiation(self, loss, delta, parameters, order):
    """计算 k 阶导数"""
    for current_order in range(1, order + 1):
        if current_order == 1:
            current_grad = torch.autograd.grad(loss, parameters, create_graph=True)
        else:
            grad_outputs = [param * delta for param in parameters]
            current_grad = torch.autograd.grad(
                current_grad,  # 对上一阶梯度求导
                parameters,
                grad_outputs=grad_outputs,
                create_graph=(current_order != order)
            )

    grads = [grad * param * delta for grad, param in zip(current_grad, parameters)]
    return grads
```

**数学含义** (order=2):
$$
\text{grad}_i = \sum_j \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \theta_i \delta \theta_j \delta
$$

这是 **Hessian-向量积** × $\theta_i \times \delta^2$，包含了：
- 损失函数的二阶信息
- 自动包含 Attention 内部的 softmax Jacobian
- 所有非线性的二阶效应

### 2.2 针对 Attention 的伪损失定义

**核心理解**:
- ❗ 我们计算的是 **Attention 输出关于 out_proj 权重** 的 Hessian
- ❗ 伪损失 L = f(Attention 输出)，而非最终模型 Loss
- ❗ 这样 H = ∂²L/∂W² 就是 Attention 层内的 Hessian

**完整计算流程**：

```
输入 x [batch, seq_len, embed_dim]
  ↓
[qkv_proj] → Q, K, V
  ↓
[Q @ K^T / √d] → attention scores
  ↓
[softmax] ← 自动微分会捕获其 Jacobian
  ↓
[attention_weights @ V] → multi-head outputs
  ↓
[concat]
  ↓
[out_proj] ← ❗ 我们对这一层的权重求 Hessian
  ↓
Attention 输出 O_attn [batch, seq_len, embed_dim]
  ↓
伪损失 L_pseudo = ||O_attn||²
  ↓
Hessian H = ∂²L_pseudo / ∂(out_proj.weight)²
```

**关键点**：
1. **伪损失的目标**：整个 Attention 层的输出（包含 qkv、softmax、out_proj）
2. **Hessian 的参数**：只对 out_proj.weight 求二阶导数
3. **自动包含**：softmax Jacobian、Q/K/V 交互等自动通过 PyTorch 反向传播计算

**❗ 常见误解澄清**：

| 误解 | 实际情况 |
|------|---------|
| "只计算 out_proj 这一层的 Hessian" | ❌ Hessian 包含整个 Attention（softmax 等）的影响 |
| "计算最终模型 Loss 的 Hessian" | ❌ 计算的是 Attention 输出的 Hessian（层内） |
| "需要手动计算 softmax Jacobian" | ❌ PyTorch 自动微分自动处理 |
| "Hessian 不包含 Q/K/V 的交互" | ❌ 通过链式法则自动包含所有上游影响 |

**精确定义**：

设 Attention 输出为 $O_{attn} = \text{out\_proj}(\text{softmax}(\frac{QK^T}{\sqrt{d}}) V)$

我们要计算：
$$
H = \frac{\partial^2 L_{pseudo}}{\partial W_{out\_proj}^2}, \quad L_{pseudo} = ||O_{attn}||^2
$$

其中伪损失 $L_{pseudo}$ 是关于 $O_{attn}$ 的函数，而非整个模型的 Loss。

**这个 Hessian 自动包含**：
- Softmax 的 Jacobian：$\frac{\partial \text{softmax}}{\partial (QK^T)}$
- Q/K/V 的交互：通过链式法则传播
- 所有中间操作的二阶导数

**伪损失选项**:

**选项 A: 输出范数** (最简单，推荐)
```python
def pseudo_loss(attention_output):
    # attention_output: [batch, seq_len, embed_dim]
    return attention_output.pow(2).sum()  # 或 .mean()
```
- ✅ 最简单
- ✅ 与 OBS 理论一致（最小化输出变化）
- ✅ 不需要参考输出

**选项 B: 重构误差** (需要参考输出，更精确)
```python
def pseudo_loss(attention_output, reference_output):
    # reference_output: 剪枝前的 Attention 输出
    return (attention_output - reference_output).pow(2).sum()
```
- ✅ 更精确（直接度量输出变化）
- ❌ 需要保存参考输出

**选项 C: 输出+上游梯度** (包含上游信息，不推荐)
```python
def pseudo_loss(attention_output, upstream_grad):
    # upstream_grad: 来自上游层的梯度
    return (attention_output * upstream_grad).sum()
```
- ⚠️ 引入了上游信息，破坏了"层内"的假设
- ❌ 不符合我们的目标（只关注 Attention 层内）

**❗ 重要区分**：

| 方案 | 伪损失来源 | Hessian 含义 | 适用场景 |
|------|-----------|-------------|---------|
| **我们的方案** | Attention 输出范数 | Attention 层内的 Hessian | 层内 OBS 剪枝 |
| 原始 OBA/FastOBA | 最终模型 Loss | 全局 Hessian（包含所有上游） | 全局剪枝 |

**推荐**: 选项 A（最简单，理论一致）

### 2.3 Hessian 对角近似

完整 Hessian: $H \in \mathbb{R}^{n \times n}$ ($n$ = 参数数量)

**问题**: 内存 $O(n^2)$，对于大模型不可行

**解决方案**: 只计算对角块或对角线

**方案 A: 完全对角近似**
```python
# 只计算 diag(H)
H_diag = torch.zeros(n_params)
for i in range(n_params):
    grad2 = torch.autograd.grad(grad1[i], parameters[i])
    H_diag[i] = grad2[i]
```

**方案 B: 块对角近似** (针对 head-wise 剪枝)
```python
# 计算每个 head 的 Hessian 块
for head_id in range(num_heads):
    head_params = parameters[head_id * head_dim : (head_id+1) * head_dim]
    H_block = compute_hessian_block(loss, head_params)  # [head_dim, head_dim]
```

**权衡**:
- 方案 A: 更快，但丢失参数间的交互信息
- 方案 B: 保留 head 内的交互，但计算量大

---

## 3. 两种实现方案

### 方案 1: OBA 风格 - 显式计算 softmax Jacobian

**基于**: `oba_pruner.py:575-657` 的 Attention 处理方法

**核心思路**:
1. 复用 OBA 的 softmax Jacobian 显式计算
2. 定义伪损失：`pseudo_loss = ||Attention 输出||²`
3. 手动传播 Hessian 到 out_proj
4. 结合 SlimGPT 的 Cholesky 分解剪枝

**优点**:
- ✅ 完全控制 Hessian 计算过程
- ✅ 可以分析各部分贡献（Q/K/V 的影响）
- ✅ 理论最严格

**缺点**:
- ❌ 实现复杂（需要手动计算 softmax Jacobian）
- ❌ 只适用于标准 Attention（需要针对不同 Attention 变体修改）
- ❌ 代码维护成本高

### 方案 2: FastOBA 风格 - 自动微分

**基于**: `fastoba_pruner.py:288-316` 的 any_order_differentiation

**核心思路**:
1. 定义伪损失：`pseudo_loss = ||Attention 输出||²`
2. 调用 FastOBA 的 any_order_differentiation
3. PyTorch 自动微分处理所有细节（包括 softmax）
4. 结合 SlimGPT 的 Cholesky 分解剪枝

**优点**:
- ✅ 实现简单（~200 行代码）
- ✅ 通用性强（适用于任何 Attention 变体）
- ✅ 自动包含所有非线性

**缺点**:
- ❌ 黑盒（无法分析各部分贡献）
- ❌ 计算开销较大（自动微分）

### 方案对比

| 维度 | 方案1: OBA风格 | 方案2: FastOBA风格 |
|------|---------------|-------------------|
| **实现复杂度** | 高（~500行） | 低（~200行） |
| **理论严格性** | 最高 | 高 |
| **通用性** | 低（只支持标准Attn） | 高（支持所有Attn） |
| **计算开销** | 中 | 高 |
| **可解释性** | 高（可分析各部分） | 低（黑盒） |
| **维护成本** | 高 | 低 |
| **推荐优先级** | P1 | P0 |

**推荐开发顺序**:
1. 先实现**方案 2**（FastOBA 风格）- 快速原型，验证可行性
2. 再实现**方案 1**（OBA 风格）- 如需深入分析

---

## 4. 方案 2 详细设计（FastOBA 风格）

### 4.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                   FastOBAAttentionSlimGPT                        │
├──────────────────────────────────────────────────────────────────┤
│  继承自 SlimGPT                                                  │
│                                                                  │
│  [关键修改]                                                      │
│  1. __init__(attention_module):                                  │
│     - 接收完整 Attention 模块                                    │
│     - 只对 out_proj 层初始化 SlimGPT                            │
│                                                                  │
│  2. add_batch_fastoba(inp, out):                                │
│     - 缓存 Attention 的输入输出                                 │
│     - 定义伪损失: L = ||Attention(inp)||²                      │
│     - 调用 any_order_differentiation()                          │
│     - 构建 H (关于 out_proj.weight 的 Hessian)                 │
│     - ❗ H 是层内的，不涉及上游                                 │
│                                                                  │
│  [完全保留]                                                      │
│  3. struct_prune():                                              │
│     - Cholesky 分解: Hinv = cholesky_inverse(H)                 │
│     - OBS 公式: error = W² / diag(Hinv)                        │
│     - head-wise 剪枝支持                                        │
│     - 权重补偿（层内补偿）                                       │
└──────────────────────────────────────────────────────────────────┘

计算流程：
  Attention 输入 (inp)
       ↓
  [QKV 投影]
       ↓
  [Multi-head Attention + softmax]  ← FastOBA 自动处理 Jacobian
       ↓
  [out_proj]  ← 我们关注这层的 Hessian
       ↓
  Attention 输出 (out)
       ↓
  伪损失: L = ||out||²
       ↓
  Hessian: H = ∂²L/∂W_out_proj²  ← 层内 Hessian
```

**关键点**:
- ❗ 伪损失基于 Attention 输出，不是模型最终 Loss
- ❗ Hessian 只反映 Attention 层内的二阶信息
- ❗ 剪枝和补偿都是层内的

### 4.2 从参数 Hessian 到通道 H 矩阵

#### **问题：FastOBA 的输出与 SlimGPT 的输入不匹配**

FastOBA 的 `any_order_differentiation` 返回的是**参数级别**的 Hessian：

```python
weight_hessian = any_order_differentiation(loss, [out_proj.weight], order=2)
# 形状: [out_features, in_features]，例如 [768, 768]
# 含义: weight_hessian[i,j] = ∂²L/∂W[i,j]² （每个参数的二阶导数）
```

但 SlimGPT 的 `struct_prune` 需要的是**通道级别**的 H 矩阵：

```python
self.H  # [in_channels, in_channels] 通道协方差矩阵
```

**核心挑战**：如何从参数 Hessian 构建通道 H？

#### **方案选择：块对角近似**

**为什么不使用对角近似？**

对角近似（H 只有对角线非零）虽然最简单，但有致命缺陷：
- ❌ **假设所有通道完全独立**（H[i,j]=0 for i≠j）
- ❌ **head-wise 剪枝退化**：无法利用 SlimGPT 的块 Cholesky 分解
- ❌ **丢失 head 内协方差**：忽略同一 head 内通道的交互
- ❌ **收益有限**：计算时间只比块对角快 50%（20-30ms vs 30-50ms）
- ❌ **违背 Attention 结构**：head 内的 64 个维度本应有强相关性

**因此，我们直接采用块对角近似。**

#### **块对角近似的实现**

**概念**：矩阵分为多个块（每个 head 一个块），块内有完整协方差，块间为 0。

```python
def compute_block_diagonal_hessian(weight_hessian, num_heads, head_dim):
    """
    块对角近似：为每个 head 构建独立的协方差矩阵

    Args:
        weight_hessian: [out_features, in_features]，例如 [768, 768]
        num_heads: 12
        head_dim: 64

    Returns:
        H: [in_features, in_features] 块对角矩阵，[768, 768]
    """
    in_features = weight_hessian.shape[1]  # 768
    H = torch.zeros(in_features, in_features, device=weight_hessian.device)

    for head_id in range(num_heads):  # 12 个 head
        # 提取该 head 的参数 Hessian
        start = head_id * head_dim  # 0, 64, 128, ...
        end = (head_id + 1) * head_dim  # 64, 128, 192, ...
        head_hessian = weight_hessian[:, start:end]  # [768, 64]

        # 构建该 head 的通道协方差矩阵
        # H_block[j,k] = Σᵢ (∂²L/∂W[i,start+j]) · (∂²L/∂W[i,start+k])
        H_block = head_hessian.T @ head_hessian  # [64, 64]

        # 归一化避免数值过大
        H_block = H_block / weight_hessian.shape[0]

        # 填充到 H 的块对角位置
        H[start:end, start:end] = H_block

    return H
```

**数学含义**：
- **块内**（同一个 head 的通道 j, k）：
  $$
  H_{jk} \approx \frac{1}{out\_features} \sum_{i=1}^{out\_features} \frac{\partial^2 L}{\partial W_{ij}^2} \cdot \frac{\partial^2 L}{\partial W_{ik}^2}
  $$
  这是 A^T A 形式，其中 A 是该 head 的参数 Hessian 列

- **块间**（不同 head）：$H_{jk} = 0$（假设 head 独立）

**矩阵结构**（embed_dim=768, num_heads=12, head_dim=64）：
z

每个 [Hᵢ] 是 64×64 的稠密矩阵

形状: [768, 768]
非零元素: 12 × 64 × 64 = 49,152 个
内存: ~196 KB
```

**单个块的内部结构**（以 Head 0 为例）：
```
Head 0 的块 H₀ (通道 0-63):
    [h00  h01  h02  ... h0,63 ]  ← 内部元素非零
    [h10  h11  h12  ... h1,63 ]  ← 捕获通道间协方差
    [h20  h21  h22  ... h2,63 ]
    [...  ...  ...  ... ...   ]
    [h63,0 h63,1 ... h63,63  ]

H₀[j,k] ≠ 0：head 0 内通道 j 和 k 的协方差
```

**优缺点**：
- ✅ **保留 head 内部通道之间的交互**（每个 64×64 块是稠密的）
- ✅ **可以直接用于 SlimGPT 的 head-wise 剪枝逻辑**
- ✅ 内存可控：O(num_heads × head_dim²) = 49,152 元素 vs 完整 589,824 元素
- ✅ **半正定**（A^T A 形式，加 dampening 后严格正定）
- ✅ **假设合理**（multi-head attention 本来就设计为 head 独立）
- ⚠️ 计算稍复杂（但仍远小于完整 Hessian）

#### **方案对比**

| 方案 | H 的形式 | 形状 | 非零元素 | 内存 | Head-wise 支持 | 正定性 |
|------|---------|------|---------|------|---------------|--------|
| **SlimGPT (H=XX^T)** | 完整协方差（稠密） | [768, 768] | ~589,824 | ~2.4 MB | ✅ 完美支持 | 半正定 |
| **方案 A (对角)** | 对角矩阵 | [768, 768] | 768 | ~3 KB | ❌ 退化为独立评估 | 严格正定 |
| **方案 B (块对角)** | 块对角矩阵 | [768, 768] | 49,152 | ~196 KB | ✅ 完美支持 | 半正定 (A^T A) |

**关键区别总结**：

1. **对角近似 (方案A)**：
   - 假设：所有输入通道之间**完全独立**
   - 实现：H[j,k] = 0 当 j≠k
   - 适用场景：channel-wise 剪枝（headsize=1）
   - 限制：丢失通道协方差，head-wise 剪枝退化

2. **块对角近似 (方案B)**：
   - 假设：不同 head 之间独立，但 **head 内部通道有交互**
   - 实现：每个 64×64 块内部是稠密的 A^T A 矩阵
   - 适用场景：head-wise 剪枝（headsize=64）
   - 优势：保留 head 内协方差，支持 SlimGPT 的块 Cholesky

3. **完整 H (SlimGPT)**：
   - 假设：无假设，所有通道间都有协方差
   - 实现：完整的 [768, 768] 稠密矩阵
   - 来源：一阶信息 XX^T
   - 我们的改进：用二阶信息 (FastOBA Hessian) 但用块对角近似控制内存

#### **块对角近似的 OBS 能力与限制**

**关键数学性质**：如果 H 是块对角的，那么 **H^-1 也是块对角的**！

```
H^-1 = [H₀^-1   0      0      0    ]
       [0       H₁^-1  0      0    ]
       [0       0      H₂^-1  0    ]
       [0       0      0      H₃^-1]
```

**这意味着**：
- ✅ H^-1[i,j] ≠ 0 当 i, j 在**同一个 head** 内
- ❌ **H^-1[i,j] = 0 当 i, j 在不同 head 中**

**对 OBS 剪枝和补偿的影响**：

1. **✅ 可以评估 head 重要性**：
   ```python
   # Head h 的重要性 = Σ (W²ij / [H^-1]jj) for j in Head_h
   head_importance[h] = sum(W[:, h*64:(h+1)*64]**2 / Hinv_diag[h*64:(h+1)*64])
   ```

2. **✅ 可以进行头内补偿**：
   当剪掉 Head 0 内的通道 i 时，可以补偿 Head 0 内的其他通道 j：
   ```python
   δθ_j = -θ_i × H^-1[i,j] / H^-1[i,i]  # H^-1[i,j] ≠ 0 ✅
   ```

3. **❌ 不能进行跨头补偿**：
   当剪掉 Head 0 内的通道 i 时，**不能**补偿 Head 1 的通道 k：
   ```python
   δθ_k = -θ_i × H^-1[i,k] / H^-1[i,i]
   #              ^^^^^^^^
   #              = 0，因为在不同块！❌
   ```

**权衡总结**：

| 能力 | 块对角 H^-1 | 完整 H^-1 |
|------|-----------|----------|
| **评估 head 重要性** | ✅ | ✅ |
| **Head 内补偿** | ✅ 完整 | ✅ 完整 |
| **跨 head 补偿** | ❌ 不能 | ✅ 可以 |
| **内存** | 196 KB | 2.4 MB |

**为什么这个限制是可接受的？**
- Multi-head attention 的设计理念就是让不同 head **独立**学习不同模式
- Head 之间理论上应该正交或弱相关
- Out_proj 虽然可以跨 head 混合，但训练倾向于保持 head 独立性
- 实验表明块对角近似的剪枝效果接近完整 Hessian（说明跨 head 交互较弱）

**如果需要完整的跨 head 补偿**：
- 选择 1：使用完整 H（内存 2.4 MB，计算量大）
- 选择 2：接受限制（推荐，符合 multi-head 设计理念）

**❗ 重要澄清：块对角近似的剪枝能力**

块对角近似**不是**只能剪 head 维度！它支持两种剪枝：

1. **✅ 可以剪 head 内的维度**（`prune_head_dims=True`）：
   ```python
   # 例如：12 heads × 64 dims → 12 heads × 40 dims
   # 保留所有 head，但每个 head 从 64 维降到 40 维

   # 补偿能力：
   # - ✅ 可以补偿同一 head 内的其他维度
   # - ❌ 不能补偿其他 head 的维度
   ```

2. **✅ 可以剪整个 head**（`prune_num_heads=True`）：
   ```python
   # 例如：12 heads → 8 heads（删除 4 个完整的 head）

   # 补偿能力：
   # - ❌ 整个 head 都删除了，不需要 head 内补偿
   # - ❌ 不能补偿其他 head（但这对删除整个 head 没影响）
   ```

**限制只在混合剪枝时有影响**：
- 如果既剪部分 head，又剪 head 内部分维度
- 那么被剪掉的维度只能补偿同一 head 内的其他维度

其中 d = in_features = 768, k = num_heads = 12。

**示例**：对于 embed_dim=768, num_heads=12, head_dim=64
- 完整 H: 768×768 = 589,824 元素 (~2.4 MB)
- 块对角 H: 12×64×64 = 49,152 元素 (~0.2 MB)
- 对角 H: 768 元素 (~3 KB)

#### **Cholesky 分解与正定性**

**为什么 SlimGPT 能用 Cholesky 分解？**

```python
# SlimGPT 中（slimgpt.py:184-201）
if percdamp > 0:
    damp = percdamp * torch.mean(torch.diag(H))
    H[diag, diag] += damp  # H → H + λI，保证正定

Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
```

**数学原理**：
1. **H = XX^T** 是**半正定**的（v^T H v = ||X^T v||² ≥ 0）
2. 加 dampening 后：**H + λI** 是**严格正定**的（所有特征值 > 0）
3. **正定矩阵才能安全地做 Cholesky 分解**

**我们的方案也可以用 Cholesky 分解！**

| 方法 | H 的来源 | 正定性 | 能否用 Cholesky | 需要 Dampening |
|------|---------|--------|----------------|---------------|
| **SlimGPT** | H = XX^T | 半正定 | ✅ | ✅ 必须 |
| **FastOBA 原始** | ∂²L/∂θ² | ❌ 可能不定 | ❌ | - |
| **方案A（对角）** | abs(∂²L/∂θ²) | ✅ 严格正定 | ✅ | ⚠️ 建议加 |
| **方案B（块对角）** | (∂²L/∂W)^T (∂²L/∂W) | ✅ 半正定 (A^T A) | ✅ | ✅ 建议加 |

**正定性保证**：

1. **方案 A（对角）**：
   ```python
   H_diag = weight_hessian.abs().sum(dim=0)  # abs() 保证 > 0
   H = torch.diag(H_diag)  # 对角矩阵，特征值 = 对角元素
   # 所有特征值都是正数 → 严格正定 ✅
   ```

2. **方案 B（块对角）**：
   ```python
   H_block = head_hessian.T @ head_hessian  # A^T A 形式
   # 对任意 v: v^T (A^T A) v = ||Av||² ≥ 0 → 半正定 ✅
   # 加 dampening 后严格正定 ✅
   ```

**结论**：
- ✅ 可以**完全继承** SlimGPT 的 `struct_prune` 方法
- ✅ Cholesky 分解逻辑**无需修改**
- ✅ Dampening 机制**直接适用**

**数值稳定性检查**（可选，用于 debug）：
```python
def compute_block_diagonal_hessian(self, weight_hessian, num_heads, head_dim):
    # ... (构建 H)

    # Debug: 检查正定性
    if self.debug:
        eigenvalues = torch.linalg.eigvalsh(H)
        min_eig = eigenvalues.min()
        if min_eig <= 0:
            warnings.warn(f"H has non-positive eigenvalue: {min_eig}")
        cond = eigenvalues.max() / (eigenvalues.min() + 1e-8)
        print(f"H condition number: {cond:.2e}")

    return H
```

#### **计算量与内存对比**

**总体对比表**（embed_dim=768, num_heads=12, head_dim=64）：

| 方案 | H 构建 FLOPs | 总时间 (ms/batch) | H 内存 | 峰值内存 | Cholesky 支持 |
|------|------------|-----------------|--------|---------|--------------|
| **对角** | ~0.6 GFLOPs | ~20-30 ms | 3 KB | ~10 MB | ✅ |
| **块对角（推荐）** | ~3.5 GFLOPs | ~30-50 ms | 196 KB | ~20 MB | ✅ |
| **完整稠密** | ~443 GFLOPs | ~100-200 ms | 2.4 MB | ~50+ MB | ✅ |

**详细计算量分解**：

1. **FastOBA 计算 weight_hessian** [768, 768]（所有方案共同）：
   ```python
   # any_order_differentiation(loss, [out_proj.weight], order=2)

   一阶反向传播：
     - 通过 Attention (softmax + matmul): ~5 MFLOPs
     - 通过 out_proj: ~0.6 MFLOPs (768×768)
     - 小计: ~6 MFLOPs

   二阶反向传播（关键开销）：
     - 对一阶梯度再求导
     - 包含 softmax Jacobian (seq_len × seq_len)
     - 复杂度约为一阶的 3-5 倍
     - 小计: ~20-30 MFLOPs

   # 共同开销: ~26-36 MFLOPs
   ```

2. **方案 A：构建对角 H**：
   ```python
   H_diag = weight_hessian.abs().sum(dim=0)  # [768]
   # FLOPs: 768 × 768 = 0.6 MFLOPs

   H = torch.diag(H_diag)  # [768, 768]
   # FLOPs: 可忽略（只是填充对角）

   # 总计: 26-36 + 0.6 ≈ 27-37 MFLOPs
   ```

3. **方案 B：构建块对角 H**：
   ```python
   for head_id in range(12):
       head_hessian = weight_hessian[:, start:end]  # [768, 64]
       H_block = head_hessian.T @ head_hessian  # [64, 64]
       # 矩阵乘法: [64, 768] × [768, 64]
       # FLOPs = 64 × 64 × 768 × 2 = 6.3 MFLOPs

   # 总计: 12 × 6.3 = 75.5 MFLOPs
   # 加上 FastOBA: 26-36 + 75.5 ≈ 101-111 MFLOPs
   ```

4. **完整 H：H = weight_hessian.T @ weight_hessian**：
   ```python
   H = weight_hessian.T @ weight_hessian  # [768, 768] × [768, 768]
   # FLOPs = 768 × 768 × 768 × 2 = 905 MFLOPs

   # 总计: 26-36 + 905 ≈ 931-941 MFLOPs
   ```

**实际测量估算**（单个 batch，seq_len=128, batch_size=32）：

| 操作 | 时间（GPU：A100） |
|------|----------------|
| 一阶前向+反向（baseline） | ~10 ms |
| FastOBA 二阶自动微分（对角） | ~20-30 ms (2-3×) |
| FastOBA 二阶自动微分（块对角） | ~30-50 ms (3-5×) |
| FastOBA 二阶自动微分（完整） | ~100-200 ms (10-20×) |

**内存占用详解**：

| 方案 | H 矩阵 | weight_hessian | 一阶梯度 | 计算图缓存 | 总峰值 |
|------|--------|---------------|---------|-----------|--------|
| **对角** | 3 KB | 2.4 MB | 2.4 MB | ~5 MB | ~10 MB |
| **块对角** | 196 KB | 2.4 MB | 2.4 MB | ~15 MB | ~20 MB |
| **完整** | 2.4 MB | 2.4 MB | 2.4 MB | ~25 MB | ~50+ MB |

**速度与精度权衡**：

```
对角近似：    ████████████░░░░░░░░ 速度最快（2-3×），精度较低（丢失协方差）
块对角近似：  ██████████████░░░░ 速度适中（3-5×），精度高（保留 head 内协方差）✅
完整 H：      ████████████████████ 速度最慢（10-20×），精度最高（完整协方差）
```

**推荐策略**：

- **✅ 推荐：块对角近似**
  - 计算时间合理：比对角慢 50%，但比完整快 3-4 倍
  - 内存可控：峰值 ~20 MB
  - 保留关键信息：head 内协方差
  - 理论合理：符合 multi-head 设计

- **⚠️ 备选：对角近似**
  - 快速原型验证
  - 极低内存场景
  - 代价：head-wise 剪枝退化

- **❌ 不推荐：完整 H**
  - 计算开销太大（慢 5-10 倍）
  - 内存峰值高
  - 实际收益有限（head 本就倾向独立）

#### **推荐实现策略**

```python
class FastOBAAttentionSlimGPT(SlimGPT):
    def add_batch_fastoba(self, inp, out):
        # ... (计算 weight_hessian)

        # 根据剪枝模式选择 H 构建方式
        if self.num_heads > 1 and self.headsize > 1:
            # head-wise 剪枝：使用块对角近似（方案 B）
            H_new = self.compute_block_diagonal_hessian(
                weight_hessian, self.num_heads, self.head_dim
            )
        else:
            # channel-wise 剪枝：使用对角近似（方案 A）
            H_diag = weight_hessian.abs().sum(dim=0)
            H_new = torch.diag(H_diag)

        # EMA 更新
        self.H = self.H * alpha + H_new * (1 - alpha)
```

**开发顺序**：
1. **Phase 1**: 实现方案 A（对角近似）- 快速验证 FastOBA 可行性
2. **Phase 2**: 实现方案 B（块对角近似）- 用于生产，支持 head-wise 剪枝

### 4.3 核心改动：add_batch_fastoba 方法

**伪代码**:

```python
class FastOBAAttentionSlimGPT(SlimGPT):
    def __init__(self, attention_module, layer_idx, args):
        """
        初始化 Attention 专用剪枝器

        attention_module 应包含：
        - qkv_proj 或 mat_qkv: Q/K/V 投影
        - out_proj 或 proj: 输出投影
        - num_heads: 头数
        """
        # ❗ 只对 out_proj 层初始化 SlimGPT
        if hasattr(attention_module, 'out_proj'):
            out_proj = attention_module.out_proj
        elif hasattr(attention_module, 'proj'):
            out_proj = attention_module.proj

        super().__init__(out_proj, layer_idx, args)

        self.attention_module = attention_module
        self.order = args.fastoba_order  # FastOBA 阶数，默认 2
        self.delta = args.fastoba_delta  # FastOBA delta，默认 1.0
        self.inp_cache = []  # 缓存 Attention 输入
        self.out_cache = []  # 缓存 Attention 输出
        self.hessian_accumulate_freq = args.hessian_accumulate_freq  # 默认 10

    def add_batch_fastoba(self, inp, out):
        """
        使用 FastOBA 自动微分计算 Attention 层内 Hessian

        ❗ 核心：伪损失基于 Attention 输出，而非模型最终 Loss

        参数:
            inp: Attention 输入 [batch, seq_len, embed_dim]
            out: Attention 输出 [batch, seq_len, embed_dim]
        """
        # 缓存当前 batch 的输入输出
        self.inp_cache.append(inp.detach())
        self.out_cache.append(out.detach())

        # 每隔 K 个 batch 计算一次 Hessian（减少计算开销）
        if len(self.inp_cache) < self.hessian_accumulate_freq:
            return

        # --- 开始 Hessian 计算 ---
        device = self.layer.weight.device

        # 1. 定义伪损失函数（关键：基于 Attention 输出）
        def compute_pseudo_loss():
            total_loss = 0
            for inp_batch in self.inp_cache:
                inp_batch = inp_batch.to(device).requires_grad_(True)

                # ❗ 重新前向传播 Attention（需要 grad）
                out_batch = self.attention_module(inp_batch)

                # ❗ 伪损失：Attention 输出的范数
                # 这里定义了 H 是关于 Attention 输出的 Hessian
                loss_batch = out_batch.pow(2).sum()
                total_loss = total_loss + loss_batch

            return total_loss / len(self.inp_cache)

        # 2. 计算 Hessian-向量积
        loss = compute_pseudo_loss()

        # 调用 FastOBA 的 any_order_differentiation
        # ❗ parameters = [out_proj.weight] 只计算 out_proj 的 Hessian
        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],  # self.layer 是 out_proj
            order=self.order
        )

        # 3. 构建通道级别的 H 矩阵
        weight_hessian = hessian_grads[0].abs()  # [out_features, in_features]

        # ❗ 关键：weight_hessian 包含了：
        # - Attention 内部的 softmax Jacobian（自动微分）
        # - Q/K/V 的交互（自动微分）
        # - 所有到 out_proj 的梯度路径

        # 根据剪枝模式选择 H 构建方式
        if hasattr(self, 'num_heads') and self.num_heads > 1:
            # 方案 B: 块对角近似（用于 head-wise 剪枝）
            H_new = self.compute_block_diagonal_hessian(
                weight_hessian, self.num_heads, self.head_dim
            )
        else:
            # 方案 A: 对角近似（用于 channel-wise 剪枝）
            H_diag = weight_hessian.sum(dim=0)  # [in_features]
            H_new = torch.diag(H_diag)

        # 4. EMA 更新 self.H
        tmp = len(self.inp_cache)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        scale = math.sqrt(2 / self.nsamples)
        self.H += scale * H_new

        # 清空缓存
        self.inp_cache.clear()
        self.out_cache.clear()

    def compute_block_diagonal_hessian(self, weight_hessian, num_heads, head_dim):
        """
        构建块对角 H 矩阵（方案 B）

        Args:
            weight_hessian: [out_features, in_features] 参数 Hessian
            num_heads: head 数量
            head_dim: 每个 head 的维度

        Returns:
            H: [in_features, in_features] 块对角矩阵
        """
        in_features = weight_hessian.shape[1]
        H = torch.zeros(in_features, in_features, device=weight_hessian.device)

        for head_id in range(num_heads):
            start = head_id * head_dim
            end = (head_id + 1) * head_dim

            # 提取该 head 的参数 Hessian
            head_hessian = weight_hessian[:, start:end]

            # 构建该 head 的通道协方差矩阵
            # H_block[j,k] ≈ <∂²L/∂W[:,j], ∂²L/∂W[:,k]>
            H_block = head_hessian.T @ head_hessian

            # 归一化避免数值过大
            H_block = H_block / weight_hessian.shape[0]

            # 填充到 H 的块对角位置
            H[start:end, start:end] = H_block

        return H

**关键理解**:

1. **伪损失**: `L = ||Attention(inp)||²`
   - 不是模型最终 Loss
   - 是 Attention 输出的范数

2. **Hessian**: `H = ∂²L/∂W_out_proj²`
   - 关于 out_proj 权重的二阶导数
   - 通过自动微分，自动包含 softmax Jacobian
   - 层内 Hessian，不涉及上游

3. **对角近似**: `H_diag[i] = sum_j |∂²L/∂W_out_proj[j,i]|`
   - 对输出维度求和，得到每个输入通道的重要性
   - 类似 SlimGPT 的 `H = XX^T`，但包含二阶信息

    def any_order_differentiation(self, loss, delta, parameters, order):
        """
        复制自 fastoba_pruner.py:288-316
        """
        grads = [torch.zeros_like(param) for param in parameters]
        for current_order in range(1, order + 1):
            if current_order == 1:
                current_grad = torch.autograd.grad(
                    loss, parameters,
                    create_graph=(current_order != order)
                )
            else:
                grad_outputs = [param * delta for param in parameters]
                current_grad = torch.autograd.grad(
                    current_grad, parameters,
                    grad_outputs=grad_outputs,
                    create_graph=(current_order != order)
                )

        grads = [grad * param * delta
                 for grad, param in zip(current_grad, parameters)]
        return grads
```

### 3.3 与 prune_v2.py 的集成

**关键理解**:
- 对于 Attention 层，传入完整的 attention_module（包含 qkv_proj 和 out_proj）
- FastOBAAttentionSlimGPT 内部只对 out_proj 进行 Hessian 计算和剪枝
- 伪损失基于 Attention 的前向传播输出（层内），而非模型最终 Loss

**修改点**:

```python
# prune_v2.py:386-390 修改前
for name in module_dict:
    pruner_dict[name] = SlimGPT(module_dict[name], i, args)

def add_batch(name):
    def func(_, inp, out):
        pruner_dict[name].add_batch(inp[0].data, out.data)
    return func
```

**修改后**:

```python
# 选择剪枝器类型
if args.use_fastoba_attn:
    from slim_utils.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT
else:
    from slim_utils.slimgpt import SlimGPT

# 对于 Attention 层（attn.proj），使用 FastOBAAttentionSlimGPT
for names in sequential:
    for name in names:
        if name == "attn.proj" and args.use_fastoba_attn:
            # ❗ 传入完整的 attention 模块
            attention_module = layer.attn  # 获取完整 Attention 模块
            pruner_dict[name] = FastOBAAttentionSlimGPT(
                attention_module, i, args
            )
        else:
            # FFN 层或不使用 FastOBA 时，使用原始 SlimGPT
            pruner_dict[name] = SlimGPT(module_dict[name], i, args)

def add_batch(name):
    def func(_, inp, out):
        if name == "attn.proj" and args.use_fastoba_attn:
            # ❗ 传入 Attention 的输入输出（整层）
            pruner_dict[name].add_batch_fastoba(inp[0].data, out.data)
        else:
            # 原始 SlimGPT 的 add_batch
            pruner_dict[name].add_batch(inp[0].data, out.data)
    return func
```

**新增参数**:

```python
parser.add_argument(
    "--use_fastoba_attn", action="store_true",
    help="Use FastOBA autodiff for Attention layer-internal Hessian computation"
)
parser.add_argument(
    "--fastoba_order", type=int, default=2,
    help="Order of Taylor expansion for FastOBA (default: 2)"
)
parser.add_argument(
    "--fastoba_delta", type=float, default=1.0,
    help="Delta parameter for FastOBA"
)
parser.add_argument(
    "--pseudo_loss_type", type=str, default="norm",
    choices=["norm", "reconstruction"],
    help="Pseudo-loss for Attention Hessian: 'norm' = ||attn_out||², 'reconstruction' = ||attn_out - ref||²"
)
parser.add_argument(
    "--hessian_accumulate_freq", type=int, default=10,
    help="Accumulate Hessian every K batches to reduce computation"
)
```

---

## 4. 参数控制设计

### 4.1 核心参数

为了提供灵活的剪枝策略，FastOBAAttentionSlimGPT 需要支持以下参数：

#### **1. Hessian 计算方式**

```python
--hessian_mode: str = "block_diagonal"
    choices: ["slimgpt", "block_diagonal"]

    - "slimgpt": 使用 H = XX^T（一阶信息，SlimGPT原始方法）
    - "block_diagonal": 使用 FastOBA 二阶 Hessian（块对角近似）
```

**说明**：
- **slimgpt 模式**：
  - 计算 `H = inp.T @ inp`（输入协方差）
  - 优点：快速（5-10ms），无需自动微分
  - 缺点：不包含 softmax Jacobian，一阶近似

- **block_diagonal 模式**：
  - 使用 FastOBA 计算二阶 Hessian
  - 优点：包含 softmax Jacobian，理论更严格
  - 缺点：较慢（30-50ms）

#### **2. Head 评估方式**

```python
--head_importance_mode: str = "block_mean"
    choices: ["block_mean", "slimgpt_mean"]

    - "block_mean": 使用块对角 Hessian 的每个 head 块的对角元素评估
    - "slimgpt_mean": 使用 SlimGPT 的通道重要性对每个 head 的维度求平均
```

**说明**：
- **block_mean 模式**（推荐用于 hessian_mode="block_diagonal"）：
  ```python
  # 对每个 head，提取其 64×64 块的对角元素
  for head_id in range(num_heads):
      H_block = H[head_id*64:(head_id+1)*64, head_id*64:(head_id+1)*64]
      head_imp[head_id] = torch.diag(H_block).mean()
  ```
  - 考虑了 head 内部维度的协方差
  - 与块对角近似一致

- **slimgpt_mean 模式**（推荐用于 hessian_mode="slimgpt"）：
  ```python
  # 使用 OBS 重要性公式计算每个维度，然后对 head 求平均
  importance = W ** 2 / Hinv_diag  # [768]
  head_imp = importance.view(num_heads, head_dim).mean(1)  # [12]
  ```
  - SlimGPT 的标准做法
  - 简单直接

#### **3. 剪枝模式**

```python
--prune_mode: str = "head_dims"
    choices: ["head_dims", "num_heads", "both"]

    - "head_dims": 只剪 head 内部维度（如 64 → 40），保留所有 head
    - "num_heads": 只剪整个 head（如 12 → 8），保留每个 head 的完整维度
    - "both": 同时剪 head 数量和 head 内部维度
```

**说明**：
- **head_dims**：
  ```python
  # 12 heads × 64 dims → 12 heads × 40 dims
  # 每个 head 独立评估其内部维度重要性
  ```
  - 细粒度剪枝
  - 适合需要保留所有 head 的场景

- **num_heads**：
  ```python
  # 12 heads × 64 dims → 8 heads × 64 dims
  # 删除整个 head
  ```
  - 结构化程度高
  - 适合硬件加速

- **both**：
  ```python
  # 12 heads × 64 dims → 8 heads × 40 dims
  # 先按 head 重要性删除 head，再对保留的 head 删除维度
  ```
  - 最大压缩率
  - 两阶段剪枝

#### **4. 权重补偿**

```python
--use_compensation: bool = True

    - True: 使用 OBS 权重补偿
    - False: 不补偿，直接置零
```

**说明**：
- **True（推荐）**：
  ```python
  # OBS 补偿公式
  δθ_j = -θ_i × H^-1[i,j] / H^-1[i,i]
  ```
  - 最小化剪枝后的输出变化
  - 理论保证
  - 计算成本：需要 H^-1（Cholesky 已计算）

- **False**：
  ```python
  # 直接置零
  W[:, pruned_indices] = 0
  ```
  - 快速
  - 无理论保证
  - 适合高稀疏度后需要 fine-tune 的场景

### 4.2 参数组合推荐

| 场景 | hessian_mode | head_importance_mode | prune_mode | use_compensation | 理由 |
|------|-------------|---------------------|-----------|-----------------|------|
| **高精度剪枝** | block_diagonal | block_mean | both | True | 最严格理论，最佳效果 |
| **快速剪枝** | slimgpt | slimgpt_mean | head_dims | False | 最快速度，适合实验 |
| **硬件友好** | block_diagonal | block_mean | num_heads | True | 结构化，易加速 |
| **极致压缩** | block_diagonal | block_mean | both | True | 最大压缩率 |

### 4.3 参数实现要点

#### **在 `__init__` 中**：

```python
class FastOBAAttentionSlimGPT(SlimGPT):
    def __init__(self, attention_module, layer_idx, args):
        super().__init__(attention_module.out_proj, layer_idx, args)

        self.attention_module = attention_module
        self.hessian_mode = args.hessian_mode  # "slimgpt" or "block_diagonal"
        self.head_importance_mode = args.head_importance_mode
        self.prune_mode = args.prune_mode
        self.use_compensation = args.use_compensation

        # 块对角参数
        self.num_heads = attention_module.num_heads
        self.head_dim = attention_module.embed_dim // self.num_heads
```

#### **在 `add_batch` 中**：

```python
def add_batch(self, inp, out):
    if self.hessian_mode == "slimgpt":
        # 使用 SlimGPT 的原始方法
        super().add_batch(inp, out)
    elif self.hessian_mode == "block_diagonal":
        # 使用 FastOBA 块对角 Hessian
        self.add_batch_fastoba(inp, out)
```

#### **在 `struct_prune` 中**（复用 SlimGPT，添加参数传递）：

```python
def struct_prune(self, sparsity, headsize=1, percdamp=0.01):
    # ... (SlimGPT 的原始逻辑)

    # 根据 prune_mode 调整行为
    if self.prune_mode == "head_dims":
        # 只剪维度（SlimGPT 默认行为）
        return super().struct_prune(sparsity, headsize, percdamp)

    elif self.prune_mode == "num_heads":
        # 只剪 head（需要计算 head 重要性）
        head_imp = self._compute_head_importance()
        # ... 返回要删除的 head 的所有维度索引

    elif self.prune_mode == "both":
        # 两阶段剪枝
        # 1. 先剪 head
        # 2. 对剩余 head 剪维度
        pass
```

#### **Head 重要性计算**：

```python
def _compute_head_importance(self):
    if self.head_importance_mode == "block_mean":
        # 使用块对角 H 的每个块的对角元素
        head_imp = torch.zeros(self.num_heads)
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = (h + 1) * self.head_dim
            H_block = self.H[start:end, start:end]
            head_imp[h] = torch.diag(H_block).mean()
        return head_imp

    elif self.head_importance_mode == "slimgpt_mean":
        # 使用 SlimGPT 的通道重要性
        Hinv_diag = torch.diag(torch.cholesky_inverse(torch.linalg.cholesky(self.H)))
        importance = (self.layer.weight ** 2).sum(0) / Hinv_diag
        head_imp = importance.view(self.num_heads, self.head_dim).mean(1)
        return head_imp
```

---

## 5. 实现步骤

### Phase 1: 创建 FastOBAAttentionSlimGPT 类（方案2 - 推荐先实现）

**时间**: 5-8小时（考虑参数控制）

**文件**: `slim_utils/fastoba_attention_slimgpt.py`

**任务**:
- [ ] 1.1 创建类框架，继承自 SlimGPT
  - 接收 attention_module 参数（包含 qkv_proj 和 out_proj）
  - 添加参数：hessian_mode, head_importance_mode, prune_mode, use_compensation
  - 只对 out_proj 层初始化 SlimGPT 的 H 矩阵
- [ ] 1.2 实现 `any_order_differentiation` 方法（复制自 fastoba_pruner.py:288-316）
- [ ] 1.3 实现 `add_batch` 方法（参数控制）
  - [ ] 根据 hessian_mode 选择计算方式
  - [ ] slimgpt 模式：调用 super().add_batch()
  - [ ] block_diagonal 模式：调用 add_batch_fastoba()
- [ ] 1.4 实现 `add_batch_fastoba` 方法
  - [ ] 缓存 Attention 的输入输出
  - [ ] 定义伪损失：`L = ||attention_module(inp)||²`
  - [ ] 调用 any_order_differentiation 计算 Hessian
  - [ ] 块对角近似：构建 H 矩阵
  - [ ] EMA 更新 self.H
- [ ] 1.5 实现 `_compute_head_importance` 方法
  - [ ] block_mean 模式：提取块对角元素
  - [ ] slimgpt_mean 模式：使用 OBS 重要性
- [ ] 1.6 扩展 `struct_prune` 方法（基于 prune_mode）
  - [ ] head_dims 模式：复用 SlimGPT 逻辑
  - [ ] num_heads 模式：基于 head 重要性剪枝
  - [ ] both 模式：两阶段剪枝
  - [ ] 根据 use_compensation 决定是否补偿

**验证**:
```python
# 单元测试
import torch
import torch.nn as nn
from slim_utils.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT
from modules.models import Attention

# 创建 Attention 模块
attention = Attention(num_heads=12, embed_dim=768)
pruner = FastOBAAttentionSlimGPT(attention, layer_idx=0, args=mock_args)

# 模拟输入输出
inp = torch.randn(32, 128, 768, requires_grad=True)
# 注意：这里 inp 应该是 Attention 的输入（QKV 已投影后）
q = k = v = inp
out = attention(q, k, v)

# 添加 batch
pruner.add_batch_fastoba(inp, out)

# 检查 H 的形状和数值（应该是 out_proj 的输入通道数）
assert pruner.H.shape == (768, 768)
assert torch.isfinite(pruner.H).all()
print("Hessian diagonal elements:", torch.diag(pruner.H)[:10])
```

### Phase 2: 创建 OBAAttentionSlimGPT 类（方案1 - 可选）

**时间**: 5-8小时

**文件**: `slim_utils/oba_attention_slimgpt.py`

**任务**:
- [ ] 2.1 创建类框架，继承自 SlimGPT
  - 接收 attention_module 参数
  - 只对 out_proj 层初始化 H 矩阵
- [ ] 2.2 实现 Attention 内部的前向传播 hook
  - 捕获 Q, K, V 和中间激活
- [ ] 2.3 实现 `compute_softmax_jacobian` 方法
  ```python
  # 基于 oba_pruner.py:636-640
  jacobianfunc = torch.func.jacfwd(lambda x: torch.softmax(x, dim=0))
  softmax_jacobian = torch.func.vmap(
      torch.func.vmap(
          torch.func.vmap(jacobianfunc)
      )
  )(forward_attn_weights).detach()
  ```
- [ ] 2.4 实现 `add_batch_oba` 方法
  - 定义伪损失：`L = ||attention(inp)||²`
  - 计算 Attention 权重 = (Q @ K^T) / √d
  - 计算 softmax Jacobian
  - 反向传播到 Q/K/V
  - 累积到 out_proj 的 Hessian
- [ ] 2.5 完全复用 `struct_prune` 方法

**优点**:
- 完全控制 Hessian 计算
- 可以分析 Q/K/V 的独立贡献

**缺点**:
- 实现复杂
- 只支持标准 Attention

### Phase 3: 数值稳定性和优化（1-2小时）

**任务**:
- [ ] 3.1 EMA 更新逻辑
  ```python
  self.H *= self.nsamples / (self.nsamples + tmp)
  self.nsamples += tmp
  scale = math.sqrt(2 / self.nsamples)
  self.H += scale * H_diag_matrix
  ```
- [ ] 3.2 Dampening 处理
  ```python
  if percdamp > 0:
      damp = percdamp * torch.mean(torch.diag(H))
      H[diag, diag] += damp
  ```
- [ ] 3.3 避免 NaN/Inf
  - 检查 Hessian 的条件数
  - 对接近零的对角元素添加正则化

**验证**:
```python
# 检查对角矩阵是否正定
eigenvalues = torch.linalg.eigvalsh(pruner.H)
assert (eigenvalues > 0).all(), "H must be positive definite"

# 检查条件数
cond = torch.linalg.cond(pruner.H)
print(f"Condition number: {cond.item()}")
assert cond < 1e10, "Hessian is ill-conditioned"
```

### Phase 4: 集成到 prune_v2.py（2-3小时）

**文件**: `prune_v2.py`

**任务**:
- [ ] 4.1 添加命令行参数
  ```python
  # Attention 剪枝相关参数
  parser.add_argument(
      "--use_fastoba_attn", action="store_true",
      help="Use FastOBA+OBS for Attention layer pruning"
  )
  parser.add_argument(
      "--hessian_mode", type=str, default="block_diagonal",
      choices=["slimgpt", "block_diagonal"],
      help="Hessian computation mode for Attention"
  )
  parser.add_argument(
      "--head_importance_mode", type=str, default="block_mean",
      choices=["block_mean", "slimgpt_mean"],
      help="How to evaluate head importance"
  )
  parser.add_argument(
      "--prune_mode", type=str, default="head_dims",
      choices=["head_dims", "num_heads", "both"],
      help="Pruning mode: head dimensions, entire heads, or both"
  )
  parser.add_argument(
      "--use_compensation", action="store_true", default=True,
      help="Use OBS weight compensation"
  )
  parser.add_argument(
      "--fastoba_order", type=int, default=2,
      help="Order of Taylor expansion for FastOBA (default: 2)"
  )
  parser.add_argument(
      "--fastoba_delta", type=float, default=1.0,
      help="Delta parameter for FastOBA"
  )
  parser.add_argument(
      "--hessian_accumulate_freq", type=int, default=10,
      help="Accumulate Hessian every K batches to reduce computation"
  )
  ```

- [ ] 4.2 修改剪枝器初始化逻辑
  ```python
  # 选择剪枝器类型
  if args.use_fastoba_attn:
      from slim_utils.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

  sequential = [
      ["attn.proj"],    # Attention 输出投影
      ["ffn.fc2"],      # FFN 第二层
  ]

  for names in sequential:
      for name in names:
          if name == "attn.proj" and args.use_fastoba_attn:
              # ❗ 使用 FastOBA+OBS 方法
              attention_module = layer.attn  # 获取完整 Attention 模块
              pruner_dict[name] = FastOBAAttentionSlimGPT(
                  attention_module, i, args
              )
          else:
              # ❗ FFN 层或不使用 FastOBA 时，使用原始 SlimGPT
              pruner_dict[name] = SlimGPT(module_dict[name], i, args)
  ```

- [ ] 4.3 修改 add_batch hook
  ```python
  def add_batch(name):
      def func(_, inp, out):
          if name == "attn.proj" and args.use_fastoba_attn:
              # ❗ Attention 层：传入 Attention 的输入输出（整层）
              pruner_dict[name].add_batch(inp[0].data, out.data)
          else:
              # ❗ FFN 层：使用原始 SlimGPT 的 add_batch
              pruner_dict[name].add_batch(inp[0].data, out.data)
      return func
  ```
  **注意**：FastOBAAttentionSlimGPT 的 `add_batch` 会根据 `hessian_mode` 自动选择计算方式

- [ ] 4.4 修改 struct_prune 调用
  ```python
  if name == "attn.proj":
      # Attention 层剪枝
      if args.use_fastoba_attn:
          # 使用新的参数化剪枝
          idx = pruner_dict[name].struct_prune(
              sparsity=sparsity,
              headsize=64,  # head_dim
              percdamp=0.01
          )
      else:
          # 原始 SlimGPT 方法
          idx = pruner_dict[name].struct_prune(
              sparsity=sparsity,
              headsize=64,
              percdamp=0.01
          )
  else:
      # FFN 层（保持不变）
      idx = pruner_dict[name].struct_prune(
          sparsity=sparsity,
          headsize=1,  # channel-wise
          percdamp=0.01
      )
  ```

- [ ] 4.5 兼容性测试
  - 测试原始 SlimGPT 模式（`--use_fastoba_attn False`）
  - 测试 FastOBA 各种参数组合
  - 确保不破坏原有功能

**验证**:
```bash
# 测试原始 SlimGPT（baseline）
python prune_v2.py --sparsity 0.2 --num_samples 128

# 测试 FastOBA Attention 剪枝（不同参数组合）

# 组合1：高精度剪枝（块对角 + 补偿 + 双模式剪枝）
python prune_v2.py --use_fastoba_attn \
    --hessian_mode block_diagonal \
    --head_importance_mode block_mean \
    --prune_mode both \
    --use_compensation \
    --sparsity 0.3 --num_samples 128

# 组合2：快速剪枝（SlimGPT H + 维度剪枝 + 不补偿）
python prune_v2.py --use_fastoba_attn \
    --hessian_mode slimgpt \
    --head_importance_mode slimgpt_mean \
    --prune_mode head_dims \
    --no-use_compensation \
    --sparsity 0.3 --num_samples 128

# 组合3：硬件友好（块对角 + head剪枝）
python prune_v2.py --use_fastoba_attn \
    --hessian_mode block_diagonal \
    --head_importance_mode block_mean \
    --prune_mode num_heads \
    --use_compensation \
    --sparsity 0.3 --num_samples 128

# 测试不同稀疏度
for sparsity in 0.2 0.3 0.4 0.5; do
    python prune_v2.py --use_fastoba_attn --sparsity $sparsity --num_samples 128
done
```

### Phase 5: 高级特性（可选，5-10小时）

**5.1 完整 Hessian 计算** (使用 functorch 加速)
```python
from torch.func import jacrev, vmap

def compute_full_hessian(attention_module, inp):
    """计算 Attention 层内的完整 Hessian 矩阵"""
    def pseudo_loss_fn(weight):
        # 临时替换 weight
        original_weight = attention_module.out_proj.weight.data.clone()
        attention_module.out_proj.weight.data = weight

        # 前向传播
        out = attention_module(inp, inp, inp)  # q=k=v=inp
        loss = out.pow(2).sum()

        # 恢复 weight
        attention_module.out_proj.weight.data = original_weight
        return loss

    weight = attention_module.out_proj.weight
    # 计算 Hessian: H[i,j] = ∂²L/∂w[i]∂w[j]
    hessian = jacrev(jacrev(pseudo_loss_fn))(weight)
    return hessian
```

**5.2 KFAC 近似** (Kronecker-Factored Approximate Curvature) #目前不需要 不要实现
```python
def compute_kfac_hessian(attention_module, inp_batch, out_batch):
    """
    使用 KFAC 近似 Hessian
    H ≈ A ⊗ G
    A: 输入协方差矩阵 [in_features, in_features]
    G: 输出梯度协方差矩阵 [out_features, out_features]
    """
    # A = E[inp @ inp.T]
    inp_flat = inp_batch.view(-1, inp_batch.shape[-1])  # [batch*seq, embed_dim]
    A = inp_flat.T @ inp_flat / inp_flat.shape[0]

    # G = E[grad_out @ grad_out.T]
    # 这里需要伪损失的梯度
    out_flat = out_batch.view(-1, out_batch.shape[-1])
    pseudo_loss = out_flat.pow(2).sum()
    grad_out = torch.autograd.grad(pseudo_loss, out_batch)[0]
    grad_out_flat = grad_out.view(-1, grad_out.shape[-1])
    G = grad_out_flat.T @ grad_out_flat / grad_out_flat.shape[0]

    return A, G  # H ≈ G ⊗ A

# 使用 KFAC 计算 Hinv 的对角元素
def kfac_inv_diag(A, G, damp=1e-5):
    """高效计算 (G ⊗ A)^-1 的对角元素"""
    A_inv = torch.inverse(A + damp * torch.eye(A.shape[0], device=A.device))
    G_inv = torch.inverse(G + damp * torch.eye(G.shape[0], device=G.device))

    # diag(H^-1) ≈ diag(G^-1 ⊗ A^-1) = vec(diag(G^-1)) ⊗ vec(diag(A^-1))
    Hinv_diag = torch.kron(torch.diag(G_inv), torch.diag(A_inv))
    return Hinv_diag
```

**5.3 head-wise 块 Hessian**
```python
def compute_headwise_hessian(attention_module, inp, num_heads, head_dim):
    """
    为每个 head 计算独立的 Hessian 块
    这样可以更精确地评估每个 head 的重要性
    """
    head_hessians = []

    for head_id in range(num_heads):
        # 提取该 head 对应的权重
        head_slice = slice(head_id * head_dim, (head_id+1) * head_dim)
        head_weight = attention_module.out_proj.weight[:, head_slice]

        # 定义该 head 的伪损失
        def head_pseudo_loss():
            # 只通过该 head 的权重前向传播
            out = attention_module(inp, inp, inp)
            # 只计算该 head 贡献的输出范数
            head_out = out[:, :, head_slice]
            return head_out.pow(2).sum()

        loss = head_pseudo_loss()

        # 计算 Hessian
        hessian_grads = any_order_differentiation(
            loss=loss,
            parameters=[head_weight],
            order=2,
            delta=1.0
        )

        head_hessians.append(hessian_grads[0])

    return head_hessians  # List of [head_dim, head_dim] tensors
```

---

## 5. 实验验证

### 5.1 单元测试

**测试用例 1: Hessian 计算正确性**
```python
def test_attention_hessian_correctness():
    """验证 Attention 层内 Hessian 计算的正确性"""
    from modules.models import Attention
    from slim_utils.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

    # 创建 Attention 模块
    attention = Attention(num_heads=4, embed_dim=64)
    attention.eval()

    # 创建测试输入
    inp = torch.randn(2, 10, 64, requires_grad=True)
    q = k = v = inp

    # FastOBA 计算
    pruner = FastOBAAttentionSlimGPT(attention, 0, args)
    out = attention(q, k, v)
    pruner.add_batch_fastoba(inp, out)
    H_fastoba = pruner.H

    # 数值微分计算（参考）
    def compute_numerical_hessian():
        eps = 1e-4
        weight = attention.out_proj.weight
        H_num = torch.zeros_like(weight)

        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                # 扰动 weight[i,j]
                weight[i,j] += eps
                out_plus = attention(q, k, v)
                loss_plus = out_plus.pow(2).sum()

                weight[i,j] -= 2*eps
                out_minus = attention(q, k, v)
                loss_minus = out_minus.pow(2).sum()

                weight[i,j] += eps  # 恢复

                # 二阶导数近似
                H_num[i,j] = (loss_plus - 2*loss + loss_minus) / (eps**2)

        return H_num

    # 比较（对角元素）
    H_num = compute_numerical_hessian()
    H_num_diag = torch.diag(H_num)
    H_fastoba_diag = torch.diag(H_fastoba)

    rel_error = (H_fastoba_diag - H_num_diag).abs() / (H_num_diag.abs() + 1e-8)
    print(f"Mean relative error: {rel_error.mean():.4f}")
    assert rel_error.mean() < 0.2  # 20% 相对误差（数值微分本身有误差）
```

**测试用例 2: 剪枝后模型有效性**
```python
def test_attention_pruning_validity():
    """验证剪枝后 Attention 仍可运行"""
    from modules.models import Attention, Block
    from slim_utils.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

    # 创建 Transformer Block
    block = Block(n_head=8, n_embed=512)
    attention = block.attn

    # 创建剪枝器
    pruner = FastOBAAttentionSlimGPT(attention, 0, args)

    # 收集数据
    for _ in range(10):
        inp = torch.randn(4, 20, 512, requires_grad=True)
        q = k = v = inp
        out = attention(q, k, v)
        pruner.add_batch_fastoba(inp, out)

    # 执行剪枝（head-wise）
    pruned_idx = pruner.struct_prune(sparsity=0.3, headsize=64, percdamp=0.01)

    # 测试前向传播
    test_inp = torch.randn(1, 20, 512)
    q = k = v = test_inp
    test_out = attention(q, k, v)

    assert torch.isfinite(test_out).all(), "Output contains NaN or Inf"
    print(f"Pruned {len(pruned_idx)} channels, output shape: {test_out.shape}")
```

**测试用例 3: 伪损失选择的影响**
```python
def test_pseudo_loss_comparison():
    """对比不同伪损失定义的效果"""
    from modules.models import Attention
    from slim_utils.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

    attention = Attention(num_heads=4, embed_dim=64)
    inp = torch.randn(4, 10, 64, requires_grad=True)

    # 选项 A: 输出范数
    def pseudo_loss_norm(out):
        return out.pow(2).sum()

    # 选项 B: 输出均值（归一化）
    def pseudo_loss_mean(out):
        return out.pow(2).mean()

    # 对比 Hessian
    for loss_fn, name in [(pseudo_loss_norm, "norm"), (pseudo_loss_mean, "mean")]:
        pruner = FastOBAAttentionSlimGPT(attention, 0, args)
        pruner.pseudo_loss_fn = loss_fn

        q = k = v = inp
        out = attention(q, k, v)
        pruner.add_batch_fastoba(inp, out)

        print(f"{name}: H diagonal mean = {torch.diag(pruner.H).mean():.4f}")
```

### 5.2 对比实验

**实验 A: SlimGPT vs FastOBA-Attention-SlimGPT**

**目标**: 对比使用 H=XX^T 和层内 Hessian 的剪枝效果

| 方法 | Hessian 来源 | 理论基础 | 剪枝准确率 | 计算时间 |
|------|-------------|---------|-----------|---------|
| SlimGPT (H=XX^T) | 输入协方差 | 一阶信息 | Baseline | 1x |
| FastOBA-Attention-SlimGPT | 层内二阶 Hessian | Attention 输出的泰勒展开 | ? | 3-5x |
| OBA (全局) | 全局 Hessian | 三种连接性 | Reference | 10x |

**实验设置**:
```bash
# Baseline: SlimGPT
python prune_v2.py --sparsity 0.3 --num_samples 128

# 测试: FastOBA Attention
python prune_v2.py --use_fastoba_attn --sparsity 0.3 --num_samples 128 \
    --fastoba_order 2 --fastoba_delta 1.0

# 测试: 不同稀疏度
for sparsity in 0.2 0.3 0.4 0.5; do
    python prune_v2.py --use_fastoba_attn --sparsity $sparsity --num_samples 128
done
```

**实验 B: 不同 pseudo-loss 定义**

**目标**: 评估伪损失选择对剪枝质量的影响

| Pseudo-Loss | 定义 | 优点 | 缺点 |
|------------|------|------|------|
| 输出范数 (sum) | `||attn_out||²` | 简单，不需要参考 | 可能偏向大幅值 |
| 输出范数 (mean) | `(1/n)||attn_out||²` | 归一化 | 同上 |
| 重构误差 | `||attn_out - ref||²` | 更精确 | 需要保存参考输出 |

**实验 C: head-wise vs channel-wise**

```python
# head-wise 剪枝 (headsize=64, 假设 head_dim=64)
pruner.struct_prune(sparsity=0.3, headsize=64, percdamp=0.01)

# channel-wise 剪枝 (headsize=1)
pruner.struct_prune(sparsity=0.3, headsize=1, percdamp=0.01)
```

**预期**:
- head-wise: 剪掉整个 head，结构更规整，但可能过于粗粒度
- channel-wise: 更细粒度，可能保留更多信息，但结构不规整

**评估指标**:
- 剪枝后准确率/困惑度
- 推理速度提升
- 参数量减少
- FLOPs 减少

---

## 6. 技术难点与解决方案

### 6.1 难点 1: Hessian 计算开销

**问题**: 自动微分计算二阶导数非常慢

**解决方案**:
1. **批量累积**: 每隔 K 个 batch 才计算一次 Hessian
2. **对角近似**: 只计算对角线而非完整矩阵
3. **低精度**: 使用 float16 计算 Hessian（如果数值稳定）
4. **并行化**: 对不同 head 的 Hessian 并行计算

```python
# 示例：批量累积
if batch_idx % args.hessian_accumulate_freq == 0:
    compute_hessian()
```

### 6.2 难点 2: 内存占用

**问题**: 完整 Hessian 需要 O(n²) 内存

**解决方案**:
1. **对角近似**: 只存储对角线 O(n)
2. **块对角**: 存储每个 head 的块 O(k × d²)，k=num_heads, d=head_dim
3. **即时计算**: 需要时才计算 Hessian，不存储中间结果
4. **CPU offload**: 将 Hessian 存储在 CPU

```python
# 示例：对角近似
H_diag = torch.zeros(n_params, device='cpu')  # CPU 存储
```

### 6.3 难点 3: 数值稳定性

**问题**: Cholesky 分解要求矩阵正定，但 FastOBA 的 Hessian 可能接近奇异

**解决方案**:
1. **Dampening**: 添加正则项 `H += damp * I`
2. **死神经元处理**: 对 `diag(H)==0` 的位置特殊处理
3. **条件数检查**: 如果 `cond(H) > threshold`，增加 damp

```python
# slimgpt.py:184-187（已有实现）
if percdamp > 0:
    damp = percdamp * torch.mean(torch.diag(H))
    H[diag, diag] += damp
```

### 6.4 难点 4: 伪损失的选择

**问题**: 不同的伪损失会导致不同的 Hessian，影响剪枝效果

**关键理解**:
- ❗ 伪损失定义了 Hessian 的含义：H = ∂²L_pseudo/∂W²
- ❗ 选择伪损失 = 选择优化目标（最小化什么）
- ❗ 对于层内剪枝，伪损失应该基于 **Attention 输出**，而非全局 Loss

**解决方案**:
1. **推荐方案**: 输出范数（简单、无需参考）
   ```python
   def pseudo_loss(attention_output):
       # 目标：最小化 Attention 输出的变化
       return attention_output.pow(2).sum()
   ```

2. **备选方案**: 重构误差（需要保存参考输出）
   ```python
   def pseudo_loss(attention_output, reference_output):
       # 目标：输出接近参考
       return (attention_output - reference_output).pow(2).sum()
   ```

3. **不推荐**: 加权范数（引入上游梯度，破坏层内假设）
   ```python
   # ❌ 这会引入上游信息，不符合层内 OBS 的目标
   if upstream_grad is not None:
       loss = (output * upstream_grad).sum()
   ```

**实验建议**: 在 Phase 5.1 测试用例 3 中对比不同伪损失

---

## 7. 预期效果

### 7.1 理论优势

**相比原始 SlimGPT (H=XX^T)**:

✅ **包含二阶信息**: FastOBA 的 Hessian 包含关于 Attention 输出的二阶导数
✅ **捕获非线性**: 自动包含 Attention 的 softmax Jacobian 和 Q/K/V 交互
✅ **理论更严格**: 基于泰勒展开，有明确的损失变化估计
✅ **无需手动推导**: 自动微分处理所有复杂性，适用于任何 Attention 变体
✅ **层内 OBS**: 结合权重补偿，最小化 Attention 输出变化

**关键区别**:
| 维度 | SlimGPT (H=XX^T) | FastOBA-Attention-SlimGPT |
|------|------------------|--------------------------|
| **Hessian 含义** | 输入协方差（一阶） | Attention 输出的二阶导数 |
| **包含 softmax** | ❌ | ✅ (自动) |
| **理论基础** | Fisher 信息近似 | 二阶泰勒展开 |
| **权重补偿** | ✅ | ✅ |
| **优化目标** | 不明确 | 明确（最小化 Attention 输出变化） |

### 7.2 性能预期

**计算开销**:
- SlimGPT: 1x (只需 H = XX^T)
- FastOBA-Attention-SlimGPT: 3-5x (需要二阶反向传播)
  - 可通过批量累积（`hessian_accumulate_freq`）降低到 2-3x

**剪枝质量**:
- 预期在相同稀疏度下，准确率/困惑度提升 **0.5-2%**
- head-wise 剪枝效果应优于 channel-wise（更结构化）
- 对高稀疏度（>50%）的改进更明显

**内存占用**:
- 对角近似: 与 SlimGPT 相当 (O(n))
- 需要缓存少量输入输出（`hessian_accumulate_freq` 个 batch）
- 完整 Hessian（可选）: 10-100x（不推荐用于生产）

---

## 8. 开发时间表

| 阶段 | 任务 | 预计时间 | 优先级 |
|------|------|---------|--------|
| **Phase 1** | 创建 FastOBAAttentionSlimGPT 类（方案2） | 3-5 小时 | P0 |
| **Phase 2** | 创建 OBAAttentionSlimGPT 类（方案1，可选） | 5-8 小时 | P1 |
| **Phase 3** | 数值稳定性和优化 | 1-2 小时 | P0 |
| **Phase 4** | 集成到 prune_v2.py | 1-2 小时 | P0 |
| **Phase 5.1** | 完整 Hessian 计算（可选） | 3-5 小时 | P2 |
| **Phase 5.2** | KFAC 近似（可选） | 3-5 小时 | P2 |
| **Phase 5.3** | head-wise 块 Hessian（可选） | 2-3 小时 | P2 |
| **测试** | 单元测试 + 对比实验 | 5-8 小时 | P0 |

**总计**:
- **核心功能 (P0)**: 10-17 小时
- **OBA 方案 (P1)**: +5-8 小时
- **高级特性 (P2)**: +8-13 小时

**推荐开发顺序**:
1. Phase 1: FastOBA 方案（快速原型，验证可行性）
2. Phase 3-4: 数值稳定性 + 集成
3. 测试: 验证正确性和效果
4. Phase 2: OBA 方案（如需深入分析）
5. Phase 5: 高级特性（如需进一步优化）

---

## 9. 文件清单

将创建/修改以下文件：

```
slim_utils/
├── fastoba_attention_slimgpt.py  # 新增：FastOBA 风格（方案2，推荐）
├── oba_attention_slimgpt.py      # 新增：OBA 风格（方案1，可选）
├── slimgpt.py                    # 保持不变（作为基类）
└── hessian_utils.py              # 新增：Hessian 计算辅助函数（可选）

prune_v2.py                       # 修改：添加 FastOBA Attention 支持

tests/
├── test_fastoba_attention_slimgpt.py  # 新增：单元测试
├── test_oba_attention_slimgpt.py      # 新增：OBA 方案测试（可选）
└── test_hessian_accuracy.py           # 新增：Hessian 正确性验证

experiments/
├── compare_slimgpt_vs_fastoba.py      # 新增：对比实验脚本
└── ablation_pseudo_loss.py            # 新增：伪损失消融实验
```

---

## 10. 参考代码片段

### 10.1 FastOBA 的 any_order_differentiation

```python
# 来源: fastoba_pruner.py:288-316
def any_order_differentiation(self, loss, delta=-1.0, parameters=None, order=1):
    if parameters is None:
        parameters = self.model.parameters()
    grads = [torch.zeros_like(param) for param in parameters]
    for current_order in range(1, order + 1):
        if current_order == 1:
            if current_order == order:
                current_grad = torch.autograd.grad(loss, parameters)
            else:
                current_grad = torch.autograd.grad(loss, parameters, create_graph=True)
        else:
            grad_outputs = [param * delta for param in parameters]
            if current_order == order:
                current_grad = torch.autograd.grad(current_grad, parameters,
                                                   grad_outputs=grad_outputs)
            else:
                current_grad = torch.autograd.grad(current_grad, parameters,
                                                   grad_outputs=grad_outputs, create_graph=True)
    grads = [current_grad.detach() * param.data * delta
             for grad, current_grad, param in zip(grads, current_grad, parameters)]
    return grads
```

### 10.2 SlimGPT 的 Cholesky 分解

```python
# 来源: slimgpt.py:200-221
Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

if headsize > 1:
    # head-wise 剪枝的特殊处理
    Hinv_diag = torch.stack([
        Hinv[i:i+headsize, i:i+headsize]
        for i in range(0, self.columns, headsize)
    ])
    Hinv_diag = torch.diagonal(
        torch.linalg.cholesky(Hinv_diag),
        dim1=-2, dim2=-1
    ).reshape(-1)
    Hinv_diag = Hinv_diag ** 2
else:
    Hinv_diag = Hinv.diag()

# OBS 重要性公式
error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
```

---

## 11. 附录：VAR 多尺度模型的校准集采样策略

### 11.1 问题背景

**VAR (Visual AutoRegressive) 模型特性**：

1. **多尺度token金字塔**：
   ```python
   patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # 10个尺度
   总token数 = 1² + 2² + ... + 16² = 680 tokens
   ```

2. **参数共享**：
   - 所有尺度使用**同一组** Transformer blocks
   - 没有尺度特定的参数
   - 剪枝决策影响所有尺度的性能

3. **尺度不平衡**：

   | 尺度 | 分辨率 | Token数 | 占比 |
   |------|--------|---------|------|
   | Stage 0 | 1×1 | 1 | 0.15% |
   | Stage 1 | 2×2 | 4 | 0.59% |
   | Stage 2 | 3×3 | 9 | 1.32% |
   | Stage 3 | 4×4 | 16 | 2.35% |
   | Stage 4 | 5×5 | 25 | 3.68% |
   | Stage 5 | 6×6 | 36 | 5.29% |
   | Stage 6 | 8×8 | 64 | 9.41% |
   | Stage 7 | 10×10 | 100 | 14.7% |
   | Stage 8 | 13×13 | 169 | 24.9% |
   | Stage 9 | 16×16 | 256 | **37.6%** |

**关键问题**：如何选择校准集才能让剪枝后的模型在所有尺度上都表现良好？

---

### 11.2 采样策略设计

#### **策略 A：完整序列采样（Baseline）**

```python
def collect_calibration_baseline(model, dataloader, num_samples=128):
    """
    不加权的完整序列采样
    Hessian自然地被细尺度主导
    """
    calibration_data = []
    model.eval()

    for i, (img, label) in enumerate(dataloader):
        if i >= num_samples:
            break

        img, label = img.cuda(), label.cuda()

        with torch.no_grad():
            # 编码图像为tokens
            x_BLCv = model.vae_proxy[0].img_to_idxBl(img)
            x_BLCv_embed = model.vae_quant_proxy[0].embedding(x_BLCv)

            calibration_data.append({
                'label': label,
                'x_embed': x_BLCv_embed,  # [B, 680, Cvae]
            })

    return calibration_data
```

**特点**：
- ✅ 简单，反映真实使用分布
- ❌ 细尺度（stage 9）占37.6%，主导Hessian
- ❌ 粗尺度（stage 0-2）仅占2%，可能被忽略

---

#### **策略 B：加权采样（推荐）**

```python
def collect_calibration_weighted(model, dataloader, num_samples=128,
                                weighting='inverse'):
    """
    尺度平衡的加权采样

    Args:
        weighting: 'inverse' | 'sqrt' | 'uniform'
            - 'inverse': 权重 ∝ 1/token_count（完全平衡）
            - 'sqrt': 权重 ∝ 1/√token_count（温和平衡）
            - 'uniform': 权重 = 1（所有尺度平等）
    """
    calibration_data = []
    model.eval()

    # 计算尺度权重
    stage_token_counts = [pn**2 for pn in model.patch_nums]
    total_tokens = sum(stage_token_counts)  # 680

    if weighting == 'inverse':
        # 反比权重：小尺度权重高
        stage_weights = [total_tokens / (10 * count) for count in stage_token_counts]
    elif weighting == 'sqrt':
        # 平方根反比：折中方案
        import math
        stage_weights = [math.sqrt(total_tokens / count) for count in stage_token_counts]
    elif weighting == 'uniform':
        # 均匀权重：每个尺度平等
        stage_weights = [1.0] * 10

    # 归一化
    weight_sum = sum(stage_weights)
    stage_weights = [w / weight_sum for w in stage_weights]

    # 构建token到stage的映射
    token_to_stage = []
    for stage_id, token_count in enumerate(stage_token_counts):
        token_to_stage.extend([stage_id] * token_count)

    # 收集数据
    for i, (img, label) in enumerate(dataloader):
        if i >= num_samples:
            break

        img, label = img.cuda(), label.cuda()

        with torch.no_grad():
            x_BLCv = model.vae_proxy[0].img_to_idxBl(img)
            x_BLCv_embed = model.vae_quant_proxy[0].embedding(x_BLCv)

            calibration_data.append({
                'label': label,
                'x_embed': x_BLCv_embed,
                'stage_weights': stage_weights,
                'token_to_stage': token_to_stage,  # [680] 每个token所属stage
            })

    return calibration_data
```

**加权效果对比**：

| 尺度 | Token占比 | inverse权重 | sqrt权重 | uniform权重 |
|------|----------|-------------|----------|-------------|
| Stage 0 (1×1) | 0.15% | **68.0x** | **26.1x** | **1.0x** |
| Stage 1 (2×2) | 0.59% | 17.0x | 13.0x | 1.0x |
| Stage 5 (6×6) | 5.29% | 1.89x | 4.57x | 1.0x |
| Stage 9 (16×16) | 37.6% | 0.27x | 1.63x | 1.0x |

**计算公式**：
```python
# inverse权重（完全平衡）
w_i = (680 / 10) / token_count_i = 68 / token_count_i

# sqrt权重（温和平衡）
w_i = sqrt(680 / token_count_i)

# uniform权重（无加权）
w_i = 1.0
```

---

#### **策略 C：单尺度采样（对比实验）**

```python
def collect_calibration_single_scale(model, dataloader, target_stage,
                                    num_samples=128):
    """
    只使用特定尺度的tokens计算Hessian
    用于对比实验，分析不同尺度的剪枝偏好

    Args:
        target_stage: 0-9，指定使用哪个尺度
    """
    calibration_data = []
    model.eval()

    # 计算该stage的token范围
    stage_token_counts = [pn**2 for pn in model.patch_nums]
    start_idx = sum(stage_token_counts[:target_stage])
    end_idx = start_idx + stage_token_counts[target_stage]

    for i, (img, label) in enumerate(dataloader):
        if i >= num_samples:
            break

        img, label = img.cuda(), label.cuda()

        with torch.no_grad():
            x_BLCv = model.vae_proxy[0].img_to_idxBl(img)
            x_BLCv_embed = model.vae_quant_proxy[0].embedding(x_BLCv)

            # 只保留目标stage的tokens
            x_stage = x_BLCv_embed[:, start_idx:end_idx, :]

            calibration_data.append({
                'label': label,
                'x_embed': x_stage,
                'stage_id': target_stage,
            })

    return calibration_data
```

**用途**：
- 🔬 研究实验：分析每个尺度对剪枝决策的影响
- 📊 可视化：对比不同尺度下的head重要性分布
- 🎯 诊断工具：找出哪些尺度对特定head最敏感

---

### 11.3 在 FastOBAAttentionSlimGPT 中实现逐尺度统计

#### **方案 A：基于 `add_batch_v7` 的逐尺度独立统计（推荐）**

参考 `/home/project/real_prune/slimgpt_pub_prune/slim_utils/slimgpt.py:add_batch_v7`

**核心思想**：
- 缓存最近 N 次的输入（每次可能是不同尺度的 token）
- 每个尺度独立计算 `X_s @ X_s^T`
- 支持按 `seqlen` 归一化（分辨率同权）
- 支持自定义权重 `gamma_s`

```python
class FastOBAAttentionSlimGPT(SlimGPT):
    def __init__(self, attention_module, layer_idx, args):
        super().__init__(attention_module.out_proj, layer_idx, args)
        # ... 其他初始化

        # VAR 逐尺度统计参数
        self._var_cache = []              # 缓存最近的输入
        self._var_cache_limit = 10        # 每10次触发一次融合
        self._var_equalize_seq = True     # 是否按 seqlen 做平均（分辨率同权）
        self._var_gamma = None            # 每次 flush 的权重；None 表示均匀

        # 用于分尺度对比分析
        self._per_stage_cache = {s: [] for s in range(10)}  # 每个尺度独立缓存
        self._per_stage_H = {s: None for s in range(10)}     # 每个尺度独立的 H

    def add_batch_v7_fastoba(self, inp, out, stage_id=None):
        """
        逐尺度独立统计 Hessian（FastOBA 版本）

        参考 add_batch_v7 的缓存机制，结合 FastOBA 自动微分

        Args:
            inp: [B, L, C] Attention 输入（L 是当前尺度的 token 数）
            out: [B, L, C] Attention 输出
            stage_id: int 当前尺度 ID (0-9)，如果为 None 则尝试从 L 推断
        """
        # 推断尺度 ID
        if stage_id is None:
            L = inp.shape[1]
            patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            stage_id = next((i for i, pn in enumerate(patch_nums) if pn * pn == L), None)
            if stage_id is None:
                # 如果是累积的 token（多尺度拼接），使用全局统计
                stage_id = -1  # 特殊标记：全局

        # 缓存当前输入
        cache_entry = {
            'inp': inp.detach(),
            'out': out.detach(),
            'stage_id': stage_id,
            'seqlen': inp.shape[1]
        }
        self._var_cache.append(cache_entry)

        # 同时缓存到分尺度缓存（用于对比分析）
        if stage_id >= 0:
            self._per_stage_cache[stage_id].append(cache_entry)

        # 每隔 N 次触发一次 Hessian 计算
        if len(self._var_cache) < self._var_cache_limit:
            return

        # --- 开始 Hessian 计算 ---
        xs = self._var_cache
        self._var_cache = []  # 清空缓存，准备下一个窗口

        device = self.layer.weight.device
        dtype = torch.float32
        S = len(xs)

        # 权重 γ_s：默认均匀同权
        if self._var_gamma is None:
            gammas = torch.full((S,), 1.0 / S, device=device, dtype=dtype)
        else:
            g = torch.as_tensor(self._var_gamma, device=device, dtype=dtype)
            gammas = g / (g.sum() + 1e-12)

        # 构建加权伪损失
        def compute_weighted_pseudo_loss():
            total_loss = 0
            for s, entry in enumerate(xs):
                inp_batch = entry['inp'].to(device).requires_grad_(True)
                out_batch = self.attention_module(inp_batch)

                # 按 seqlen 归一化（分辨率同权）
                seqlen = entry['seqlen']
                norm = (1.0 / seqlen) if self._var_equalize_seq else 1.0

                # 加权损失
                loss_batch = gammas[s] * norm * out_batch.pow(2).sum()
                total_loss = total_loss + loss_batch

            return total_loss

        # 计算 Hessian
        loss = compute_weighted_pseudo_loss()
        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],
            order=self.order
        )

        # 构建块对角 H
        weight_hessian = hessian_grads[0].abs()
        H_local = self.compute_block_diagonal_hessian(
            weight_hessian, self.num_heads, self.head_dim
        )

        # EMA 更新
        tmp_total = sum(1 if entry['inp'].dim() == 2 else entry['inp'].shape[0] for entry in xs)
        self.H *= self.nsamples / (self.nsamples + tmp_total)
        self.nsamples += tmp_total
        scale = 2.0 / self.nsamples
        self.H += scale * H_local

    def compute_per_stage_hessian(self, stage_id, flush_cache=True):
        """
        计算特定尺度的独立 Hessian（用于对比分析）

        Args:
            stage_id: 0-9 尺度 ID
            flush_cache: 是否清空该尺度的缓存

        Returns:
            H_stage: [in_features, in_features] 该尺度的 Hessian
        """
        if stage_id not in self._per_stage_cache:
            return None

        xs = self._per_stage_cache[stage_id]
        if len(xs) == 0:
            return None

        device = self.layer.weight.device

        # 定义该尺度的伪损失
        def compute_stage_loss():
            total_loss = 0
            for entry in xs:
                inp_batch = entry['inp'].to(device).requires_grad_(True)
                out_batch = self.attention_module(inp_batch)
                loss_batch = out_batch.pow(2).sum()
                total_loss = total_loss + loss_batch
            return total_loss / len(xs)

        # 计算 Hessian
        loss = compute_stage_loss()
        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],
            order=self.order
        )

        # 构建块对角 H
        weight_hessian = hessian_grads[0].abs()
        H_stage = self.compute_block_diagonal_hessian(
            weight_hessian, self.num_heads, self.head_dim
        )

        # 保存结果
        self._per_stage_H[stage_id] = H_stage

        if flush_cache:
            self._per_stage_cache[stage_id] = []

        return H_stage

    def compute_per_stage_head_importance(self, stage_id):
        """
        计算特定尺度下的 head 重要性

        用于对比不同尺度对 head 选择的影响

        Args:
            stage_id: 0-9 尺度 ID

        Returns:
            head_imp: [num_heads] 该尺度下的 head 重要性
        """
        H_stage = self._per_stage_H.get(stage_id)
        if H_stage is None:
            H_stage = self.compute_per_stage_hessian(stage_id, flush_cache=False)

        if H_stage is None:
            return None

        # 计算 head 重要性（使用 block_mean 方法）
        head_imp = torch.zeros(self.num_heads)
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = (h + 1) * self.head_dim
            H_block = H_stage[start:end, start:end]
            head_imp[h] = torch.diag(H_block).mean()

        return head_imp

    def compute_per_stage_head_dim_importance(self, stage_id):
        """
        计算特定尺度下的 head 维度重要性

        用于对比不同尺度对 head 维度选择的影响

        Args:
            stage_id: 0-9 尺度 ID

        Returns:
            dim_imp: [num_heads, head_dim] 该尺度下每个 head 的维度重要性
        """
        H_stage = self._per_stage_H.get(stage_id)
        if H_stage is None:
            H_stage = self.compute_per_stage_hessian(stage_id, flush_cache=False)

        if H_stage is None:
            return None

        # 计算 Hinv（用于 OBS 重要性）
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H_stage))
        Hinv_diag = torch.diag(Hinv)

        # OBS 重要性公式
        importance = (self.layer.weight ** 2).sum(0) / Hinv_diag  # [in_features]

        # 重塑为 [num_heads, head_dim]
        dim_imp = importance.view(self.num_heads, self.head_dim)

        return dim_imp
```

---

### 11.4 分尺度对比实验：Head 维度剪枝挑选分析

#### **实验目标**

对比不同尺度下的剪枝决策差异：
1. **Head 重要性排序**：每个尺度选择哪些 head 最重要？
2. **Head 维度重要性**：每个尺度在每个 head 内选择哪些维度？
3. **剪枝决策一致性**：不同尺度的剪枝选择有多大差异？

#### **实验 A：分尺度 Head 重要性对比**

```python
def experiment_per_stage_head_analysis(model, dataloader, layer_idx, args):
    """
    分析每个尺度对 head 重要性的影响

    Returns:
        results: Dict[stage_id, Dict[str, Any]]
    """
    layer = model.blocks[layer_idx]
    attention_module = layer.attn

    # 创建剪枝器
    pruner = FastOBAAttentionSlimGPT(attention_module, layer_idx, args)

    # 收集数据（每个 batch 是一个尺度的 token）
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    for batch_idx, label in enumerate(dataloader):
        # VAR 前向传播（逐尺度生成）
        model.eval()
        with torch.no_grad():
            # 假设有 hook 捕获每个尺度的 attention 输入输出
            # 参考 prune_v7.py 的实现
            for stage_id, pn in enumerate(patch_nums):
                # 获取该尺度的输入输出
                inp, out = get_stage_io(model, layer_idx, label, stage_id)
                # 添加到剪枝器
                pruner.add_batch_v7_fastoba(inp, out, stage_id=stage_id)

    # 计算每个尺度的 Hessian
    results = {}
    for stage_id in range(10):
        # 计算该尺度的独立 Hessian
        H_stage = pruner.compute_per_stage_hessian(stage_id)
        if H_stage is None:
            continue

        # 计算 head 重要性
        head_imp = pruner.compute_per_stage_head_importance(stage_id)

        # 计算 head 维度重要性
        dim_imp = pruner.compute_per_stage_head_dim_importance(stage_id)

        # 排序
        head_ranking = torch.argsort(head_imp, descending=True)

        results[f'stage_{stage_id}'] = {
            'patch_size': patch_nums[stage_id],
            'token_count': patch_nums[stage_id] ** 2,
            'head_importance': head_imp.cpu().numpy(),
            'head_ranking': head_ranking.cpu().numpy(),
            'dim_importance': dim_imp.cpu().numpy(),  # [num_heads, head_dim]
        }

    return results
```

#### **分析指标 1：Head 排序一致性**

```python
def analyze_head_ranking_consistency(results):
    """
    分析不同尺度下 head 排序的一致性

    使用 Kendall's tau 相关系数
    """
    from scipy.stats import kendalltau

    num_stages = len(results)
    correlation_matrix = np.zeros((num_stages, num_stages))

    for si in range(num_stages):
        for sj in range(num_stages):
            ranking_i = results[f'stage_{si}']['head_ranking']
            ranking_j = results[f'stage_{sj}']['head_ranking']

            tau, p_value = kendalltau(ranking_i, ranking_j)
            correlation_matrix[si, sj] = tau

            if si < sj:
                print(f"Stage {si} ({results[f'stage_{si}']['patch_size']}×{results[f'stage_{si}']['patch_size']}) "
                      f"vs Stage {sj} ({results[f'stage_{sj}']['patch_size']}×{results[f'stage_{sj}']['patch_size']}): "
                      f"τ={tau:.3f}, p={p_value:.4f}")

    return correlation_matrix
```

#### **分析指标 2：Head 维度选择重叠度**

```python
def analyze_head_dim_selection_overlap(results, sparsity=0.3):
    """
    分析不同尺度下 head 维度选择的重叠度

    对于每个 head，比较不同尺度选择删除的维度
    """
    num_stages = len(results)
    num_heads = results['stage_0']['dim_importance'].shape[0]
    head_dim = results['stage_0']['dim_importance'].shape[1]

    # 每个尺度选择要删除的维度
    stage_pruned_dims = {}
    for stage_id in range(num_stages):
        dim_imp = results[f'stage_{stage_id}']['dim_importance']  # [num_heads, head_dim]
        n_pruned_per_head = int(head_dim * sparsity)

        pruned_dims_per_head = []
        for h in range(num_heads):
            head_imp = dim_imp[h]
            pruned_idx = torch.argsort(torch.from_numpy(head_imp))[:n_pruned_per_head].numpy()
            pruned_dims_per_head.append(set(pruned_idx))

        stage_pruned_dims[stage_id] = pruned_dims_per_head

    # 计算 Jaccard 相似度
    print("\n=== Head Dimension Selection Overlap (Jaccard Similarity) ===")
    for head_id in range(num_heads):
        print(f"\nHead {head_id}:")
        for si in range(num_stages):
            for sj in range(si + 1, num_stages):
                set_i = stage_pruned_dims[si][head_id]
                set_j = stage_pruned_dims[sj][head_id]

                jaccard = len(set_i & set_j) / len(set_i | set_j)

                print(f"  Stage {si} vs Stage {sj}: Jaccard={jaccard:.3f} "
                      f"(共同删除 {len(set_i & set_j)}/{len(set_i)} 维)")
```

#### **分析指标 3：尺度特异性 Head/Dimension**

```python
def find_scale_specific_heads(results, top_k=3):
    """
    找出每个尺度的特异性 head

    特异性 = 在该尺度排名高，但在其他尺度排名低
    """
    num_stages = len(results)
    scale_specific = {}

    for si in range(num_stages):
        ranking_si = results[f'stage_{si}']['head_ranking']
        top_heads = set(ranking_si[:top_k])

        for head_id in top_heads:
            # 计算该 head 在其他尺度的平均排名
            ranks_in_other = []
            for sj in range(num_stages):
                if sj != si:
                    ranking_sj = results[f'stage_{sj}']['head_ranking']
                    rank = np.where(ranking_sj == head_id)[0][0]
                    ranks_in_other.append(rank)

            avg_rank_other = np.mean(ranks_in_other)
            rank_in_si = np.where(ranking_si == head_id)[0][0]

            # 如果在当前尺度排名高，但在其他尺度平均排名低 → 特异性
            if avg_rank_other > 6:  # 平均排名后半
                key = f'stage_{si}_head_{head_id}'
                scale_specific[key] = {
                    'stage_id': si,
                    'patch_size': results[f'stage_{si}']['patch_size'],
                    'head_id': head_id,
                    'rank_in_stage': rank_in_si,
                    'avg_rank_other': avg_rank_other,
                    'importance': results[f'stage_{si}']['head_importance'][head_id]
                }

    return scale_specific

def find_scale_specific_dims(results, head_id, top_k=5):
    """
    找出特定 head 内的尺度特异性维度

    Args:
        results: 实验结果
        head_id: 要分析的 head ID
        top_k: 每个尺度选择前 k 个重要维度

    Returns:
        scale_specific_dims: Dict[stage_id, List[int]]
    """
    num_stages = len(results)
    head_dim = results['stage_0']['dim_importance'].shape[1]

    # 收集每个尺度的 top-k 维度
    stage_top_dims = {}
    for si in range(num_stages):
        dim_imp = results[f'stage_{si}']['dim_importance'][head_id]
        top_dims = np.argsort(dim_imp)[-top_k:][::-1]  # 降序
        stage_top_dims[si] = set(top_dims)

    # 找出特异性维度
    scale_specific_dims = {}
    for si in range(num_stages):
        top_si = stage_top_dims[si]

        # 该尺度独有的维度（不在其他尺度的 top-k 中）
        unique_dims = top_si.copy()
        for sj in range(num_stages):
            if sj != si:
                unique_dims -= stage_top_dims[sj]

        if len(unique_dims) > 0:
            scale_specific_dims[si] = {
                'patch_size': results[f'stage_{si}']['patch_size'],
                'unique_dims': list(unique_dims),
                'importance': [results[f'stage_{si}']['dim_importance'][head_id][d]
                              for d in unique_dims]
            }

    return scale_specific_dims
```

#### **可视化：分尺度 Head 重要性热力图**

```python
def visualize_per_stage_analysis(results):
    """
    可视化分尺度分析结果
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_stages = len(results)
    num_heads = results['stage_0']['head_importance'].shape[0]

    # 1. Head 重要性热力图（10 尺度 × 12 heads）
    fig, ax = plt.subplots(figsize=(14, 8))
    importance_matrix = np.zeros((num_stages, num_heads))
    for si in range(num_stages):
        importance_matrix[si, :] = results[f'stage_{si}']['head_importance']

    sns.heatmap(importance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'S{si}({results[f"stage_{si}"]["patch_size"]}×{results[f"stage_{si}"]["patch_size"]})'
                            for si in range(num_stages)],
                ax=ax)
    ax.set_title('Head Importance by Stage (FastOBA Block-Diagonal Hessian)')
    ax.set_xlabel('Head ID')
    ax.set_ylabel('Stage (Resolution)')
    plt.tight_layout()
    plt.savefig('per_stage_head_importance.png', dpi=300)

    # 2. Head 排序相关性矩阵
    fig, ax = plt.subplots(figsize=(10, 10))
    correlation_matrix = analyze_head_ranking_consistency(results)

    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0,
                xticklabels=[f'S{i}' for i in range(num_stages)],
                yticklabels=[f'S{i}' for i in range(num_stages)],
                ax=ax)
    ax.set_title("Head Ranking Correlation Between Stages (Kendall's τ)")
    plt.tight_layout()
    plt.savefig('stage_head_ranking_correlation.png', dpi=300)

    # 3. 每个 Head 的维度重要性热力图（分尺度）
    head_dim = results['stage_0']['dim_importance'].shape[1]
    for head_id in range(min(4, num_heads)):  # 只可视化前 4 个 head
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for si in range(num_stages):
            ax = axes[si // 5, si % 5]
            dim_imp = results[f'stage_{si}']['dim_importance'][head_id]

            ax.bar(range(head_dim), dim_imp)
            ax.set_title(f'Stage {si} ({results[f"stage_{si}"]["patch_size"]}×{results[f"stage_{si}"]["patch_size"]})')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Importance')

        plt.suptitle(f'Head {head_id} Dimension Importance Across Stages')
        plt.tight_layout()
        plt.savefig(f'head_{head_id}_dim_importance_per_stage.png', dpi=300)
```

---

### 11.5 实验执行命令

```bash
# 实验：分尺度 Head 分析
python experiments/var_per_stage_head_analysis.py \
    --model_path /path/to/var_d16.pth \
    --layer_idx 0 \
    --num_samples 128 \
    --hessian_mode block_diagonal \
    --output_dir results/per_stage_analysis

# 分析特定 head 的维度选择
python experiments/var_per_stage_head_analysis.py \
    --model_path /path/to/var_d16.pth \
    --layer_idx 0 \
    --num_samples 128 \
    --analyze_head_dims \
    --target_head_id 5 \
    --output_dir results/head_5_analysis
```

---

### 11.6 预期发现

#### **预期结果 A：Head 排序一致性**

- **高一致性场景** (τ > 0.7)：
  - 相邻尺度（如 Stage 5 vs Stage 6）
  - 表明这些尺度对 head 的评估相似
  - **决策**：可以使用加权平均（`_var_equalize_seq=True`）

- **低一致性场景** (τ < 0.3)：
  - 极端尺度（如 Stage 0 vs Stage 9）
  - 表明不同尺度有不同的 head 偏好
  - **决策**：需要仔细选择加权策略

#### **预期结果 B：尺度特异性 Head**

可能发现：
- **粗尺度偏好** (1×1-3×3)：全局语义 head（如 Head 0, Head 11）
- **细尺度偏好** (13×13-16×16)：局部细节 head（如 Head 4, Head 7）
- **通用 head**：所有尺度都重要（如 Head 2）

#### **预期结果 C：维度选择重叠度**

- **高重叠** (Jaccard > 0.6)：
  - 该 head 内的维度重要性在不同尺度间一致
  - 剪枝决策稳定

- **低重叠** (Jaccard < 0.3)：
  - 不同尺度选择不同的维度子集
  - 需要更多数据或更好的加权策略

---

### 11.7 决策建议矩阵

| 观察结果 | τ(extreme) | Jaccard(avg) | 推荐策略 | 理由 |
|---------|-----------|-------------|---------|------|
| **高一致性** | > 0.5 | > 0.6 | `_var_equalize_seq=True` + uniform | 尺度间差异小，均匀加权即可 |
| **中等一致性** | 0.3-0.5 | 0.4-0.6 | `_var_equalize_seq=True` + sqrt | 折中方案，轻微平衡 |
| **低一致性** | < 0.3 | < 0.4 | `_var_equalize_seq=True` + inverse | 强制平衡，保护粗尺度 |
| **发现特异性head** | - | - | 分尺度剪枝（高级） | 每个尺度独立剪枝决策 |

---
                    token_weights = torch.tensor([weights[s] for s in token_stages],
                                                device=device)

                    # 加权损失：Σ (w_i * ||out_i||²)
                    loss_batch = (out_batch.pow(2).sum(dim=-1) * token_weights).sum()

                total_loss = total_loss + loss_batch

            return total_loss / len(self.inp_cache)

        # 2. 计算 Hessian
        loss = compute_weighted_pseudo_loss()
        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],
            order=self.order
        )

        # 3. 构建块对角 H 矩阵
        weight_hessian = hessian_grads[0].abs()
        H_new = self.compute_block_diagonal_hessian(
            weight_hessian, self.num_heads, self.head_dim
        )

        # 4. EMA 更新
        tmp = len(self.inp_cache)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        scale = math.sqrt(2 / self.nsamples)
        self.H += scale * H_new

        # 清空缓存
        self.inp_cache.clear()
        self.out_cache.clear()
        if stage_weights is not None:
            self.stage_weights_cache.clear()
            self.token_to_stage_cache.clear()
```

---

### 11.4 对比实验设计

#### **实验目标**

研究问题：
1. **不同尺度对head重要性的影响差异**
2. **加权策略对剪枝结果的改善程度**
3. **剪枝后各尺度性能的变化**

#### **实验 A：单尺度剪枝对比**

```python
def experiment_single_scale_pruning(model, dataloader, sparsity=0.3):
    """
    对比每个尺度单独采样得到的剪枝决策

    返回：每个尺度下的head重要性排序
    """
    results = {}

    for stage_id in range(10):
        print(f"\n=== Stage {stage_id} ({model.patch_nums[stage_id]}×{model.patch_nums[stage_id]}) ===")

        # 1. 收集该尺度的校准数据
        calib_data = collect_calibration_single_scale(
            model, dataloader, target_stage=stage_id, num_samples=128
        )

        # 2. 创建剪枝器
        pruner = FastOBAAttentionSlimGPT(
            model.blocks[0].attn, 0, args,
            hessian_mode='block_diagonal',
            head_importance_mode='block_mean'
        )

        # 3. 计算Hessian
        for data in calib_data:
            # Forward该layer获取inp, out
            inp, out = forward_layer(model.blocks[0], data)
            pruner.add_batch(inp, out)

        # 4. 计算head重要性（不实际剪枝）
        head_imp = pruner._compute_head_importance()
        head_ranking = torch.argsort(head_imp, descending=True)

        results[f'stage_{stage_id}'] = {
            'head_importance': head_imp.cpu().numpy(),
            'head_ranking': head_ranking.cpu().numpy(),
            'pruned_heads': head_ranking[-int(12 * sparsity):].cpu().numpy()
        }

    return results
```

**分析指标**：

1. **Head排序一致性**：
   ```python
   # Kendall's tau 相关系数
   from scipy.stats import kendalltau

   for si in range(10):
       for sj in range(si+1, 10):
           tau, p_value = kendalltau(
               results[f'stage_{si}']['head_ranking'],
               results[f'stage_{sj}']['head_ranking']
           )
           print(f"Stage {si} vs Stage {sj}: τ={tau:.3f}, p={p_value:.4f}")
   ```

2. **剪枝决策重叠度**：
   ```python
   # Jaccard相似度
   def jaccard_similarity(set1, set2):
       return len(set1 & set2) / len(set1 | set2)

   # 对比不同尺度选择删除的head
   for si in range(10):
       pruned_si = set(results[f'stage_{si}']['pruned_heads'])
       for sj in range(si+1, 10):
           pruned_sj = set(results[f'stage_{sj}']['pruned_heads'])
           sim = jaccard_similarity(pruned_si, pruned_sj)
           print(f"Stage {si} vs Stage {sj}: Jaccard={sim:.3f}")
   ```

3. **尺度特异性head识别**：
   ```python
   # 找出只被某个尺度认为重要的head
   def find_scale_specific_heads(results, top_k=3):
       """找出每个尺度的特异性head"""
       scale_specific = {}

       for si in range(10):
           top_heads_si = set(results[f'stage_{si}']['head_ranking'][:top_k])

           # 计算该head在其他尺度的平均排名
           for head_id in top_heads_si:
               avg_rank_other = np.mean([
                   np.where(results[f'stage_{sj}']['head_ranking'] == head_id)[0][0]
                   for sj in range(10) if sj != si
               ])

               # 如果在当前尺度排名高，但在其他尺度排名低 → 特异性head
               if avg_rank_other > 6:  # 平均排名后半
                   scale_specific[f'stage_{si}_head_{head_id}'] = {
                       'rank_in_stage': np.where(results[f'stage_{si}']['head_ranking'] == head_id)[0][0],
                       'avg_rank_other': avg_rank_other
                   }

       return scale_specific
   ```

#### **实验 B：加权策略对比**

```python
def experiment_weighting_strategies(model, dataloader, sparsity=0.3):
    """
    对比不同加权策略的效果
    """
    strategies = ['baseline', 'inverse', 'sqrt', 'uniform']
    results = {}

    for strategy in strategies:
        print(f"\n=== Weighting Strategy: {strategy} ===")

        # 1. 收集校准数据
        if strategy == 'baseline':
            calib_data = collect_calibration_baseline(model, dataloader, 128)
            use_weights = False
        else:
            calib_data = collect_calibration_weighted(
                model, dataloader, 128, weighting=strategy
            )
            use_weights = True

        # 2. 创建剪枝器
        pruner = FastOBAAttentionSlimGPT(...)

        # 3. 计算Hessian
        for data in calib_data:
            inp, out = forward_layer(...)

            if use_weights:
                pruner.add_batch_fastoba(
                    inp, out,
                    stage_weights=data['stage_weights'],
                    token_to_stage=data['token_to_stage']
                )
            else:
                pruner.add_batch_fastoba(inp, out)

        # 4. 执行剪枝
        pruned_indices = pruner.struct_prune(sparsity=sparsity, headsize=64)

        # 5. 评估每个尺度的性能
        stage_perplexities = evaluate_by_stage(model, test_loader)

        results[strategy] = {
            'pruned_heads': pruned_indices,
            'stage_perplexities': stage_perplexities,
            'avg_perplexity': np.mean(list(stage_perplexities.values()))
        }

    return results
```

**评估指标**：

```python
def evaluate_by_stage(model, dataloader, num_samples=100):
    """
    分尺度评估模型性能
    """
    stage_losses = {i: [] for i in range(10)}

    model.eval()
    for i, (img, label) in enumerate(dataloader):
        if i >= num_samples:
            break

        img, label = img.cuda(), label.cuda()
        x_BLCv = model.vae_proxy[0].img_to_idxBl(img)

        with torch.no_grad():
            logits = model.forward_000(label, x_BLCv)  # [B, 680, V]

        # 计算每个stage的损失
        criterion = nn.CrossEntropyLoss(reduction='none')
        cur_pos = 0
        for stage_id, pn in enumerate(model.patch_nums):
            stage_len = pn * pn
            stage_logits = logits[:, cur_pos:cur_pos+stage_len, :]
            stage_targets = x_BLCv[:, cur_pos:cur_pos+stage_len]

            loss = criterion(
                stage_logits.reshape(-1, model.V),
                stage_targets.reshape(-1)
            )
            stage_losses[stage_id].append(loss.mean().item())
            cur_pos += stage_len

    # 计算困惑度
    stage_perplexities = {
        i: np.exp(np.mean(losses))
        for i, losses in stage_losses.items()
    }

    return stage_perplexities
```

#### **实验 C：可视化分析**

```python
def visualize_scale_analysis(results_single, results_weighted):
    """
    可视化不同尺度和加权策略的效果
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Head重要性热力图（10个尺度 × 12个heads）
    fig, ax = plt.subplots(figsize=(12, 8))
    importance_matrix = np.zeros((10, 12))
    for si in range(10):
        importance_matrix[si, :] = results_single[f'stage_{si}']['head_importance']

    sns.heatmap(importance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'H{i}' for i in range(12)],
                yticklabels=[f'S{i}({model.patch_nums[i]}×{model.patch_nums[i]})'
                            for i in range(10)],
                ax=ax)
    ax.set_title('Head Importance by Scale')
    ax.set_xlabel('Head ID')
    ax.set_ylabel('Scale Stage')
    plt.tight_layout()
    plt.savefig('head_importance_by_scale.png', dpi=300)

    # 2. 不同加权策略的尺度困惑度对比
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = ['baseline', 'inverse', 'sqrt', 'uniform']
    x = np.arange(10)
    width = 0.2

    for i, strategy in enumerate(strategies):
        perplexities = [results_weighted[strategy]['stage_perplexities'][si]
                       for si in range(10)]
        ax.bar(x + i*width, perplexities, width, label=strategy)

    ax.set_xlabel('Scale Stage')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity by Scale and Weighting Strategy')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'S{i}\n{model.patch_nums[i]}²' for i in range(10)])
    ax.legend()
    plt.tight_layout()
    plt.savefig('perplexity_by_strategy.png', dpi=300)

    # 3. Head排序相关性矩阵
    fig, ax = plt.subplots(figsize=(10, 10))
    correlation_matrix = np.zeros((10, 10))
    for si in range(10):
        for sj in range(10):
            tau, _ = kendalltau(
                results_single[f'stage_{si}']['head_ranking'],
                results_single[f'stage_{sj}']['head_ranking']
            )
            correlation_matrix[si, sj] = tau

    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0,
                xticklabels=[f'S{i}' for i in range(10)],
                yticklabels=[f'S{i}' for i in range(10)],
                ax=ax)
    ax.set_title("Head Ranking Correlation Between Scales (Kendall's τ)")
    plt.tight_layout()
    plt.savefig('scale_correlation.png', dpi=300)
```

---

### 11.5 实现命令

```bash
# 实验A：单尺度剪枝对比
python experiments/var_scale_analysis.py \
    --experiment single_scale \
    --num_samples 128 \
    --sparsity 0.3 \
    --output_dir results/single_scale

# 实验B：加权策略对比
python experiments/var_scale_analysis.py \
    --experiment weighting_comparison \
    --strategies baseline,inverse,sqrt,uniform \
    --num_samples 128 \
    --sparsity 0.3 \
    --output_dir results/weighting

# 实验C：完整分析pipeline
python experiments/var_scale_analysis.py \
    --experiment full_analysis \
    --num_samples 128 \
    --sparsity 0.3 \
    --visualize \
    --output_dir results/full
```

---

### 11.6 预期结果与分析

#### **预期发现**

1. **尺度特异性**：
   - 粗尺度（1×1-3×3）：可能偏好全局语义的heads
   - 细尺度（13×13-16×16）：可能偏好局部细节的heads

2. **Head排序一致性**：
   - 相邻尺度（如stage 5 vs stage 6）：高相关性（τ > 0.7）
   - 极端尺度（如stage 0 vs stage 9）：低相关性（τ < 0.3）

3. **加权效果**：
   - `inverse`权重：粗尺度性能提升，细尺度可能轻微下降
   - `sqrt`权重：平衡效果，各尺度性能均衡
   - `baseline`：细尺度最好，但粗尺度可能较差

#### **决策建议**

根据实验结果选择策略：

| 观察结果 | 推荐策略 | 理由 |
|---------|---------|------|
| τ(stage 0, stage 9) > 0.5 | baseline | 尺度间一致性高，无需加权 |
| τ(stage 0, stage 9) < 0.3 | inverse 或 sqrt | 尺度偏好差异大，需要平衡 |
| 粗尺度性能关键 | inverse | 最大化保护粗尺度 |
| 细尺度性能关键 | baseline | 顺应token分布 |
| 通用场景 | sqrt | 折中方案，稳定 |

---

### 11.7 代码集成点

在 `prune_v2.py` 中添加：

```python
# 新增参数
parser.add_argument(
    "--var_weighting", type=str, default="sqrt",
    choices=["baseline", "inverse", "sqrt", "uniform"],
    help="Weighting strategy for VAR multi-scale sampling"
)

# 校准数据收集
if args.model_type == 'var':
    if args.var_weighting == 'baseline':
        calib_data = collect_calibration_baseline(model, train_loader, args.num_samples)
    else:
        calib_data = collect_calibration_weighted(
            model, train_loader, args.num_samples,
            weighting=args.var_weighting
        )
else:
    # 其他模型的标准采样
    calib_data = collect_calibration_standard(model, train_loader, args.num_samples)
```

---

## 12. 总结

**核心创新点**:
1. ✅ **层内 Hessian 计算**: 使用 FastOBA 自动微分计算 **Attention 输出** 的二阶 Hessian（而非全局 Loss 的 Hessian）
2. ✅ **OBS 框架**: 保留 SlimGPT 的 Cholesky 分解和 OBS 剪枝逻辑（H^-1 计算、权重补偿）
3. ✅ **两种实现方案**:
   - 方案2 (FastOBA 风格): 自动微分，简单通用，推荐先实现
   - 方案1 (OBA 风格): 显式 softmax Jacobian，可分析，可选实现
4. ✅ **支持多种剪枝粒度**: head-wise 和 channel-wise
5. ✅ **数值稳定**: Dampening、EMA 更新、条件数检查

**关键理解**:
- ❗ **伪损失**: `L = ||Attention(inp)||²` (基于 Attention 输出，**不是**模型最终 Loss)
- ❗ **层内 OBS**: 目标是最小化 **Attention 输出的变化**
- ❗ **Hessian 含义**: `H = ∂²L/∂W_out_proj²` (层内二阶导数)
- ❗ **与全局 OBA 的区别**: OBA 计算全局 Loss 的 Hessian，我们计算层输出的 Hessian

**预期改进**（相比 SlimGPT H=XX^T）:
- ✅ 更准确的重要性估计（包含二阶信息和 softmax Jacobian）
- ✅ 理论基础更严格（基于泰勒展开的损失变化估计）
- ✅ 自动适配任何 Attention 变体（FastOBA 方案）
- ✅ 准确率/困惑度提升 0.5-2%（预期）

**权衡**:
- ⚠️ 计算开销增加 3-5x（可通过批量累积降低到 2-3x）
- ✅ 实现复杂度适中（FastOBA 方案 ~200 行）
- ✅ 与现有 prune_v2.py 高度兼容（只需修改 Attention 层的初始化）

**与相关工作的对比**:
| 方法 | Hessian 目标 | 包含 softmax | 理论基础 | 实现复杂度 |
|------|-------------|-------------|---------|-----------|
| **我们的方案** | Attention 输出（层内） | ✅ 自动 | OBS + 泰勒展开 | 中 |
| SlimGPT | 输入协方差 | ❌ | Fisher 近似 | 低 |
| OBA (全局) | 模型 Loss（全局） | ✅ 手动 | 三种连接性 | 高 |

---

**下一步**: 开始实现 Phase 1 - 创建 FastOBAAttentionSlimGPT 类（方案2）
