# KFAC完整技术指南
## Kronecker-Factored Approximate Curvature for Neural Network Pruning

**作者**: Claude
**日期**: 2025-01-04
**版本**: 1.0

**参考实现**: `/home/project/real_prune/OBA/torch_pruning/`

---

## 目录

1. [理论基础](#理论基础)
2. [数学推导](#数学推导)
3. [实现细节](#实现细节)
4. [代码解析](#代码解析)
5. [方法对比](#方法对比)
6. [VAR应用](#VAR应用)

---

## 理论基础

### 1.1 问题引入

在神经网络剪枝中，我们需要评估每个参数的**重要性**。一个经典的方法是**Optimal Brain Surgeon (OBS)**，它基于**Fisher信息矩阵**：

```
F = E[(∇_θ L) (∇_θ L)^T]
```

其中 θ 是所有参数的向量。

**问题**：对于一个简单的线性层 `y = Wx`，假设：
- 输入维度：64
- 输出维度：64
- 总参数数：64 × 64 = **4096**

Fisher矩阵大小：`[4096, 4096]`
- **存储**：4096² × 4 bytes ≈ **64 MB** （单层！）
- **求逆**：4096³ ≈ **68.7 G FLOPs** ≈ **10秒** （GPU）

对于深度网络（成百上千万参数），这是**不可行**的。

### 1.2 KFAC的核心思想

**关键洞察**：线性层的权重梯度有**特殊结构**。

对于 `y = Wx + b`，损失 `L(y)` 对权重的梯度是：

```
∇_W L = ∇_y L · x^T
```

这是一个**外积** (outer product)！

KFAC利用这个结构，将巨大的Fisher矩阵分解为两个小矩阵的**Kronecker积**：

```
F_W ≈ G ⊗ A
```

其中：
- **A** = E[x x^T]：输入激活的协方差矩阵 `[in_dim, in_dim]`
- **G** = E[(∇_y L) (∇_y L)^T]：输出梯度的协方差矩阵 `[out_dim, out_dim]`
- **⊗**：Kronecker积

**优势**：
- 存储：`O(m² + n²)` 而非 `O(m²n²)`
- 计算：`O(m³ + n³)` 而非 `O(m³n³)`

对于 m=n=64：
- 存储：**2 × 64² = 8K** vs 4096² = 16M （**减少2000倍**）
- 计算：**2 × 64³ ≈ 0.5M** vs 4096³ = 68G （**减少136000倍**）

---

## 数学推导

### 2.1 梯度的结构

#### 线性层的梯度

对于 `y = Wx + b`：

```
∂L/∂W[i,j] = (∂L/∂y[i]) · x[j]
```

**向量形式**：
```
∇_W L = (∇_y L) ⊗ x = [[g₁x₁, g₁x₂, ...],
                        [g₂x₁, g₂x₂, ...],
                        [...            ]]
```

其中 `g = ∇_y L` 是输出梯度。

#### 具体例子

```python
# 输入
x = [x₁, x₂, x₃]          # shape: [3]

# 输出梯度
g = ∇_y L = [g₁, g₂]      # shape: [2]

# 权重梯度
∇_W L = [[g₁·x₁, g₁·x₂, g₁·x₃],
         [g₂·x₁, g₂·x₂, g₂·x₃]]   # shape: [2, 3]
      = g ⊗ x^T
```

### 2.2 Fisher矩阵的展开

将权重矩阵 `W: [m, n]` 展平为向量 `w: [mn]`：

```
w = [W[0,0], W[0,1], ..., W[0,n-1], W[1,0], ..., W[m-1,n-1]]
```

**索引映射**：
```
w[k] = W[i, j]
其中：i = k // n  （行索引）
      j = k % n   （列索引）
```

**梯度对应**：
```
∇_w[k] L = g[i] · x[j]
```

### 2.3 Fisher矩阵的Kronecker结构

Fisher矩阵的第 `(k, k')` 个元素：

```
F[k, k'] = E[(∇_w[k] L) · (∇_w[k'] L)]
         = E[(g[i] · x[j]) · (g[i'] · x[j'])]
```

**关键假设**：输入 `x` 和梯度 `g` **独立**（在大多数情况下近似成立）。

因此：
```
F[k, k'] = E[g[i] · g[i']] · E[x[j] · x[j']]
         = G[i, i'] · A[j, j']
```

其中：
- **G** = E[g g^T]：输出梯度协方差 `[m, m]`
- **A** = E[x x^T]：输入协方差 `[n, n]`

**这正是Kronecker积的定义！**

```
F = G ⊗ A
```

### 2.4 Kronecker积详解

#### 定义

对于矩阵 `G: [m, m]` 和 `A: [n, n]`，它们的Kronecker积 `G ⊗ A` 是 `[mn, mn]` 矩阵：

```
G ⊗ A = [[G[0,0]·A, G[0,1]·A, ..., G[0,m-1]·A],
         [G[1,0]·A, G[1,1]·A, ..., G[1,m-1]·A],
         [...                                 ],
         [G[m-1,0]·A, ..., G[m-1,m-1]·A      ]]
```

每个块 `G[i,i']·A` 是 `n×n` 矩阵。

#### 具体例子

```
G = [[g₁₁, g₁₂],      # [2, 2]
     [g₂₁, g₂₂]]

A = [[a₁₁, a₁₂],      # [2, 2]
     [a₂₁, a₂₂]]

G ⊗ A = [[g₁₁·A, g₁₂·A],    # [4, 4]
         [g₂₁·A, g₂₂·A]]

      = [[g₁₁a₁₁, g₁₁a₁₂, g₁₂a₁₁, g₁₂a₁₂],
         [g₁₁a₂₁, g₁₁a₂₂, g₁₂a₂₁, g₁₂a₂₂],
         [g₂₁a₁₁, g₂₁a₁₂, g₂₂a₁₁, g₂₂a₁₂],
         [g₂₁a₂₁, g₂₁a₂₂, g₂₂a₂₁, g₂₂a₂₂]]
```

#### 索引规则

```
(G ⊗ A)[k, k'] = G[i, i'] · A[j, j']
```

其中：
- `k = i·n + j`
- `k' = i'·n + j'`

### 2.5 关键性质

#### 性质1：Kronecker积的逆

```
(G ⊗ A)^(-1) = G^(-1) ⊗ A^(-1)
```

**证明**：
```
(G ⊗ A) · (G^(-1) ⊗ A^(-1))
= (G · G^(-1)) ⊗ (A · A^(-1))    ← Kronecker积的混合积性质
= I_m ⊗ I_n
= I_{mn}
```

**意义**：
- 不需要求逆 `[mn, mn]` 的巨大矩阵
- 只需分别求逆两个小矩阵：`[m, m]` 和 `[n, n]`
- 然后用Kronecker积组合

#### 性质2：对角元素

```
(G^(-1) ⊗ A^(-1))[k, k] = G^(-1)[i, i] · A^(-1)[j, j]
```

**推导**：
```
k = i·n + j  →  对角元素对应 (i,j) = (i,j)
(F^(-1))[k, k] = G^(-1)[i, i] · A^(-1)[j, j]
```

#### 性质3：非对角元素

```
(G^(-1) ⊗ A^(-1))[k, k'] = G^(-1)[i, i'] · A^(-1)[j, j']
```

其中 `k = i·n + j`，`k' = i'·n + j'`。

### 2.6 应用到OBS剪枝

#### OBS公式

删除参数 `w_k`，其他参数的最优更新（最小化loss增加）：

```
Δw = -(w_k / (F^(-1))[k,k]) · (F^(-1))[:,k]
```

重要性评分：
```
importance(w_k) = w_k² / (F^(-1))[k,k]
```

#### 使用KFAC

对于权重 `W[i, j]`（对应参数索引 `k = i·n + j`）：

**对角元素**（重要性评分）：
```
(F^(-1))[k,k] = G^(-1)[i,i] · A^(-1)[j,j]

importance(W[i,j]) = W[i,j]² / (G^(-1)[i,i] · A^(-1)[j,j])
```

**非对角元素**（权重更新）：
```
ΔW[i', j'] = -(W[i,j] / (G^(-1)[i,i] · A^(-1)[j,j]))
             · (G^(-1)[i',i] · A^(-1)[j',j])
```

**关键优势**：不需要显式构造 `[mn, mn]` 的矩阵！

---

## 实现细节

### 3.1 KFAC因子计算

基于 `torch_pruning/pruner/kfac_utils/kfac_utils.py` 的实现。

#### 输入激活协方差 A

```python
class ComputeCovA:
    @staticmethod
    def linear(a, layer):
        """
        计算线性层的输入协方差矩阵

        Args:
            a: 输入激活 [batch_size, in_features]
            layer: nn.Linear层

        Returns:
            A: 协方差矩阵 [in_features, in_features]
               如果有bias，则 [in_features+1, in_features+1]
        """
        batch_size = a.size(0)

        # 添加bias项（常数1）
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

        # 计算协方差：A = (1/N) · a^T @ a
        return a.t() @ (a / batch_size)

    @staticmethod
    def conv2d(a, layer):
        """
        计算卷积层的输入协方差矩阵

        Args:
            a: 输入特征 [batch, in_channels, H, W]
            layer: nn.Conv2d层

        Returns:
            A: 协方差矩阵 [in_channels*kH*kW, in_channels*kH*kW]
               如果有bias，则 [in_channels*kH*kW+1, in_channels*kH*kW+1]
        """
        batch_size = a.size(0)

        # 提取所有卷积窗口
        # a: [batch, out_h, out_w, in_channels*kh*kw]
        a = _extract_patches(
            a,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding
        )

        spatial_size = a.size(1) * a.size(2)  # out_h * out_w
        a = a.view(-1, a.size(-1))            # [batch*out_h*out_w, in_c*kh*kw]

        # 添加bias项
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

        # 归一化并计算协方差
        a = a / spatial_size
        return a.t() @ (a / batch_size)
```

**关键点**：
1. **卷积转线性**：通过 `_extract_patches` 将卷积展开为矩阵乘法
2. **Bias处理**：添加常数1作为额外维度
3. **归一化**：除以spatial_size和batch_size

#### 输出梯度协方差 G

```python
class ComputeCovG:
    @staticmethod
    def linear(g, layer, batch_averaged=False):
        """
        计算线性层的输出梯度协方差矩阵

        Args:
            g: 输出梯度 [batch_size, out_features]
            layer: nn.Linear层
            batch_averaged: 梯度是否已经过batch平均

        Returns:
            G: 协方差矩阵 [out_features, out_features]
        """
        batch_size = g.size(0)

        if batch_averaged:
            # 如果已经平均，需要乘回batch_size
            cov_g = g.t() @ (g * batch_size)
        else:
            # 未平均，直接计算
            cov_g = g.t() @ (g / batch_size)

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged=False):
        """
        计算卷积层的输出梯度协方差矩阵

        Args:
            g: 输出梯度 [batch, out_channels, H, W]
            layer: nn.Conv2d层
            batch_averaged: 梯度是否已经过batch平均

        Returns:
            G: 协方差矩阵 [out_channels, out_channels]
        """
        # 重排维度：[batch, H, W, out_channels]
        g = g.transpose(1, 2).transpose(2, 3)
        g = g.contiguous()

        spatial_size = g.size(1) * g.size(2)  # H * W
        batch_size = g.size(0)

        # 展平：[batch*H*W, out_channels]
        g = g.view(-1, g.size(-1))

        # 处理batch平均
        if batch_averaged:
            g = g * batch_size

        # 乘上spatial size（因为卷积在所有空间位置共享权重）
        g = g * spatial_size

        # 计算协方差
        cov_g = g.t() @ (g / g.size(0))

        return cov_g
```

**关键点**：
1. **spatial size**：卷积的每个输出位置贡献一次梯度
2. **batch_averaged**：处理不同框架的梯度归一化方式

#### Patch模式（结构化剪枝）

```python
class ComputeCovAPatch(ComputeCovA):
    @staticmethod
    def conv2d(a, layer):
        """
        Patch模式：保持通道维度独立
        用于结构化（通道级）剪枝

        Returns:
            A: [in_channels+bias, in_channels+bias]
            （而非标准模式的 [in_c*kh*kw+bias, in_c*kh*kw+bias]）
        """
        batch_size = a.size(0)

        # 提取通道级patch: [b, oh, ow, kh, kw, in_c]
        a = _extract_channel_patches(
            a,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding
        )

        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))  # [b*oh*ow*kh*kw, in_c]

        patch_size = layer.kernel_size[0] * layer.kernel_size[1]

        # Bias项归一化
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1./patch_size)], 1)

        # 归一化
        a = a / spatial_size
        return a.t() @ (a / batch_size / patch_size)
```

**区别**：
- **标准模式**：展平空间维度，A维度为 `[in_c*kh*kw, in_c*kh*kw]`
- **Patch模式**：保持通道独立，A维度为 `[in_c, in_c]`
- **应用**：Patch模式更适合**filter/channel剪枝**

### 3.2 特征值分解

基于 `torch_pruning/pruner/algorithms/kfac_pruner.py`。

```python
def _update_inv(self):
    """
    对A和G进行特征值分解，准备计算逆矩阵

    对于每个模块：
        A = Q_a @ diag(d_a) @ Q_a^T
        G = Q_g @ diag(d_g) @ Q_g^T

    然后：
        A^(-1) = Q_a @ diag(1/d_a) @ Q_a^T
        G^(-1) = Q_g @ diag(1/d_g) @ Q_g^T
    """
    eps = 1e-15

    for m in self.modules:
        # 获取累积的A和G
        m_aa = self.m_aa[m] / self.steps  # 平均
        m_gg = self.m_gg[m] / self.steps

        # 特征值分解 A
        try:
            self.d_a[m], self.Q_a[m] = torch.linalg.eigh(m_aa)
        except:
            # 不是正定矩阵，添加正则化
            self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
                m_aa + eps * torch.eye(m_aa.size(0), device=m_aa.device)
            )

        # 特征值分解 G
        try:
            self.d_g[m], self.Q_g[m] = torch.linalg.eigh(m_gg)
        except:
            self.d_g[m], self.Q_g[m] = torch.linalg.eigh(
                m_gg + eps * torch.eye(m_gg.size(0), device=m_gg.device)
            )

        # 截断负特征值（数值误差导致）
        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())
```

**数学原理**：

对于对称正定矩阵 A：
```
A = Q @ Λ @ Q^T
```
其中：
- Q：正交矩阵（特征向量）
- Λ = diag(λ₁, λ₂, ..., λₙ)：对角矩阵（特征值）

逆矩阵：
```
A^(-1) = Q @ Λ^(-1) @ Q^T
       = Q @ diag(1/λ₁, 1/λ₂, ..., 1/λₙ) @ Q^T
```

**优势**：
- 稳定性好（处理近奇异矩阵）
- 可以截断小特征值
- 支持低秩近似

### 3.3 重要性评分

#### 方法1：基础KFAC（完整逆矩阵）

```python
def _get_unit_importance(self):
    """
    使用完整的逆矩阵计算重要性
    基于OBS公式：importance = w² / F^(-1)_diag
    """
    eps = 1e-10

    for m in self.modules:
        w = fetch_mat_weights(m, self.use_patch)  # [out_dim, in_dim]

        # 计算 A^(-1) 和 G^(-1)
        A_inv = self.Q_a[m] @ torch.diag(1.0 / (self.d_a[m] + eps)) @ self.Q_a[m].t()
        G_inv = self.Q_g[m] @ torch.diag(1.0 / (self.d_g[m] + eps)) @ self.Q_g[m].t()

        # 提取对角元素
        A_inv_diag = torch.diag(A_inv)  # [in_dim]
        G_inv_diag = torch.diag(G_inv)  # [out_dim]

        # OBS重要性：w_ij² / (G_inv[i,i] * A_inv[j,j])
        # 使用外积： G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0)
        F_inv_diag = G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0)  # [out, in]

        w_imp = w ** 2 / F_inv_diag

        # 存储
        self.importance[m] = w_imp
```

**复杂度**：
- A^(-1) 和 G^(-1)：O(m³ + n³)（特征值分解已完成）
- 重要性计算：O(mn)

#### 方法2：特征空间旋转（KFAC Eigen）

```python
def _get_unit_importance(self):
    """
    在特征空间中计算重要性
    避免显式计算完整的逆矩阵

    数学原理：
        W_star = Q_g^T @ W @ Q_a  （权重旋转到特征空间）
        importance = W_star² ⊙ (d_g ⊗ d_a)  （特征值即为重要性权重）
    """
    for m in self.modules:
        w = fetch_mat_weights(m, self.use_patch)

        # 权重旋转
        # W_star = Q_g^T @ W @ Q_a
        w_star = self.Q_g[m].t() @ w @ self.Q_a[m]

        # 重要性 = W_star² * (d_g ⊗ d_a)
        # d_g ⊗ d_a = d_g.unsqueeze(1) @ d_a.unsqueeze(0)
        importance_weight = self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0)

        w_imp = (w_star ** 2) * importance_weight

        # 保存旋转后的权重（用于迭代剪枝）
        self.W_star[m] = w_star
        self.importance[m] = w_imp
```

**数学推导**：

在特征空间中：
```
F = G ⊗ A = (Q_g @ Λ_g @ Q_g^T) ⊗ (Q_a @ Λ_a @ Q_a^T)

旋转权重：
W_star = Q_g^T @ W @ Q_a

在特征空间中，Fisher矩阵是对角的：
F_eigen = Λ_g ⊗ Λ_a

重要性（在特征空间）：
importance = W_star² / (Λ_g ⊗ Λ_a)
```

但在剪枝时，我们实际上寻找的是：
```
importance = W_star² * (Λ_g ⊗ Λ_a)
```
（数值越大越重要，因为特征值大意味着该方向信息量大）

**优势**：
- 不需要显式计算A^(-1)和G^(-1)
- 支持迭代剪枝（保持W_star）
- 更稳定（避免求逆）

#### 方法3：OBD对角近似

```python
def _get_unit_importance(self):
    """
    Optimal Brain Damage (OBD): 只使用Fisher矩阵的对角元素
    更快但精度较低

    公式：
        importance_out[i] = Σ_j w_ij² * A[j,j] * G[i,i]
        importance_in[j] = Σ_i w_ij² * A[j,j] * G[i,i]
    """
    for m in self.modules:
        w = fetch_mat_weights(m, False)  # [out_dim, in_dim]

        A = self.m_aa[m]  # [in_dim, in_dim]
        G = self.m_gg[m]  # [out_dim, out_dim]

        # 只使用对角元素
        A_diag = torch.diag(A)  # [in_dim]
        G_diag = torch.diag(G)  # [out_dim]

        # 输出通道重要性：(w² @ A) ⊙ G_diag
        out_neuron_imp = (w**2 @ A_diag) * G_diag

        # 输入通道重要性：(G @ w²) ⊙ A_diag
        in_neuron_imp = (G_diag @ w**2) * A_diag

        self.importance_out[m] = out_neuron_imp
        self.importance_in[m] = in_neuron_imp
```

**复杂度**：O(mn)（最快）

**适用场景**：
- 快速剪枝
- 对精度要求不高
- 大规模模型

---

## 代码解析

### 4.1 完整剪枝流程

基于 `torch_pruning/pruner/algorithms/kfac_pruner.py`。

```python
class KFACMetaPruner(MetaPruner):
    def obtain_importance(self, dataloader, criterion, device,
                          iter_steps=1000, fisher_type='true'):
        """
        主流程：计算所有层的重要性

        Steps:
            1. 注册hooks记录激活和梯度
            2. 前向+反向传播计算Fisher信息
            3. 特征值分解
            4. 计算重要性
            5. 清理hooks
        """
        # 1. 准备模型：注册hooks
        self._prepare_model()
        self.init_step()

        # 2. 计算Fisher信息
        self._compute_fisher(
            dataloader,
            criterion,
            device,
            fisher_type=fisher_type,
            iter_steps=iter_steps
        )

        # 3. 特征值分解
        self._update_inv()

        # 4. 计算重要性
        self._get_unit_importance()

        # 5. 清理
        self._rm_hooks()
        self._clear_buffer()

    def _prepare_model(self):
        """注册前向和反向hooks"""
        for module in self.model.modules():
            classname = module.__class__.__name__

            if classname in self.known_modules:  # ['Linear', 'Conv2d']
                self.modules.append(module)

                # 前向hook：保存输入激活
                module.register_forward_pre_hook(self._save_input)

                # 反向hook：保存输出梯度
                module.register_backward_hook(self._save_grad_output)

    def _save_input(self, module, input):
        """前向hook：保存输入激活"""
        if self.steps % self.freq == 0:
            self.a[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        """反向hook：保存输出梯度"""
        if self.steps % self.freq == 0:
            self.g[module] = grad_output[0].data

    def _compute_fisher(self, dataloader, criterion, device,
                       fisher_type='true', iter_steps=1000):
        """
        计算Fisher信息矩阵

        两种Fisher类型：
            - 'true': 从模型输出分布采样（理论正确）
            - 'empirical': 使用真实标签（计算简单）
        """
        self.model.eval()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= iter_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = self.model(inputs)

            # 计算loss
            if fisher_type == 'true':
                # True Fisher: 从预测分布采样
                sampled_y = torch.multinomial(
                    F.softmax(outputs.cpu().data, dim=1),
                    num_samples=1
                ).squeeze().to(device)
                loss = criterion(outputs, sampled_y)
            else:
                # Empirical Fisher: 使用真实标签
                loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 累积Fisher信息
            self.kfac_step()

    def kfac_step(self):
        """累积一个batch的KFAC因子"""
        for m in self.modules:
            classname = m.__class__.__name__

            # 计算A（输入协方差）
            if classname == 'Conv2d':
                a = ComputeCovA.conv2d(self.a[m], m)
            elif classname == 'Linear':
                a = ComputeCovA.linear(self.a[m], m)

            # 计算G（输出梯度协方差）
            if classname == 'Conv2d':
                g = ComputeCovG.conv2d(self.g[m], m, batch_averaged=False)
            elif classname == 'Linear':
                g = ComputeCovG.linear(self.g[m], m, batch_averaged=False)

            # 累积
            if self.steps == 0:
                self.m_aa[m] = a
                self.m_gg[m] = g
            else:
                self.m_aa[m] += a
                self.m_gg[m] += g

        self.steps += 1
```

**关键点**：
1. **Hooks**：自动记录每层的激活和梯度
2. **True Fisher**：采样保证理论正确性
3. **累积**：多个batch平均提高估计精度

### 4.2 迭代剪枝支持

基于 `torch_pruning/pruner/algorithms/kfac_eigen_pruner.py`。

```python
class KFACEigenPruner(KFACMetaPruner):
    def step(self, interactive=False, re_init=False):
        """
        执行一次剪枝步骤
        支持迭代剪枝：多轮逐步剪枝
        """
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers):
            # 剪枝这一组相关的层
            group.prune()

            # 更新KFAC因子以反映剪枝
            for grp in group:
                module = grp.dep.layer

                if 'out' in str(grp.dep):
                    # 删除输出通道：更新Q_g和W_star
                    self._update_factors_output(module, grp.idxs)

                elif 'in' in str(grp.dep):
                    # 删除输入通道：更新Q_a和W_star
                    self._update_factors_input(module, grp.idxs)

    def _update_factors_output(self, module, pruned_idxs):
        """
        删除输出通道后更新因子

        Args:
            module: 被剪枝的层
            pruned_idxs: 被删除的输出通道索引
        """
        # 创建保留mask
        out_dim = module.weight.data.shape[0]
        reserve_flag = torch.ones(out_dim, dtype=torch.bool)
        reserve_flag[pruned_idxs] = False

        # 更新Q_g：只保留未被剪枝的列
        self.Q_g[module] = self.Q_g[module][:, reserve_flag]

        # 更新d_g：只保留未被剪枝的特征值
        self.d_g[module] = self.d_g[module][reserve_flag]

        # 更新W_star：删除对应行
        self.W_star[module] = self.W_star[module][reserve_flag, :]

    def _update_factors_input(self, module, pruned_idxs):
        """
        删除输入通道后更新因子

        Args:
            module: 被剪枝的层
            pruned_idxs: 被删除的输入通道索引
        """
        # 创建保留mask
        in_dim = module.weight.data.shape[1]
        reserve_flag = torch.ones(in_dim, dtype=torch.bool)

        # 如果有bias，需要额外考虑
        if module.bias is not None:
            reserve_flag = torch.cat([reserve_flag, torch.ones(1, dtype=torch.bool)])

        reserve_flag[pruned_idxs] = False

        # 更新Q_a
        self.Q_a[module] = self.Q_a[module][:, reserve_flag]

        # 更新d_a
        self.d_a[module] = self.d_a[module][reserve_flag]

        # 更新W_star：删除对应列
        self.W_star[module] = self.W_star[module][..., reserve_flag]
```

**原理**：
- 剪枝后，权重维度减小
- 特征向量矩阵Q和特征值d也需要相应更新
- W_star在特征空间中，也需要删除对应维度
- 保持一致性，支持多轮剪枝

### 4.3 OBA和FastOBA

#### OBA：使用JVP计算Hessian

基于 `torch_pruning/pruner/algorithms/oba_pruner.py`。

```python
class OBAPruner(MetaPruner):
    def obtain_importance(self, loss):
        """
        使用Optimal Brain Apoptosis方法计算重要性

        不使用KFAC的Kronecker近似
        而是通过自动微分直接计算Hessian信息
        """
        # 初始化重要性
        current_group_importances = self.initialize_importance()

        # 1. 一阶泰勒展开
        current_group_importances = self.first_order_taylor(
            loss, current_group_importances
        )

        # 2. 二阶Hessian信息
        current_group_importances = self.both_connectivity_hessian(
            current_group_importances
        )

        # 3. 更新全局重要性
        self.update_group_importance(current_group_importances)

    def first_order_taylor(self, loss, current_group_importances):
        """
        一阶重要性：importance = δw * ∇L
        """
        # 反向传播获取梯度
        loss.backward(retain_graph=True)

        for module in self.model.modules():
            if hasattr(module, "weight"):
                # 扰动量
                delta_w = self.delta * module.weight.data

                # 梯度
                dw = module.weight.grad.data

                # 一阶重要性
                importance = delta_w * dw

                current_group_importances[module]["weight"] = importance.detach()

        return current_group_importances

    def both_connectivity_hessian(self, current_group_importances):
        """
        二阶Hessian信息（向上和向下连接）
        """
        # 向上连接：从输入到输出的二阶效应
        current_group_importances = self.upward_direct_connectivity_hessian(
            current_group_importances
        )

        # 向下连接：从输出到输入的二阶效应
        current_group_importances = self.downward_direct_connectivity_hessian(
            current_group_importances
        )

        return current_group_importances

    def upward_direct_connectivity_hessian(self, current_group_importances):
        """
        向上连接Hessian：模拟权重扰动的二阶效应

        数学原理：
            H @ δw ≈ ∇(∇L^T @ f(x, w+δw))
        其中 f 是前向传播
        """
        # 清除之前的梯度
        self.model.zero_grad()

        for module in self.model.modules():
            if hasattr(module, "weight"):
                # 权重扰动
                delta_w = self.upward_delta * module.weight.data

                # 获取保存的输入
                x = module.X

                # "替代前向"：用扰动后的权重
                zero_bias = module.bias is None
                y = self.surrogate_forward(module, x, delta_w, zero_bias)

                # 用当前的输出梯度反向传播
                output_grad = current_group_importances[module]["output_gradient"]
                y.backward(output_grad)

                # Hessian-vector积：dw = ∇(output_grad^T @ y) 对权重的导数
                dw = module.weight.grad.data

                # Hessian重要性
                upward_hessian_importance = dw * delta_w

                # 累加到总重要性
                current_group_importances[module]["weight"] += upward_hessian_importance

        return current_group_importances
```

**关键技术**：
- **JVP** (Jacobian-Vector Product)：计算Hessian-vector积
- **Surrogate forward**：模拟权重扰动的效果
- **避免显式Hessian**：不需要构造 O(p²) 的矩阵

#### FastOBA：高阶自动微分

基于 `torch_pruning/pruner/algorithms/fastoba_pruner.py`。

```python
class FastOBAPruner(MetaPruner):
    def any_order_differentiation(self, loss, delta=-1.0,
                                  parameters=None, order=1):
        """
        计算任意阶导数

        Args:
            loss: 标量损失
            delta: 扰动系数
            parameters: 参数列表
            order: 导数阶数（1=梯度，2=Hessian，...）

        Returns:
            grads: order阶导数 × 参数 × delta

        数学公式（二阶情况）：
            importance = ∂²L/∂w² × w × δ
        """
        grads = [torch.zeros_like(param) for param in parameters]

        # 逐阶计算导数
        for current_order in range(1, order + 1):
            if current_order == 1:
                # 一阶：∂L/∂w
                if current_order == order:
                    current_grad = torch.autograd.grad(
                        loss, parameters, create_graph=False
                    )
                else:
                    current_grad = torch.autograd.grad(
                        loss, parameters, create_graph=True
                    )
            else:
                # k阶：∂(grad^(k-1))/∂w
                grad_outputs = [param * delta for param in parameters]

                if current_order == order:
                    current_grad = torch.autograd.grad(
                        current_grad, parameters,
                        grad_outputs=grad_outputs,
                        create_graph=False
                    )
                else:
                    current_grad = torch.autograd.grad(
                        current_grad, parameters,
                        grad_outputs=grad_outputs,
                        create_graph=True
                    )

        # 最终重要性：grad^(order) × w × δ^order
        grads = [g.detach() * param.data * delta
                 for g, param in zip(current_grad, parameters)]

        return grads

    def obtain_importance(self, loss, order=2, distributed=False):
        """
        计算重要性（使用高阶导数）

        Args:
            loss: 损失函数
            order: 导数阶数（推荐2）
            distributed: 是否使用分布式训练
        """
        # 计算高阶导数
        grads = self.any_order_differentiation(
            loss,
            delta=self.delta,
            parameters=self.target_parameters,
            order=order,
            distributed=distributed
        )

        # 分布式：all-reduce
        if distributed:
            world_size = dist.get_world_size()
            for i, g in enumerate(grads):
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                grads[i] = g.div(world_size)

        # 存储重要性
        self.update_importance(grads)
```

**特点**：
- **最简单**：纯PyTorch自动微分
- **最快**：不需要KFAC因子计算
- **灵活**：支持任意阶导数
- **DDP友好**：内置分布式支持

---

## 方法对比

### 5.1 复杂度对比

| 方法 | 存储复杂度 | 计算复杂度 | 精度 | 特点 |
|------|-----------|----------|-----|------|
| **完整Fisher** | O(p²) | O(p³) | 理论最优 | 不可行（p太大） |
| **KFAC基础** | O(m²+n²) | O(m³+n³) | 高 | Kronecker分解 |
| **KFAC Eigen** | O(m²+n²+mn) | O(m³+n³+kmn) | 高 | 特征空间，支持迭代 |
| **KFAC-OBD** | O(m+n) | O(mn) | 中 | 对角近似，最快 |
| **KFAC-OBS** | O(m²+n²) | O(m³+n³) | 高 | 完整逆矩阵 |
| **OBA** | O(mn) | O(kmn) | 中高 | JVP，无需Fisher |
| **FastOBA** | O(mn) | O(kmn) | 中高 | 纯自动微分 |

其中：
- p：总参数数（对于64×64线性层，p=4096）
- m：输出维度（64）
- n：输入维度（64）
- k：样本数/迭代数

### 5.2 数值例子

对于典型的 64×64 线性层：

| 方法 | 存储 | 计算 | 时间估计（GPU） |
|------|------|------|---------------|
| 完整Fisher | 64 MB | 68.7 G FLOPs | ~10秒 |
| KFAC | 32 KB | 0.5 M FLOPs | ~1ms |
| KFAC-OBD | 512 B | 0.25 M FLOPs | ~0.5ms |
| FastOBA | 32 KB | 0.5 M FLOPs | ~1ms |

**加速比**：KFAC比完整Fisher快 **10,000倍**！

### 5.3 精度对比

基于实验观察（非严格测试）：

| 方法 | 精度排名 | 适用场景 |
|------|---------|---------|
| KFAC-OBS F2 | ⭐⭐⭐⭐⭐ | 高精度剪枝，充足计算资源 |
| KFAC Eigen | ⭐⭐⭐⭐⭐ | 迭代剪枝，需要保存特征空间 |
| OBA | ⭐⭐⭐⭐ | 无需Fisher，中等精度 |
| FastOBA | ⭐⭐⭐⭐ | 快速原型，分布式训练 |
| KFAC基础 | ⭐⭐⭐⭐ | 标准应用，平衡精度和速度 |
| KFAC-OBD F2 | ⭐⭐⭐ | 大规模模型，速度优先 |

### 5.4 选择指南

```
┌─────────────────────────────────────────┐
│  需要最高精度？                          │
│  └─ Yes → KFAC-OBS F2 或 KFAC Eigen   │
│  └─ No → 继续                           │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│  需要迭代剪枝（多轮）？                  │
│  └─ Yes → KFAC Eigen                    │
│  └─ No → 继续                           │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│  计算资源有限（大模型）？                │
│  └─ Yes → KFAC-OBD F2 或 FastOBA       │
│  └─ No → 继续                           │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│  使用分布式训练？                        │
│  └─ Yes → FastOBA                       │
│  └─ No → KFAC基础 或 OBA               │
└─────────────────────────────────────────┘
```

---

## VAR应用

### 6.1 VAR模型特点

VAR (Visual AutoRegressive) 模型的attention层特点：

```python
class AdaLNSelfAttn(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, ...):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 64

        # QKV权重：head分离存储
        self.Q_heads = nn.ModuleList([
            nn.Linear(head_dim, head_dim) for _ in range(num_heads)
        ])
        self.K_heads = nn.ModuleList([...])
        self.V_heads = nn.ModuleList([...])

        # O投影
        self.O_proj = nn.Linear(embed_dim, embed_dim)
```

**关键特性**：
1. ✅ **Head分离存储**：每个head有独立的QKV权重
2. ✅ **固定head_dim=64**：标准维度
3. ✅ **渐进式训练**：10个scale逐步训练

### 6.2 Head维度剪枝方案

#### 目标

对每个attention head，从64维剪枝到 (1-sparsity)×64 维，例如：
- sparsity=0.25 → 剪到48维
- sparsity=0.40 → 剪到38维

#### 方案1：KFAC Eigen（推荐）

```python
# 伪代码
def prune_var_attention_head_dims(var_model, dataloader, sparsity=0.4):
    """
    使用KFAC Eigen剪枝VAR的attention head维度
    """
    for layer_idx, attn_layer in enumerate(var_model.blocks):
        print(f"Pruning Layer {layer_idx}")

        for head_id in range(attn_layer.num_heads):
            # 1. 准备该head的QKV权重
            Q_weight = attn_layer.Q_heads[head_id].weight  # [64, 64]
            K_weight = attn_layer.K_heads[head_id].weight
            V_weight = attn_layer.V_heads[head_id].weight

            # 2. 计算KFAC因子（基于输入激活和输出梯度）
            A, G = compute_kfac_factors(
                attn_layer.Q_heads[head_id],
                dataloader,
                num_samples=100
            )

            # 3. 特征值分解
            d_a, Q_a = torch.linalg.eigh(A)
            d_g, Q_g = torch.linalg.eigh(G)

            # 4. 旋转到特征空间
            Q_star = Q_g.t() @ Q_weight @ Q_a
            K_star = Q_g.t() @ K_weight @ Q_a
            V_star = Q_g.t() @ V_weight @ Q_a

            # 5. 计算重要性
            importance_Q = (Q_star ** 2) * (d_g.unsqueeze(1) @ d_a.unsqueeze(0))
            importance_K = (K_star ** 2) * (d_g.unsqueeze(1) @ d_a.unsqueeze(0))
            importance_V = (V_star ** 2) * (d_g.unsqueeze(1) @ d_a.unsqueeze(0))

            # 总重要性
            total_importance = importance_Q + importance_K + importance_V

            # 6. 选择要剪枝的维度（输出维度）
            dims_to_remove = int(64 * sparsity)

            # 按行求和（每个输出维度的总重要性）
            row_importance = total_importance.sum(dim=1)

            # 选择最不重要的维度
            pruned_dims = torch.argsort(row_importance)[:dims_to_remove]
            kept_dims = torch.argsort(row_importance)[dims_to_remove:]

            # 7. 剪枝（删除对应的行）
            Q_weight.data = Q_weight.data[kept_dims, :]
            K_weight.data = K_weight.data[kept_dims, :]
            V_weight.data = V_weight.data[kept_dims, :]

            print(f"  Head {head_id}: {64} → {len(kept_dims)} dims")
```

**优势**：
- 理论严格（完整Fisher信息）
- 考虑QKV的联合重要性
- 支持迭代剪枝

#### 方案2：KFAC-OBS + 层级联动（创新）

结合之前讨论的**链式剪枝**思想：

```python
def cascading_prune_var_with_kfac_obs(var_model, inputs, sparsity=0.4):
    """
    链式层级剪枝 + KFAC-OBS  head维度剪枝

    关键：使用层间输出差异作为loss，通过KFAC计算Fisher信息矩阵
    """
    # 记录原始输出（作为target）
    original_outputs = {}
    with torch.no_grad():
        current_input = inputs
        for layer_idx, layer in enumerate(var_model.blocks):
            current_output = layer(current_input)
            original_outputs[layer_idx] = current_output.detach()
            current_input = current_output

    # 逐层剪枝
    current_input = inputs
    for layer_idx, layer in enumerate(var_model.blocks):
        print(f"Pruning Layer {layer_idx}")

        # 1. 先剪枝O矩阵（SlimGPT方法）
        prune_O_matrix_slimgpt(layer.O_proj, current_input)

        # 2. 前向传播
        pruned_output = layer(current_input)

        # 3. 如果有下一层，剪枝下一层的QKV head维度
        if layer_idx + 1 < len(var_model.blocks):
            next_layer = var_model.blocks[layer_idx + 1]
            target_output = original_outputs[layer_idx + 1]

            # 对每个head独立剪枝
            for head_id in range(next_layer.num_heads):
                print(f"  Processing Head {head_id}")

                # 获取该head的Q/K/V层
                Q_layer = next_layer.Q_heads[head_id]
                K_layer = next_layer.K_heads[head_id]
                V_layer = next_layer.V_heads[head_id]

                # 4. 计算KFAC因子（基于loss的梯度）
                # 注册hooks记录激活和梯度
                activations = {}
                gradients = {}

                def save_activation(name):
                    def hook(module, input, output):
                        activations[name] = input[0].data
                    return hook

                def save_gradient(name):
                    def hook(module, grad_input, grad_output):
                        gradients[name] = grad_output[0].data
                    return hook

                handle_q_fwd = Q_layer.register_forward_hook(save_activation('Q'))
                handle_q_bwd = Q_layer.register_backward_hook(save_gradient('Q'))
                handle_k_fwd = K_layer.register_forward_hook(save_activation('K'))
                handle_k_bwd = K_layer.register_backward_hook(save_gradient('K'))
                handle_v_fwd = V_layer.register_forward_hook(save_activation('V'))
                handle_v_bwd = V_layer.register_backward_hook(save_gradient('V'))

                # 前向+反向传播
                next_output = next_layer(pruned_output)
                loss = F.mse_loss(next_output, target_output)
                loss.backward(retain_graph=True)

                # 移除hooks
                handle_q_fwd.remove()
                handle_q_bwd.remove()
                handle_k_fwd.remove()
                handle_k_bwd.remove()
                handle_v_fwd.remove()
                handle_v_bwd.remove()

                # 5. 计算A和G（KFAC因子）
                # 假设Q/K/V共享输入（attention head的输入）
                A_q = activations['Q'].t() @ activations['Q'] / activations['Q'].size(0)
                G_q = gradients['Q'].t() @ gradients['Q'] / gradients['Q'].size(0)

                A_k = activations['K'].t() @ activations['K'] / activations['K'].size(0)
                G_k = gradients['K'].t() @ gradients['K'] / gradients['K'].size(0)

                A_v = activations['V'].t() @ activations['V'] / activations['V'].size(0)
                G_v = gradients['V'].t() @ gradients['V'] / gradients['V'].size(0)

                # 6. 计算逆矩阵
                eps = 1e-10
                A_inv_q = torch.cholesky_inverse(torch.linalg.cholesky(A_q + eps * torch.eye(A_q.size(0))))
                G_inv_q = torch.cholesky_inverse(torch.linalg.cholesky(G_q + eps * torch.eye(G_q.size(0))))

                A_inv_k = torch.cholesky_inverse(torch.linalg.cholesky(A_k + eps * torch.eye(A_k.size(0))))
                G_inv_k = torch.cholesky_inverse(torch.linalg.cholesky(G_k + eps * torch.eye(G_k.size(0))))

                A_inv_v = torch.cholesky_inverse(torch.linalg.cholesky(A_v + eps * torch.eye(A_v.size(0))))
                G_inv_v = torch.cholesky_inverse(torch.linalg.cholesky(G_v + eps * torch.eye(G_v.size(0))))

                # 7. 计算OBS重要性（输出维度）
                W_q = Q_layer.weight.data  # [64, 64]
                W_k = K_layer.weight.data
                W_v = V_layer.weight.data

                # 使用KFAC-OBS F2公式：importance_out[i] = Σ_j w_ij² * A_jj / G_inv[i,i]
                importance_q = torch.sum(W_q**2 @ A_q, dim=1) / torch.diag(G_inv_q)
                importance_k = torch.sum(W_k**2 @ A_k, dim=1) / torch.diag(G_inv_k)
                importance_v = torch.sum(W_v**2 @ A_v, dim=1) / torch.diag(G_inv_v)

                # 合并QKV重要性
                total_importance = importance_q + importance_k + importance_v

                # 8. 选择剪枝维度
                dims_to_remove = int(64 * sparsity)
                pruned_dims = torch.argsort(total_importance)[:dims_to_remove]
                kept_dims = torch.argsort(total_importance)[dims_to_remove:]

                # 9. 剪枝（删除对应的输出维度/行）
                W_q.data = W_q.data[kept_dims, :]
                W_k.data = W_k.data[kept_dims, :]
                W_v.data = W_v.data[kept_dims, :]

                print(f"    Head {head_id}: {64} → {len(kept_dims)} dims")

        # 更新输入
        current_input = pruned_output
```

**为什么不能用FastOBA？**

FastOBA的输出是：
```python
importance = ∂²L/∂w² × w × δ  # 标量或向量
```
- 这是每个参数的**独立重要性分数**
- **没有提供G和A的协方差矩阵结构**
- 无法应用Kronecker分解

**KFAC-OBS F2的优势**：
- 通过loss的梯度计算完整的G和A矩阵
- 使用Kronecker分解：`F ≈ G ⊗ A`
- 理论上更准确（考虑参数间相关性）
- 适合head维度剪枝（利用[64,64]的矩阵结构）


**层级联动的好处**：
- 考虑上游剪枝对下游的影响
- Loss基于实际的输出差异（而非假设的pseudo-loss）
- 更符合VAR的链式结构

### 6.3 实现建议

#### 数据准备

```python
# VAR的多scale输入
def prepare_var_calibration_data(var_model, dataset, num_samples=100):
    """
    准备VAR剪枝的校准数据
    考虑10个scale的渐进式训练
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True
    )

    calibration_data = []
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_samples:
            break

        # VAR的输入处理
        with torch.no_grad():
            # 通过VAE编码器
            z = var_model.vae_encode(images)

            # 转换为VAR的token序列
            tokens = var_model.quantize(z)

        calibration_data.append(tokens)

    return calibration_data
```

#### 渐进式训练兼容

```python
def prune_var_progressive(var_model, dataloader, sparsity=0.4):
    """
    兼容VAR的渐进式训练（10个scale）
    """
    for scale_id in range(10):  # 10 scales
        print(f"Processing Scale {scale_id}")

        # 只剪枝当前active的scale
        if scale_id <= var_model.prog_si:
            # 获取该scale的layers
            scale_layers = var_model.get_scale_layers(scale_id)

            # 对该scale的layers剪枝
            for layer in scale_layers:
                prune_attention_head_dims(layer, dataloader, sparsity)
```

#### 性能估计

假设VAR-d16模型（16层，16个head）：

| 阶段 | 操作 | 时间估计 |
|------|------|---------|
| KFAC因子计算 | 16层 × 16头 × 1ms | ~0.25秒 |
| 特征值分解 | 16层 × 16头 × 0.5ms | ~0.13秒 |
| 重要性计算 | 16层 × 16头 × 0.2ms | ~0.05秒 |
| 剪枝执行 | 16层 × 16头 | ~0.1秒 |
| **总计** | | **~0.5秒** |

相比完整Fisher方法的 ~2560秒（16层×16头×10秒），加速 **5000倍**！

### 6.4 完整代码示例

```python
import torch
import torch.nn as nn
from torch_pruning import KFACEigenPruner, FastOBAPruner

def prune_var_model(var_model, dataloader, method='kfac_eigen', sparsity=0.4):
    """
    完整的VAR模型head维度剪枝流程

    Args:
        var_model: VAR模型
        dataloader: 校准数据
        method: 'kfac_eigen' 或 'fastoba'
        sparsity: 稀疏度（0.4 = 剪枝40%）
    """
    # 准备example inputs
    example_inputs = next(iter(dataloader))[0].cuda()

    if method == 'kfac_eigen':
        # 方法1：KFAC Eigen
        pruner = KFACEigenPruner(
            model=var_model,
            example_inputs=example_inputs,
            importance=PostLayerKFACImportance(),
            pruning_ratio=sparsity,
            num_heads={'attn': 16},  # 指定每个attention的head数
            prune_head_dims=True,     # 启用head维度剪枝
            use_patch=False           # 不使用patch模式
        )

        # 计算重要性
        pruner.obtain_importance(
            dataloader=dataloader,
            criterion=nn.CrossEntropyLoss(),
            device='cuda',
            fisher_type='true',
            iter_steps=100
        )

        # 执行剪枝
        pruner.step()

    elif method == 'fastoba':
        # 方法2：FastOBA
        pruner = FastOBAPruner(
            model=var_model,
            example_inputs=example_inputs,
            importance=FastOBAImportance(normalizer='mean'),
            pruning_ratio=sparsity,
            order=2,          # 二阶导数
            delta=1.0
        )

        # 计算重要性
        for inputs, _ in dataloader:
            inputs = inputs.cuda()
            outputs = var_model(inputs)
            loss = outputs.sum()  # 或使用实际损失

            pruner.obtain_importance(loss, order=2)

        # 执行剪枝
        pruner.step()

    return var_model

# 使用示例
if __name__ == '__main__':
    # 加载VAR模型
    var_model = torch.hub.load('FoundationVision/var', 'var_d16').cuda()

    # 准备数据
    calibration_data = prepare_var_calibration_data(var_model, dataset)

    # 剪枝
    pruned_model = prune_var_model(
        var_model,
        calibration_data,
        method='kfac_eigen',
        sparsity=0.4
    )

    # 保存
    torch.save(pruned_model.state_dict(), 'var_d16_pruned_40%.pth')
```

---

## 总结

### 理论贡献

KFAC通过**Kronecker积分解**，将Fisher信息矩阵的计算从 O(p²) 降到 O(m²+n²)，使得基于Fisher信息的剪枝方法在实践中可行。

### 关键公式

```
F_W ≈ G ⊗ A
F_W^(-1) = G^(-1) ⊗ A^(-1)
importance(w_ij) = w_ij² / (G^(-1)[i,i] × A^(-1)[j,j])
```

### 实践建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 高精度剪枝 | KFAC Eigen / OBS F2 | 完整Fisher信息 |
| 迭代剪枝 | KFAC Eigen | 保存特征空间 |
| 大规模模型 | FastOBA / OBD F2 | 计算高效 |
| 分布式训练 | FastOBA | 内置DDP支持 |
| VAR head维度 | KFAC Eigen + 链式剪枝 | 结合层间联动 |

### 未来方向

1. **自适应稀疏度**：不同层/head使用不同稀疏度
2. **混合方法**：结合KFAC和FastOBA的优势
3. **硬件感知**：考虑实际硬件的计算特性
4. **量化剪枝**：同时进行剪枝和量化

---

## 参考文献

1. Martens, J., & Grosse, R. (2015). **Optimizing neural networks with Kronecker-factored approximate curvature**. ICML.
2. LeCun, Y., Denker, J., & Solla, S. (1990). **Optimal brain damage**. NeurIPS.
3. Hassibi, B., & Stork, D. G. (1993). **Second order derivatives for network pruning: Optimal brain surgeon**. NeurIPS.
4. Wang, C., et al. (2019). **Eigendamage: Structured pruning in the Kronecker-factored eigenbasis**. ICML.
5. Singh, S., & Alistarh, D. (2020). **WoodFisher: Efficient second-order approximation for neural network compression**. NeurIPS.

---

**文档版本**: 1.0
**最后更新**: 2025-01-04
**维护者**: Claude (Anthropic)
**代码参考**: `/home/project/real_prune/OBA/torch_pruning/`

---

## 附录：常见问题

### Q1: KFAC适用于所有层吗？

**A**: KFAC主要适用于**线性层**和**卷积层**。对于其他层（如LayerNorm、Activation），可以使用其他方法（如magnitude pruning）。

### Q2: True Fisher和Empirical Fisher哪个更好？

**A**:
- **True Fisher**：理论更正确，但计算略慢
- **Empirical Fisher**：实践中效果接近，计算更快
- **建议**：先用Empirical快速测试，精细调优时用True

### Q3: 如何选择calibration数据量？

**A**:
- **最少**：50-100 batch（快速原型）
- **推荐**：200-500 batch（标准精度）
- **最多**：1000+ batch（最高精度）
- **边际收益**：超过500 batch后提升有限

### Q4: 剪枝后需要fine-tune吗？

**A**:
- **KFAC/OBS剪枝**：包含权重补偿，直接使用即可
- **建议**：轻量fine-tune（1-5 epoch）可进一步提升
- **VAR模型**：可能需要在各scale上分别fine-tune

### Q5: 如何处理batch normalization？

**A**:
- BN层的统计量（running_mean/var）在剪枝时保持不变
- 剪枝后需要更新这些统计量（前向传播几个batch）
- 或者fine-tune时自动更新

---

**END OF DOCUMENT**
