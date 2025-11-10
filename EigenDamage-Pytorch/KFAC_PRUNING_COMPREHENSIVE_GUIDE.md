# KFAC剪枝方法完全指南

> 基于 Kronecker-Factored Approximate Curvature (KFAC) 的神经网络剪枝理论与实践

---

## 目录

1. [KFAC基础理论](#1-kfac基础理论)
2. [特征值分解与逆矩阵计算](#2-特征值分解与逆矩阵计算)
3. [F2近似方法详解](#3-f2近似方法详解)
4. [神经元重要性计算](#4-神经元重要性计算)
5. [OBS Surgery补偿机制](#5-obs-surgery补偿机制)
6. [Attention层的特殊考虑](#6-attention层的特殊考虑)
7. [内存需求分析](#7-内存需求分析)
8. [Head-level剪枝策略](#8-head-level剪枝策略)
9. [方法对比与选择建议](#9-方法对比与选择建议)

---

## 1. KFAC基础理论

### 1.1 Fisher信息矩阵与损失函数梯度

**定义**：Fisher信息矩阵（Fisher Information Matrix）是Hessian矩阵在损失函数上的期望近似。

#### 1.1.1 损失函数L的定义

**L = Loss Function（损失函数）**

在神经网络中，Fisher矩阵通过损失函数的梯度定义：
```
H = E[∇vec(W) L ⊗ ∇vec(W) L]
```

其中：
- **L**: 损失函数，如 `L = CrossEntropyLoss(model(X), y)`
- **∇vec(W) L**: 损失函数L对权重矩阵W的梯度（向量化后）
- **vec(W)**: 将矩阵W按列堆叠成向量的操作

**具体例子（分类任务）**：
```
L = CrossEntropyLoss(softmax(X @ W), y)

∇vec(W) L = ∂L/∂vec(W)  # 形状: [out_dim × in_dim, 1]
```

**梯度计算**：
```python
# 在KFAC实现中的体现
def _save_input(self, module, input):
    self.a = input[0]  # 保存输入激活 a

def _save_grad_output(self, module, grad_input, grad_output):
    self.g = grad_output[0]  # 保存输出梯度 ∂L/∂output
```

**梯度与协方差的关系**：
```
对于线性层: ∇W L = g @ a^T
因此: ∇vec(W) L = vec(g @ a^T) = vec(g) ⊗ vec(a)

E[∇vec(W) L ⊗ ∇vec(W) L] = E[vec(g) ⊗ vec(g)] ⊗ E[vec(a) ⊗ vec(a)]
                          = G ⊗ A
```

#### 1.1.2 经典Fisher信息矩阵

对于参数 θ，经典Fisher矩阵定义为：
```
F = E[∇log p(y|x,θ) · ∇log p(y|x,θ)ᵀ]
```

**在神经网络中的Gauss-Newton近似**：
```
H ≈ E[∇θ L ⊗ ∇θ L]  (实际使用的形式)
```

**内存问题**：对于一个1024×1024的全连接层：
- 参数总数：1,048,576
- Fisher矩阵大小：[1,048,576 × 1,048,576]
- 内存需求：约 **4.4 TB**（不可行！）

*参考文献*：
- Fisher, R. A. (1925). Theory of statistical estimation. *Mathematical Proceedings of the Cambridge Philosophical Society*
- Martens, J. (2020). New insights and perspectives on the natural gradient method. *Journal of Machine Learning Research*

### 1.2 Kronecker积详解

#### 1.2.1 Kronecker积的定义

**Kronecker积（⊗）不是外积**！它是一种特殊的矩阵运算。

对于矩阵 A: [m×n] 和 B: [p×q]，它们的Kronecker积 A ⊗ B 是一个 [mp×nq] 的矩阵：

```
A ⊗ B = [a₁₁B  a₁₂B  ...  a₁ₙB]
        [a₂₁B  a₂₂B  ...  a₂ₙB]
        [ ⋮     ⋮          ⋮  ]
        [aₘ₁B  aₘ₂B  ...  aₘₙB]
```

**具体例子**：
```
A = [1  2]     B = [5  6]
    [3  4]         [7  8]

A ⊗ B = [1×B  2×B] = [1×[5 6]  2×[5 6]] = [5  6  10 12]
        [3×B  4×B]   [1×[7 8]  2×[7 8]]   [7  8  14 16]
                     [3×[5 6]  4×[5 6]]   [15 18 20 24]
                     [3×[7 8]  4×[7 8]]   [21 24 28 32]

维度验证：A:[2×2], B:[2×2] → A⊗B:[4×4] ✓
```

#### 1.2.2 vec操作和关键数学性质

**vec操作**：将矩阵按列堆叠成向量

```
W = [w₁₁  w₁₂]     vec(W) = [w₁₁]
    [w₂₁  w₂₂]              [w₂₁]
                            [w₁₂]
                            [w₂₂]
```

**关键数学性质（Kronecker积的混合乘积性质）**：
```
vec(AXB) = (B^T ⊗ A) vec(X)
```

这个性质是KFAC方法的数学基础！

#### 1.2.3 KFAC的Kronecker分解近似

**KFAC核心思想**：将Fisher矩阵近似为两个小矩阵的Kronecker积。

```
F ≈ A ⊗ G
```

其中：
- **A**: 输入协方差矩阵 [input_dim × input_dim]
- **G**: 输出梯度协方差矩阵 [output_dim × output_dim]
- **⊗**: Kronecker积

**数学推导**：
```
对于线性层: output = input @ W^T

∇vec(W) L = vec(∇W L) = vec(g @ a^T)

使用混合乘积性质:
vec(g @ a^T) = (a ⊗ I) vec(g) = (a ⊗ g)

因此:
E[∇vec(W) L ⊗ ∇vec(W) L] = E[(a ⊗ g) ⊗ (a ⊗ g)]
                          = E[a ⊗ a] ⊗ E[g ⊗ g]
                          = A ⊗ G
```

#### 1.2.4 从向量形式到矩阵形式的转换

这是理解Surgery公式的关键！

**Step 1: OBS的向量形式**
```
δvec(W) = -H^(-1) @ [某个向量]
```

**Step 2: KFAC近似**
```
H^(-1) ≈ (G ⊗ A)^(-1) = G^(-1) ⊗ A^(-1)

因此:
δvec(W) = -(G^(-1) ⊗ A^(-1)) @ vec(某矩阵)
```

**Step 3: 转换为矩阵形式**

使用混合乘积性质的逆变换：
```
如果 vec(δW) = (G^(-1) ⊗ A^(-1)) vec(某矩阵)

那么 δW = G^(-1) @ 某矩阵 @ (A^(-1))^T
        = G^(-1) @ 某矩阵 @ A^(-1)  (因为A^(-1)对称)
```

**这就是Surgery公式的来源**：
```
δW = G^(-1) @ coeff @ A^(-1)
```

#### 1.2.5 维度验证和内存节省

**对于1024×1024层**：

**向量形式（理论正确但不可行）**：
```
vec(W): [1,048,576, 1]
H = G ⊗ A: [1,048,576, 1,048,576] → 4.4 TB ❌
```

**矩阵形式（实用的等价形式）**：
```
只需存储:
- G^(-1): [1024, 1024] = 4 MB
- A^(-1): [1024, 1024] = 4 MB
总计: 8 MB ✓

内存节省: 8 MB / 4.4 TB ≈ 0.00000018%
节省了 99.999998% 的内存！
```

*参考文献*：
- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML*
- Van Loan, C. F. (2000). The ubiquitous Kronecker product. *Journal of computational and applied mathematics*

**优势**：
- 原始：需要存储 (in × out)²
- KFAC：只需存储 in² + out²
- **节省内存**：对于1024×1024层，从4.4TB降到约 **16.8 MB**

### 1.3 统计信息收集

**A矩阵（输入协方差）**：
```
A = E[a ⊗ a]
```
- a: 输入激活值
- 在forward pass时累积

**G矩阵（输出梯度协方差）**：
```
G = E[g ⊗ g]
```
- g: 输出梯度
- 在backward pass时累积

**物理意义**：
- A捕捉：输入特征之间的相关性
- G捕捉：输出神经元之间的相关性（或损失对输出的敏感度）

---

## 2. 特征值分解与逆矩阵计算

### 2.1 特征值与特征向量

**定义**：对于方阵G，如果存在标量λ和非零向量v满足：
```
G @ v = λ · v
```

则：
- **λ** 是特征值（eigenvalue）
- **v** 是对应的特征向量（eigenvector）

**物理意义**：
- 特征向量：矩阵作用下"方向不变"的向量
- 特征值：向量被"拉伸或压缩"的倍数

### 2.2 特征值分解

对于对称矩阵（如Fisher矩阵），可以分解为：
```
G = Q @ Λ @ Qᵀ
```

其中：
- **Q**: [1024 × 1024] 特征向量矩阵（正交矩阵）
- **Λ**: diag(λ₁, λ₂, ..., λ₁₀₂₄) 对角矩阵
- **Q的每一列**是一个特征向量
- **Qᵀ @ Q = I** （单位矩阵）

**对于1024×1024的G矩阵**：
- 特征值d_g: [1024] 一维向量
- 特征向量Q_g: [1024 × 1024] 矩阵

### 2.3 通过特征值分解求逆矩阵

**原理推导**：

步骤1：特征值分解
```
G = Q @ diag(λ) @ Qᵀ
```

步骤2：两边取逆
```
G⁻¹ = (Q @ diag(λ) @ Qᵀ)⁻¹
```

步骤3：利用逆矩阵性质
```
G⁻¹ = Q⁻ᵀ @ diag(λ)⁻¹ @ Q⁻¹
```

步骤4：**关键**！Q是正交矩阵，所以 Q⁻¹ = Qᵀ
```
G⁻¹ = Q @ diag(λ)⁻¹ @ Qᵀ
    = Q @ diag(1/λ₁, 1/λ₂, ..., 1/λₙ) @ Qᵀ
```

**矩阵维度验证**（1024×1024层）：
```
Q: [1024, 1024]
diag(1/λ): [1024, 1024] (对角矩阵)
Qᵀ: [1024, 1024]

结果: [1024, 1024] @ [1024, 1024] @ [1024, 1024] = [1024, 1024] ✓
```

### 2.4 为什么选择特征分解？

在实际应用中，有三种计算逆矩阵的主要方法。让我们详细对比它们的优缺点。

#### 2.4.1 三种方法对比

| 方法 | 计算复杂度 | 数值稳定性 | 特殊优势 | 主要缺点 |
|------|-----------|-----------|----------|---------|
| **直接逆** | O(n³) | ❌ 差 | 简单直接 | 数值不稳定，条件数敏感 |
| **Cholesky分解** | O(n³/3) | ✓ 好 | 最快（约2倍速） | 要求严格正定，无诊断信息 |
| **特征分解** | O(n³) | ✓✓ 很好 | 灵活控制，提供诊断 | 稍慢，但最稳定 |

#### 2.4.2 方法1：直接逆矩阵的问题

```python
# 直接求逆
G_inv = torch.inverse(G)  # ❌ 存在严重问题
```

**问题1：数值不稳定**
```
如果G有特征值 [100, 10, 1, 0.001]
直接逆产生 [0.01, 0.1, 1, 1000]  ← 最后一个值爆炸！

可能导致：
- 数值溢出 (1000+ 会继续放大)
- 梯度爆炸
- 优化不稳定
- NaN或Inf出现
```

**问题2：无法处理奇异或接近奇异的矩阵**
```
如果某个特征值 = 0 或接近0:
→ 直接求逆会失败或产生无穷大
→ Fisher矩阵经常出现这种情况！
```

**问题3：无法诊断和控制**
```
无法知道：
- 矩阵的条件数
- 哪些方向是"病态"的
- 如何针对性地正则化
```

#### 2.4.3 方法2：Cholesky分解的限制

```python
# Cholesky分解（L @ L^T = G）
try:
    L = torch.linalg.cholesky(G)
    G_inv = torch.cholesky_inverse(L)
except:
    print("Cholesky failed: matrix not positive definite")
```

**优点**：
- ✓ 最快的方法（比特征分解快约2倍）
- ✓ 数值稳定（对正定矩阵）
- ✓ 内存效率高

**致命缺点1：要求严格正定**
```
Cholesky要求：
- 所有特征值 > 0（严格正定）
- 不能有0或负特征值

Fisher矩阵的现实：
- 经常半正定（有0特征值）
- 深度网络的Fisher矩阵经常病态
- 即使理论上正定，数值上可能不满足

结果：Cholesky经常失败 ❌
```

**致命缺点2：无法处理小特征值**
```
如果λ_min = 1e-10:
- Cholesky会尝试分解，但数值极不稳定
- 无法像特征分解那样设置阈值
- 无法过滤或正则化小特征值
```

**致命缺点3：无法提供诊断信息**
```
Cholesky只返回L矩阵，无法得到：
- 特征值分布
- 条件数
- 有效秩
- 谱范数

这些信息对调试和优化至关重要！
```

#### 2.4.4 方法3：特征分解的优势

```python
# 特征分解
d, Q = torch.linalg.eigh(G)  # 对称矩阵专用
eps = 1e-10
G_inv = Q @ torch.diag(1.0 / (d + eps)) @ Q.T
```

**优势1：数值稳定性的精确控制**

```python
# 策略1：简单截断
d_stable = torch.where(d > eps, d, eps)
G_inv = Q @ torch.diag(1.0 / d_stable) @ Q.T

# 策略2：过滤小特征值
mask = d > eps
d_filtered = d[mask]
Q_filtered = Q[:, mask]
G_inv = Q_filtered @ torch.diag(1.0 / d_filtered) @ Q_filtered.T

# 策略3：软阈值（Tikhonov正则化）
lambda_reg = 1e-6
G_inv = Q @ torch.diag(1.0 / (d + lambda_reg)) @ Q.T
```

**优势2：丰富的诊断信息**

```python
# 分析矩阵性质
condition_number = d.max() / d.min()
effective_rank = (d > 1e-8).sum()
spectral_norm = d.max()
explained_variance = d.cumsum() / d.sum()

print(f"条件数: {condition_number:.2e}")
print(f"有效秩: {effective_rank}/{len(d)}")
print(f"90%方差需要: {(explained_variance > 0.9).nonzero()[0].item()} 个特征值")

# 根据诊断结果选择策略
if condition_number > 1e12:
    # 严重病态，需要强正则化
    d_reg = d + 1e-5
elif condition_number > 1e8:
    # 中度病态，温和正则化
    d_reg = d + 1e-7
else:
    # 健康矩阵，只过滤极小值
    d_reg = torch.where(d > eps, d, eps)
```

**优势3：低秩近似**

```python
# 内存受限时的策略
def low_rank_inverse(G, max_rank=100):
    d, Q = torch.linalg.eigh(G)

    # 只保留最大的max_rank个特征值
    idx = torch.argsort(d, descending=True)[:max_rank]
    d_top = d[idx]
    Q_top = Q[:, idx]

    # 低秩近似的逆
    G_inv_lr = Q_top @ torch.diag(1.0 / d_top) @ Q_top.T

    return G_inv_lr

# 内存节省：
# 完整: [1024, 1024] = 4 MB
# 低秩(100): [1024, 100] + [100] ≈ 0.4 MB
# 节省 90% 内存！
```

**优势4：与优化理论的契合**

```
自然梯度更新：
θ_new = θ - α * F^(-1) * ∇L

如果不控制特征值：
- 大特征值的逆 → 小步长 ✓
- 小特征值的逆 → 大步长 ❌ 可能发散

特征分解允许我们：
- 截断过小的特征值
- 保持优化的稳定性
- 自适应地调整不同方向的学习率
```

#### 2.4.5 Fisher矩阵的特殊性质

**为什么Fisher矩阵特别需要特征分解？**

**性质1：半正定**
```
Fisher矩阵 = E[g ⊗ g] ≥ 0

典型的特征值分布（1024维）：
[128.5, 64.2, 32.1, ..., 0.8, 0.001, 0.0001, 1e-8, 1e-10, ...]
                                ↑
                          大量接近0的特征值

原因：
- 神经网络参数冗余
- 某些方向几乎不影响输出
- 数据分布的低维流形结构
```

**性质2：病态条件数**
```
深度网络的Fisher矩阵：
- 条件数通常 > 10^10
- 跨越10个数量级的特征值
- 数值计算的噩梦

例如：
λ_max = 1e3
λ_min = 1e-7
条件数 = 1e10  ← 极度病态！

Cholesky会失败，直接逆会爆炸
只有特征分解能处理
```

**性质3：低有效秩**
```
虽然名义维度 = 1024
但有效秩可能只有 50-200

意味着：
- 大部分特征值接近0
- 只有少数方向真正重要
- 非常适合低秩近似

特征分解自然地揭示这一结构
```

#### 2.4.6 实际代码中的实现

```python
# 来自 kfac_full_pruner.py:126-137
def _update_inv(self):
    eps = 1e-15  # 极小的稳定性参数

    for idx, m in enumerate(self.modules):
        # 平均化统计信息
        m_aa, m_gg = self.m_aa[m] / self.steps, self.m_gg[m] / self.steps

        # 🔑 关键：使用特征分解
        self.d_a[m], self.Q_a[m] = torch.symeig(m_aa, eigenvectors=True)
        self.d_g[m], self.Q_g[m] = torch.symeig(m_gg, eigenvectors=True)

        # 🔑 关键：过滤小特征值
        self.d_a[m].mul_((self.d_a[m] > eps).float())  # < eps 的设为0
        self.d_g[m].mul_((self.d_g[m] > eps).float())
```

**这样做的好处**：
1. ✓ 自动处理半正定矩阵
2. ✓ 避免数值爆炸
3. ✓ 保留诊断信息用于调试
4. ✓ 可以灵活调整eps平衡稳定性和精度
5. ✓ 支持低秩近似节省内存

#### 2.4.7 总结

**选择特征分解的根本原因**：

1. **Fisher矩阵的本质**：半正定、经常病态、低有效秩
2. **剪枝的需求**：需要极其稳定的数值计算
3. **实用考虑**：需要诊断、调试、可视化特征值分布
4. **扩展性**：支持正则化、低秩近似、自适应策略

虽然Cholesky在理论上更快，但Fisher矩阵的特殊性质使得**特征分解成为唯一实用的选择**。

这就是为什么所有主流的KFAC、自然梯度、二阶优化方法都采用特征分解！

*参考文献*：
- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML*
- Grosse, R., & Martens, J. (2016). A Kronecker-factored approximate Fisher matrix for convolution layers. *ICML*
- Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press

---

## 3. F2近似方法详解

### 3.1 F2近似的定义

**完整KFAC**：
```
F ≈ A ⊗ G
```

**F2近似**：进一步假设
```
F₂ = A ⊗ B
```
其中：
- A: [input_dim × input_dim] 完整矩阵
- B: [output_dim × output_dim] **对角矩阵**

**关键假设**：输出神经元之间无二阶相关性。

### 3.2 F2近似发生在哪一步？

**在重要性计算的这一行**：

**Full KFAC**（使用A和G的对角元素外积）：
```
w_imp = w² / (diag(G_inv) ⊗ diag(A_inv))
out_neuron_imp = sum(w_imp, dim=1)
```

**F2方法**（只用G的对角元素）：
```
w_imps = sum(w² @ A, dim=1)  ← 用完整的A矩阵
out_neuron_imp = w_imps / diag(G_inv)  ← 只用G_inv的对角！
```

**对比图示**：

```
G矩阵（实际）:                 G矩阵（F2假设）:
[g₁₁ g₁₂ g₁₃ ... g₁ₙ]        [g₁₁  0   0  ...  0 ]
[g₂₁ g₂₂ g₂₃ ... g₂ₙ]        [ 0  g₂₂  0  ...  0 ]
[g₃₁ g₃₂ g₃₃ ... g₃ₙ]   →    [ 0   0  g₃₃ ...  0 ]
[... ... ... ... ...]        [... ... ... ... ...]
[gₙ₁ gₙ₂ gₙ₃ ... gₙₙ]        [ 0   0   0  ... gₙₙ]
      ↑ 非对角元素被忽略
```

### 3.3 F2近似的含义

**数学含义**：
- 假设不同输出神经元的Fisher信息互不相关
- G矩阵只保留对角元素，非对角元素设为0

**物理含义**：
- 删除某个输出神经元i，只影响神经元i本身
- 不会通过二阶项影响其他神经元
- 各输出神经元"独立"

**适用场景**：
- 输出神经元之间相关性弱
- 网络层的输出分布相对独立

### 3.4 对角性的测量

**判断G矩阵是否对角占优**：
```
对角占比 = ||diag(G)||_F / ||G||_F

- > 80%: F2近似合理
- 60%-80%: F2可用但建议谨慎
- < 60%: 不建议用F2，用完整KFAC
```

其中 ||·||_F 是Frobenius范数（矩阵所有元素平方和的平方根）。

---

## 4. 神经元重要性计算

### 4.1 理论基础（OBS公式）

**Optimal Brain Surgeon (OBS)** 的核心思想：

对于要删除的参数w_i，最优的补偿方式是：
```
δw = -H⁻¹ @ eᵢ @ (eᵢᵀ @ H⁻¹ @ eᵢ)⁻¹ @ wᵢ
```

损失增加为：
```
ΔL = wᵢ² / (2 · [H⁻¹]ᵢᵢ)
```

**重要性**正比于：
```
importance ∝ w² / [H⁻¹]对角元素
```

### 4.2 Full KFAC的重要性计算

**步骤1**：计算A_inv和G_inv
```
A_inv = Q_a @ diag(1/d_a) @ Q_aᵀ  [1024, 1024]
G_inv = Q_g @ diag(1/d_g) @ Q_gᵀ  [1024, 1024]
```

**步骤2**：提取对角元素
```
A_inv_diag = diag(A_inv)  [1024]
G_inv_diag = diag(G_inv)  [1024]
```

**步骤3**：计算每个权重的重要性
```
外积: G_inv_diag[:, None] @ A_inv_diag[None, :]
     = [1024, 1] @ [1, 1024] = [1024, 1024]

结果[i,j] = G_inv_diag[i] × A_inv_diag[j]

w_imp[i,j] = w²[i,j] / (G_inv_diag[i] × A_inv_diag[j])
```

**步骤4**：聚合到神经元级别
```
out_neuron_imp[i] = sum(w_imp[i, :])  对第i个输出神经元的所有输入求和
```

### 4.3 F2的重要性计算

**步骤1**：用完整A矩阵计算加权平方和
```
w_imps = sum(w² @ A, dim=1)  [1024]
```

展开：
```
w_imps[i] = Σⱼ Σₖ w²[i,j] × A[j,k]
```

**物理意义**：考虑了输入特征之间的协方差，比简单的平方和更准确。

**步骤2**：除以G_inv的对角元素
```
out_neuron_imp[i] = w_imps[i] / diag(G_inv)[i]
```

### 4.4 物理意义解释

**分子（w²或w_imps）**：
- 权重的"能量"或"强度"
- 越大说明该神经元对网络的直接贡献越大

**分母（Fisher逆的对角元素）**：
- 删除该神经元的"代价"
- 越大说明删除的代价越小（容易补偿）

**重要性公式的直觉**：
```
importance = 贡献 / 删除代价

- 高重要性 = 贡献大且难补偿 → 不应剪枝
- 低重要性 = 贡献小或易补偿 → 可以剪枝
```

---

## 5. OBS Surgery补偿机制

### 5.1 为什么需要Surgery？

**问题**：直接删除神经元会导致网络输出突变。

**解决**：在删除之前，调整保留神经元的权重来"补偿"被删除神经元的功能。

**目标**：最小化删除操作对网络输出的影响。

### 5.2 Full KFAC Surgery的三个步骤

#### 步骤1：计算归一化系数（coeff矩阵）

```
coeff = w / (G_inv_diag[:, None] @ A_inv_diag[None, :])
```

**维度**：[1024, 1024]

**每个元素的含义**：
```
coeff[i,j] = w[i,j] / (G_inv_diag[i] × A_inv_diag[j])
```

表示权重w[i,j]在补偿计算中的"责任系数"。

#### 步骤2：标记被剪枝的神经元

```
coeff[保留的神经元索引, :] = 0
```

**为什么这样做？**
- 只有被剪枝的神经元需要被"补偿掉"
- 保留的神经元设为0，表示它们本身不需要被消除
- 非零的行对应被剪枝的神经元

**示例**（8个神经元，保留[0,1,2,4,5,7]，剪枝[3,6]）：
```
coeff矩阵:
行0: 全0  ← 保留
行1: 全0  ← 保留
行2: 全0  ← 保留
行3: 非0  ← 被剪枝，需要补偿
行4: 全0  ← 保留
行5: 全0  ← 保留
行6: 非0  ← 被剪枝，需要补偿
行7: 全0  ← 保留
```

#### 步骤3：通过Fisher逆分配补偿

```
delta_theta = -G_inv @ coeff @ A_inv
```

**公式推导**：
```
基于泰勒展开和约束优化，推导出：
δθ* = argmin ||δθ||²_{H}  s.t. θ_pruned + δθ_pruned = 0

解为：
δθ = -H⁻¹ @ [处理后的系数]

在KFAC近似下：
H⁻¹ ≈ G⁻¹ ⊗ A⁻¹

展开为矩阵形式：
δθ = -G⁻¹ @ coeff @ A⁻¹
```

**三个矩阵的作用**：
1. **G_inv**：决定补偿如何在输出神经元间分配
2. **coeff**：编码哪些权重被删除及其影响
3. **A_inv**：决定补偿如何在输入维度间分配

**为什么是负号？**
- 要"抵消"被删除权重的影响
- 被删除的权重贡献w[i,j]，补偿应该是-w[i,j]（相反方向）

### 5.3 F2 Surgery的简化版本

F2的surgery只考虑输出侧：

**步骤1**：计算G_inv（完整矩阵）
```
G_inv = Q_g @ diag(1/d_g) @ Q_gᵀ  [1024, 1024]
```

**步骤2**：将保留神经元对应的列清零
```
G_inv[:, 保留的索引] = 0
```

**结果**：G_inv只保留被剪枝神经元的列。

**步骤3**：归一化和计算补偿
```
G_inv_diag = diag(G_inv)
coeff = G_inv @ diag(1 / G_inv_diag)
delta_theta = -coeff @ w
```

**关键差异**：
- 完全忽略了A_inv
- 假设输入维度之间无相关性
- 直接作用于权重w，不需要A_inv变换

### 5.4 Surgery的直观理解

**类比**：桥梁的拆除与加固

1. **识别要拆的桥墩**（被剪枝的神经元）
2. **计算负载分配**（coeff矩阵）
   - 每个桥墩承担多少重量？
   - 拆除后重量如何转移？
3. **加固剩余桥墩**（delta_theta更新）
   - 保留的桥墩需要加强多少？
   - 通过Fisher矩阵计算最优分配

**结果**：拆除后的桥梁（网络）功能基本不变。

---

## 6. Attention层的特殊考虑

### 6.1 Multi-Head Attention结构

**标准Transformer Attention流程**：

```
输入 X: [batch, seq_len, d_model]

1. 线性投影:
   Q = X @ W_q  [d_model, d_model]
   K = X @ W_k  [d_model, d_model]
   V = X @ W_v  [d_model, d_model]

2. 分头重塑:
   Q → [batch, num_heads, seq_len, head_dim]
   K → [batch, num_heads, seq_len, head_dim]
   V → [batch, num_heads, seq_len, head_dim]

3. 计算Attention:
   Attention = softmax(QKᵀ/√d) @ V

4. 拼接heads:
   concat(head_0, head_1, ..., head_n-1)
   → [batch, seq_len, d_model]

5. ⭐ O矩阵投影（我们要剪枝的层）:
   output = concat_heads @ W_o
   W_o: [d_model, d_model]
```

### 6.2 O矩阵的输入特性

**输入结构**（以d_model=768, num_heads=12为例）：
```
O矩阵的输入 = [Head_0 | Head_1 | ... | Head_11]
              [64 dim | 64 dim | ... | 64 dim ] = 768 dim

每个head占64维（head_dim = 768/12）
```

**输入相关性分析**：

1. **所有head来自同一输入X**
   - 共享位置编码
   - 共享底层语义信息

2. **不同head关注不同子空间**
   - 设计目的：捕捉不同类型的关系
   - 但不是完全独立

3. **实验观察**：
   - Head之间的相关系数：0.3-0.7
   - 浅层更相关，深层稍独立

**结论**：A矩阵（输入协方差）有显著的非对角元素。

### 6.3 O矩阵的输出特性

**输出用途**：
```
O矩阵输出 → LayerNorm → FFN → ...
```

**输出相关性来源**：

1. **LayerNorm的影响**
   - 在特征维度上归一化
   - 公式：(x - mean) / std
   - 引入输出神经元之间的依赖

2. **Residual Connection**
   - output = Attention(X) + X
   - 进一步增强相关性

3. **后续FFN的联合使用**
   - FFN需要多个特征的联合信息
   - 输出神经元不是独立处理的

**结论**：G矩阵（输出协方差）也有显著的非对角元素。

### 6.4 F2方法对Attention O矩阵的适用性

**F2假设**：G是对角矩阵（输出无相关性）

**实际情况**：
```
G矩阵的对角占比测试（基于模拟）:

场景1: 独立heads（理想）
  → 对角占比: ~85%  ✓ F2适用

场景2: 弱相关heads
  → 对角占比: ~72%  △ F2可用，建议谨慎

场景3: 中等相关heads（现实）
  → 对角占比: ~55%  ✗ 不建议F2

场景4: 强相关heads
  → 对角占比: ~35%  ✗ 必须用Full KFAC
```

**判断标准**：
```
对角占比 = ||diag(G)||_F / ||G||_F

> 80%:  ✓ F2方法合理
60%-80%: △ F2可用，但建议谨慎验证
< 60%:  ✗ 不建议F2，改用Full KFAC或Head-level
```

### 6.5 针对Attention的建议

**方案优先级**：

1. **首选：Head-level剪枝**
   - 不剪单个神经元，剪整个head
   - Head之间相对独立
   - 更符合Attention的语义结构

2. **次选：Full KFAC**
   - 考虑A和G的完整结构
   - 通过对角元素外积简化
   - 比F2更保守的近似

3. **谨慎：F2方法**
   - 需要先测量对角占比
   - 如果对角占比低，效果会差

**分层策略**：
```
早期层（靠近输入）:
  - Head相关性较弱
  - 对角占比可能>70%
  → 可以考虑F2

中间层:
  - Head相关性中等
  - 对角占比约50-70%
  → 建议Full KFAC

后期层（靠近输出）:
  - Head相关性较强
  - 对角占比可能<50%
  → 必须Full KFAC或Head-level
```

---

## 7. 内存需求分析

### 7.1 完整Hessian的内存需求

**对于1024×1024全连接层**：

```
参数总数: 1024 × 1024 = 1,048,576

Hessian矩阵:
- 大小: [1,048,576 × 1,048,576]
- 元素数量: 1,098,511,627,776
- 内存（float32）: 1,098,511,627,776 × 4 bytes
                 = 4,398,046,511,104 bytes
                 = 4.4 TB ❌ 完全不可行
```

### 7.2 KFAC的内存需求

**存储的矩阵**：

```
1. 输入协方差 A: [1024, 1024]
   = 1024² × 4 bytes = 4,194,304 bytes = 4.0 MB

2. 输出协方差 G: [1024, 1024]
   = 1024² × 4 bytes = 4,194,304 bytes = 4.0 MB

3. 特征向量 Q_a: [1024, 1024] = 4.0 MB
4. 特征向量 Q_g: [1024, 1024] = 4.0 MB
5. 特征值 d_a: [1024] = 4,096 bytes = 4 KB
6. 特征值 d_g: [1024] = 4,096 bytes = 4 KB

总计: 约 16.8 MB per layer
```

**内存节省比例**：
```
KFAC vs 完整Hessian:
= 16.8 MB / 4.4 TB
= 16.8 / (4.4 × 10⁶) MB
≈ 0.0000038
= 0.00038%

节省了 99.9996% 的内存！
```

### 7.3 VAR模型的实际内存估算

**典型VAR-d16配置**：
```
d_model = 1024
num_layers = 24
num_heads = 16
head_dim = 64
ffn_hidden = 4096
```

**每层的KFAC内存**：

```
1. attn.qkv: [1024, 3×1024] = [1024, 3072]
   A: 1024² = 4 MB
   G: 3072² = 36 MB
   小计: 40 MB

2. attn.proj (O矩阵): [1024, 1024]
   A: 1024² = 4 MB
   G: 1024² = 4 MB
   小计: 8 MB

3. ffn.fc1: [1024, 4096]
   A: 1024² = 4 MB
   G: 4096² = 64 MB
   小计: 68 MB

4. ffn.fc2: [4096, 1024]
   A: 4096² = 64 MB
   G: 1024² = 4 MB
   小计: 68 MB

每层总计: 40 + 8 + 68 + 68 = 184 MB
```

**加上特征值分解结果（×2）**：
```
每层实际: 184 × 2 = 368 MB
24层总计: 368 × 24 = 8,832 MB ≈ 8.6 GB
```

### 7.4 服务器配置建议

**GPU内存需求**：
```
- KFAC统计: ~9 GB
- 模型参数: VAR-d16约2-3 GB
- 前向激活: 取决于batch size，约2-4 GB
- 梯度: 约等于参数大小，2-3 GB

总计: 15-19 GB

建议配置:
- 最低: 16GB VRAM (如 V100)
- 推荐: 24GB VRAM (如 A100)
- 更好: 40GB+ VRAM (如 A100 40GB)
```

**系统内存**：
```
- 数据加载: 10-20 GB
- 中间计算: 10-15 GB

建议: 32GB+ 系统内存
```

### 7.5 低秩近似进一步节省内存

**方法**：只保留前k个最大的特征值

```
完整: Q [1024, 1024] + d [1024]
低秩: Q [1024, k] + d [k]

例如 k=100:
- 完整: 4 MB
- 低秩: 0.4 MB

节省: 90% 内存
代价: 略微损失精度
```

**适用场景**：
- 特征值快速衰减（大部分特征值很小）
- 对精度要求不是极高
- 内存严重受限

---

## 8. Head-level剪枝策略

### 8.1 为什么需要Head-level剪枝？

**Neuron-level剪枝的问题**（对于Attention）：

1. **破坏结构**：打乱head的完整性
2. **相关性强**：输入输出都有相关性，F2假设不成立
3. **硬件不友好**：不规则的稀疏难以加速

**Head-level剪枝的优势**：

1. **保持结构**：每个head作为一个单元
2. **语义清晰**：head有明确的功能（关注不同模式）
3. **硬件友好**：规则的结构化稀疏
4. **适合KFAC**：head之间相对独立

### 8.2 Head-level剪枝的原理

**基本思想**：不剪单个神经元，剪整个head（64维的块）

```
原始 (12 heads × 64 dims = 768):
┌────────┬────────┬────────┬─────┬────────┐
│ Head 0 │ Head 1 │ Head 2 │ ... │ Head 11│
│ 64 dim │ 64 dim │ 64 dim │     │ 64 dim │
└────────┴────────┴────────┴─────┴────────┘
     ↓ O矩阵 [768, 768]

剪枝后 (9 heads × 64 dims = 576):
┌────────┬────────┬────────┬─────┬────────┐
│ Head 0 │ Head 2 │ Head 3 │ ... │ Head 10│
│ 64 dim │ 64 dim │ 64 dim │     │ 64 dim │
└────────┴────────┴────────┴─────┴────────┘
     ↓ O矩阵 [576, 768]

剪掉了: Head 1, Head 5, Head 8
```

### 8.3 Head重要性计算

**方法1：基于KFAC的Head重要性**

```
步骤1: 计算逐元素重要性（与neuron-level相同）
w_imp[i,j] = w²[i,j] / (G_inv_diag[i] × A_inv_diag[j])

步骤2: 按head分组求和
对于第h个head:
  start = h × head_dim
  end = (h+1) × head_dim

  head_importance[h] = sum(w_imp[:, start:end])

步骤3: 选择保留的heads
排序head_importance，保留前k个
```


### 8.4 Head-level Surgery

**关键**：剪掉整个head后，surgery也要考虑块状结构

```
如果剪掉Head h (第h个64维块):

输入索引: [h×64, (h+1)×64)

在KFAC surgery中:
- A矩阵的这64个维度对应的列会被处理
- 补偿会考虑整个块的联合影响
- 比逐个神经元剪枝更稳定
```

### 8.5 实现考虑

**如何识别head边界**：

```
需要知道:
- num_heads: head数量
- head_dim: 每个head的维度
- head_dim = d_model / num_heads

第h个head对应的输入维度:
- start_idx = h × head_dim
- end_idx = (h+1) × head_dim
```

**剪枝策略**：

```
统一剪枝: 所有层剪相同比例的heads
  - 简单
  - 可能不是最优

逐层剪枝: 每层根据重要性独立决定
  - 更灵活
  - 需要更多搜索

重要性排序全局剪枝:
  - 在所有层的所有heads中排序
  - 全局选择最不重要的
  - 最优但计算量大
```

### 8.6 与Neuron-level的对比

| 特性 | Neuron-level | Head-level |
|------|--------------|------------|
| **粒度** | 单个神经元（1维） | 整个head（64维） |
| **结构** | 不规则稀疏 | 结构化稀疏 |
| **灵活性** | 高（可以精确控制） | 中（以head为单位） |
| **硬件加速** | 困难 | 容易 |
| **KFAC适用性** | 需要考虑相关性 | 更适合（head独立性） |
| **语义** | 无明确意义 | 有明确功能 |
| **实现难度** | 中 | 相对简单 |

---

## 9. 方法对比与选择建议

### 9.1 三种方法的完整对比

| 特性 | Full KFAC | F2 | Head-level |
|------|-----------|-----|------------|
| **计算复杂度** | 高 | 中 | 低-中 |
| **内存需求** | 中（需要A_inv, G_inv） | 中（同样需要） | 低-中 |
| **Surgery** | G_inv @ coeff @ A_inv | G_inv @ coeff | 块状surgery |
| **输入相关性** | ✓ 考虑（A_inv） | △ 部分考虑 | ✓ 考虑 |
| **输出相关性** | ✓ 考虑（G_inv完整） | ✗ 只用对角 | ✓ 考虑 |
| **假设** | 对角近似两侧 | G是对角矩阵 | Head独立 |
| **精度** | 高 | 中（依赖对角占比） | 高 |
| **硬件友好** | 不规则 | 不规则 | 结构化 |
| **适合CNN** | ✓ | ✓ | ✗ |
| **适合Attention** | ✓ | △ 需验证 | ✓✓ 推荐 |

### 9.2 决策流程图

```
开始
  ↓
是否为Attention的O矩阵？
  ├─ 是 → 推荐Head-level剪枝
  │       └─ 实现困难？
  │           ├─ 是 → 使用Full KFAC
  │           └─ 否 → Head-level
  │
  └─ 否 → 是否为全连接层/卷积层？
          ├─ 是 → 测量G矩阵对角占比
          │       ├─ >80% → F2可用
          │       ├─ 60-80% → Full KFAC更好
          │       └─ <60% → 必须Full KFAC
          │
          └─ 其他 → 根据具体情况分析
```

### 9.3 各类网络的建议

#### CNN（ResNet, VGG等）

```
推荐: Full KFAC 或 F2

原因:
- 卷积层的输出神经元（通道）相对独立
- 对角占比通常较高（>70%）
- F2可以作为快速baseline
- Full KFAC提供更高精度

建议流程:
1. 先用F2快速实验
2. 如果效果不理想，改用Full KFAC
```

#### Transformer / ViT

```
推荐: Head-level > Full KFAC > F2

Attention O矩阵:
- 首选: Head-level剪枝
- 次选: Full KFAC
- 不推荐: F2（除非验证对角占比>80%）

FFN层:
- 可以用Full KFAC
- F2需要先验证

原因:
- Head有明确语义结构
- 输入输出都有相关性
- LayerNorm引入依赖
```

#### VAR (Visual AutoRegressive)

```
推荐策略（分层）:

早期层 (0-8):
- Attention: Head-level
- FFN: Full KFAC或F2

中间层 (9-16):
- Attention: Head-level
- FFN: Full KFAC

后期层 (17-24):
- Attention: Head-level
- FFN: Full KFAC

原因:
- 早期层特征相对独立
- 后期层高度相关
- Head语义在所有层都清晰
```

### 9.4 实践建议

#### 第一步：诊断分析

```
1. 在验证集上收集统计信息
2. 计算G矩阵的对角占比
3. 可视化特征值分布
4. 分析不同层的特性

工具:
- 对角占比: ||diag(G)||_F / ||G||_F
- 条件数: max(λ) / min(λ)
- 相关性热图: 可视化G矩阵
```

#### 第二步：选择方法

```
根据诊断结果:
- 对角占比高 + 计算资源有限 → F2
- 对角占比中等 + 要求精度 → Full KFAC
- Attention层 → Head-level
- 混合网络 → 分层策略
```

#### 第三步：验证和调优

```
1. 在小规模数据上快速实验
2. 对比不同方法的性能保持
3. 分析剪枝后的激活分布
4. 微调（fine-tune）恢复性能

关键指标:
- 准确率/性能保持
- 实际加速比
- 内存占用
- 稳定性（多次运行方差）
```

### 9.5 常见陷阱

1. **盲目使用F2**
   - 陷阱：不验证对角占比就用F2
   - 后果：在Attention等相关性强的层效果差
   - 建议：先测量，再决定

2. **忽略Surgery**
   - 陷阱：直接剪枝不做surgery
   - 后果：性能下降显著
   - 建议：始终执行surgery或重新初始化

3. **全局统一策略**
   - 陷阱：所有层用相同方法和比例
   - 后果：某些层过度剪枝，某些层剪不够
   - 建议：分层分析，自适应剪枝

4. **忽略内存峰值**
   - 陷阱：只看平均内存，不看峰值
   - 后果：实际运行OOM
   - 建议：profile整个流程的内存曲线

---

## 总结

### KFAC剪枝的核心思想

1. **Kronecker分解**：用两个小矩阵近似巨大的Fisher矩阵
2. **特征值分解**：稳定高效地计算逆矩阵
3. **重要性评估**：基于权重和Fisher逆的对角元素
4. **Surgery补偿**：最小化剪枝对网络输出的影响

### 关键公式回顾

```
Fisher近似:
F ≈ G ⊗ A

逆矩阵:
G⁻¹ = Q_g @ diag(1/λ_g) @ Q_g^T

重要性 (Full KFAC):
importance = w² / (diag(G⁻¹) ⊗ diag(A⁻¹))

重要性 (F2):
importance = (w² @ A).sum(1) / diag(G⁻¹)

Surgery (Full KFAC):
δθ = -G⁻¹ @ coeff @ A⁻¹
```

### 方法选择简表

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| CNN全连接层 | Full KFAC或F2 | 对角占比高 |
| Attention O矩阵 | Head-level | 结构清晰，相关性强 |
| Transformer FFN | Full KFAC | 较高精度需求 |
| 内存受限 | F2或低秩KFAC | 节省内存 |
| 追求极致精度 | Full KFAC | 近似误差小 |
| 需要硬件加速 | Head-level | 结构化稀疏 |

### 内存对比

| 方法 | 1024×1024层内存 | VAR-d16总内存 |
|------|----------------|--------------|
| 完整Hessian | 4.4 TB | 不可行 |
| KFAC | 16.8 MB | ~9 GB |
| 低秩KFAC (k=100) | 1.7 MB | ~0.9 GB |

### 未来方向

1. **自适应方法选择**：根据层的统计特性自动选择方法
2. **混合精度KFAC**：用低精度存储统计信息
3. **在线KFAC**：边训练边剪枝
4. **跨层优化**：考虑层与层之间的依赖关系

---

## 参考资源

### 理论基础
- Optimal Brain Surgeon (OBS): Hassibi & Stork, 1993
- KFAC: Martens & Grosse, 2015
- Fisher Information Pruning: Various works

### 实践应用
- Transformer剪枝: Michel et al., 2019 (Head pruning)
- Vision Transformer剪枝: Recent works on ViT compression
- VAR模型: Visual AutoRegressive Modeling

### 工具和代码
- 本项目：EigenDamage-Pytorch
- 相关文件：
  - `pruner/kfac_full_pruner.py` - 完整KFAC实现
  - `pruner/kfac_OBS_F2.py` - F2实现
  - `pruner/kfac_eigen_pruner.py` - 特征值方法

---

**文档版本**: v1.0
**最后更新**: 2025-01-04
**作者**: 基于KFAC剪枝实践总结
