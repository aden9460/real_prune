# KFAC剪枝方法补充说明

> 对KFAC_PRUNING_COMPREHENSIVE_GUIDE.md的重要补充

---

## 补充1：F2方法的适用性澄清

### F2在重要性计算vs Surgery中的差异

这是一个**非常重要的澄清**！F2方法在两个阶段的适用性是不同的。

#### 阶段1：重要性计算 - F2理论上OK ✓

**OBS重要性公式**（只需对角元素）：
```
importance_i = w_i² / [H^(-1)]_ii

只需要Hessian逆矩阵的对角元素！
```

**Full KFAC重要性计算**：
```python
# kfac_full_pruner.py:148-152
A_inv = Q_a @ diag(1 / d_a) @ Q_a.T
G_inv = Q_g @ diag(1 / d_g) @ Q_g.T
A_inv_diag = diag(A_inv)  # 提取对角
G_inv_diag = diag(G_inv)  # 提取对角

w_imp = w² / (G_inv_diag[:, None] @ A_inv_diag[None, :])
importance = w_imp.sum(1)
```

**F2重要性计算**：
```python
# kfac_OBS_F2.py:24-26
G_inv = Q_g @ diag(1 / d_g) @ Q_g.T
w_imps = sum(w² @ A, dim=1)  # 用完整的A
importance = w_imps / diag(G_inv)  # 只用G_inv的对角
```

**对比**：
- Full KFAC：用 `diag(A_inv)` 和 `diag(G_inv)`
- F2：用完整的 `A` 和 `diag(G_inv)`

**结论**：在重要性计算阶段，F2的假设（G是对角）是合理的，因为只需要对角元素。

---

#### 阶段2：Surgery补偿 - F2有问题 ❌

**Full KFAC Surgery**：
```python
# kfac_full_pruner.py:180-182
coeff = w / (G_inv_diag[:, None] @ A_inv_diag[None, :])
coeff[keep_indices, :] = 0  # 保留的神经元清零

delta_theta = -G_inv @ coeff @ A_inv
               ^^^^^^         ^^^^^^
               完整矩阵！     完整矩阵！
```

**F2 Surgery**：
```python
# kfac_OBS_F2.py:41-45
G_inv = Q_g @ diag(1 / d_g) @ Q_g.T  # 完整的G_inv
G_inv[:, keep_indices] = 0  # 保留的列清零

coeff = G_inv @ diag(1 / G_inv_diag)
delta_theta = -coeff @ w
                       ^^
                   没有A_inv！
```

**关键差异**：
| 操作 | Full KFAC | F2 | 差异 |
|------|-----------|-----|------|
| 输入侧补偿 | ✓ 用 A_inv | ❌ 忽略 | F2假设输入无相关性 |
| 输出侧补偿 | ✓ 用完整 G_inv | ✓ 用完整 G_inv | 相同 |
| 最终公式 | `G_inv @ coeff @ A_inv` | `coeff @ w` | F2少了A_inv |

---

### 为什么Surgery需要完整矩阵？

**Surgery的目标**：调整保留参数来补偿被删除参数的影响

**需要的信息**：
1. 被删除参数之间的相关性（通过H^(-1)的非对角元素）
2. 被删除参数如何影响保留参数（需要完整矩阵乘法）

**数学推导**：
```
完整OBS公式：
δw = -H^(-1) @ [处理后的系数]

KFAC近似：
H^(-1) = G^(-1) ⊗ A^(-1)

转换为矩阵形式：
δW = G^(-1) @ [某矩阵] @ A^(-1)
```

**物理意义**：
- `A^(-1)`（右乘）：在输入维度上分配补偿
- `G^(-1)`（左乘）：在输出维度上分配补偿
- F2只考虑输出侧，完全忽略输入侧的协同效应

---

### 两个阶段的对比表

| 特性 | 重要性计算 | Surgery补偿 |
|------|-----------|-------------|
| **需要什么** | H^(-1)的对角元素 | H^(-1)的完整结构（非对角） |
| **Full KFAC** | 用 `diag(G_inv) ⊗ diag(A_inv)` | 用 `G_inv @ ... @ A_inv` |
| **F2假设** | ✓ 合理（只需对角） | ❌ 不合理（需要完整矩阵） |
| **F2适用性** | ✓ 理论上OK | ⚠️ 忽略输入相关性 |

---

### 结论

**F2方法的问题不在重要性计算，而在Surgery**：

1. **重要性计算**：
   - 只需要对角元素
   - F2用 `diag(G_inv)` 是合理的
   - 虽然分子用了完整的A，但最终只需对角信息

2. **Surgery补偿**：
   - 需要完整的矩阵结构
   - F2完全忽略 `A_inv`
   - 假设输入维度之间无相关性
   - **这才是F2的主要限制**

3. **对Attention O矩阵**：
   - 输入（heads拼接）有相关性
   - Surgery时忽略A_inv会导致补偿不准确
   - 建议用Full KFAC或Head-level剪枝

---

## 补充2：Surgery公式的矩阵维度详解

### 矩阵乘法 vs 内积的区别

很多人误解 `[1024,1024] @ [1024,1024] @ [1024,1024]` 会变成标量，这是因为混淆了**矩阵乘法**和**内积**。

#### 内积（Dot Product）

```
a · b，其中 a: [n], b: [n]
结果：标量（单个数字）

例如：
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1×4 + 2×5 + 3×6 = 32  ← 标量

维度：[n] · [n] → 标量
```

#### 矩阵乘法（Matrix Multiplication）

```
A @ B，其中 A: [m, n]，B: [n, p]
结果：[m, p]（还是矩阵）

规则：中间维度必须相等，结果保留两侧维度

例如：
[3, 4] @ [4, 5] = [3, 5]
[1024, 1024] @ [1024, 1024] = [1024, 1024]
```

---

### Surgery公式的维度推导

```python
delta_theta = -G_inv @ coeff @ A_inv
```

**逐步推导**：

```
步骤1：G_inv @ coeff
  G_inv: [1024, 1024]
  coeff: [1024, 1024]

  矩阵乘法规则：[m,n] @ [n,p] = [m,p]
  [1024, 1024] @ [1024, 1024] = [1024, 1024]

  中间结果：[1024, 1024]

步骤2：(G_inv @ coeff) @ A_inv
  中间结果: [1024, 1024]
  A_inv: [1024, 1024]

  [1024, 1024] @ [1024, 1024] = [1024, 1024]

  最终结果：[1024, 1024] ✓
```

---

### 3×3矩阵的可视化例子

```
G_inv = [g₁₁ g₁₂ g₁₃]     coeff = [c₁₁ c₁₂ c₁₃]     A_inv = [a₁₁ a₁₂ a₁₃]
        [g₂₁ g₂₂ g₂₃]             [c₂₁ c₂₂ c₂₃]             [a₂₁ a₂₂ a₂₃]
        [g₃₁ g₃₂ g₃₃]             [c₃₁ c₃₂ c₃₃]             [a₃₁ a₃₂ a₃₃]

步骤1：G_inv @ coeff = temp
  temp[0,0] = g₁₁×c₁₁ + g₁₂×c₂₁ + g₁₃×c₃₁  ← 第0行和第0列的点积
  temp[0,1] = g₁₁×c₁₂ + g₁₂×c₂₂ + g₁₃×c₃₂
  temp[0,2] = g₁₁×c₁₃ + g₁₂×c₂₃ + g₁₃×c₃₃
  ...（共9个元素）

  结果：[3, 3]  ← 维度不变！

步骤2：temp @ A_inv = delta
  delta[0,0] = temp[0,0]×a₁₁ + temp[0,1]×a₂₁ + temp[0,2]×a₃₁
  delta[0,1] = temp[0,0]×a₁₂ + temp[0,1]×a₂₂ + temp[0,2]×a₃₂
  ...

  结果：[3, 3]  ← 维度仍然不变！
```

---

### 为什么需要保持维度？

**delta_theta的物理意义**：

```
delta_theta[i,j] = 对权重w[i,j]的补偿量

- 每个权重都需要一个补偿值
- 矩阵形状必须和原权重W相同
- [1024, 1024] → [1024, 1024] ✓
```

**如果变成标量**：
```
如果结果是一个标量（假设内积）：
  → 所有权重得到相同的补偿 ❌
  → 无法体现不同权重的不同重要性
  → Surgery完全失效
```

---

### 数值示例验证

```python
import torch

# 创建3×3矩阵
G_inv = torch.randn(3, 3)
coeff = torch.randn(3, 3)
A_inv = torch.randn(3, 3)

# 矩阵乘法
temp = G_inv @ coeff
print(f"G_inv @ coeff: {temp.shape}")  # torch.Size([3, 3])

delta = temp @ A_inv
print(f"(G_inv @ coeff) @ A_inv: {delta.shape}")  # torch.Size([3, 3])

# 验证：一次性计算
delta_direct = G_inv @ coeff @ A_inv
print(f"Direct: {delta_direct.shape}")  # torch.Size([3, 3])

print(f"相等: {torch.allclose(delta, delta_direct)}")  # True
```

---

## 补充3：官方代码对照

### 重要性计算代码对照

#### Full KFAC实现

```python
# 文件：kfac_full_pruner.py
# 位置：第140-165行

def _get_unit_importance(self, normalize):
    eps = 1e-10
    assert self._inversed, 'Not inversed.'
    with torch.no_grad():
        for m in self.modules:
            w = fetch_mat_weights(m, False)  # [output_dim, input_dim]

            # 分支1：不使用低秩近似
            if self.S_l is None:
                # 🔑 计算完整的A_inv和G_inv
                A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()

                # 🔑 提取对角元素
                A_inv_diag = torch.diag(A_inv)  # [input_dim]
                G_inv_diag = torch.diag(G_inv)  # [output_dim]

                # 🔑 计算每个权重的重要性
                # 外积：[output_dim, 1] @ [1, input_dim] = [output_dim, input_dim]
                w_imp = w ** 2 / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))

            # 分支2：使用低秩近似
            else:
                Q_a, Q_g = self.Q_a[m], self.Q_g[m]
                S_l = self.S_l[m]
                S_l_inv = 1.0 / (S_l + eps)
                # 直接计算H_inv的对角
                H_inv_diag = (Q_g ** 2) @ S_l_inv @ (Q_a.t() ** 2)
                w_imp = w ** 2 / H_inv_diag

            self.W_pruned[m] = w
            # 🔑 聚合到神经元级别
            out_neuron_imp = w_imp.sum(1)  # [output_dim]

            if not normalize:
                out_imps = out_neuron_imp
            else:
                out_imps = out_neuron_imp / out_neuron_imp.sum()

            self.importances[m] = (tensor_to_list(out_imps), out_neuron_imp.size(0))
```

#### F2实现

```python
# 文件：kfac_OBS_F2.py
# 位置：第19-32行

def _get_unit_importance(self, normalize):
    eps = 1e-10
    with torch.no_grad():
        for m in self.modules:
            w = fetch_mat_weights(m, False)  # [output_dim, input_dim]

            # 🔑 计算G_inv（完整矩阵）
            G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()

            # 🔑 用完整的A矩阵（self.m_aa）
            w_imps = torch.sum(w**2 @ self.m_aa[m], 1)  # [output_dim]

            # 🔑 只用G_inv的对角
            out_neuron_imp = w_imps / torch.diag(G_inv)

            self.W_pruned[m] = w
            if not normalize:
                out_imps = out_neuron_imp
            else:
                out_imps = out_neuron_imp / out_neuron_imp.sum()

            self.importances[m] = (tensor_to_list(out_imps), out_neuron_imp.size(0))
```

---

### Surgery代码对照

#### Full KFAC Surgery

```python
# 文件：kfac_full_pruner.py
# 位置：第167-196行

def _do_surgery(self):
    eps = 1e-10
    assert not self.use_patch, 'Will never use patch'
    with torch.no_grad():
        for idx, m in enumerate(self.modules):
            w = fetch_mat_weights(m, False)  # [output_dim, input_dim]

            if w.size(0) == len(m.out_indices):
                continue  # 没有剪枝，跳过

            # 分支1：完整矩阵方法
            if self.S_l is None:
                # 计算A_inv和G_inv
                A_inv = self.Q_a[m] @ (torch.diag(1.0 / (self.d_a[m] + eps))) @ self.Q_a[m].t()
                G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()

                A_inv_diag = torch.diag(A_inv)
                G_inv_diag = torch.diag(G_inv)

                # 🔑 计算coeff矩阵
                coeff = w / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))

                # 🔑 保留的神经元清零（不需要补偿）
                coeff[m.out_indices, :] = 0

                # 🔑 Surgery公式：同时用G_inv和A_inv
                delta_theta = -G_inv @ coeff @ A_inv

            # 分支2：低秩方法
            else:
                Q_a, Q_g = self.Q_a[m], self.Q_g[m]
                S_l = self.S_l[m]
                S_l_inv = 1.0 / (S_l + eps)

                H_inv_diag = (Q_g ** 2) @ S_l_inv @ (Q_a.t() ** 2)
                coeff = w / H_inv_diag
                coeff[m.out_indices, :] = 0

                # 在特征空间中计算
                delta_theta = (Q_g.t() @ coeff @ Q_a) / S_l_inv
                delta_theta = Q_g @ delta_theta @ Q_a.t()

            # 🔑 应用补偿
            dw, dbias = mat_to_weight_and_bias(delta_theta, m)
            m.weight += dw
            if m.bias is not None:
                m.bias += dbias
```

#### F2 Surgery

```python
# 文件：kfac_OBS_F2.py
# 位置：第34-51行

def _do_surgery(self):
    eps = 1e-10
    with torch.no_grad():
        for idx, m in enumerate(self.modules):
            w = fetch_mat_weights(m, False)

            if w.size(0) == len(m.out_indices):
                continue

            # 计算G_inv（完整矩阵）
            G_inv = self.Q_g[m] @ (torch.diag(1.0 / (self.d_g[m] + eps))) @ self.Q_g[m].t()

            G_inv_diag = torch.diag(G_inv)

            # 🔑 关键差异：将保留神经元的列清零
            G_inv[:, m.out_indices] = 0

            # 🔑 归一化
            coeff = G_inv @ torch.diag(1.0 / G_inv_diag)

            # 🔑 Surgery：只用G，不用A_inv
            delta_theta = -coeff @ w
            #                       ^^
            #                   直接作用于w，没有A_inv变换！

            # 应用补偿
            dw, dbias = mat_to_weight_and_bias(delta_theta, m)
            m.weight += dw
            if m.bias is not None:
                m.bias += dbias
```

---

## 参考文献补充

### 核心理论文献

1. **Optimal Brain Surgeon (OBS)**
   - Hassibi, B., & Stork, D. G. (1993). Second order derivatives for network pruning: Optimal brain surgeon. *Advances in Neural Information Processing Systems (NIPS)*.

2. **KFAC方法**
   - Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *International Conference on Machine Learning (ICML)*.
   - Grosse, R., & Martens, J. (2016). A Kronecker-factored approximate Fisher matrix for convolution layers. *ICML*.

3. **Fisher信息矩阵**
   - Fisher, R. A. (1925). Theory of statistical estimation. *Mathematical Proceedings of the Cambridge Philosophical Society*.
   - Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*.

4. **Kronecker积**
   - Van Loan, C. F. (2000). The ubiquitous Kronecker product. *Journal of Computational and Applied Mathematics*.

### 数值线性代数

5. **矩阵计算**
   - Golub, G. H., & Van Loan, C. F. (2013). *Matrix computations* (4th ed.). Johns Hopkins University Press.
   - Trefethen, L. N., & Bau III, D. (1997). *Numerical linear algebra*. SIAM.

### 剪枝方法

6. **Transformer Head Pruning**
   - Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one? *Advances in Neural Information Processing Systems (NeurIPS)*.

7. **神经网络剪枝综述**
   - Blalock, D., Ortiz, J. J. G., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? *MLSys*.

### 二阶优化

8. **自然梯度方法**
   - Martens, J. (2020). New insights and perspectives on the natural gradient method. *Journal of Machine Learning Research*.

---

**文档版本**: v1.1
**补充日期**: 2025-01-04
**用途**: 与KFAC_PRUNING_COMPREHENSIVE_GUIDE.md配合阅读
