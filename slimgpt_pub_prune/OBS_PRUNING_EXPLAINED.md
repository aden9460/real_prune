# OBS结构化剪枝算法详解

> **文档目的**: 详细解释slimgpt中OBS (Optimal Brain Surgeon) 结构化剪枝算法的实现原理
>
> **适用代码**: `/home/project/real_prune/slimgpt_pub_prune/slim_utils/slimgpt.py`
>
> **作者**: Claude (Anthropic)
>
> **日期**: 2025-11-03

---

## 目录

1. [概述](#概述)
2. [第一部分：基础概念](#第一部分基础概念)
3. [第二部分：剪枝准备](#第二部分剪枝准备)
4. [第三部分：核心算法](#第三部分核心算法)
5. [第四部分：误差计算与选择](#第四部分误差计算与选择)
6. [第五部分：权重更新与误差补偿](#第五部分权重更新与误差补偿)
7. [第六部分：Hessian矩阵的维度问题](#第六部分hessian矩阵的维度问题)
8. [第七部分：扩展应用 - Head维度剪枝](#第七部分扩展应用---head维度剪枝)
9. [附录：公式速查](#附录公式速查)

---

## 概述

### OBS算法简介

**Optimal Brain Surgeon (OBS)** 是一种基于二阶导数信息的神经网络剪枝算法：

- **核心思想**: 通过Hessian矩阵（损失函数的二阶导数）评估删除每个参数的影响
- **优势**: 比简单的幅值剪枝（magnitude pruning）更准确
- **代价**: 需要计算和存储Hessian矩阵

### 结构化剪枝

在Transformer/Attention架构中，结构化剪枝按**完整的attention head**删除：

- **非结构化**: 删除单个权重参数（破坏模型结构）
- **结构化**: 删除整组参数（保持模型结构完整）

```python
# 结构化剪枝示例
# 原始: 12个head × 64维 = 768维
# 剪枝: 删除3个head → 9个head × 64维 = 576维
```

---

## 第一部分：基础概念

### 1.1 权重矩阵的形状约定

#### PyTorch标准约定

对于一个线性层：**4个输入神经元 → 2个输出神经元**

```python
import torch.nn as nn

layer = nn.Linear(in_features=4, out_features=2)
print(layer.weight.shape)  # torch.Size([2, 4])
```

**关键规则**: `weight.shape = [out_features, in_features]`

#### 前向传播计算

```python
# 批量输入
x = torch.randn(32, 4)  # [batch_size, in_features]

# 实际计算
y = x @ layer.weight.T + layer.bias
#   [32,4] @ [4,2] + [2] → [32,2]
```

#### 权重矩阵的语义

```
W.shape = [2, 4] = [输出, 输入]

         Input (4个神经元)
         IN0  IN1  IN2  IN3
OUT0  [  w00  w01  w02  w03 ]  ← 输出神经元0的权重向量
OUT1  [  w10  w11  w12  w13 ]  ← 输出神经元1的权重向量
  ↑
输出 (2个神经元)
```

- `W[i, j]`: 从输入j到输出i的连接权重
- `W[i, :]`: 第i个输出神经元的所有输入权重

#### slimgpt中的应用

```python
W.shape = [768, 192]
# 192个输入特征 → 768个输出特征
# self.rows = 768 (输出维度)
# self.columns = 192 (输入维度)

# 剪枝"列" → 减少输入维度
# 剪枝"行" → 减少输出维度（通常不做）
```

---

### 1.2 索引映射：二维 ↔ 一维

#### 问题背景

Hessian矩阵 `H_full` 需要对**所有参数**计算二阶导数：

```python
W.shape = [rows, columns]
total_params = rows × columns

H_full.shape = [total_params, total_params]
```

需要将二维索引 `(i, j)` 映射到一维索引。

#### 映射公式（Row-major顺序）

```python
flat_index = i * columns + j
```

**含义**：
- `i * columns`: 跳过前面i行（每行有columns个元素）
- `+ j`: 在当前行内偏移j个位置

#### 具体示例

假设 `W.shape = [3, 4]`（3行4列）：

```python
# 展平顺序：
W = [[W[0,0], W[0,1], W[0,2], W[0,3]],   # 第0行
     [W[1,0], W[1,1], W[1,2], W[1,3]],   # 第1行
     [W[2,0], W[2,1], W[2,2], W[2,3]]]   # 第2行

W_flat = [W[0,0], W[0,1], W[0,2], W[0,3],  # 索引0-3
          W[1,0], W[1,1], W[1,2], W[1,3],  # 索引4-7
          W[2,0], W[2,1], W[2,2], W[2,3]]  # 索引8-11
```

**映射表**：

| 二维 W[i,j] | i | j | 计算 | 一维索引 |
|-------------|---|---|------|---------|
| W[0,0] | 0 | 0 | 0×4 + 0 | 0 |
| W[0,1] | 0 | 1 | 0×4 + 1 | 1 |
| W[0,2] | 0 | 2 | 0×4 + 2 | 2 |
| W[0,3] | 0 | 3 | 0×4 + 3 | 3 |
| W[1,0] | 1 | 0 | 1×4 + 0 | **4** |
| W[1,1] | 1 | 1 | 1×4 + 1 | 5 |
| W[2,3] | 2 | 3 | 2×4 + 3 | **11** |

#### Hessian索引

```python
H_full[i*columns + j, k*columns + l] = ∂²Loss / (∂W[i,j] ∂W[k,l])
```

**示例**：计算 W[1,2] 和 W[2,3] 的二阶导数

```python
i=1, j=2 → flat_index_1 = 1×4 + 2 = 6
k=2, l=3 → flat_index_2 = 2×4 + 3 = 11

H_full[6, 11] = ∂²Loss / (∂W[1,2] ∂W[2,3])
```

---

### 1.3 死节点处理

代码位置：`slimgpt.py:179-181`

```python
dead = torch.diag(H) == 0
H[dead, dead] = 1
W[:, dead] = 0
```

#### 第一行：检测死节点

```python
dead = torch.diag(H) == 0
```

- 提取Hessian矩阵的对角线 `H[i,i]`
- 对角线为0意味着该维度对损失函数没有贡献
- `dead` 是布尔张量，标记"死亡"的维度

**物理意义**：
- `H[i,i]` 是损失函数对第i个参数的二阶导数
- `H[i,i] = 0` → 该参数已经失效或可以安全删除

#### 第二行：避免除零错误

```python
H[dead, dead] = 1
```

- 将死节点对应的对角线元素设为1
- **原因**: OBS算法需要计算 `H^(-1)`（Hessian的逆）
- 如果对角线有0，矩阵不可逆 → 数值错误
- 设为1是数值稳定性技巧，使矩阵可逆

#### 第三行：清零死节点权重

```python
W[:, dead] = 0
```

- 将权重矩阵中死节点对应的**列**全部置零
- 既然这些维度已失效，直接清零权重
- 确保后续计算中不产生影响

#### 完整流程示例

```python
# 假设 H.shape = [5, 5], W.shape = [10, 5]
H = [[2.0, ..., ...],
     [..., 0.0, ...],   # 第1维对角线为0
     [..., ..., 3.0],
     [..., ..., ...],
     [..., ..., 0.0]]   # 第4维对角线为0

dead = [False, True, False, False, True]

# 修正H（避免不可逆）
H[dead, dead] = 1
H = [[2.0, ..., ...],
     [..., 1.0, ...],   # 修正为1
     [..., ..., 3.0],
     [..., ..., ...],
     [..., ..., 1.0]]   # 修正为1

# 清零死节点的权重
W[:, [1, 4]] = 0  # 第1列和第4列全部清零
```

---

## 第二部分：剪枝准备

### 2.1 初始化剪枝掩码

代码位置：`slimgpt.py:188-190`

```python
column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
pruned_columns = column_mask.count_nonzero()
target_columns = round(self.columns // headsize * sparsity) * headsize
```

#### 第一行：创建掩码

```python
column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
```

- 形状: `[columns]`
- 类型: 布尔张量
- 初始值: 全为 `False`（0）
- **约定**: `True` (1) 表示该列需要删除，`False` (0) 表示保留

#### 第二行：统计已剪枝数

```python
pruned_columns = column_mask.count_nonzero()
```

- 统计 `column_mask` 中有多少个 `True`
- 初始时 = 0（还没开始剪枝）
- 用于迭代循环中追踪进度

#### 第三行：计算目标剪枝数

```python
target_columns = round(self.columns // headsize * sparsity) * headsize
```

**公式分解**：

```python
num_heads = self.columns // headsize       # 有多少个完整head
heads_to_prune = num_heads * sparsity      # 需要剪掉多少个head
heads_to_prune = round(heads_to_prune)     # 四舍五入到整数
target_columns = heads_to_prune * headsize # 转换回列数
```

**关键约束**: `target_columns` **必须是 `headsize` 的整数倍**

---

### 2.2 结构化剪枝的必要性

#### 为什么必须是headsize的倍数？

**Attention Head的结构**：

```python
# 假设总维度768，12个head，每个head 64维
Q = input @ W_Q  # W_Q.shape = [768, 768]

# reshape成多头
Q = Q.view(batch, seq_len, 12, 64)  # [B, L, num_heads, head_dim]

# 如果剪掉50列（不是64的倍数）
# 768 - 50 = 718
# 718 / 64 = 11.21875  ❌ 无法整除！
# Q.view(batch, seq_len, ?, 64) 会报错
```

**结构化剪枝**：

```python
# 剪掉3个完整head
768 - 3×64 = 576
576 / 64 = 9  ✓ 可以整除
Q.view(batch, seq_len, 9, 64)  ✓ 合法
```

#### 数值示例

```python
# 参数设置
self.columns = 768
headsize = 64
sparsity = 0.4  # 40%稀疏度

# 计算过程
num_heads = 768 // 64 = 12
heads_to_prune = 12 * 0.4 = 4.8
heads_to_prune = round(4.8) = 5
target_columns = 5 * 64 = 320

# 结果：需要删除320列（5个head）
```

---

## 第三部分：核心算法

### 3.1 迭代剪枝循环

代码位置：`slimgpt.py:198-205`

```python
while pruned_columns < target_columns:
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    if headsize > 1:
        Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize]
                                 for i in range(0, self.columns, headsize)])
        Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag),
                                   dim1=-2, dim2=-1).reshape(-1)
        Hinv_diag = Hinv_diag ** 2
    else:
        Hinv_diag = Hinv.diag()
```

#### 循环结构

```python
while pruned_columns < target_columns:
```

- **迭代式剪枝**: 每次迭代删除一个head（或一批列）
- **贪心策略**: 每次选择影响最小的head
- **动态更新**: 删除后重新计算Hessian和误差

---

### 3.2 第一次Cholesky分解：计算H^(-1)

```python
Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
```

#### 数学原理

给定正定矩阵 `H`，Cholesky分解：

```
H = L L^T
```

其中 `L` 是下三角矩阵。

计算逆矩阵：

```
H^(-1) = (L L^T)^(-1) = (L^T)^(-1) L^(-1) = (L^(-1))^T L^(-1)
```

#### 步骤分解

```python
# 步骤1：Cholesky分解
L = torch.linalg.cholesky(H)  # H = L L^T

# 步骤2：计算逆矩阵
Hinv = torch.cholesky_inverse(L)  # Hinv = (L^(-1))^T @ L^(-1)
```

#### 为什么不直接用 torch.inverse(H)？

**数值稳定性对比**：

| 方法 | 复杂度 | 数值稳定性 | 精度 |
|------|--------|-----------|------|
| `torch.inverse(H)` | O(n³) | ⚠️ 差 | 低 |
| `cholesky_inverse` | O(n³/3) | ✅ 好 | **高** |

**原因**：
- Cholesky分解利用了**正定性**（所有特征值>0）
- 下三角矩阵的求逆更稳定
- 避免大数值和小数值混合导致的精度损失

#### 数值示例

```python
# 假设 H 是 2×2 正定矩阵
H = [[4, 2],
     [2, 3]]

# Cholesky分解
L = [[2,   0  ],
     [1, √2 ]]

# 验证: L @ L^T = H
# [[2,0], [1,√2]] @ [[2,1], [0,√2]] = [[4,2], [2,3]] ✓

# 计算 L^(-1)
L_inv = [[0.5,      0   ],
         [-1/(2√2), 1/√2]]

# 计算 H^(-1) = L_inv^T @ L_inv
Hinv = [[0.5, -1/(2√2)],      [[0.5,      0   ],
        [0,    1/√2    ]]  @   [-1/(2√2), 1/√2]]

     = [[0.375, -0.25],
        [-0.25,  0.5 ]]
```

#### 维度追踪

假设 `H.shape = [192, 192]`：

```python
H.shape = [192, 192]
  ↓ cholesky
L.shape = [192, 192]  # 下三角矩阵
  ↓ cholesky_inverse
Hinv.shape = [192, 192]  # H的逆矩阵
```

---

### 3.3 第二次Cholesky分解：提取结构化重要性

#### 为什么需要第二次分解？

**问题**: 如何从64×64的子矩阵中提取一个标量，代表整个head的重要性？

**挑战**: 不能简单地求平均或求和，因为：
- 需要考虑维度间的相关性
- 独立的3维 vs 冗余的3维应该有不同的分数

#### 直觉示例：独立 vs 冗余

**Head A（维度独立）**：

```python
Hinv_A = [[4, 0, 0],    # 3个维度完全独立
          [0, 4, 0],
          [0, 0, 4]]

# 方法1：直接对角线求和
sum(diag) = 4 + 4 + 4 = 12

# 方法2：Cholesky再分解
L = [[2, 0, 0],
     [0, 2, 0],
     [0, 0, 2]]
sum(L.diag()^2) = 4 + 4 + 4 = 12  ✓ 相同
```

**Head B（维度高度相关，冗余）**：

```python
Hinv_B = [[4.0, 3.9, 3.8],    # 3个维度几乎线性相关
          [3.9, 4.0, 3.9],    # 本质上是同一维度的重复
          [3.8, 3.9, 4.0]]

# 方法1：直接对角线求和
sum(diag) = 4 + 4 + 4 = 12  ❌ 误以为有3个独立维度

# 方法2：Cholesky再分解
L = [[2.0,   0,     0    ],
     [1.95,  0.31,  0    ],
     [1.90,  0.61,  0.15 ]]

sum(L.diag()^2) = 4.0 + 0.096 + 0.0225 ≈ 4.12  ✓ 发现只有~1个有效维度！
```

**关键发现**: Cholesky对角线能**识别冗余**！

#### 数学原理

对于正定矩阵 `A = LL^T`：

```
det(A) = ∏[L[i,i]²]           # 行列式 = 对角线平方的乘积
log(det(A)) = 2∑log(L[i,i])   # 取对数变成和
```

**`sum(L[i,i]²)` 近似于**：
- 矩阵的"有效秩" (effective rank)
- 矩阵的"体积" (volume)
- 信息论中的"熵" (entropy)

---

### 3.4 完整的第二次分解过程

#### 步骤1：提取各head的子矩阵

```python
Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize]
                         for i in range(0, self.columns, headsize)])
```

**维度变化**：

假设 `Hinv.shape = [192, 192]`, `headsize = 64`：

```python
# 切片操作
Head_0 = Hinv[0:64, 0:64]       # 左上角64×64
Head_1 = Hinv[64:128, 64:128]   # 中间64×64
Head_2 = Hinv[128:192, 128:192] # 右下角64×64

# stack堆叠
Hinv_diag.shape = [3, 64, 64]  # 3个head的子矩阵
```

**为什么提取对角块？**
- 每个head是相对独立的单元
- 对角块包含head内部64个维度的关系
- 用于评估**整个head作为整体**的重要性

#### 步骤2：对每个子矩阵做Cholesky分解

```python
L2 = torch.linalg.cholesky(Hinv_diag)
```

**维度**：

```python
Hinv_diag.shape = [3, 64, 64]
  ↓ cholesky (对每个子矩阵独立分解)
L2.shape = [3, 64, 64]  # 3个下三角矩阵
```

**每个L2矩阵的含义**：

```python
# Head 0的Hinv子矩阵分解
L2[0] = [[l00, 0,   0,   ..., 0  ],    # 下三角
         [l10, l11, 0,   ..., 0  ],
         [..., ..., ..., ..., ...],
         [l630,l631,..., ..., l6363]]

# 满足: L2[0] @ L2[0].T = Hinv[0:64, 0:64]
```

#### 步骤3：提取对角线并平方

```python
Hinv_diag = torch.diagonal(L2, dim1=-2, dim2=-1).reshape(-1)
Hinv_diag = Hinv_diag ** 2
```

**维度追踪**：

```python
L2.shape = [3, 64, 64]
  ↓ diagonal (提取对角线)
diag.shape = [3, 64]  # 每个L2矩阵的64个对角线元素
  ↓ reshape(-1) (展平)
Hinv_diag.shape = [192]  # 3×64 = 192
  ↓ 平方
Hinv_diag.shape = [192]  # 最终的重要性分数
```

**数值示例**：

```python
# Head 0的对角线
L2[0].diag() = [2.0, 1.9, 1.8, ..., 2.1]  # 64个值

# Head 1的对角线
L2[1].diag() = [1.5, 1.6, ..., 1.4]       # 64个值

# Head 2的对角线
L2[2].diag() = [3.2, 3.1, ..., 3.3]       # 64个值

# 拼接
Hinv_diag = [2.0, 1.9, ..., 2.1,    # Head 0
             1.5, 1.6, ..., 1.4,    # Head 1
             3.2, 3.1, ..., 3.3]    # Head 2

# 平方
Hinv_diag = [4.0, 3.61, ..., 4.41,  # Head 0
             2.25, 2.56, ..., 1.96, # Head 1
             10.24, 9.61, ..., 10.89] # Head 2
```

---

### 3.5 两次Cholesky分解对比

| 特性 | 第一次分解 | 第二次分解 |
|------|-----------|-----------|
| **输入** | 完整H矩阵 [192,192] | 各head子矩阵 [3,64,64] |
| **目的** | 计算H的逆矩阵 | 提取有效维度数 |
| **输出** | Hinv [192,192] | 重要性分数 [192] |
| **数学意义** | H^(-1) = (L^(-1))^T @ L^(-1) | sum(L.diag()²) ≈ 有效秩 |
| **用途** | OBS算法核心 | 识别冗余维度 |
| **能否省略** | ❌ 不可（算法必需） | ⚠️ 可以但不准确 |

---

### 3.6 完整流程示意图

```
输入: H [192, 192]
  │
  ├─→ 第一次 Cholesky: H = L₁ L₁ᵀ
  │     ↓
  │   计算 Hinv = (L₁⁻¹)ᵀ @ L₁⁻¹  [192, 192]
  │     │
  │     ├─→ 提取对角块: Hinv[0:64,0:64], Hinv[64:128,64:128], ...
  │     │                            [3, 64, 64]
  │     │
  │     ├─→ 第二次 Cholesky: 对每个子矩阵 Hinv_sub = L₂ L₂ᵀ
  │     │                                    [3, 64, 64]
  │     │
  │     ├─→ 提取对角线: L₂.diag()  [3, 64]
  │     │
  │     └─→ 展平并平方: [192]
  │
  └─→ 输出: Hinv_diag [192] - 每个维度的重要性分数
```

---

## 第四部分：误差计算与选择

### 4.1 OBS剪枝误差公式

代码位置：`slimgpt.py:207`

```python
error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
```

#### 理论公式

OBS算法的损失增量近似：

```
ΔLoss_j ≈ ∑ᵢ [ W[i,j]² / (2 * Hinv[j,j]) ]
```

**符号说明**：
- `W[i,j]`: 第i行、第j列的权重
- `Hinv[j,j]`: Hessian逆矩阵的第j个对角线元素
- `ΔLoss_j`: 删除第j列后的损失增加

**物理意义**：
- **分子** `W[i,j]²`: 权重的重要性（越大越重要）
- **分母** `Hinv[j,j]`: 删除后的补偿能力（越大越容易补偿）
- **比值**: 删除该列的"性价比"

#### 代码实现解析

```python
W.shape = [768, 192]          # 权重矩阵
Hinv_diag.shape = [192]       # Hessian逆对角线

# 步骤1: unsqueeze扩展维度
Hinv_diag.unsqueeze(0).shape = [1, 192]

# 步骤2: W² / Hinv_diag 广播计算
W ** 2 / Hinv_diag.unsqueeze(0)
→ [768, 192] / [1, 192]
→ [768, 192]  # 每个元素: W[i,j]² / Hinv_diag[j]

# 步骤3: sum(dim=0) 按列求和
torch.sum(..., dim=0) → [192]

# 最终结果
error[j] = ∑ᵢ [ W[i,j]² / Hinv_diag[j] ]
```

#### 数值示例

```python
# 假设只有2行3列
W = [[0.5, 0.2, 0.8],
     [0.3, 0.1, 0.6]]

Hinv_diag = [2.0, 4.0, 1.0]

# 计算第0列的误差
error[0] = (0.5² + 0.3²) / 2.0 = (0.25 + 0.09) / 2.0 = 0.17

# 计算第1列的误差
error[1] = (0.2² + 0.1²) / 4.0 = (0.04 + 0.01) / 4.0 = 0.0125  ← 最小，应优先删除

# 计算第2列的误差
error[2] = (0.8² + 0.6²) / 1.0 = (0.64 + 0.36) / 1.0 = 1.0

# 排序: [0.0125, 0.17, 1.0] → 应该先删第1列
```

**结论**: `error` 越小的列，删除后对模型影响越小，应该优先剪掉。

---

### 4.2 排除已剪枝的列

```python
error[column_mask] = torch.inf
```

- `column_mask`: 布尔数组，`True` 表示该列已被删除
- 设为**无穷大**，确保排序时不会再次选中
- 迭代剪枝的防重复机制

**示例**：

```python
error = [0.5, 2.0, 1.5, 0.8, 3.0]
column_mask = [False, False, True, False, False]  # 第2列已删

error[column_mask] = torch.inf
# 结果: [0.5, 2.0, inf, 0.8, 3.0]

# 后续排序时，第2列会被排到最后，不会再选中
```

---

### 4.3 结构化剪枝的排序策略

代码位置：`slimgpt.py:210-217`

```python
if headsize > 1:
    head_sort_idx = error.view(-1, headsize).sum(1).argsort()
    column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize)
                                     for x in head_sort_idx])
    cnt = headsize
else:
    column_sort_idx = error.argsort()
    cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)
```

#### 情况A：结构化剪枝（headsize > 1）

##### 第一步：按head计算总误差

```python
head_sort_idx = error.view(-1, headsize).sum(1).argsort()
```

假设 `error.shape = [192]`, `headsize = 64`：

```python
# 1. reshape成head视图
error.view(-1, 64) → [3, 64]
# [[e0, e1, ..., e63],      # Head 0的64个误差
#  [e64, e65, ..., e127],   # Head 1的64个误差
#  [e128, e129, ..., e191]] # Head 2的64个误差

# 2. sum(1) 按行求和
.sum(1) → [3]
# [head_0_total, head_1_total, head_2_total]
# 例如: [80.5, 45.2, 120.8]

# 3. argsort() 从小到大排序
.argsort() → [1, 0, 2]
# Head 1误差最小 → Head 0次之 → Head 2最大
```

**含义**: 应该按 `[Head 1, Head 0, Head 2]` 的顺序删除

##### 第二步：生成列索引优先级队列

```python
column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize)
                                 for x in head_sort_idx])
```

继续上面的例子：

```python
head_sort_idx = [1, 0, 2]
headsize = 64

# 列表推导式:
for x in [1, 0, 2]:
    # x=1: arange(64, 128)   → [64, 65, ..., 127]    Head 1的所有列
    # x=0: arange(0, 64)     → [0, 1, ..., 63]       Head 0的所有列
    # x=2: arange(128, 192)  → [128, 129, ..., 191]  Head 2的所有列

# hstack拼接
column_sort_idx = [64, 65, ..., 127,    # 优先删除Head 1
                   0, 1, ..., 63,       # 其次删除Head 0
                   128, 129, ..., 191]  # 最后删除Head 2
```

**意义**:
- 生成了一个"待删除列的优先级队列"
- 从左到右依次是应该删除的列
- **按整个head一起排列**，保证结构完整性

##### 第三步：每次删除一个head

```python
cnt = headsize  # 64
```

- 每次迭代删除**一个完整head**（64列）
- 删除后重新计算Hessian，再决定下一个

---

#### 情况B：非结构化剪枝（headsize = 1）

##### 第一步：直接按列排序

```python
column_sort_idx = error.argsort()
```

**示例**：

```python
error = [2.5, 0.8, 3.1, 0.5, 1.2]

argsort() → [3, 1, 4, 0, 2]
# 第3列误差最小(0.5) → 第1列(0.8) → 第4列(1.2) → ...
```

##### 第二步：动态确定批量删除数

```python
cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)
```

取**三者中的最小值**：

1. `target_columns - pruned_columns`: 还需删除多少列
2. `max(blocksize, 64)`: 至少删64列
3. `1024`: 最多删1024列（防止过大）

**示例**：

```python
target_columns = 500
pruned_columns = 300
blocksize = 32

cnt = min(500-300, max(32,64), 1024)
    = min(200, 64, 1024)
    = 64  # 本次删除64列
```

**为什么批量删除？**
- 逐列删除太慢（需迭代500次）
- 批量删除提高效率（只需 500/64 ≈ 8次）
- 但不能一次删太多，否则误差累积

---

### 4.4 对比总结

| 特性 | 结构化剪枝 (headsize>1) | 非结构化剪枝 (headsize=1) |
|------|------------------------|--------------------------|
| **排序单位** | 整个head（64列一组） | 单列 |
| **排序依据** | head总误差的和 | 单列误差 |
| **每次删除数** | 固定1个head（64列） | 动态64~1024列 |
| **迭代次数** | 多（精确） | 少（快速） |
| **精度** | 高（每次重算H） | 中（批量近似） |
| **模型兼容性** | ✅ 保持结构 | ❌ 破坏结构 |
| **适用场景** | Attention模型 | 全连接层 |

---

## 第五部分：权重更新与误差补偿

### 5.1 重排矩阵准备删除

代码位置：`slimgpt.py:219-225`

```python
W = W[:, column_sort_idx]
Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
Hinv = torch.linalg.cholesky(Hinv, upper=True)[:cnt]

W1 = W[:, :cnt].clone()
Hinv1 = Hinv[:, :cnt]
Err1 = torch.zeros_like(W1)
```

#### 第一行：重排权重矩阵的列

```python
W = W[:, column_sort_idx]
```

**作用**：将权重矩阵的列按**删除优先级重新排列**

- 最应该删除的列排在最前面
- 最重要的列排在最后面

**维度**：假设 `W.shape = [768, 192]`, `cnt = 64`

```python
# column_sort_idx 示例
column_sort_idx = [64, 65, ..., 127,    # Head 1（误差最小）
                   0, 1, ..., 63,       # Head 0
                   128, 129, ..., 191]  # Head 2（误差最大）

W = W[:, column_sort_idx]  # [768, 192]，列重排
# 前64列：最应删除
# 中64列：次之
# 后64列：最重要保留
```

#### 第二行：重排Hessian逆矩阵

```python
Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
```

**作用**：对Hessian逆矩阵进行**行和列的同步重排**，保持对称性

**步骤分解**：

```python
# 步骤1：重排行
temp = Hinv[column_sort_idx, :]  # [192, 192]

# 步骤2：重排列
Hinv = temp[:, column_sort_idx]  # [192, 192]

# 结果：Hinv保持对称，且与W的列顺序一致
```

#### 第三行：第三次Cholesky分解

```python
Hinv = torch.linalg.cholesky(Hinv, upper=True)[:cnt]
```

**这是第三次Cholesky分解**：

| 次数 | 输入 | 输出 | 目的 |
|------|------|------|------|
| 第1次 | H [192,192] | Hinv [192,192] | 计算H^(-1) |
| 第2次 | Hinv子矩阵 [3,64,64] | 重要性分数 [192] | 识别冗余 |
| **第3次** | **重排后Hinv [192,192]** | **上三角 [64,192]** | **准备权重更新** |

**参数说明**：

- `upper=True`：返回**上三角**矩阵（默认是下三角）
- `[:cnt]`：只取前cnt行（对应待删除的列）

**维度变化**：

```python
Hinv.shape = [192, 192]  # 重排后
  ↓ cholesky(upper=True)
U.shape = [192, 192]     # 上三角矩阵
  ↓ [:64]
Hinv.shape = [64, 192]   # 只保留前64行
```

#### 第四~六行：提取子矩阵

```python
W1 = W[:, :cnt].clone()      # [768, 64] 待删除列的副本
Hinv1 = Hinv[:, :cnt]        # [64, 64] 上三角子矩阵
Err1 = torch.zeros_like(W1)  # [768, 64] 误差矩阵（初始化为0）
```

---

### 5.2 Hessian逆矩阵的分块结构

**理解Local和Global Update的关键**：不同的更新使用Hessian的不同部分！

#### 重排后的矩阵布局

假设：总列数192，待删除前64列，保留后128列

```python
W: [768, 192]
   ├─────┬─────────┤
   待删除  保留
   64列   128列

Hinv_original: [192, 192]（未经第三次Cholesky）
   ┌─────────┬─────────────┐
   │ Hinv_dd │  Hinv_dk    │ 64行（待删除对应）
   ├─────────┼─────────────┤
   │ Hinv_kd │  Hinv_kk    │ 128行（保留对应）
   └─────────┴─────────────┘
     64列      128列
   待删除      保留
```

**分块含义**：
- `Hinv_dd` [64, 64]：**待删除列**之间的Hessian关系
- `Hinv_dk` [64, 128]：**待删除列**与**保留列**的关系
- `Hinv_kd` [128, 64]：对称，等于 `Hinv_dk.T`
- `Hinv_kk` [128, 128]：**保留列**之间的关系

#### 第三次Cholesky后的结构

```python
Hinv_original = U^T @ U  # 上三角分解

U: [192, 192] 上三角矩阵
   ┌─────────┬─────────────┐
   │  U_dd   │   U_dk      │ 64行
   ├─────────┼─────────────┤
   │    0    │   U_kk      │ 128行（上三角，左下为0）
   └─────────┴─────────────┘
     64列      128列

# 只取前64行
Hinv = U[:64, :] = [U_dd, U_dk]
Hinv.shape = [64, 192]
```

**分块说明**：
- `U_dd` [64, 64]：上三角，待删除块的Cholesky因子
- `U_dk` [64, 128]：待删除与保留之间的关系

#### 提取Hinv1

```python
Hinv: [64, 192]
   ┌──────┬────────┐
   │ Hinv1│        │
   │[64,64]│[64,128]│
   └──────┴────────┘
     :cnt   cnt:end
```

**`Hinv1 = U_dd`**：待删除块的上三角Cholesky因子

---

### 5.3 Local Update：局部权重更新

代码位置：`slimgpt.py:227-230`

```python
for i in range(cnt):
    Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
    if not self.no_compensate:
        W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])  # local update
```

#### 循环结构

逐列处理待删除的每一列（i = 0 到 63）

#### 计算误差补偿量

```python
Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
```

**OBS公式**：

```
Err[i] = W[i] / Hinv[i,i]
```

**维度分析**：

```python
W1[:, i:i+1].shape = [768, 1]   # 第i列权重
Hinv1[i, i] = scalar             # 对角线元素

Err1[:, i:i+1] = [768, 1] / scalar = [768, 1]
```

**切片 `i:i+1` 的作用**：保持二维形状，便于矩阵乘法

#### 局部权重更新

```python
W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])
```

**OBS局部更新公式**：

```
W[j] = W[j] - Err[i] × (Hinv[i,j] / Hinv[i,i])
```

对 j ≥ i 的所有列。

**维度追踪**（假设 i=0）：

```python
Err1[:, 0:1].shape = [768, 1]       # 第0列的误差
Hinv1[0:1, 0:].shape = [1, 64]      # Hinv第0行

# 矩阵乘法
Err1[:, 0:1] @ Hinv1[0:1, 0:]
= [768, 1] @ [1, 64]
= [768, 64]

# 更新W1的所有列
W1[:, 0:] -= [768, 64]
```

#### 使用的Hessian部分：`Hinv1` = `U_dd`

```python
Hinv1: [64, 64] (上三角矩阵)

   列0  列1  列2  ...  列63
 ┌───────────────────────┐
0│ *    *    *   ...  *  │ ← i=0时用这行
 ├───────────────────────┤
1│ 0    *    *   ...  *  │ ← i=1时用这行
 ├───────────────────────┤
2│ 0    0    *   ...  *  │ ← i=2时用这行
 │ ...                   │
 └───────────────────────┘

# 第i次迭代使用：Hinv1[i:i+1, i:]
# 即第i行从第i列到最后
```

**物理意义**：
- 只更新待删除块内部（W1）
- 删除第i列时，调整后续列（i到63）
- 级联更新：i → i+1 → ... → 63

---

### 5.4 Global Update：全局权重更新

代码位置：`slimgpt.py:232-235`

```python
W[:, :cnt] = 0
if not self.no_compensate:
    end = self.columns - pruned_columns
    W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # global update
```

#### 清零待删除列

```python
W[:, :cnt] = 0
```

将前cnt列（待删除）全部设为0。

#### 计算有效列数

```python
end = self.columns - pruned_columns
```

- `self.columns = 192`：原始总列数
- `pruned_columns`：之前已删除的列数
- `end`：当前有效的列数

#### 全局更新公式

```python
W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])
```

**OBS全局更新**：

```
W_kept = W_kept - Err_deleted @ Hinv_cross
```

**维度分析**（第一次迭代）：

```python
Err1.shape = [768, 64]              # 所有删除列的误差
Hinv[:, 64:192].shape = [64, 128]   # Hinv的前64行，第64-191列

# 矩阵乘法
Err1 @ Hinv[:, 64:192]
= [768, 64] @ [64, 128]
= [768, 128]  # 对保留的128列的补偿量

# 更新保留的列
W[:, 64:192] -= [768, 128]
```

#### 使用的Hessian部分：`Hinv[:, cnt:end]` = `U_dk`

```python
完整Hinv: [64, 192]
   ┌──────┬────────┐
   │ U_dd │  U_dk  │
   │[64,64]│[64,128]│
   └──────┴────────┘
           ↑
      这部分用于Global Update
```

**`Hinv[:, 64:192]`** [64, 128]：

```python
U_dk: [64, 128]

   列64 列65 ... 列191
 ┌─────────────────────┐
0│ *    *    ...  *    │
1│ *    *    ...  *    │
 │ ...                 │
63│*    *    ...  *    │
 └─────────────────────┘

# 全部64行，对应保留的128列
```

**物理意义**：
- 更新所有保留的列
- 跨块补偿：从删除块影响到保留块
- 一次性批量更新

---

### 5.5 Local vs Global Update对比

| 特性 | Local Update | Global Update |
|------|-------------|---------------|
| **更新范围** | `W1[:, i:]`（待删除块内部） | `W[:, cnt:end]`（保留块） |
| **更新对象** | 后续待删除列 | 所有保留列 |
| **Hessian使用** | `Hinv1[i:i+1, i:]`<br>`U_dd`（左上块） | `Hinv[:, cnt:end]`<br>`U_dk`（右半部分） |
| **执行方式** | 逐列循环（64次） | 一次性批量 |
| **目的** | 内部调整，准备删除 | 补偿保留列 |
| **数学含义** | 列间依赖传播 | 跨块补偿 |

#### 可视化示意图

```
Hessian逆矩阵（Cholesky分解后）:

Hinv [64, 192]:
   ┌────────────────┬──────────────────────┐
   │    Hinv1       │                      │
   │   [64, 64]     │   Hinv[:, 64:192]   │
   │                │     [64, 128]        │
   │   U_dd         │       U_dk           │
   │  (上三角)       │                      │
   │                │                      │
   │ Local Update   │   Global Update      │
   │    使用        │       使用           │
   │                │                      │
   │ 逐行迭代:      │   一次性批量:        │
   │ Hinv1[i, i:]   │   全部64行          │
   └────────────────┴──────────────────────┘
    待删除列相关        待删除→保留的关系
```

---

### 5.6 恢复原始列顺序

代码位置：`slimgpt.py:237-238`

```python
column_sort_idx_inv = torch.argsort(column_sort_idx)
W = W[:, column_sort_idx_inv]
```

#### 为什么需要恢复？

- 之前按误差排序，打乱了原始顺序
- 删除和更新完成后，需要恢复
- 保持与Hessian矩阵的一致性

#### argsort的逆操作

```python
column_sort_idx_inv = torch.argsort(column_sort_idx)
```

**数学原理**：

```python
# 原始数组
error = [2.5, 0.8, 3.1, 0.5, 1.2]
#       idx 0   1    2    3    4

# 排序索引
column_sort_idx = [3, 1, 4, 0, 2]
# 第3个最小 → 第1个 → 第4个 → 第0个 → 第2个

# 逆索引：告诉你"原始第i列现在在哪"
column_sort_idx_inv = torch.argsort([3, 1, 4, 0, 2])
                    = [3, 1, 4, 0, 2]

# 验证
sorted_array = array[column_sort_idx]
original = sorted_array[column_sort_idx_inv]  # 恢复原始顺序
```

#### 完整恢复过程

```python
# 原始顺序
W_original = [列0, 列1, 列2, 列3, 列4]

# 重排（按误差）
column_sort_idx = [3, 1, 4, 0, 2]
W_sorted = [列3, 列1, 列4, 列0, 列2]

# 删除前2列并清零
W_sorted[:, :2] = 0
W_sorted = [0, 0, 列4, 列0, 列2]

# 恢复原始顺序
W_restored = W_sorted[:, column_sort_idx_inv]
           = [列0, 0, 列2, 0, 列4]
           # 恢复到原始位置，被删除的列变成0
```

---

### 5.7 更新Hessian矩阵

代码位置：`slimgpt.py:240-243`

```python
pruned_idx = column_sort_idx[:cnt]
H[pruned_idx, :] = H[:, pruned_idx] = 0
H[pruned_idx, pruned_idx] = 1
column_mask[pruned_idx] = 1
pruned_columns += cnt
```

#### 获取被删除列的原始索引

```python
pruned_idx = column_sort_idx[:cnt]
```

**示例**：

```python
column_sort_idx = [64, 65, ..., 127,  # Head 1
                   0, 1, ..., 63,     # Head 0
                   128, ..., 191]     # Head 2

pruned_idx = [64, 65, ..., 127]  # Head 1在原始矩阵中的位置
```

#### 清零Hessian的行和列

```python
H[pruned_idx, :] = H[:, pruned_idx] = 0
```

**示意图**（假设 pruned_idx = [1, 3]）：

```python
原始H:
    0   1   2   3   4
0 [[10, 2,  3,  1,  4],
1  [2, 20,  5,  3,  6],
2  [3,  5, 30,  2,  7],
3  [1,  3,  2, 40,  8],
4  [4,  6,  7,  8, 50]]

清零后:
    0   1   2   3   4
0 [[10, 0,  3,  0,  4],
1  [0,  0,  0,  0,  0],   ← 第1行全零
2  [3,  0, 30,  0,  7],
3  [0,  0,  0,  0,  0],   ← 第3行全零
4  [4,  0,  7,  0, 50]]
     ↑       ↑
   第1列   第3列全零
```

#### 对角线设为1

```python
H[pruned_idx, pruned_idx] = 1
```

**为什么？**
- 避免矩阵奇异（保持可逆）
- 数值稳定性
- 下次迭代可继续使用

**最终结果**：

```python
    0   1   2   3   4
0 [[10, 0,  3,  0,  4],
1  [0,  1,  0,  0,  0],   ← 对角线为1
2  [3,  0, 30,  0,  7],
3  [0,  0,  0,  1,  0],   ← 对角线为1
4  [4,  0,  7,  0, 50]]
```

#### 更新剪枝状态

```python
column_mask[pruned_idx] = 1  # 标记已删除
pruned_columns += 64         # 累计删除数
```

- `column_mask`：防止重复选中
- `pruned_columns`：追踪进度，用于循环终止

---

### 5.8 完整算法流程总结

```python
# 假设 cnt=64（删除1个head）

# === 准备阶段 ===
W = W[:, column_sort_idx]              # 重排权重列
Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]  # 重排Hessian
Hinv = cholesky(Hinv, upper=True)[:64] # 第三次分解，上三角
W1 = W[:, :64].clone()                 # 待删除列副本
Hinv1 = Hinv[:, :64]                   # U_dd [64,64]
Err1 = zeros([768, 64])                # 误差矩阵

# === Local Update（64次迭代）===
for i in range(64):
    Err1[:, i] = W1[:, i] / Hinv1[i,i]
    W1[:, i:] -= Err1[:, i] @ Hinv1[i, i:]  # 使用U_dd

# === Global Update ===
W[:, :64] = 0                          # 清零待删除列
W[:, 64:192] -= Err1 @ Hinv[:, 64:192] # 使用U_dk

# === 恢复与更新 ===
column_sort_idx_inv = argsort(column_sort_idx)
W = W[:, column_sort_idx_inv]          # 恢复原始列顺序
H[pruned_idx, :] = H[:, pruned_idx] = 0  # 清零H
H[pruned_idx, pruned_idx] = 1          # 对角线为1
column_mask[pruned_idx] = 1            # 标记已删除
pruned_columns += 64                   # 更新计数
```

---

### 5.9 关键理解点

#### 1. 为什么要三次Cholesky分解？

| 次数 | 输入 | 输出 | 目的 |
|------|------|------|------|
| 第1次 | H [192,192] | Hinv [192,192] | 计算H^(-1)（算法基础） |
| 第2次 | Hinv子矩阵 [3,64,64] | 重要性分数 [192] | 识别冗余（选择删除列） |
| 第3次 | 重排后Hinv [192,192] | 上三角 [64,192] | 准备权重更新（OBS补偿） |

#### 2. 为什么Local和Global用不同部分？

- **Local**：处理待删除块内部依赖，使用 `U_dd`
- **Global**：补偿保留列，使用 `U_dk`（跨块关系）

#### 3. 为什么恢复原始顺序？

- 重排是为了计算方便（连续删除）
- 恢复是为了与H对应（H没有重排）
- 保持数据一致性

#### 4. no_compensate的权衡

| 特性 | 补偿（False） | 不补偿（True） |
|------|-------------|--------------|
| **精度** | 高 | 低 |
| **速度** | 慢 | 快 |
| **适用** | 高稀疏度 | 低稀疏度 |
| **计算量** | O(cnt×rows×cols) | O(1) |

---

## 第六部分：Hessian矩阵的维度问题

### 6.1 理论上的完整Hessian

#### 定义

给定权重矩阵 `W.shape = [rows, columns]`：

```python
total_params = rows × columns

H_full[p, q] = ∂²Loss / (∂W_flat[p] ∂W_flat[q])

H_full.shape = [total_params, total_params]
```

#### 具体示例

假设 `W.shape = [768, 192]`：

```python
total_params = 768 × 192 = 147,456

H_full.shape = [147456, 147456]
```

**存储和计算代价**：

```python
# 内存占用
memory = 147456² × 4 bytes (float32)
       = 21,743,271,936 × 4
       = 86,973,087,744 bytes
       ≈ 87 GB  😱

# 求逆复杂度
O(n³) = O(147456³) ≈ 3.2 × 10¹⁵ 次浮点运算
# 即使1 TFLOPS的GPU也需要 3200秒 ≈ 53分钟（仅一次求逆）
```

**结论**: 完全不可行！

---

### 6.2 实际使用的近似Hessian

#### 代码中的H

```python
H.shape = [columns, columns] = [192, 192]
```

**存储代价**：

```python
memory = 192² × 4 bytes = 147,456 bytes ≈ 144 KB
# 相比87GB，缩小了 600,000 倍！
```

#### 物理含义

```python
H[i, j] = E[(∂Loss/∂W[:,i])ᵀ @ (∂Loss/∂W[:,j])]
```

**解释**：
- `W[:,i]` 是第i列的所有权重（长度rows）
- `∂Loss/∂W[:,i]` 是该列的梯度向量
- `H[i, j]` 是第i列和第j列梯度的**协方差**

#### Fisher信息矩阵近似

在实际实现中（推测）：

```python
H = torch.zeros(columns, columns)

for batch in dataloader:
    # X是该层的输入激活
    X = get_layer_input()  # shape: [batch_size, rows]

    # 累积Fisher信息矩阵
    H += X.T @ X  # [rows, batch] @ [batch, rows] → [rows, rows]
                   # 错误：应该是不同的计算方式

# 更可能的实现：
for batch in dataloader:
    grad = compute_gradient()  # [rows, columns]

    # 对每列计算外积
    for i in range(columns):
        for j in range(columns):
            H[i,j] += grad[:,i].T @ grad[:,j]

H /= len(dataloader)
```

---

### 6.3 简化假设

#### 假设1：行间独立性

```python
∂²Loss / (∂W[i,j] ∂W[k,l]) ≈ 0  当 i ≠ k
```

**含义**: 不同行（对应不同输出特征）的权重相互独立

**为什么合理？**
- 每行对应一个输出特征通道
- 神经网络训练中，输入特征归一化后统计独立性较强
- 剪枝目标是"哪些输入维度可删除"，行间耦合不是关键

#### 假设2：列间局部相关性

```python
H[i, j] ≈ strong  当 i,j 属于同一head
H[i, j] ≈ weak    当 i,j 属于不同head
```

**含义**: 同一head内的列强相关，不同head间弱相关

---

### 6.4 维度降维的合理性

#### 从完整H到简化H

**完整Hessian**（理论）：

```python
H_full[147456, 147456] =
    [[H_00,  H_01,  ..., H_0,767 ],
     [H_10,  H_11,  ..., H_1,767 ],
     [...,   ...,   ..., ...     ],
     [H_767,0, ...,  ..., H_767,767]]

# 每个 H_ij 都是 [192, 192] 的子矩阵
```

**简化假设**（实际）：

```python
# 假设 H_ij ≈ 0 当 i ≠ j（行间独立）
# 只保留对角块：
H_simplified = average([H_00, H_11, ..., H_767,767])
               # 平均所有对角块，得到 [192, 192]
```

**物理意义**：
- 计算每一行权重的局部Hessian
- 平均所有行的Hessian
- 得到"平均意义下的列间关系"

---

### 6.5 形象比喻

#### 完整H：公司所有员工的协作网络

```
H_full[所有员工×所有员工] =
    技术部A ↔ 技术部B：强
    技术部A ↔ 市场部C：中
    技术部A ↔ 财务部D：弱
    市场部C ↔ 财务部D：中
    ... 所有人对所有人
```

#### 简化H：只关注部门内协作

```
H_simplified[部门×部门] =
    技术部内部协作矩阵
    市场部内部协作矩阵
    财务部内部协作矩阵

假设：不同部门员工相对独立
目标：决定裁掉哪个部门
```

**对于"部门级裁员"决策，简化版已经足够！**

---

### 6.6 总结

| 特性 | 完整H | 简化H |
|------|-------|-------|
| **理论维度** | `[rows×columns, rows×columns]` | `[columns, columns]` |
| **示例大小** | `[147456, 147456]` | `[192, 192]` |
| **内存占用** | 87 GB | 144 KB |
| **计算时间** | 不可行 | 毫秒级 |
| **假设** | 无 | 行间独立 |
| **适用性** | 理论完美 | 实践有效 |
| **剪枝任务** | 过度设计 | 恰好够用 |

---

## 附录：公式速查

### A.1 核心公式

| 名称 | 公式 | 说明 |
|------|------|------|
| **二维→一维索引** | `flat_idx = i×cols + j` | W[i,j]的展平位置 |
| **Hessian定义** | `H[i,j] = ∂²L/(∂wᵢ∂wⱼ)` | 损失的二阶导数 |
| **Cholesky分解** | `H = LLᵀ` | L是下三角矩阵 |
| **Hessian逆** | `H⁻¹ = (L⁻¹)ᵀL⁻¹` | 通过Cholesky计算 |
| **OBS误差** | `error[j] = Σᵢ[W[i,j]²/Hᵢₙᵥ[j,j]]` | 删除第j列的损失 |
| **目标剪枝数** | `round(cols/hs × s) × hs` | 保证headsize倍数 |
| **有效秩近似** | `Σ(L.diag()²)` | 矩阵的有效维度数 |

### A.2 维度速查表

假设 `W.shape = [768, 192]`, `headsize = 64`, `num_heads = 3`

| 变量 | 形状 | 说明 |
|------|------|------|
| `W` | `[768, 192]` | 权重矩阵 |
| `H` | `[192, 192]` | Hessian矩阵 |
| `L` | `[192, 192]` | Cholesky因子（下三角） |
| `Hinv` | `[192, 192]` | Hessian逆矩阵 |
| `Hinv_sub` | `[3, 64, 64]` | 各head的子矩阵 |
| `L2` | `[3, 64, 64]` | 子矩阵的Cholesky因子 |
| `Hinv_diag` | `[192]` | 重要性分数向量 |
| `error` | `[192]` | OBS剪枝误差 |
| `column_mask` | `[192]` | 布尔掩码（已剪枝标记） |
| `head_sort_idx` | `[3]` | head排序索引 |
| `column_sort_idx` | `[192]` | 列排序索引 |

### A.3 PyTorch操作速查

```python
# Cholesky分解
L = torch.linalg.cholesky(H)  # H必须正定

# Cholesky逆
Hinv = torch.cholesky_inverse(L)

# 提取对角线
diag = torch.diagonal(A)  # 或 A.diag()

# 多维对角线
diag = torch.diagonal(A, dim1=-2, dim2=-1)  # 最后两维的对角线

# 维度扩展
x_expanded = x.unsqueeze(0)  # 在第0维添加维度

# 数组切片
sub = A[i:i+size, j:j+size]  # 提取子矩阵

# 堆叠
stacked = torch.stack([A, B, C])  # 沿新维度堆叠

# 拼接
concatenated = torch.hstack([A, B])  # 横向拼接

# 排序索引
sorted_idx = x.argsort()  # 从小到大的索引

# reshape
reshaped = x.view(-1, headsize)  # -1自动推断

# 求和
sum_all = x.sum()
sum_dim = x.sum(dim=0)  # 沿第0维求和
```

---

## 第七部分：扩展应用 - Head维度剪枝

### 7.1 从Head剪枝到Head_dim剪枝

#### 当前实现：Head剪枝

slimgpt当前实现的是**完整head剪枝**：

```python
# 删除整个head
原始: 12个head × 64维 = 768维
剪枝: 删除3个head → 9个head × 64维 = 576维
```

**特点**：
- 删除完整的attention head
- 保持每个head的维度不变
- 需要Global Update补偿跨head影响

#### 扩展方案：Head_dim剪枝

**Head维度剪枝**减少每个head的维度，但保持head数量：

```python
# 减少每个head的维度
原始: 12个head × 64维 = 768维
剪枝: 12个head × 48维 = 576维  # 每个head删除16维
```

**特点**：
- 保持head数量不变
- 每个head独立评估，删除相同数量的维度
- **无需Global Update**（head内操作）

---

### 7.2 Head_dim剪枝的优势

#### 1. 个性化剪枝

每个head根据自己的特征分布决定删除哪些维度：

```python
Head 0: 删除 [dim3, dim17, dim25, dim48, ...]   # 根据Head 0的误差
Head 1: 删除 [dim7, dim42, dim55, dim60, ...]   # 根据Head 1的误差
Head 2: 删除 [dim1, dim31, dim45, dim60, ...]   # 根据Head 2的误差
```

**对比统一位置删除**：

```python
# ❌ 次优方案：所有head删除相同位置
Head 0: 删除 [dim48, dim49, ..., dim63]  # 但dim48可能对Head 0很重要
Head 1: 删除 [dim48, dim49, ..., dim63]  # 统一位置
Head 2: 删除 [dim48, dim49, ..., dim63]

# ✅ 优化方案：每个head独立评估
Head 0: 删除自己最不重要的16维
Head 1: 删除自己最不重要的16维
Head 2: 删除自己最不重要的16维
```

#### 2. 无需Global Update

**关键假设**：Head间影响较小

```python
完整Hessian [192, 192]:

        Head0    Head1    Head2
      ┌──────┬──────┬──────┐
Head0 │ H_00 │ H_01 │ H_02 │  H_00 强相关（head内部）
      │[64,64]│      │      │  H_01 弱相关（head间）
      ├──────┼──────┼──────┤
Head1 │      │ H_11 │ H_12 │
      ├──────┼──────┼──────┤
Head2 │      │      │ H_22 │
      └──────┴──────┴──────┘
```

**推论**：
- Head 0删除dim3 → 主要影响Head 0内部（通过`H_00`）
- 对Head 1的影响通过`H_01`，但`H_01`很小
- **结论**：只需在head内做Local Update

#### 3. 保持结构规整

```python
原始: [num_heads=12, head_dim=64]
剪枝: [num_heads=12, head_dim=48]

# reshape仍然有效
Q = Q.view(batch, seq_len, 12, 48)  ✓
attention_output = attention_output.view(batch, seq_len, 576)  ✓
```

#### 4. 计算效率高

```python
# 每个head独立处理小矩阵
for head in range(12):
    H_head [64, 64]   # 小矩阵，快速求逆
    W_head [768, 64]

# 可并行化
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(prune_head, i) for i in range(12)]
```

---

### 7.3 算法实现

#### 核心流程

```python
def head_dim_pruning(W, H, num_heads, headsize, sparsity):
    """
    对每个head单独评估，删除相同数量的维度

    Args:
        W: [rows, columns] 权重矩阵
        H: [columns, columns] Hessian矩阵
        num_heads: head数量（例如12）
        headsize: 每个head的维度（例如64）
        sparsity: 稀疏度（例如0.25表示删除25%维度）

    Returns:
        W_pruned: [rows, new_columns] 剪枝后的权重
        H_pruned: [new_columns, new_columns] 更新后的Hessian
    """

    dims_to_remove_per_head = round(headsize * sparsity)

    # 逐个head处理
    for head_idx in range(num_heads):
        start_col = head_idx * headsize
        end_col = start_col + headsize

        # 1. 提取该head的权重和Hessian
        W_head = W[:, start_col:end_col]           # [rows, headsize]
        H_head = H[start_col:end_col, start_col:end_col]  # [headsize, headsize]

        # 2. 计算该head内的OBS误差
        Hinv_head = cholesky_inverse(cholesky(H_head))
        error_head = (W_head ** 2 / Hinv_head.diag().unsqueeze(0)).sum(0)

        # 3. 选择该head内最不重要的维度
        dim_sort_idx = error_head.argsort()
        dims_to_remove = dim_sort_idx[:dims_to_remove_per_head]

        # 4. 在head内部应用OBS（只用Local Update）
        prune_dims_in_head(W_head, H_head, dims_to_remove)

    return W, H
```

#### Head内部剪枝详细步骤

```python
def prune_dims_in_head(W_head, H_head, dims_to_remove):
    """
    在单个head内部应用OBS剪枝（只用Local Update）

    Args:
        W_head: [rows, 64] 该head的权重
        H_head: [64, 64] 该head的Hessian
        dims_to_remove: [16] 要删除的维度索引（head内相对位置）
    """

    # 1. 重排：将要删除的维度移到前面
    keep_dims = [i for i in range(64) if i not in dims_to_remove]
    reorder_idx = list(dims_to_remove) + keep_dims

    W_head = W_head[:, reorder_idx]
    H_head = H_head[reorder_idx, :][:, reorder_idx]

    # 2. 计算Hessian逆并做Cholesky分解（上三角）
    Hinv_head = cholesky_inverse(cholesky(H_head))
    Hinv_head = cholesky(Hinv_head, upper=True)[:16]  # 前16行

    # 3. Local Update（只在head内部）
    W1 = W_head[:, :16].clone()   # 待删除的16维
    Hinv1 = Hinv_head[:, :16]     # [16, 16] 上三角
    Err1 = torch.zeros_like(W1)

    for i in range(16):
        Err1[:, i] = W1[:, i] / Hinv1[i, i]
        W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

    # 4. 清零待删除维度（无需Global Update）
    W_head[:, :16] = 0

    # 5. 恢复原始顺序
    reorder_idx_inv = torch.argsort(reorder_idx)
    W_head = W_head[:, reorder_idx_inv]

    # 6. 更新Hessian（只在head内部）
    H_head[dims_to_remove, :] = 0
    H_head[:, dims_to_remove] = 0
    H_head[dims_to_remove, dims_to_remove] = 1
```

---

### 7.4 算法流程图

```
开始
  ↓
for each head in [0, 1, ..., 11]:
  ↓
  提取该head的子空间
  ├─ W_head [768, 64]
  └─ H_head [64, 64]
  ↓
  计算head内OBS误差
  error_head [64]
  ↓
  选择最不重要的16维
  dims_to_remove = [3, 17, 25, ...]
  ↓
  重排（待删除维度移到前面）
  ↓
  ┌─────────────────────┐
  │ Local Update循环    │
  │ for i in range(16): │
  │   计算 Err[:, i]    │
  │   更新 W_head[:, i:]│
  └─────────────────────┘
  ↓
  清零 W_head[:, :16] = 0
  ↓
  恢复原始顺序
  ↓
  更新 H_head（清零+对角线1）
  ↓
next head
  ↓
结束

# 结果：每个head从64维 → 48维
```

---

### 7.5 与Head剪枝的对比

| 特性 | Head剪枝 | Head_dim剪枝 |
|------|---------|-------------|
| **删除对象** | 完整的head | 每个head内的部分维度 |
| **head数量** | 减少（12→9） | 不变（12） |
| **head维度** | 不变（64） | 减少（64→48） |
| **个性化** | ❌ 删除整个head | ✅ 每个head独立评估 |
| **Global Update** | ✅ 需要 | ❌ 不需要 |
| **计算复杂度** | 高（192×192矩阵） | 低（12个64×64矩阵） |
| **并行化** | 困难 | ✅ 容易 |
| **精度** | 好 | **更好**（个性化） |
| **适用场景** | 减少head数量 | 减少每个head的复杂度 |

---

### 7.6 实现要点

#### 修改1：参数添加

```python
class SlimGPT:
    def __init__(self, ..., prune_type='head'):
        """
        Args:
            prune_type: 'head' 或 'head_dim'
        """
        self.prune_type = prune_type
```

#### 修改2：目标计算

```python
if self.prune_type == 'head':
    # Head剪枝：删除整个head
    target_columns = round(self.columns // headsize * sparsity) * headsize
elif self.prune_type == 'head_dim':
    # Head_dim剪枝：每个head删除相同数量的维度
    dims_per_head = headsize
    dims_to_remove_per_head = round(dims_per_head * sparsity)
    target_dims_per_head = dims_per_head - dims_to_remove_per_head
    target_columns = num_heads * target_dims_per_head
```

#### 修改3：循环结构

```python
if self.prune_type == 'head':
    # 原始逻辑：全局迭代删除head
    while pruned_columns < target_columns:
        # ... 全局排序和删除

elif self.prune_type == 'head_dim':
    # 新逻辑：逐head处理
    for head_idx in range(num_heads):
        # ... head内部剪枝
```

---

### 7.7 使用示例

```python
from slim_utils.slimgpt import SlimGPT

# Head剪枝（原始方法）
pruner_head = SlimGPT(
    layer,
    dataloader,
    prune_type='head',      # 删除完整head
    headsize=64
)
pruner_head.prune(sparsity=0.25)  # 删除25%的head（12→9）

# Head_dim剪枝（新方法）
pruner_dim = SlimGPT(
    layer,
    dataloader,
    prune_type='head_dim',  # 减少head维度
    headsize=64
)
pruner_dim.prune(sparsity=0.25)   # 每个head删除25%维度（64→48）
```

---

### 7.8 数值示例

假设 3个head，每个64维，删除25%维度（16维）：

```python
# 初始状态
W.shape = [768, 192]  # 3个head × 64维
H.shape = [192, 192]

# 逐head处理
Head 0 (columns 0-63):
  error_head = [5.2, 0.8, 3.1, 0.5, ...]
  删除: [dim3(0.5), dim17(0.6), ..., 共16维]

Head 1 (columns 64-127):
  error_head = [2.5, 0.3, 1.2, 8.9, ...]
  删除: [dim1(0.3), dim42(0.4), ..., 共16维]

Head 2 (columns 128-191):
  error_head = [7.1, 1.5, 0.9, 4.3, ...]
  删除: [dim2(0.9), dim31(1.1), ..., 共16维]

# 最终状态
W.shape = [768, 144]  # 3个head × 48维
H.shape = [144, 144]

# 每个head保留了自己最重要的48维
```

---

### 7.9 理论基础

#### OBS在子空间中的应用

Head_dim剪枝本质上是在**独立子空间**中应用OBS：

```
完整空间: R^192

分解为独立子空间:
  Head 0: R^64
  Head 1: R^64  ⊕  独立
  Head 2: R^64

在每个子空间独立应用OBS:
  Head 0: R^64 → R^48 （删除16维）
  Head 1: R^64 → R^48
  Head 2: R^64 → R^48
```

**关键假设验证**：
- Attention机制中，不同head学习不同的特征子空间
- Head间通过残差连接和层归一化交互，但直接影响较小
- 实验表明head间Hessian非对角块 `H_ij (i≠j)` 通常较小

---

### 7.10 总结

Head_dim剪枝是OBS算法的**创新应用**：

1. ✅ **更精确**：每个head独立优化，保留最重要的维度
2. ✅ **更简单**：无需Global Update，计算效率高
3. ✅ **更灵活**：可以精确控制每个head的最终维度
4. ✅ **理论严谨**：基于OBS在独立子空间的应用

**适用场景**：
- 减少attention复杂度而不改变head数量
- 对每个head进行精细化优化
- 需要保持多头结构的模型压缩

---

## 结语

本文档详细解释了slimgpt中OBS结构化剪枝算法的实现原理，包括：

1. ✅ **基础概念**：权重形状、索引映射、死节点处理
2. ✅ **剪枝准备**：掩码初始化、目标计算、结构化约束
3. ✅ **核心算法**：三次Cholesky分解的数学原理和必要性
4. ✅ **误差计算**：OBS公式、排序策略、批量删除
5. ✅ **权重更新**：Local和Global Update的Hessian分块使用、误差补偿机制
6. ✅ **维度问题**：理论vs实践、简化假设的合理性
7. ✅ **扩展应用**：Head维度剪枝的理论基础、算法实现和个性化优化

通过数值示例、维度追踪、矩阵分块图和直觉解释，帮助理解这一精妙的剪枝算法及其创新扩展。

---

**参考文献**：
- Hassibi, B., & Stork, D. G. (1993). Second order derivatives for network pruning: Optimal brain surgeon. NIPS.
- Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One? NeurIPS.

**代码位置**：`/home/project/real_prune/slimgpt_pub_prune/slim_utils/slimgpt.py`
