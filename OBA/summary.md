# OBA 剪枝方法深度解析总结

## 目录
1. [OBA 总体解释](#1-oba-总体解释)
2. [OBA 的 Attention 处理方法](#2-oba-的-attention-处理方法)
3. [FastOBA 的实现与改进](#3-fastoba-的实现与改进)
4. [FastOBA 的 Attention 处理](#4-fastoba-的-attention-处理)
5. [与结构化 OBS 的结合方法](#5-与结构化-obs-的结合方法)

---

## 1. OBA 总体解释

### 1.1 核心思想

**Optimal Brain Apoptosis (OBA)** 是一种基于 Hessian 矩阵的神经网络结构化剪枝方法。

**关键创新**：
- 直接计算 **Hessian-向量积**（二阶泰勒展开项）
- 不像传统方法（如 OBD）使用 Fisher 矩阵的一阶近似
- **不计算 Hessian 逆矩阵**（区别于 OBS）

### 1.2 数学基础

对于参数 $\theta$，损失函数的二阶泰勒展开：

$$
L(\theta - \delta\theta) \approx L(\theta) - \sum_i \frac{\partial L}{\partial \theta_i}\delta\theta_i + \frac{1}{2}\sum_{i,j}\frac{\partial^2 L}{\partial\theta_i\partial\theta_j}\delta\theta_i\delta\theta_j
$$

**OBA 的重要性度量**：

$$
\text{Importance}_i = \left| \sum_{j} \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \delta\theta_i \delta\theta_j \right|
$$

**注意**：这是 Hessian-向量积的**幅值**，而非 OBS 的 $\frac{\theta_i^2}{[H^{-1}]_{ii}}$！

### 1.3 连接性分解

OBA 的核心创新：将整个网络的 Hessian 分解为三种连接性：

```
总 Hessian = 上游连接性 (Upward) + 下游连接性 (Downward) + 并行连接性 (Parallel)
```

#### **1. 上游连接性 (Upward Direct Connectivity)**

**含义**：参数 $\theta_i$ 如何影响其**上游**层（损失函数方向）

**计算方法** (`oba_pruner.py:340-375`)：
```python
def upward_direct_connectivity_hessian(self, current_group_importances):
    for module in self.model.modules():
        # 使用替代前向传播
        delta_w = self.upward_delta * module.weight.data
        y = self.surrogate_forward(module, x, delta_w, zero_bias)

        # 反向传播
        y.backward(current_group_importances[module]["output_gradient"])

        # 计算 Hessian
        dw = module.weight.grad.data
        upward_hessian = dw * delta_w
```

**权重控制**：`--upward_delta`（默认 1.0）

#### **2. 下游连接性 (Downward Both Connectivity)**

**含义**：**下游**层的参数如何通过 $\theta_i$ 影响损失

**计算方法** (`oba_pruner.py:453-657`)：
```python
def downward_both_connectivity_hessian(self, current_group_importances):
    # 从根节点开始，逐层向下传播
    for current_node in topology_order:
        # 获取输入的 Hessian 特征
        input_hessian = self.obtain_feature(node, passed_nodes_feature)

        # 替代前向传播
        delta_w = self.downward_delta * module.weight.data
        output = self.surrogate_forward(module, input_hessian, delta_w)

        # 计算 Hessian
        downward_hessian = grad(output, delta_w) * delta_w
```

**权重控制**：`--downward_delta`（默认 1.0）

**特殊处理**：Attention 层的 softmax Jacobian 在此计算（见下节）

#### **3. 并行连接性 (Parallel Connectivity)**

**含义**：参数 $\theta_i$ **自身**的二阶效应

**计算方法** (`oba_pruner.py:299-330`)：
```python
def first_order_taylor(self, loss, current_group_importances):
    loss.backward(retain_graph=True)

    for module in self.model.modules():
        dw = module.weight.grad.data
        delta_w = self.delta * module.weight.data
        parallel_hessian = dw * delta_w  # 这就是并行 Hessian
```

**权重控制**：`--parallel_delta`（默认 1.0）

### 1.4 从 Hessian 到通道重要性

#### **步骤 1：计算权重 Hessian**

```python
# 累积三种连接性
group_importances[module]["weight"] = (
    upward_hessian + downward_hessian + parallel_hessian
)
```

权重 Hessian 形状：
- Conv2d: `[out_ch, in_ch, kh, kw]`
- Linear: `[out_features, in_features]`

#### **步骤 2：提取输出通道重要性**

```python
# oba_importance.py:47-50
if prune_fn == prune_linear_out_channels:
    local_imp = group_importances[layer]['output']

# 如何从 weight Hessian 得到 output？
# 对于 Linear: [out, in]
output_imp = weight_hessian.sum(dim=1)  # 对输入维度求和
```

#### **步骤 3：跨 batch 平均**

```python
# oba_importance.py:103-105
group_imp = list(map(lambda imp: imp.mean(0).abs(), group_imp))
```

#### **步骤 4：归一化**

```python
# oba_importance.py:109
group_imp = self._normalize(group_imp, self.normalizer)
# 支持：max, mean, sum, gaussian, norm 等
```

### 1.5 依赖图与剪枝执行

**依赖图** (`torch_pruning/dependency.py`)：
- 追踪层之间的通道依赖关系
- 确保剪枝后网络结构有效

**剪枝流程**：
1. 构建依赖图：`DG.get_all_groups()`
2. 对每个组计算重要性：`estimate_importance(group)`
3. 全局阈值：拼接所有重要性，找到 topk
4. 执行剪枝：`group.prune()`
5. 依赖传播：自动剪掉相关层的对应通道

---

## 2. OBA 的 Attention 处理方法

### 2.1 Attention 的挑战

**标准 Attention 结构**：

```python
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)  # Q, K, V 投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)      # 输出投影

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # Multi-head attention
        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        attn = softmax(attn, dim=-1)  # ← 非线性！

        # Output
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, embed_dim)
        out = self.out_proj(out)
        return out
```

**问题**：
1. **softmax 的 Hessian 非常复杂**（需要 Jacobian 矩阵）
2. **Q/K/V 之间有复杂交互**
3. **多头结构**需要特殊处理

### 2.2 OBA 的解决方案：显式计算 softmax Jacobian

**位置**：`oba_pruner.py:575-657`（在 `downward_both_connectivity_hessian` 中）

#### **步骤 1：识别 Attention 模块**

```python
# oba_pruner.py:542-558
if isinstance(module, Attention):
    # 获取 Q, K, V 的 Hessian 特征
    q, k, v = ...
    forward_q_grad, forward_k_grad, forward_v_grad = ...
```

#### **步骤 2：计算 Attention 权重的 Hessian**

```python
# oba_pruner.py:629-635
forward_attn_weights = (forward_q @ forward_k.transpose(2, 3)) / sqrt(head_dim)

# Q 的贡献
attn_weights_q = (forward_q_grad @ forward_k.transpose(2, 3)) / sqrt(head_dim)

# K 的贡献
attn_weights_k = (forward_q @ forward_k_grad.transpose(2, 3)) / sqrt(head_dim)

# 总 Hessian
attn_weights_grad = attn_weights_q + attn_weights_k
```

#### **步骤 3：计算 softmax 的 Jacobian**

这是 OBA 的**核心难点**！

```python
# oba_pruner.py:636-640
# 定义 softmax 的 forward Jacobian
jacobianfunc = torch.func.jacfwd(lambda x: torch.softmax(x, dim=0))

# 批量计算（对每个 batch, head, position）
softmax_jacobian = torch.func.vmap(
    torch.func.vmap(
        torch.func.vmap(jacobianfunc)
    )
)(forward_attn_weights).detach()

# softmax 输出的 Hessian
softmax_attn_weights_grad = (
    softmax_jacobian * attn_weights_grad.unsqueeze(3)
).sum(4)
```

**数学含义**：

softmax 的 Jacobian：
$$
J_{ij} = \frac{\partial \text{softmax}(x)_i}{\partial x_j} = \begin{cases}
s_i(1 - s_i) & \text{if } i = j \\
-s_i s_j & \text{if } i \neq j
\end{cases}
$$

其中 $s_i = \text{softmax}(x)_i$。

#### **步骤 4：反向传播到 Q/K/V**

```python
# oba_pruner.py:641-657
before_linear_grad = (
    current_group_importances[module]['output_gradient'].unsqueeze(-1)
    * module.out_proj.weight
).sum(2).detach()

# Q, K 的梯度
out = softmax_attn_weights_grad @ forward_v.detach()
out = out.reshape(b, num_heads, n, head_dim).permute(0, 2, 1, 3).reshape(...)
out.backward(before_linear_grad, retain_graph=True)

# V 的梯度
out = softmax_forward_attn_weights @ forward_v_grad
out.backward(before_linear_grad, retain_graph=True)

# Attention weights 的梯度
out = softmax_attn_weights_grad.detach() @ forward_v
out.backward(before_linear_grad, retain_graph=True)
```

**最终结果**：`out_proj` 的权重梯度累积了来自 Q/K/V 的所有 Hessian 信息。

#### **补充：伪损失的聚合方式（sum vs mean）**

在 OBA 的 Attention 处理中，`before_linear_grad` 的构造使用了 `sum` 聚合：

```python
# oba_pruner.py:641-657
before_linear_grad = (
    current_group_importances[module]['output_gradient'].unsqueeze(-1)
    * module.out_proj.weight
).sum(2).detach()  # ← 对 embed_dim 维度求和
```

**使用 sum vs mean 的区别分析**：

| 方面 | sum | mean |
|------|-----|------|
| **数学含义** | 总梯度 = Σ(所有维度的贡献) | 总梯度 = (1/n) × Σ(所有维度的贡献) |
| **Hessian 幅值** | 与维度数成正比 | 被归一化到固定范围 |
| **维度公平性** | 大 embed_dim 层的重要性偏高 | 不同维度的层具有可比性 |

**对剪枝结果的实际影响**：

1. **局部剪枝（Local Pruning）**：**完全没有区别**
   ```python
   # 每层独立剪枝，只关心排序
   importance = compute_importance(layer)  # [ch1, ch2, ch3, ...]
   pruned_indices = argsort(importance)[:n_prune]
   ```
   - sum 和 mean 只差常数因子 (1/n)
   - 排序结果相同：`argsort([100, 200, 300])` = `argsort([10, 20, 30])`

2. **全局剪枝 + 归一化**：**可能没有区别**

   OBA 默认使用归一化（`normalizer='max'`）：
   ```python
   # oba_importance.py:148-149
   group_imp = self._normalize(group_imp, self.normalizer)
   # 'max': importance / importance.max()
   ```

   归一化后效果：
   - **sum 版本**：`[100, 200, 300]` → normalize → `[0.33, 0.67, 1.0]`
   - **mean 版本**：`[10, 20, 30]` → normalize → `[0.33, 0.67, 1.0]`
   - 结果相同！

3. **全局剪枝 + 无归一化**：**有区别**

   如果 `normalizer='none'`：
   ```python
   # Layer A (embed_dim=768): sum → 重要性 ~1000, mean → 重要性 ~1.3
   # Layer B (embed_dim=256): sum → 重要性 ~300,  mean → 重要性 ~1.2

   # 全局阈值剪枝
   all_importance = concat([layer_A_imp, layer_B_imp])
   threshold = percentile(all_importance, pruning_ratio)
   ```
   - **使用 sum**：Layer A 的重要性会偏高（因为维度大）
   - **使用 mean**：更公平地比较不同层

**推荐选择**：

**使用 mean 更好**，原因：
- ✅ **维度无关性**：不同 embed_dim 的层具有可比性
- ✅ **数值稳定性**：避免大维度层的数值过大
- ✅ **理论一致性**：期望的概念（平均影响）比总和更合理

**但在 OBA 当前实现中**：
- 由于使用了**归一化**（默认 `normalizer='max'`）
- **sum 和 mean 的剪枝结果完全相同**

**实验验证**：

```python
# 修改 oba_pruner.py:641 测试
# 原版（sum）
before_linear_grad_sum = (...).sum(2).detach()

# 测试版（mean）
before_linear_grad_mean = (...).mean(2).detach()

# 预测：如果使用默认归一化，剪枝结果应该完全相同
```

### 2.3 结构化剪枝：评估哪个矩阵？

**关键发现**：OBA 使用 **`out_proj`** 的 Hessian！

#### **为什么是 out_proj？**

1. **依赖图的选择**：
```python
# metapruner.py:310-321
def _downstream_node_as_root_if_attention(self, group):
    is_attention = False
    for _dep in group:
        if _dep.source.module in self.num_heads:
            is_attention = True

    if is_attention:
        # 使用下游节点（out_proj）作为根
        group = self.DG.get_pruning_group(downstream_module, ...)
    return group
```

2. **数据流方向**：
```
qkv_proj → Q/K/V reshape → Attention 计算 → out_proj → 输出
                                            ↑
                                    评估这里的 Hessian
```

3. **维度对应**：
   - `out_proj.weight`: `[embed_dim, embed_dim]`
   - `embed_dim = num_heads × head_dim`
   - 前 `head_dim` 维对应 Head 0，接下来 `head_dim` 维对应 Head 1，依此类推

#### **从 out_proj Hessian 到 Head 重要性**

```python
# oba_pruner.py:1102-1110
_is_attn, qkv_layers = self._is_attn_group(group)
if _is_attn and self.prune_num_heads:
    # imp 形状: [embed_dim]，例如 [768]
    # num_heads = 12, head_dim = 64

    # 重塑为 [num_heads, head_dim]
    # [[i1, i2, ..., i64],      ← Head 0
    #  [i65, i66, ..., i128],   ← Head 1
    #  ...
    #  [i705, ..., i768]]       ← Head 11

    # 每个 head 取平均
    head_imp = imp.view(num_heads, head_dim).mean(1)
    # head_imp 形状: [num_heads]

    global_head_importance[group] = (qkv_layers, head_imp)
```

**数学公式**：

$$
\text{Importance}(\text{Head}_h) = \frac{1}{d_{\text{head}}} \sum_{i=h \times d_{\text{head}}}^{(h+1) \times d_{\text{head}} - 1} \left| H_{\text{out\_proj}}^{(i)} \right|
$$

### 2.4 两种剪枝模式

#### **模式 A：剪枝 Head 维度** (`--prune_head_dims True`)

```python
# oba_pruner.py:1160-1168
if not _is_attn or self.prune_head_dims == True:
    raw_imp = self.estimate_importance(group, ch_groups=num_heads)

    # 对每个 head 独立剪枝
    for head_id in range(num_heads):
        sub_group_imp = raw_imp[head_id * head_dim : (head_id+1) * head_dim]
        # 剪掉该 head 内重要性低的维度
        pruning_idxs = argsort(sub_group_imp)[:n_pruned_per_head]
```

**效果**：
- 保留所有 head
- 减少每个 head 的维度
- 例如：12 heads × 64 dims → 12 heads × 40 dims

#### **模式 B：剪枝整个 Head** (`--prune_num_heads True`)

```python
# oba_pruner.py:1184-1193
if group in global_head_importance:
    qkv_layers, head_imp = global_head_importance[group]

    # 找到重要性低于阈值的 head
    head_pruning_indices = (head_imp <= head_thres).nonzero().view(-1)

    for head_id in head_pruning_indices:
        # 剪掉该 head 的所有维度
        pruning_indices.append(
            torch.arange(head_id * head_dim, (head_id+1) * head_dim)
        )

    # 更新 head 数量
    for qkv_layer in qkv_layers:
        self.num_heads[qkv_layer] -= len(head_pruning_indices)
```

**效果**：
- 移除整个 head
- 例如：12 heads → 8 heads

---

## 3. FastOBA 的实现与改进

### 3.1 核心创新：任意阶泰勒展开

**FastOBA 的最大突破**：从固定 2 阶扩展到任意 k 阶！

#### **数学基础**

完整的泰勒展开：

$$
L(\theta - \delta\theta) = L(\theta) + \sum_{k=1}^{\infty} \frac{1}{k!} \sum_{i_1, \ldots, i_k} \frac{\partial^k L}{\partial \theta_{i_1} \cdots \partial \theta_{i_k}} \delta\theta_{i_1} \cdots \delta\theta_{i_k}
$$

**FastOBA 可以计算任意 k 阶项**！

#### **实现：`any_order_differentiation`**

```python
# fastoba_pruner.py:288-316
def any_order_differentiation(self, loss, delta, parameters, order):
    """计算 k 阶泰勒展开项"""

    for current_order in range(1, order + 1):
        if current_order == 1:
            # 1 阶：∂L/∂θ
            current_grad = torch.autograd.grad(
                loss,
                parameters,
                create_graph=True  # 保留计算图用于下一阶
            )
        else:
            # k 阶：对 (k-1) 阶梯度再求导
            grad_outputs = [param * delta for param in parameters]
            current_grad = torch.autograd.grad(
                current_grad,  # 对上一阶的梯度求导
                parameters,
                grad_outputs=grad_outputs,
                create_graph=(current_order != order)
            )

    # k 阶泰勒项 = grad^(k) × θ × δ^k
    grads = [grad * param * delta for grad, param in zip(current_grad, parameters)]
    return grads
```

**递归计算过程**：

```
order=1: grad1 = ∂L/∂θ
order=2: grad2 = ∂(grad1)/∂θ × (θ × δ)  ← OBA 的能力
order=3: grad3 = ∂(grad2)/∂θ × (θ × δ)
order=k: gradk = ∂(grad_{k-1})/∂θ × (θ × δ)
```

**数学含义**（以 order=2 为例）：

$$
\text{grad2}_i = \sum_j \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \theta_i \delta \theta_j \delta
$$

这正是 **Hessian-向量积** × $\theta_i \times \delta^2$！

### 3.2 架构简化

#### **OBA 的复杂性**

```python
# OBA 需要：
1. register_hooks()  # 捕获激活和梯度
2. first_order_taylor()  # 一阶 + 并行 Hessian
3. upward_direct_connectivity_hessian()  # 上游 Hessian
4. downward_both_connectivity_hessian()  # 下游 Hessian
   └─ 包含 Attention 的特殊处理
5. 手动累积三种连接性

# 总代码量：~1200 行
```

#### **FastOBA 的优雅**

```python
# FastOBA 只需要：
1. any_order_differentiation()  # 一次性计算所有
2. obtain_importance()  # 映射到各层
3. update_group_importance()  # 累积

# 总代码量：~500 行
```

**关键差异**：

| 维度 | OBA | FastOBA |
|------|-----|---------|
| **Hooks** | 必须注册 | 不需要 |
| **连接性** | 手动分解（3种） | 自动处理 |
| **Attention** | 手动计算 softmax Jacobian | 自动微分 |
| **通用性** | 需要为每种层编写代码 | 适用于任意层 |
| **维护** | 高（复杂度高） | 低（简洁） |

### 3.3 其他改进

#### **1. 分布式训练支持**

```python
# fastoba_pruner.py:310-315
def any_order_differentiation(self, ..., distributed=False):
    if distributed:
        world_size = dist.get_world_size()
        for i, g in enumerate(grads):
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            grads[i] = g.div(world_size)
```

**OBA 不支持分布式！**

#### **2. 内存优化：滑动窗口**

```python
# fastoba_pruner.py:368-375
self.group_importances[module]["output"] = torch.cat([
    self.group_importances[module]["output"],
    output_imp.unsqueeze(0).cpu()
])

# 限制历史长度
if len(self.group_importances[module]["output"]) > self.record_length:
    self.group_importances[module]["output"] = \
        self.group_importances[module]["output"][1:]  # 删除最旧的
```

**OBA 无限累积**，可能导致内存溢出！

#### **3. 数值稳定性**

```python
# fastoba_pruner.py:22-25
def safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """避免除以零"""
    safe_denom = torch.where(denom != 0, denom, 1.0)
    return numer / safe_denom
```

在归一化时使用 `safe_div` 而非直接除法。

### 3.4 完整流程对比

#### **OBA 流程**

```
1. 注册 hooks
   ↓
2. 前向传播（捕获 X, Y）
   ↓
3. 反向传播（捕获梯度）
   ↓
4. 计算一阶 Taylor
   ↓
5. 计算上游 Hessian（替代前向传播）
   ↓
6. 计算下游 Hessian（替代前向传播 + Attention 特殊处理）
   ↓
7. 累积三种连接性
   ↓
8. 提取输出重要性
   ↓
9. 执行剪枝
```

#### **FastOBA 流程**

```
1. 前向传播
   ↓
2. any_order_differentiation(loss, order=k)
   └─ 内部自动处理所有连接性和 Attention
   ↓
3. 映射到各模块
   ↓
4. 提取输出重要性（滑动窗口）
   ↓
5. 执行剪枝
```

**时间复杂度估计**：
- OBA: ~5-10秒/iteration（需要多次反向传播）
- FastOBA: ~2-4秒/iteration（只需 k 次反向传播）

---

## 4. FastOBA 的 Attention 处理

### 4.1 核心哲学：黑盒自动微分

**FastOBA 的关键洞察**：
> 不需要知道 Attention 的内部细节，让 PyTorch 自动处理！

#### **OBA vs FastOBA 的对比**

| 维度 | OBA | FastOBA |
|------|-----|---------|
| **Softmax 处理** | 手动计算 Jacobian | PyTorch 自动 |
| **Q/K/V 交互** | 显式追踪和反向传播 | 自动微分自动处理 |
| **代码复杂度** | 82 行专门处理 Attention | 0 行（通用代码） |
| **维护性** | 需要适配不同 Attention 实现 | 自动适配 |
| **灵活性** | 只支持标准 Attention | 支持任何 Attention 变体 |

### 4.2 计算图自动追踪

#### **Attention 的计算图**

```
Loss → Classifier → ... → out_proj(x) → Attention(Q,K,V) → qkv_proj → Input
                              ↑
                         这里评估 Hessian
```

其中 Attention 内部：

```
Q, K, V → matmul(Q, K^T) → scale(1/√d) → softmax → matmul(·, V) → concat → out
```

#### **PyTorch 自动微分的处理**

当调用 `torch.autograd.grad(loss, out_proj.weight)` 时：

**第 1 阶（梯度）**：
```python
∂L/∂(out_proj.weight) = ∂L/∂(out) × ∂(out)/∂(out_proj.weight)
```

其中 `∂L/∂(out)` 包含：
- 从 Loss 到 out_proj 输出的所有梯度
- **包括通过 softmax 的梯度**（PyTorch 自动计算）
- **包括通过 Q/K/V 交互的梯度**

**第 2 阶（Hessian）**：

```python
# fastoba_pruner.py:303-308
current_grad = torch.autograd.grad(
    current_grad,  # 这是一阶梯度
    parameters,
    grad_outputs=[param * delta for param in parameters]
)
```

这计算：

$$
\frac{\partial^2 L}{\partial (\text{out\_proj.weight})^2}
$$

**PyTorch 会自动**：
1. 追踪一阶梯度的计算图（包含 softmax 的反向传播）
2. 对这个计算图再求一次导数
3. 得到包含 **softmax Jacobian 影响** 的 Hessian
4. 全程无需手动计算！

### 4.3 FastOBA 不区分"内部"和"外部"

**关键理解**：FastOBA 计算的是 **总体 Hessian**，而非分解的 Hessian。

#### **OBA 的视角**

```
Total Hessian = ┌─────────────────────┐
                │   Upward Hessian    │  ← 可单独分析
                ├─────────────────────┤
                │  Downward Hessian   │
                │    ├─ Attention 内部 │  ← 显式计算
                │    ├─ Linear 传播    │
                │    └─ 其他操作       │
                ├─────────────────────┤
                │  Parallel Hessian   │
                └─────────────────────┘
```

#### **FastOBA 的视角**

```
Total Hessian = ┌───────────────────────────────┐
                │    PyTorch 自动微分计算        │
                │  （自动包含 Attention 的所有影响）│
                │    - softmax Jacobian        │
                │    - Q/K/V 交互              │
                │    - 上游/下游/并行连接       │
                └───────────────────────────────┘
```

**不需要分解，一次性得到总 Hessian！**

### 4.4 如何确保包含 Attention 影响？

#### **问题**：FastOBA 如何确定获得的 Hessian 包含 Attention 内部的影响？

**答案**：通过 **计算图的完整性**！

#### **详细说明**

1. **前向传播时**，PyTorch 构建完整的计算图：
```python
# 所有操作都被记录
x = qkv_proj(input)
q, k, v = x.chunk(3)
attn = softmax(q @ k.T / sqrt(d))  # ← softmax 节点
out = attn @ v  # ← matmul 节点
out = out_proj(out)  # ← linear 节点
loss = criterion(out, target)
```

2. **一阶反向传播时**：
```python
grad1 = torch.autograd.grad(loss, out_proj.weight, create_graph=True)
```
PyTorch 自动应用链式法则：
```
∂L/∂W = ∂L/∂out × ∂out/∂W
        ↑
   包含通过 softmax 的梯度
   （PyTorch 内置了 softmax 的反向传播，包含 Jacobian）
```

3. **二阶反向传播时**：
```python
grad2 = torch.autograd.grad(grad1, out_proj.weight, ...)
```
对一阶梯度的计算图再求导，得到 Hessian。

**关键**：因为 PyTorch 的 softmax 反向传播已经正确实现了 Jacobian 计算，所以二阶导数**自动包含** softmax 的二阶效应！

#### **实验验证**

你可以验证 FastOBA 是否正确捕获 Attention 影响：

```python
# 方法 1：对比实验
# 在同一模型上运行 OBA 和 FastOBA，比较剪枝后的准确率

# 方法 2：梯度检查
# 手动计算简单 Attention 的 Hessian，与 PyTorch 自动微分对比

# 方法 3：消融实验
# 移除 Attention，观察 Hessian 的变化
```

根据代码和论文，FastOBA 的效果与 OBA 相当或更好，这**间接证明**了自动微分正确处理了 Attention！

### 4.5 FastOBA 处理 Attention 的优缺点

#### **优点**

1. **简单**：无需理解 Attention 内部细节
2. **通用**：适用于任何 Attention 变体
   - 标准 Attention
   - FlashAttention
   - Sparse Attention
   - Multi-Query Attention
3. **正确性**：由 PyTorch 保证
4. **可扩展**：自动支持未来的新 Attention 机制

#### **缺点**

1. **无法单独分析**：不能分离 Attention 的贡献
2. **无法调整权重**：不能像 OBA 那样调整 upward/downward/parallel 的权重
3. **黑盒**：无法深入理解各部分的影响
4. **对 Attention 没有特殊优化**：可能错过一些特定于 Attention 的优化机会

---

## 5. 与结构化 OBS 的结合方法

### 5.1 动机：为什么需要 OBS？

#### **OBA/FastOBA 的局限**

**OBA/FastOBA 的重要性度量**：
$$
\text{Importance}_i = \left| \sum_j \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \delta\theta_i \delta\theta_j \right|
$$

这是 **Hessian 幅值**，但：
- ❌ 不是 **损失变化** 的直接估计
- ❌ 没有 **理论保证**（为什么幅值大就重要？）
- ❌ 无法 **补偿** 剪枝后的误差

#### **OBS (Optimal Brain Surgeon) 的优势**

**OBS 的重要性度量**：
$$
\Delta L_i = \frac{\theta_i^2}{2[H^{-1}]_{ii}}
$$

这是 **损失增加** 的二阶近似：
- ✅ 有明确的理论保证
- ✅ 可以通过 **权重补偿** 最小化误差
- ✅ 考虑了 Hessian 逆（参数之间的交互）

### 5.2 结合方案：Attention 输出的 OBS 剪枝

#### **目标**

最小化 **Attention 层输出** 的变化，而非最终 Loss：

$$
\min_{\text{pruned}} \quad \Delta O_{\text{attn}} = \|O_{\text{attn}}(\theta - \delta\theta) - O_{\text{attn}}(\theta)\|
$$

#### **核心思想**

使用 OBS 的框架，但 Hessian 是关于 **Attention 输出** 的：

$$
H_{\text{attn}} = \frac{\partial^2 O_{\text{attn}}}{\partial \theta_i \partial \theta_j}
$$

而非：

$$
H_{\text{loss}} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}
$$

### 5.3 方案 1：基于 OBA 的 Attention Hessian

#### **核心思路**

复用 OBA 的 Attention Hessian 计算代码，但改变目标函数。

#### **实现步骤**

**步骤 1：定义伪损失**

```python
# 不使用真实的 Loss，而是 Attention 输出的度量
def compute_attention_pseudo_loss(attention_module, input_data):
    output = attention_module(input_data)

    # 选项 A：输出的范数
    pseudo_loss = output.norm()

    # 选项 B：输出的平方和
    pseudo_loss = output.pow(2).sum()

    # 选项 C：与原始输出的差异（如果已剪枝一次）
    pseudo_loss = (output - original_output).pow(2).sum()

    return pseudo_loss
```

**步骤 2：复用 OBA 的 Attention Hessian 计算**

```python
# 基于 oba_pruner.py:575-657
def compute_attention_hessian_oba_style(attention_module, input_data):
    """使用 OBA 的方式计算 Attention 内部的 Hessian"""

    # 1. 注册 hooks
    hooks = []
    hooks.append(attention_module.register_forward_hook(forward_hook))
    hooks.append(attention_module.register_full_backward_hook(backward_hook))

    # 2. 前向传播
    output = attention_module(input_data)

    # 3. 定义伪损失
    pseudo_loss = output.pow(2).sum()

    # 4. 一阶反向传播
    pseudo_loss.backward(create_graph=True)

    # 5. 获取 Q, K, V（从 hooks）
    q, k, v = extract_qkv_from_hooks(attention_module)

    # 6. 计算 softmax Jacobian（复用 OBA 代码）
    forward_attn_weights = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
    jacobianfunc = torch.func.jacfwd(lambda x: torch.softmax(x, dim=-1))
    softmax_jacobian = torch.func.vmap(
        torch.func.vmap(
            torch.func.vmap(jacobianfunc)
        )
    )(forward_attn_weights)

    # 7. 计算 Q/K/V 的 Hessian
    # ... (类似 OBA 的代码)

    # 8. 传播到 out_proj
    out_proj_hessian = propagate_hessian_to_out_proj(...)

    # 9. 清理 hooks
    for hook in hooks:
        hook.remove()

    return out_proj_hessian
```

**步骤 3：计算 Hessian 逆的对角元素**

```python
def compute_hessian_inverse_diag(hessian, method='kfac'):
    """计算 Hessian 逆的对角元素"""

    if method == 'direct':
        # 方法 1：直接求逆（只适用于小矩阵）
        hessian_inv = torch.inverse(hessian + 1e-5 * torch.eye(hessian.shape[0]))
        return torch.diag(hessian_inv)

    elif method == 'kfac':
        # 方法 2：KFAC 近似
        # H ≈ A ⊗ G （Kronecker 分解）
        A, G = kfac_factorization(hessian)

        # H^-1 ≈ A^-1 ⊗ G^-1
        A_inv = torch.inverse(A + 1e-5 * torch.eye(A.shape[0]))
        G_inv = torch.inverse(G + 1e-5 * torch.eye(G.shape[0]))

        # 对角元素
        hessian_inv_diag = torch.kron(torch.diag(G_inv), torch.diag(A_inv))
        return hessian_inv_diag

    elif method == 'diagonal':
        # 方法 3：对角近似（最快，但最不准确）
        return 1.0 / (torch.diag(hessian) + 1e-8)
```

**步骤 4：应用 OBS 公式**

```python
def compute_obs_importance(attention_module, input_data):
    """计算 OBS 重要性分数"""

    # 1. 计算 Hessian
    hessian = compute_attention_hessian_oba_style(attention_module, input_data)

    # 2. 计算 Hessian 逆的对角元素
    hessian_inv_diag = compute_hessian_inverse_diag(hessian, method='kfac')

    # 3. OBS 重要性公式
    params = attention_module.out_proj.weight
    importance = params.pow(2) / (hessian_inv_diag.view(params.shape) + 1e-8)

    return importance
```

**步骤 5：剪枝和权重补偿**

```python
def prune_by_obs(attention_module, importance, pruning_ratio, hessian_inv):
    """基于 OBS 重要性剪枝"""

    # 1. 选择要剪掉的通道
    n_prune = int(pruning_ratio * importance.numel())
    _, indices = torch.topk(importance.view(-1), k=n_prune, largest=False)

    # 2. 权重补偿（OBS 的核心）
    # 当剪掉 θ_i 时，其他权重的补偿：
    # δθ_j = -θ_i × [H^-1]_{ij} / [H^-1]_{ii}

    for idx in indices:
        theta_i = attention_module.out_proj.weight.view(-1)[idx]
        h_inv_ii = hessian_inv[idx, idx]

        for j in range(attention_module.out_proj.weight.numel()):
            if j != idx:
                h_inv_ij = hessian_inv[idx, j]
                compensation = -theta_i * h_inv_ij / (h_inv_ii + 1e-8)
                attention_module.out_proj.weight.view(-1)[j] += compensation

    # 3. 置零被剪掉的权重
    mask = torch.ones_like(attention_module.out_proj.weight.view(-1))
    mask[indices] = 0
    attention_module.out_proj.weight.data *= mask.view(attention_module.out_proj.weight.shape)

    return mask
```

#### **优点**

- ✅ 有理论保证（OBS 框架）
- ✅ 包含权重补偿
- ✅ 考虑参数之间的交互（Hessian 逆）
- ✅ 针对 Attention 输出优化

#### **缺点**

- ❌ 需要计算 Hessian 逆（计算量大）
- ❌ 依赖 OBA 的复杂代码
- ❌ 仍需要手动处理 Attention

### 5.4 方案 2：基于 FastOBA 的自动微分 + OBS

#### **核心思路**

使用 PyTorch 自动微分计算 Hessian，然后应用 OBS。

#### **实现步骤**

**步骤 1：使用自动微分计算 Hessian**

```python
def compute_attention_hessian_autodiff(attention_module, input_data):
    """使用自动微分计算 Attention 输出的 Hessian"""

    params = list(attention_module.parameters())
    n_params = sum(p.numel() for p in params)

    # 初始化 Hessian 矩阵（对角近似）
    hessian_diag = torch.zeros(n_params)

    # 定义输出函数
    def output_func():
        output = attention_module(input_data)
        return output.pow(2).sum()  # 伪损失

    # 计算 Hessian 对角线
    param_idx = 0
    for param in params:
        for i in range(param.numel()):
            # 一阶导数
            grad1 = torch.autograd.grad(
                output_func(),
                param,
                create_graph=True,
                retain_graph=True
            )[0].flatten()[i]

            # 二阶导数
            grad2 = torch.autograd.grad(
                grad1,
                param,
                retain_graph=True
            )[0].flatten()[i]

            hessian_diag[param_idx] = grad2.item()
            param_idx += 1

    return hessian_diag
```

**注意**：完整 Hessian 矩阵太大（$O(n^2)$ 空间），通常只计算**对角近似**。

**步骤 2：使用 functorch 加速**

```python
from torch.func import jacrev, vmap

def compute_hessian_functorch(attention_module, input_data):
    """使用 functorch 高效计算 Hessian"""

    def output_func(*params):
        # 临时设置参数
        param_dict = {name: param for name, param in zip(
            attention_module.state_dict().keys(), params
        )}
        attention_module.load_state_dict(param_dict)

        output = attention_module(input_data)
        return output.pow(2).sum()

    params = tuple(attention_module.parameters())

    # 计算 Jacobian of gradient（即 Hessian）
    hessian = jacrev(jacrev(output_func))(*params)

    return hessian
```

**步骤 3-5：与方案 1 相同**

应用 OBS 公式和权重补偿。

#### **优点**

- ✅ 简单（使用自动微分）
- ✅ 通用（适用于任何模块）
- ✅ 有 OBS 的理论保证

#### **缺点**

- ❌ 完整 Hessian 太大，通常只能用对角近似
- ❌ 对角近似可能不够准确

### 5.5 方案 3：重构误差法（最实用）

#### **核心思路**

直接度量剪枝每个通道后的输出变化，无需 Hessian 逆！

#### **实现**

```python
def importance_by_reconstruction_error(attention_module, input_data, n_samples=100):
    """基于重构误差的重要性评估"""

    # 1. 获取原始输出
    attention_module.eval()
    with torch.no_grad():
        original_outputs = []
        for _ in range(n_samples):
            original_outputs.append(attention_module(input_data))
        original_output = torch.stack(original_outputs).mean(0)

    # 2. 对每个通道，计算移除后的输出变化
    importance_scores = {}

    # out_proj 的输出通道
    n_channels = attention_module.out_proj.out_features
    channel_importance = torch.zeros(n_channels)

    for ch in range(n_channels):
        # 暂时置零该通道
        original_weight = attention_module.out_proj.weight[ch].clone()
        attention_module.out_proj.weight.data[ch] = 0

        if attention_module.out_proj.bias is not None:
            original_bias = attention_module.out_proj.bias[ch].clone()
            attention_module.out_proj.bias.data[ch] = 0

        # 计算输出变化
        with torch.no_grad():
            pruned_outputs = []
            for _ in range(n_samples):
                pruned_outputs.append(attention_module(input_data))
            pruned_output = torch.stack(pruned_outputs).mean(0)

            reconstruction_error = (original_output - pruned_output).pow(2).sum()

        channel_importance[ch] = reconstruction_error.item()

        # 恢复权重
        attention_module.out_proj.weight.data[ch] = original_weight
        if attention_module.out_proj.bias is not None:
            attention_module.out_proj.bias.data[ch] = original_bias

    return channel_importance
```

#### **优点**

- ✅ 最直接（直接度量输出变化）
- ✅ 不需要 Hessian 逆
- ✅ 易于理解和实现
- ✅ 可以结合权重补偿

#### **缺点**

- ❌ 计算成本高（每个通道需要一次前向传播）
- ❌ 对于大模型，可能很慢

#### **优化：批量评估**

```python
def importance_by_reconstruction_error_batch(attention_module, input_data):
    """批量评估多个通道"""

    # 1. 原始输出
    with torch.no_grad():
        original_output = attention_module(input_data)

    n_channels = attention_module.out_proj.out_features
    channel_importance = torch.zeros(n_channels)

    # 2. 批量置零多个通道
    batch_size = 32  # 一次评估 32 个通道
    for start in range(0, n_channels, batch_size):
        end = min(start + batch_size, n_channels)

        # 创建 mask
        mask = torch.ones(n_channels, 1)
        mask[start:end] = 0

        # 临时应用 mask
        original_weight = attention_module.out_proj.weight.data.clone()
        attention_module.out_proj.weight.data *= mask

        # 计算输出
        with torch.no_grad():
            pruned_output = attention_module(input_data)
            errors = (original_output - pruned_output).pow(2).sum(dim=(0,1))

        channel_importance[start:end] = errors

        # 恢复
        attention_module.out_proj.weight.data = original_weight

    return channel_importance
```

### 5.6 方案对比与推荐

| 方案 | 理论保证 | 计算复杂度 | 实现难度 | 推荐指数 |
|------|---------|-----------|---------|---------|
| **方案1：OBA风格+OBS** | ⭐⭐⭐⭐⭐ | 高（需要 H^-1） | 难 | ⭐⭐⭐ |
| **方案2：FastOBA+OBS** | ⭐⭐⭐⭐ | 中（对角近似） | 中 | ⭐⭐⭐⭐ |
| **方案3：重构误差** | ⭐⭐⭐ | 高（多次前向） | 易 | ⭐⭐⭐⭐⭐ |

**推荐策略**：

1. **快速原型**：使用方案 3（重构误差）
   - 最直接，易于实现
   - 可以快速验证想法

2. **性能优化**：使用方案 2（FastOBA + 对角 OBS）
   - 平衡了理论和效率
   - 使用对角近似降低计算量

3. **研究探索**：使用方案 1（OBA风格 + 完整 OBS）
   - 最完整的理论框架
   - 可以深入分析 Attention 内部机制

### 5.7 完整实现示例

```python
class AttentionOBSPruner:
    """结合 OBS 思想的 Attention 剪枝器"""

    def __init__(self, attention_module, method='reconstruction'):
        self.attention = attention_module
        self.method = method

    def compute_importance(self, input_data):
        """计算通道重要性"""

        if self.method == 'reconstruction':
            # 方案 3：重构误差
            return self.importance_by_reconstruction(input_data)

        elif self.method == 'autodiff_obs':
            # 方案 2：自动微分 + OBS
            hessian_diag = self.compute_hessian_diag(input_data)
            params = self.attention.out_proj.weight
            importance = params.pow(2) / (hessian_diag + 1e-8)
            return importance

        elif self.method == 'oba_obs':
            # 方案 1：OBA 风格 + OBS
            hessian = self.compute_hessian_oba_style(input_data)
            hessian_inv_diag = self.compute_hessian_inverse_diag(hessian)
            params = self.attention.out_proj.weight
            importance = params.pow(2) / (hessian_inv_diag + 1e-8)
            return importance

    def prune(self, input_data, pruning_ratio, use_compensation=True):
        """执行剪枝"""

        # 1. 计算重要性
        importance = self.compute_importance(input_data)

        # 2. 选择要剪掉的通道
        n_prune = int(pruning_ratio * importance.numel())
        _, indices = torch.topk(importance.view(-1), k=n_prune, largest=False)

        # 3. 权重补偿（如果使用 OBS）
        if use_compensation and self.method != 'reconstruction':
            self.compensate_weights(indices)

        # 4. 执行剪枝
        mask = torch.ones_like(self.attention.out_proj.weight)
        mask.view(-1)[indices] = 0
        self.attention.out_proj.weight.data *= mask

        return mask

    def importance_by_reconstruction(self, input_data):
        """方案 3 实现"""
        # ... (见上文)

    def compute_hessian_diag(self, input_data):
        """方案 2 实现"""
        # ... (见上文)

    def compute_hessian_oba_style(self, input_data):
        """方案 1 实现"""
        # ... (见上文)

    def compensate_weights(self, pruned_indices):
        """OBS 权重补偿"""
        # ... (见上文)

# 使用示例
pruner = AttentionOBSPruner(model.attention, method='reconstruction')
mask = pruner.prune(input_data, pruning_ratio=0.3, use_compensation=False)
```

---

## 总结

### OBA 的核心贡献

1. **连接性分解**：将 Hessian 分解为三种可解释的部分
2. **直接计算 Hessian-向量积**：避免 Fisher 矩阵近似
3. **Attention 特殊处理**：显式计算 softmax Jacobian

### FastOBA 的改进

1. **任意阶泰勒展开**：从固定 2 阶到可配置 k 阶
2. **自动微分简化**：用 PyTorch 替代手动计算
3. **工程优化**：分布式、内存管理、数值稳定性

### 与 OBS 结合的价值

1. **理论保证**：OBS 有明确的损失变化估计
2. **权重补偿**：最小化剪枝误差
3. **针对 Attention**：可以专门优化 Attention 输出

### 实践建议

- **快速原型**：重构误差法
- **生产使用**：FastOBA（计算效率高）
- **研究探索**：OBA + OBS（理论完整）

---

## 参考代码位置

### OBA 核心代码
- 连接性分解：`oba_pruner.py:332-676`
- Attention 处理：`oba_pruner.py:575-657`
- 重要性计算：`oba_importance.py:28-110`

### FastOBA 核心代码
- 任意阶微分：`fastoba_pruner.py:288-316`
- 重要性获取：`fastoba_pruner.py:318-335`
- 更新累积：`fastoba_pruner.py:347-411`

### 依赖图
- 构建：`dependency.py:499-528`
- 下游节点选择：`metapruner.py:310-321`
- Head 识别：`metapruner.py:278-287`

---

**文档创建时间**：2025-11-02
**代码库版本**：OBA Main Branch (commit: 91e27b2)
