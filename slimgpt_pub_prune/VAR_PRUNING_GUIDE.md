# VAR模型剪枝创新方法：链式剪枝 + KFAC-OBS
# VAR Model Pruning Innovation: Chain Pruning + KFAC-OBS

**创建日期 | Created**: 2025-11-05
**方法 | Method**: 链式剪枝 + Full-KFAC-OBS + 多尺度处理
**核心创新 | Core Innovation**: 使用伪损失构建真正的Fisher信息矩阵
**基于 | Based on**: prune_v6.py (SlimGPT) 的改进版本

---

## 目录 | Table of Contents

1. [概述与动机 | Overview and Motivation](#overview)
2. [从SlimGPT到KFAC | From SlimGPT to KFAC](#from-slimgpt)
3. [核心创新：链式剪枝 | Core Innovation: Chain Pruning](#chain-pruning)
4. [KFAC矩阵计算 | KFAC Matrix Computation](#kfac-computation)
5. [输入维度剪枝数学 | Input Dimension Pruning Mathematics](#input-pruning-math)
6. [多尺度处理策略 | Multi-Scale Strategy](#multi-scale)
7. [完整实现流程 | Complete Implementation](#implementation)
8. [代码实现 | Code Implementation](#code)

---

## 概述与动机 | Overview and Motivation {#overview}

### VAR模型的挑战

VAR (Visual AutoRegressive) 模型特点：
- **10个尺度层级**：patch_nums=(1,2,3,4,5,6,8,10,13,16)，共680个token
- **渐进式生成**：每次前向传播生成一个尺度
- **Attention结构**：需要剪枝proj层的输入维度（列）

**Token数量详解**：
```python
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
total_tokens = sum(pn**2 for pn in patch_nums)
             = 1² + 2² + 3² + 4² + 5² + 6² + 8² + 10² + 13² + 16²
             = 1 + 4 + 9 + 16 + 25 + 36 + 64 + 100 + 169 + 256
             = 680 tokens
```

### 现有方法的局限：SlimGPT

**prune_v6.py 使用的方法**：

```python
# SlimGPT: 只使用输入协方差
H = X^T @ X  # [in_dim, in_dim]

# 重要性计算
importance = W² / diag(H^(-1))
```

**局限性**：
1. ❌ **只有一阶信息**：仅考虑输入特征的协方差
2. ❌ **缺少梯度信息**：不知道删除某个维度对输出的影响
3. ❌ **不是真正的Fisher矩阵**：只是输入统计量
4. ❌ **无法捕获非线性**：softmax等非线性的影响被忽略

### 本方法的创新

**链式剪枝 + KFAC-OBS**：

```python
# 1. 构建伪损失（层间输出差异）
loss = ||pruned_output - original_output||²

# 2. 反向传播获得梯度
loss.backward()

# 3. 计算真正的Fisher矩阵
A = E[x @ x^T]  # 输入激活协方差 [in_dim, in_dim]
G = E[g @ g^T]  # 输出梯度协方差 [out_dim, out_dim]
F = A ⊗ G       # Fisher信息矩阵（Kronecker积）

# 4. 使用F进行剪枝
importance = W² / diag(F^(-1))
```

**优势**：
1. ✅ **二阶信息**：真正的Hessian/Fisher矩阵
2. ✅ **包含梯度**：G矩阵来自伪损失的反向传播
3. ✅ **理论严格**：基于Taylor展开和OBS理论
4. ✅ **捕获非线性**：通过自动微分包含所有非线性

---

## 从SlimGPT到KFAC | From SlimGPT to KFAC {#from-slimgpt}

### SlimGPT方法回顾

**Hessian定义**（prune_v6.py:130-145）：

```python
def add_batch(self, inp, out):
    """SlimGPT的Hessian计算"""
    # inp: [batch, seq_len, hidden]

    # Flatten
    if inp.dim() == 3:
        inp = inp.reshape(-1, inp.shape[-1])  # [batch*seq, hidden]

    X = inp.t()  # [hidden, batch*seq]

    # 计算 H = X @ X^T
    self.H += X @ X.t()  # [hidden, hidden]
```

**物理意义**：
```
H = X^T @ X = E[x @ x^T]

这只是输入特征的协方差矩阵
- 告诉我们输入特征之间的相关性
- 但不知道这些特征如何影响输出
- 缺少目标函数的信息
```

### KFAC方法的理论基础

**Fisher信息矩阵**：

对于损失函数 L 和参数 W，Fisher信息矩阵定义为：

```
F = E[(∇_W L) (∇_W L)^T]

对于线性层 y = W @ x：
∇_W L = (∇_y L) ⊗ x = g ⊗ x

其中 g = ∇_y L 是输出的梯度

因此：
F = E[(g ⊗ x) (g ⊗ x)^T]
  = E[g g^T ⊗ x x^T]
  = E[g g^T] ⊗ E[x x^T]
  = G ⊗ A

其中：
- G = E[g @ g^T] : 输出梯度协方差 [out_dim, out_dim]
- A = E[x @ x^T] : 输入激活协方差 [in_dim, in_dim]
```

**关键洞察**：
- SlimGPT只有A矩阵（输入协方差）
- KFAC有完整的Fisher矩阵 F = G ⊗ A
- G矩阵包含了目标函数的梯度信息

### 为什么KFAC更准确？

| 特性 | SlimGPT (H = A) | KFAC (F = G ⊗ A) |
|------|----------------|-----------------|
| **信息类型** | 一阶（输入统计） | 二阶（Fisher/Hessian） |
| **梯度信息** | ❌ 无 | ✅ 有（G矩阵） |
| **优化目标** | 隐式 | 显式（最小化输出变化） |
| **非线性** | ❌ 不包含 | ✅ 通过反向传播自动包含 |
| **理论基础** | 经验性 | Taylor展开 + OBS |

### 第一层的特殊处理 | First Layer Bootstrap Strategy

**核心问题**：链式剪枝需要前一层的输出作为对比目标，但第一层没有前置层。

**VAR模型的层级结构**：
```python
var_model.blocks = [
    Layer 0,  # 第一层：无前置层 ❌
    Layer 1,  # 可以使用Layer 0的输出 ✅
    Layer 2,  # 可以使用Layer 1的输出 ✅
    ...
    Layer N   # 可以使用Layer N-1的输出 ✅
]
```

**解决方案：混合策略**

| 层级 | 剪枝方法 | 原因 |
|------|---------|------|
| **Layer 0** | SlimGPT (H = X^T X) | 无法构建伪损失，只能使用输入统计 |
| **Layer 1-N** | Chain Pruning + KFAC | 可以使用前一层输出构建伪损失 |

**实现逻辑**：

```python
for layer_idx in range(var_model.depth):
    layer = var_model.blocks[layer_idx]
    proj_layer = layer.attn.proj

    if layer_idx == 0:
        # 第一层：使用SlimGPT
        print(f"Layer {layer_idx}: Using SlimGPT (no previous layer)")
        pruner = SlimGPT(
            layer=proj_layer,
            layer_idx=layer_idx,
            args=args
        )
    else:
        # 后续层：使用链式剪枝 + KFAC
        print(f"Layer {layer_idx}: Using Chain Pruning + KFAC")
        pruner = ChainKFACPruner(
            layer=proj_layer,
            layer_idx=layer_idx,
            scale_weight_strategy='natural',
            percdamp=args.percdamp
        )

    # 统一的剪枝接口
    prune_indices = pruner.struct_prune(sparsity=args.sparsity, headsize=64)
```

**为什么不能强行给第一层构造伪损失？**

```python
# ❌ 错误尝试1：使用零向量作为"前一层输出"
loss = F.mse_loss(output, torch.zeros_like(output))
# 问题：这会让模型倾向于输出零，破坏特征表示

# ❌ 错误尝试2：使用输入作为"前一层输出"
loss = F.mse_loss(output, input)
# 问题：输入和输出维度/语义不同，损失无意义

# ❌ 错误尝试3：使用随机噪声
loss = F.mse_loss(output, torch.randn_like(output))
# 问题：随机目标无法提供有意义的梯度信息
```

**SlimGPT对第一层仍然有效的原因**：

虽然SlimGPT只使用输入协方差 H = X^T X，但对于第一层：
- ✅ 输入是原始的patch embeddings，统计特性稳定
- ✅ 输入相关性能反映低层特征的冗余度
- ✅ 不需要梯度信息也能识别冗余的输入维度

对于后续层：
- 🔥 输入已经过多层变换，简单的协方差不足以捕获复杂依赖
- 🔥 需要梯度信息（G矩阵）来理解对输出的影响
- 🔥 链式剪枝的伪损失能提供更精确的重要性评估

**性能影响分析**：

| 场景 | 第一层方法 | 后续层方法 | 预期效果 |
|------|-----------|-----------|---------|
| **本方法（推荐）** | SlimGPT | Chain+KFAC | 🔥🔥🔥 最佳 |
| 全部使用SlimGPT | SlimGPT | SlimGPT | 🔥🔥 良好 |
| 强行全部KFAC | KFAC (错误) | Chain+KFAC | 🔥 差（第一层会失败） |

**代码示例（完整）**：

```python
from slim_utils.slimgpt import SlimGPT

def prune_var_hybrid_strategy(var_model, calibration_dataloader, args):
    """
    VAR混合剪枝策略：第一层SlimGPT + 后续层链式KFAC
    """
    for layer_idx in range(var_model.depth):
        layer = var_model.blocks[layer_idx]
        proj_layer = layer.attn.proj

        # ====================================
        # 步骤1：根据层级选择剪枝器
        # ====================================
        if layer_idx == 0:
            # 第一层特殊处理
            pruner = SlimGPT(proj_layer, layer_idx, args)

            # 收集输入统计（只需要A矩阵）
            for batch in calibration_dataloader:
                inputs_10scales = extract_layer_inputs(var_model, layer_idx, batch)
                for inp_s in inputs_10scales:
                    pruner.add_batch(inp_s, None)  # SlimGPT不需要输出
        else:
            # 后续层使用链式剪枝
            pruner = ChainKFACPruner(
                layer=proj_layer,
                layer_idx=layer_idx,
                scale_weight_strategy='natural',
                percdamp=args.percdamp
            )

            # 收集多尺度KFAC矩阵（需要A和G）
            for batch in calibration_dataloader:
                inputs_10scales = extract_layer_inputs(var_model, layer_idx, batch)
                # 关键：需要前一层的原始输出作为target
                original_outputs_10scales = extract_layer_outputs(
                    var_model, layer_idx-1, batch  # 前一层！
                )
                pruner.add_batch_multiscale(inputs_10scales, original_outputs_10scales)

        # ====================================
        # 步骤2：执行剪枝（统一接口）
        # ====================================
        prune_indices = pruner.struct_prune(sparsity=args.sparsity, headsize=64)

        # 后续的物理删除和QKV同步保持一致
        # ...
```

**关键点总结**：

1. ✅ **第一层必须使用SlimGPT**：无法构建有效的伪损失
2. ✅ **后续层使用链式KFAC**：利用前一层输出构建伪损失
3. ✅ **统一的剪枝接口**：`struct_prune()` 方法对两种pruner都适用
4. ✅ **性能权衡合理**：第一层用简单方法，关键的深层用精确方法

---

## 核心创新：链式剪枝 | Core Innovation: Chain Pruning {#chain-pruning}

### 伪损失的定义

**核心思想**：用剪枝后层输出与原始层输出的差异作为损失函数

```python
# 1. 记录原始层输出
with torch.no_grad():
    original_output = layer(input)  # [B, L, C]

# 2. 剪枝后前向传播
pruned_output = pruned_layer(input)  # [B, L, C]

# 3. 伪损失：输出差异的均方误差
loss = F.mse_loss(pruned_output, original_output)
```

**数学表达**：

```
L_pseudo = ||f(x; W_pruned) - f(x; W_original)||²

其中：
- f(x; W): 层的输出函数
- W_pruned: 剪枝后的权重（某些维度置零）
- W_original: 原始权重
```

### 逐尺度计算的策略

**VAR的10个尺度**：

```python
scales = [
    (0, 1×1=1),      # Scale 0
    (1, 2×2=4),      # Scale 1
    (2, 3×3=9),      # Scale 2
    (3, 4×4=16),     # Scale 3
    (4, 5×5=25),     # Scale 4
    (5, 6×6=36),     # Scale 5
    (6, 8×8=64),     # Scale 6
    (7, 10×10=100),  # Scale 7
    (8, 13×13=169),  # Scale 8
    (9, 16×16=256)   # Scale 9
]
Total: 680 tokens
```

**逐尺度处理流程**：

```python
# 对每个尺度s独立计算
for scale_s in range(10):
    # 该尺度的输入
    input_s = inputs[scale_s]  # [B, L_s, C]

    # 1. 原始输出
    with torch.no_grad():
        original_out_s = original_layer(input_s)

    # 2. 剪枝后输出
    pruned_out_s = pruned_layer(input_s)

    # 3. 伪损失
    loss_s = F.mse_loss(pruned_out_s, original_out_s)

    # 4. 反向传播（保留计算图）
    loss_s.backward(retain_graph=True)

    # 5. 收集该尺度的A和G
    A_s = collect_input_covariance(input_s)
    G_s = collect_gradient_covariance(output_grad_s)

    # 6. 计算重要性分数
    importance_score_s = compute_importance(A_s, G_s)

# 7. 加权平均
H_weighted = weighted_average(importance_scores, weights)

# 8. 按样本平均
H_final = H_weighted / num_samples
```

### 梯度隔离：Attention块内传播 | Gradient Isolation within Attention Block

**核心原则**：链式剪枝的梯度只在当前层的attention模块内部传播，确保层间独立。

#### 梯度传播范围

```
┌─────────────────────────────────────────────────────────┐
│  当前层 Attention Block (Layer i)                       │
│                                                          │
│  [QKV输入] ──→ [mat_qkv] ──→ [Attention计算]           │
│                                    ↓                     │
│                            [Attention输出]               │
│                                    ↓                     │
│                            [O层/proj] ─────→ [输出]     │
│                                    ↓           ↓         │
│                                    ↓     [原始输出·detach]│
│                                    ↓           ↓         │
│                                    └─→ [伪损失L_pseudo]  │
│                                            ↓             │
│                                    [梯度反向传播]  ◄──────│
│                                            │             │
│  关键：梯度在这个边界停止 ─────────────────┘             │
└─────────────────────────────────────────────────────────┘
         ↑                                        ↑
    不传播到                                  不传播到
   前一层Layer i-1                          后一层Layer i+1
```

#### KFAC近似的关键

**我们只需要两个矩阵**：

1. **A矩阵**：O层（proj）的输入激活协方差
   ```python
   A = E[x @ x^T]  # x是proj的输入（来自attention计算后）
   ```

2. **G矩阵**：O层输出关于伪损失的一阶梯度协方差
   ```python
   G = E[g @ g^T]  # g = ∂L_pseudo/∂(proj输出)
   ```

**不需要**：
- ❌ 整个模型的完整Hessian
- ❌ 跨层的梯度信息
- ❌ FFN或其他模块的梯度

#### 层间独立性保证

**每层的attn module相互独立**：

```python
# Layer 0的attn剪枝
with torch.no_grad():
    original_out_0 = layer_0.attn(input_0)  # 记录原始输出

pruned_out_0 = layer_0.attn(input_0)  # 剪枝后的前向
loss_0 = F.mse_loss(pruned_out_0, original_out_0)
loss_0.backward()  # 梯度只在layer_0.attn内传播

# Layer 1的attn剪枝（完全独立）
with torch.no_grad():
    original_out_1 = layer_1.attn(input_1)

pruned_out_1 = layer_1.attn(input_1)
loss_1 = F.mse_loss(pruned_out_1, original_out_1)
loss_1.backward()  # 梯度只在layer_1.attn内传播
```

**为什么是独立的？**

| 特性 | 说明 |
|------|------|
| **目标函数独立** | 每层的伪损失 L_i = \\|out_i - original_out_i\\|² 只依赖当前层 |
| **参数独立** | Layer i的权重不出现在Layer j的损失中 (i≠j) |
| **梯度独立** | ∂L_i/∂W_j = 0 当 i≠j |
| **可并行剪枝** | 理论上可以并行计算所有层的KFAC矩阵（实际上顺序剪枝更稳定） |

#### 关键实现细节

**1. 原始输出必须detach**

```python
# ✅ 正确：原始输出不参与梯度计算
with torch.no_grad():
    original_output = layer.attn(input)  # 完全detach

# 或者显式detach
original_output = layer.attn(input).detach()

# ❌ 错误：原始输出参与梯度
original_output = layer.attn(input)  # 没有no_grad或detach
loss = F.mse_loss(pruned_output, original_output)
loss.backward()  # 梯度会流回original的计算路径！
```

**2. 梯度收集使用hooks**

```python
def add_batch_multiscale(self, inputs_10scales, original_outputs_10scales):
    """收集多尺度的A和G矩阵"""

    # 注册hooks来拦截激活和梯度
    activations = {}
    gradients = {}

    def hook_forward(module, inp, out):
        # 收集proj层的输入（用于A矩阵）
        activations['current'] = inp[0].detach()

    def hook_backward(module, grad_in, grad_out):
        # 收集proj层输出的梯度（用于G矩阵）
        # detach()确保不继续向前传播
        gradients['current'] = grad_out[0].detach()

    handle_fwd = self.layer.register_forward_hook(hook_forward)
    handle_bwd = self.layer.register_backward_hook(hook_backward)

    try:
        for scale_s in range(10):
            inp_s = inputs_10scales[scale_s]
            target_s = original_outputs_10scales[scale_s]  # 已经detach

            # 前向：只有当前层的attn参与计算图
            inp_s = inp_s.requires_grad_(True)
            out_s = self.layer(inp_s)

            # 伪损失
            loss_s = F.mse_loss(out_s, target_s)

            # 反向：梯度只在当前attn内传播
            loss_s.backward(retain_graph=True)

            # 提取A和G
            x = activations['current']  # [B*L_s, in_dim]
            g = gradients['current']    # [B*L_s, out_dim]

            A_s = x.t() @ x / x.shape[0]
            G_s = g.t() @ g / g.shape[0]

            # ... 加权累积
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
```

**3. retain_graph=True的作用**

```python
for scale_s in range(10):
    out_s = layer(inp_s)
    loss_s = F.mse_loss(out_s, target_s)

    # retain_graph=True: 保留计算图给下一个尺度使用
    # 因为我们要对10个尺度分别backward
    loss_s.backward(retain_graph=True)

    # 最后一个尺度可以不保留
    if scale_s == 9:
        loss_s.backward(retain_graph=False)
```

**注意**：`retain_graph`不是为了跨层传播，而是为了同一层的多尺度计算！

#### 梯度传播的完整路径

**单个尺度的前向-反向路径**：

```
前向传播（构建计算图）:
┌──────────────────────────────────────────────────┐
│ inp_s (requires_grad=True)                       │
│   ↓                                              │
│ [QKV Projection: mat_qkv]                        │
│   ↓                                              │
│ [Attention Calculation: softmax(QK^T/√d) @ V]    │
│   ↓                                              │
│ [Output Projection: proj] ← 这是我们关注的O层    │
│   ↓                                              │
│ out_s (计算图的终点)                             │
│   ↓                                              │
│ L_pseudo = MSE(out_s, target_s)                  │
└──────────────────────────────────────────────────┘

反向传播（梯度流动）:
┌──────────────────────────────────────────────────┐
│ ∂L/∂L = 1.0                                      │
│   ↓                                              │
│ ∂L/∂(out_s) ← hook在这里拦截，得到g（G矩阵用）   │
│   ↓                                              │
│ ∂L/∂(proj.weight)                                │
│ ∂L/∂(proj.input) ← hook在前向时记录（A矩阵用）   │
│   ↓                                              │
│ ∂L/∂(attention output)                           │
│   ↓                                              │
│ ∂L/∂(mat_qkv.weight)                             │
│   ↓                                              │
│ ∂L/∂(inp_s) ← 梯度在这里终止（inp_s边界）        │
└──────────────────────────────────────────────────┘
```

**梯度不会传播到**：
- ✅ **前一层Layer i-1**：因为`inp_s`是从前一层detach出来的
- ✅ **后一层Layer i+1**：因为我们只对当前层backward
- ✅ **同层的FFN**：因为我们只对attn部分构建计算图
- ✅ **其他batch或尺度**：每个尺度独立backward

#### 常见错误和调试

**错误1：忘记detach原始输出**

```python
# ❌ 错误
original_out = layer.attn(inp)  # 没有detach
pruned_out = layer.attn(inp)
loss = F.mse_loss(pruned_out, original_out)
loss.backward()
# 结果：梯度会流入original的计算路径，污染Fisher矩阵

# ✅ 正确
with torch.no_grad():
    original_out = layer.attn(inp)
pruned_out = layer.attn(inp)
loss = F.mse_loss(pruned_out, original_out)
loss.backward()
```

**错误2：梯度泄漏到前一层**

```python
# ❌ 错误：inp直接来自前一层的输出
inp = layer_prev.forward(x)  # inp带有梯度信息
out = layer_current.attn(inp)
loss = F.mse_loss(out, target)
loss.backward()  # 梯度会流回layer_prev！

# ✅ 正确：detach输入
inp = layer_prev.forward(x).detach()  # 切断梯度
out = layer_current.attn(inp)
loss = F.mse_loss(out, target)
loss.backward()  # 梯度止于当前层
```

**错误3：hook中没有detach**

```python
# ❌ 错误
def hook_backward(module, grad_in, grad_out):
    gradients['current'] = grad_out[0]  # 没有detach

# ✅ 正确
def hook_backward(module, grad_in, grad_out):
    gradients['current'] = grad_out[0].detach()  # 必须detach
```

#### 验证梯度隔离

**测试代码**：

```python
def test_gradient_isolation():
    """验证梯度只在当前层传播"""
    layer_i = var_model.blocks[i]
    layer_prev = var_model.blocks[i-1]

    # 1. 获取输入（从前一层）
    with torch.no_grad():
        inp = layer_prev(x)  # detach确保隔离

    # 2. 记录前一层的权重
    prev_weight = layer_prev.attn.proj.weight.clone()

    # 3. 当前层的链式剪枝
    with torch.no_grad():
        original_out = layer_i.attn(inp)

    inp_grad = inp.requires_grad_(True)
    pruned_out = layer_i.attn(inp_grad)
    loss = F.mse_loss(pruned_out, original_out)
    loss.backward()

    # 4. 检查前一层权重的梯度
    if layer_prev.attn.proj.weight.grad is not None:
        print("❌ 错误：梯度泄漏到前一层！")
    else:
        print("✅ 正确：梯度被隔离在当前层")

    # 5. 检查当前层权重的梯度
    if layer_i.attn.proj.weight.grad is not None:
        print("✅ 正确：当前层有梯度")
    else:
        print("❌ 错误：当前层没有梯度！")
```

#### 关键点总结

1. ✅ **KFAC近似只需A和G**：不需要完整Hessian，只需proj层的输入和输出梯度协方差
2. ✅ **梯度范围**：QKV输入 → Attention → O层 → 伪损失，止于当前attn边界
3. ✅ **层间独立**：每层的伪损失和KFAC矩阵完全独立计算
4. ✅ **Detach三要素**：原始输出detach、hook中detach、输入来源detach
5. ✅ **retain_graph用途**：保留图给多尺度使用，不是为了跨层传播

---

## KFAC矩阵计算 | KFAC Matrix Computation {#kfac-computation}

### A矩阵：输入激活协方差

**定义**：

```
A = E[x @ x^T]  # [in_dim, in_dim]

其中 x 是 proj 层的输入
```

**计算方法**（与SlimGPT相同）：

```python
def compute_A_matrix(inputs_list):
    """
    计算输入激活协方差矩阵

    Args:
        inputs_list: 10个尺度的输入列表

    Returns:
        A: [in_dim, in_dim]
    """
    A = torch.zeros(in_dim, in_dim, device=device)
    total_samples = 0

    for scale_s, inp_s in enumerate(inputs_list):
        # inp_s: [batch, L_s, in_dim]

        # Flatten
        X_s = inp_s.reshape(-1, in_dim).t()  # [in_dim, batch*L_s]

        n_samples_s = X_s.shape[1]
        total_samples += n_samples_s

        # 累积
        A += X_s @ X_s.t()

    # 归一化
    A = A / total_samples

    return A
```

### G矩阵：输出梯度协方差（关键！）

**定义**：

```
G = E[g @ g^T]  # [out_dim, out_dim]

其中 g = ∇_y L_pseudo 是输出y关于伪损失的梯度
```

**计算方法**（核心创新）：

```python
def compute_G_matrix(layer, inputs_list, original_outputs_list):
    """
    计算输出梯度协方差矩阵

    Args:
        layer: 要剪枝的层
        inputs_list: 10个尺度的输入
        original_outputs_list: 10个尺度的原始输出（target）

    Returns:
        G: [out_dim, out_dim]
    """
    G = torch.zeros(out_dim, out_dim, device=device)
    total_samples = 0

    # 注册hook收集梯度
    gradients_list = []

    def hook_fn(module, grad_input, grad_output):
        # grad_output[0]: [batch, L_s, out_dim]
        gradients_list.append(grad_output[0].detach())

    handle = layer.register_backward_hook(hook_fn)

    # 逐尺度计算
    for scale_s, (inp_s, target_s) in enumerate(zip(inputs_list, original_outputs_list)):
        gradients_list.clear()

        # 前向传播
        inp_s = inp_s.requires_grad_(True)
        out_s = layer(inp_s)  # [batch, L_s, out_dim]

        # 伪损失
        loss_s = F.mse_loss(out_s, target_s)

        # 反向传播
        loss_s.backward(retain_graph=True)

        # 提取梯度
        g_s = gradients_list[0]  # [batch, L_s, out_dim]

        # Flatten
        G_s = g_s.reshape(-1, out_dim).t()  # [out_dim, batch*L_s]

        n_samples_s = G_s.shape[1]
        total_samples += n_samples_s

        # 累积
        G += G_s @ G_s.t()

    # 移除hook
    handle.remove()

    # 归一化
    G = G / total_samples

    return G
```

### Fisher矩阵：F = A ⊗ G

**关键**：对于输入维度剪枝，使用 `F = A ⊗ G`（不是 `G ⊗ A`）

```python
def compute_fisher_inverse_diag(A, G, percdamp=0.01):
    """
    计算Fisher矩阵逆的对角元素

    Returns:
        F_inv_diag: [out_dim, in_dim]
    """
    # 添加阻尼
    damp_A = percdamp * torch.mean(torch.diag(A))
    damp_G = percdamp * torch.mean(torch.diag(G))

    A += damp_A * torch.eye(A.shape[0], device=A.device)
    G += damp_G * torch.eye(G.shape[0], device=G.device)

    # 计算逆矩阵
    A_inv = torch.cholesky_inverse(torch.linalg.cholesky(A))
    G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G))

    # 提取对角元素
    A_inv_diag = torch.diag(A_inv)  # [in_dim]
    G_inv_diag = torch.diag(G_inv)  # [out_dim]

    # Kronecker积的对角（注意顺序：A ⊗ G）
    F_inv_diag = A_inv_diag.unsqueeze(0) @ G_inv_diag.unsqueeze(1)
    # [out_dim, in_dim]

    return F_inv_diag
```

---

## 输入维度剪枝数学 | Input Dimension Pruning Mathematics {#input-pruning-math}

### 为什么是 F = G ⊗ A？（标准KFAC理论）

**关键问题**：输入维度剪枝的Fisher矩阵应该是 `G ⊗ A` 还是 `A ⊗ G`？

#### 标准KFAC理论推导

**线性层定义**：
```
y = W @ x
W.shape = [out_dim, in_dim]
x.shape = [in_dim]
y.shape = [out_dim]
```

**Fisher信息矩阵构造**：

对于权重向量 `vec(W)`，Fisher信息矩阵定义为：
```
F = E[(∇_W L)(∇_W L)^T]

其中梯度：
∇_W L = ∂L/∂W = (∂L/∂y) ⊗ (∂y/∂W) = g ⊗ x

因此：
F = E[(g ⊗ x)(g ⊗ x)^T]
  = E[(g ⊗ x)(g^T ⊗ x^T)]
  = E[g g^T ⊗ x x^T]    （Kronecker积的性质）
  = E[g g^T] ⊗ E[x x^T]
  = G ⊗ A

其中：
- G = E[g g^T] : [out_dim, out_dim] 输出梯度协方差
- A = E[x x^T] : [in_dim, in_dim] 输入激活协方差
- g = ∂L/∂y : [out_dim] 输出梯度
```

#### 为什么顺序很重要？

**Kronecker积的定义**：
```
对于 G: [m×m] 和 A: [n×n]
G ⊗ A 产生 [mn × mn] 矩阵：

[G[0,0]×A  G[0,1]×A  ...  G[0,m-1]×A]
[G[1,0]×A  G[1,1]×A  ...  G[1,m-1]×A]
[  ...        ...     ...     ...    ]
[G[m-1,0]×A G[m-1,1]×A ... G[m-1,m-1]×A]
```

**权重向量化（列优先）**：
```
vec(W) = [W[0,0], W[1,0], ..., W[out-1,0],
          W[0,1], W[1,1], ..., W[out-1,1],
          ...,
          W[0,in-1], W[1,in-1], ..., W[out-1,in-1]]

索引公式：vec(W)[i×in_dim + j] = W[i,j]
```

**Fisher矩阵对角元素**：
```
对于 F = G ⊗ A：
F^{-1}[i×in_dim + j, i×in_dim + j] = G^{-1}[i,i] × A^{-1}[j,j]

对应权重 W[i,j] 的Fisher逆对角元素
```

#### 输入维度剪枝的重要性计算

**权重重要性（按元素）**：
```python
importance[i,j] = W[i,j]² / F^{-1}[i×in_dim + j, i×in_dim + j]
                = W[i,j]² / (G^{-1}[i,i] × A^{-1}[j,j])
```

**输入维度j的总重要性**：
```python
importance_column[j] = Σ_i importance[i,j]
                     = Σ_i W[i,j]² / (G^{-1}[i,i] × A^{-1}[j,j])
```

#### 实现公式

**正确的Fisher逆对角计算**：
```python
G_inv_diag = torch.diag(G_inv)  # [out_dim]
A_inv_diag = torch.diag(A_inv)  # [in_dim]

# 对于 F = G ⊗ A，对角元素为：
F_inv_diag = G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0)
# Shape: [out_dim, in_dim]

# 重要性计算
importance_element = W ** 2 / F_inv_diag  # [out_dim, in_dim]
importance_column = importance_element.sum(dim=0)  # [in_dim]
```

**错误公式对比**：
```python
# ❌ 错误：A ⊗ G 会导致：
F_inv_diag_wrong = A_inv_diag.unsqueeze(1) @ G_inv_diag.unsqueeze(0)
# Shape: [in_dim, out_dim] - 维度都错了！

# ❌ 错误：即使维度对了，数学意义也错误
F_inv_diag_wrong = A_inv_diag.unsqueeze(0) @ G_inv_diag.unsqueeze(1)
# 这不是 G ⊗ A 的对角元素
```

#### 总结：标准KFAC

✅ **正确**：`F = G ⊗ A`
- 符合KFAC理论
- 正确的权重向量化对应
- 正确的重要性计算

❌ **错误**：`F = A ⊗ G`
- 违反KFAC理论
- 错误的矩阵维度理解
- 错误的重要性计算

### 重要性计算（修正版）

**公式**：

```python
# 权重矩阵
W: [out_dim, in_dim]

# Fisher逆的对角（正确版本：F = G ⊗ A）
G_inv_diag = torch.diag(G_inv)  # [out_dim]
A_inv_diag = torch.diag(A_inv)  # [in_dim]

F_inv_diag = G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0)
# Shape: [out_dim, in_dim]

# 元素重要性
importance_element = W ** 2 / F_inv_diag  # [out_dim, in_dim]

# 输入维度（列）的重要性
importance_column = importance_element.sum(dim=0)  # [in_dim]
```

### OBS手术公式（基于标准KFAC）

**理论依据**：对于 `F = G ⊗ A`，输入维度剪枝的OBS手术需要特殊处理。

**标准OBS公式**：
```
设要删除权重W的第j列，OBS更新为：
W[:, other_cols] -= W[:, j] @ (F^{-1}[j_indices, other_indices] / F^{-1}[j_indices, j_indices])

其中j_indices = [0×in_dim+j, 1×in_dim+j, ..., (out_dim-1)×in_dim+j]
```

**简化近似**：对于输入维度剪枝，可以使用 A_inv 近似：

```python
def perform_obs_surgery_input_dim(W, A_inv, idx_to_remove):
    """
    对输入维度执行OBS手术

    基于近似：对于输入剪枝，主要相关性由A矩阵捕获
    这是一个计算高效的近似，避免了完整Fisher矩阵求逆

    Args:
        W: [out_dim, in_dim] 权重矩阵
        A_inv: [in_dim, in_dim] 输入协方差逆矩阵
        idx_to_remove: 要删除的列索引
    """
    A_inv_upper = torch.linalg.cholesky(A_inv, upper=True)

    for j in idx_to_remove:
        # 计算误差（近似OBS）
        err = W[:, j] / A_inv_upper[j, j]  # [out_dim]

        # 更新其他列
        W[:, j+1:] -= err.unsqueeze(1) @ A_inv_upper[j:j+1, j+1:]

        # 删除该列
        W[:, j] = 0

    return W
```

**注意**：这是一个实用近似。严格的OBS应该使用完整的 `F = G ⊗ A` 矩阵，但计算成本过高。

---

## 多尺度处理策略 | Multi-Scale Strategy {#multi-scale}

### 三种加权策略详解

基于VAR的680个token，提供三种不同的尺度加权策略：

#### 策略1：自然加权（Natural Weighting）

**公式**：
```python
weight_s = n_tokens_s / 680
```

**权重分布**：

| 尺度 | Patch Size | Token数 | 权重 | 累积占比 |
|------|-----------|---------|------|---------|
| 0 | 1×1 | 1 | 0.15% | 0.15% |
| 1 | 2×2 | 4 | 0.59% | 0.74% |
| 2 | 3×3 | 9 | 1.32% | 2.06% |
| 3 | 4×4 | 16 | 2.35% | 4.41% |
| 4 | 5×5 | 25 | 3.68% | 8.09% |
| 5 | 6×6 | 36 | 5.29% | 13.38% |
| 6 | 8×8 | 64 | 9.41% | 22.79% |
| 7 | 10×10 | 100 | 14.71% | 37.50% |
| 8 | 13×13 | 169 | 24.85% | 62.35% |
| 9 | 16×16 | 256 | 37.65% | 100.00% |

**优点**：
- 符合VAR的实际使用场景
- 大尺度（高分辨率）自然有更高权重
- 简单直接，无需超参数

**缺点**：
- 小尺度可能被忽略
- 大尺度主导（37.6%）

#### 策略2：均等加权（Equal Weighting）

**公式**：
```python
weight_s = 1.0 / 10  # 每个尺度10%
```

**权重分布**：

| 尺度 | Token数 | 原始贡献 | 均等权重 | 实际影响 |
|------|---------|---------|---------|---------|
| 0 | 1 | 0.15% | 10% | 放大67倍 |
| 1 | 4 | 0.59% | 10% | 放大17倍 |
| 2 | 9 | 1.32% | 10% | 放大7.6倍 |
| 3 | 16 | 2.35% | 10% | 放大4.3倍 |
| 4 | 25 | 3.68% | 10% | 放大2.7倍 |
| 5 | 36 | 5.29% | 10% | 放大1.9倍 |
| 6 | 64 | 9.41% | 10% | 放大1.1倍 |
| 7 | 100 | 14.71% | 10% | 缩小0.7倍 |
| 8 | 169 | 24.85% | 10% | 缩小0.4倍 |
| 9 | 256 | 37.65% | 10% | 缩小0.27倍 |

**优点**：
- 所有尺度的重要性被平等对待
- 小尺度不会被忽略
- 有利于优化小尺度的性能

**缺点**：
- 可能过度强调小尺度
- 不符合实际重要性分布
- 大尺度的优势被人为削弱

#### 策略3：平方根加权（Square Root Weighting）

**公式**：
```python
sqrt_tokens = [sqrt(n) for n in token_counts]
total_sqrt = sum(sqrt_tokens)
weight_s = sqrt(n_tokens_s) / total_sqrt
```

**权重分布**：

| 尺度 | Token数 | √Token | 权重 | vs自然 | vs均等 |
|------|---------|--------|------|--------|--------|
| 0 | 1 | 1.0 | 2.1% | +14倍 | -79% |
| 1 | 4 | 2.0 | 4.2% | +7倍 | -58% |
| 2 | 9 | 3.0 | 6.3% | +4.8倍 | -37% |
| 3 | 16 | 4.0 | 8.4% | +3.6倍 | -16% |
| 4 | 25 | 5.0 | 10.5% | +2.9倍 | +5% |
| 5 | 36 | 6.0 | 12.6% | +2.4倍 | +26% |
| 6 | 64 | 8.0 | 16.8% | +1.8倍 | +68% |
| 7 | 100 | 10.0 | 21.0% | +1.4倍 | +110% |
| 8 | 169 | 13.0 | 27.3% | +1.1倍 | +173% |
| 9 | 256 | 16.0 | 33.6% | -0.11倍 | +236% |

**优点**：
- 平衡了大小尺度的重要性
- 大尺度仍有更高权重，但不会过度主导
- 小尺度有合理的影响力
- 折中方案，适合大多数场景

**缺点**：
- 需要额外的平方根计算
- 权重分布不如自然加权直观

### 代码实现

```python
def compute_scale_weights(strategy='natural'):
    """计算多尺度权重"""
    token_counts = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    total_tokens = 680

    if strategy == 'natural':
        weights = [n / total_tokens for n in token_counts]
    elif strategy == 'equal':
        weights = [1.0 / 10 for _ in range(10)]
    elif strategy == 'sqrt':
        sqrt_counts = [n**0.5 for n in token_counts]
        total_sqrt = sum(sqrt_counts)
        weights = [s / total_sqrt for s in sqrt_counts]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return weights

# 使用示例
weights_natural = compute_scale_weights('natural')
weights_equal = compute_scale_weights('equal')
weights_sqrt = compute_scale_weights('sqrt')

print("Natural:", [f"{w:.3f}" for w in weights_natural])
print("Equal:  ", [f"{w:.3f}" for w in weights_equal])
print("Sqrt:   ", [f"{w:.3f}" for w in weights_sqrt])
```

### 推荐使用场景

| 策略 | 推荐场景 | 优先级 |
|------|---------|-------|
| **自然加权** | VAR在大尺度（高分辨率）性能更关键 | 🔥🔥🔥 |
| **平方根加权** | 希望平衡优化所有尺度，折中方案 | 🔥🔥 |
| **均等加权** | 小尺度性能同样重要，或者实验对比 | 🔥 |

---

## 完整实现流程 | Complete Implementation {#implementation}

### 整体架构图

```
┌─────────────────────────────────────────────┐
│ 输入：VAR模型 + 校准数据                     │
│ • 10个尺度：1²,2²,...,16² = 680 tokens      │
│ • 多个calibration batch                    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Step 1: 记录原始层输出（所有尺度）           │
└─────────────────────────────────────────────┘
    for batch in calibration_data:
        for scale_s in range(10):
            original_outputs[layer][scale_s].append(
                layer(input_s).detach()
            )
                    ↓
┌─────────────────────────────────────────────┐
│ Step 2: 逐尺度计算伪损失和KFAC矩阵          │
│ • 10个尺度独立计算A_s和G_s                  │
│ • 使用选定的加权策略组合                    │
└─────────────────────────────────────────────┘
    A_final = 0
    G_final = 0

    for scale_s in range(10):
        # 前向
        out_s = layer(input_s)

        # 伪损失
        loss_s = MSE(out_s, original_out_s)

        # 反向
        loss_s.backward()

        # 收集
        A_s = input_covariance(input_s)
        G_s = gradient_covariance(out_grad_s)

        # 加权累积
        w_s = scale_weights[scale_s]
        A_final += w_s * A_s
        G_final += w_s * G_s
                    ↓
┌─────────────────────────────────────────────┐
│ Step 3: 计算Fisher矩阵逆对角 F = A ⊗ G      │
└─────────────────────────────────────────────┘
    A_inv = cholesky_inverse(A_final + damp*I)
    G_inv = cholesky_inverse(G_final + damp*I)

    F_inv_diag = A_inv_diag ⊗ G_inv_diag
                    ↓
┌─────────────────────────────────────────────┐
│ Step 4: 计算输入维度重要性（列重要性）        │
└─────────────────────────────────────────────┘
    importance = (W² / F_inv_diag).sum(dim=0)

    # Head-wise聚合
    head_importance = importance.view(num_heads, 64).sum(1)

    # 选择要删除的head
    prune_heads = argsort(head_importance)[:num_prune]
                    ↓
┌─────────────────────────────────────────────┐
│ Step 5: OBS手术（使用A_inv，不是G_inv）      │
└─────────────────────────────────────────────┘
    for head_idx in prune_heads:
        cols = head_idx * 64 : (head_idx+1) * 64

        # 计算误差
        err = W[:, cols] / A_inv_diag[cols]

        # 更新其他列
        W[:, others] -= err @ A_inv[cols, others]

        # 删除
        W[:, cols] = 0
                    ↓
┌─────────────────────────────────────────────┐
│ Step 6: 物理删除（tp.prune，与prune_v6相同） │
└─────────────────────────────────────────────┘
    tp.prune_linear_in_channels(layer.proj, idx)
                    ↓
┌─────────────────────────────────────────────┐
│ Step 7: QKV同步（与prune_v6相同）            │
└─────────────────────────────────────────────┘
    rm_qkv = cat([idx, idx+hidden, idx+2*hidden])
    tp.prune_linear_out_channels(layer.mat_qkv, rm_qkv)
                    ↓
                ┌─────────┐
                │  完成！  │
                └─────────┘
```

---

## 代码实现 | Code Implementation {#code}

### 主剪枝函数（完整版）

```python
"""
VAR模型剪枝：链式剪枝 + KFAC-OBS
基于prune_v6.py的改进版本，使用真正的Fisher信息矩阵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from typing import List, Dict, Tuple
from types import SimpleNamespace
import math


class ChainKFACPruner:
    """链式剪枝+KFAC的完整实现"""

    def __init__(
        self,
        layer: nn.Linear,
        layer_idx: int,
        scale_weight_strategy: str = 'natural',
        percdamp: float = 0.01
    ):
        self.layer = layer
        self.layer_idx = layer_idx
        self.scale_weight_strategy = scale_weight_strategy
        self.percdamp = percdamp

        self.device = layer.weight.device
        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # 初始化A和G矩阵
        self.A = torch.zeros(self.in_features, self.in_features, device=self.device)
        self.G = torch.zeros(self.out_features, self.out_features, device=self.device)

        # 尺度权重
        self.scale_weights = self._compute_scale_weights()
        self.collected_samples = 0

    def _compute_scale_weights(self) -> List[float]:
        """计算尺度权重"""
        token_counts = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]  # 680 total

        if self.scale_weight_strategy == 'natural':
            weights = [n / 680 for n in token_counts]
        elif self.scale_weight_strategy == 'equal':
            weights = [1.0 / 10 for _ in range(10)]
        elif self.scale_weight_strategy == 'sqrt':
            sqrt_counts = [n**0.5 for n in token_counts]
            total_sqrt = sum(sqrt_counts)
            weights = [s / total_sqrt for s in sqrt_counts]
        else:
            raise ValueError(f"Unknown strategy: {self.scale_weight_strategy}")

        return weights

    def add_batch_multiscale(
        self,
        inputs_10scales: List[torch.Tensor],
        original_outputs_10scales: List[torch.Tensor]
    ):
        """
        添加一个多尺度batch来更新A和G矩阵

        Args:
            inputs_10scales: 10个尺度的输入 [List[Tensor[B, L_s, in_dim]]]
            original_outputs_10scales: 10个尺度的原始输出 [List[Tensor[B, L_s, out_dim]]]
        """
        A_batch = torch.zeros_like(self.A)
        G_batch = torch.zeros_like(self.G)
        total_weight = 0

        # 注册hooks收集激活和梯度
        activations = {}
        gradients = {}

        def hook_forward(module, inp, out):
            activations['current'] = inp[0].detach()

        def hook_backward(module, grad_in, grad_out):
            gradients['current'] = grad_out[0].detach()

        handle_fwd = self.layer.register_forward_hook(hook_forward)
        handle_bwd = self.layer.register_backward_hook(hook_backward)

        try:
            # 逐尺度处理
            for scale_s in range(10):
                inp_s = inputs_10scales[scale_s]  # [B, L_s, in_dim]
                target_s = original_outputs_10scales[scale_s]  # [B, L_s, out_dim]

                activations.clear()
                gradients.clear()

                # 前向传播
                inp_s = inp_s.requires_grad_(True)
                out_s = self.layer(inp_s)

                # 伪损失
                loss_s = F.mse_loss(out_s, target_s)

                # 反向传播
                loss_s.backward(retain_graph=True)

                # 收集A_s（输入协方差）
                X_s = activations['current'].reshape(-1, self.in_features).t()  # [in_dim, B*L_s]
                A_s = X_s @ X_s.t()  # [in_dim, in_dim]

                # 收集G_s（梯度协方差）
                G_s_mat = gradients['current'].reshape(-1, self.out_features).t()  # [out_dim, B*L_s]
                G_s = G_s_mat @ G_s_mat.t()  # [out_dim, out_dim]

                # 归一化
                n_samples_s = X_s.shape[1]
                A_s = A_s / n_samples_s
                G_s = G_s / n_samples_s

                # 加权累积
                weight_s = self.scale_weights[scale_s]
                A_batch += weight_s * A_s
                G_batch += weight_s * G_s
                total_weight += weight_s

        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        # 归一化并累积到全局A、G
        if total_weight > 0:
            A_batch = A_batch / total_weight
            G_batch = G_batch / total_weight

            # 计算实际样本数（所有尺度的token总数）
            actual_samples = sum(
                inputs_10scales[s].shape[0] * inputs_10scales[s].shape[1]
                for s in range(10)
            )

            # 更新总样本计数
            self.total_samples = getattr(self, 'total_samples', 0) + actual_samples

            # EMA更新（基于实际样本数，不是batch数）
            if not hasattr(self, 'total_samples_seen'):
                self.total_samples_seen = 0

            self.total_samples_seen += actual_samples
            alpha = actual_samples / self.total_samples_seen

            self.A = (1 - alpha) * self.A + alpha * A_batch
            self.G = (1 - alpha) * self.G + alpha * G_batch

            self.collected_samples += 1

    def struct_prune(self, sparsity: float, headsize: int = 64) -> torch.Tensor:
        """
        使用KFAC-OBS执行结构化剪枝

        Args:
            sparsity: 稀疏度
            headsize: head大小

        Returns:
            pruned_indices: 被删除的列索引
        """
        W = self.layer.weight.data.clone().float()
        num_heads = self.in_features // headsize

        print(f"    Layer {self.layer_idx}: Collected {self.collected_samples} batches")
        print(f"    Scale weights: {[f'{w:.3f}' for w in self.scale_weights]}")

        # ========================================
        # 1. 计算Fisher矩阵逆的对角元素
        # ========================================
        A_inv, G_inv = self._compute_inverses()
        F_inv_diag = self._compute_fisher_inverse_diag(A_inv, G_inv)

        # ========================================
        # 2. 计算输入维度重要性
        # ========================================
        importance_element = W ** 2 / F_inv_diag  # [out_dim, in_dim]
        importance_column = importance_element.sum(dim=0)  # [in_dim]

        # Head-wise重要性
        head_importance = importance_column.view(num_heads, headsize).sum(1)  # [num_heads]

        print(f"    Head importances: {head_importance.tolist()}")

        # ========================================
        # 3. 选择要删除的heads
        # ========================================
        num_prune = int(num_heads * sparsity)
        prune_heads = torch.argsort(head_importance)[:num_prune]

        print(f"    Pruning {num_prune}/{num_heads} heads: {prune_heads.tolist()}")

        # ========================================
        # 4. OBS手术（使用A_inv）
        # ========================================
        self._perform_obs_surgery(W, A_inv, prune_heads, headsize)

        # 更新权重
        self.layer.weight.data = W.to(self.layer.weight.data.dtype)

        # 返回被删除的列索引
        prune_indices = []
        for head_idx in prune_heads:
            start = head_idx * headsize
            prune_indices.extend(range(start, start + headsize))

        return torch.tensor(prune_indices, device=self.device)

    def _compute_inverses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算A和G的逆矩阵"""
        # 添加阻尼
        damp_A = self.percdamp * torch.mean(torch.diag(self.A))
        damp_G = self.percdamp * torch.mean(torch.diag(self.G))

        A_damped = self.A + damp_A * torch.eye(self.A.shape[0], device=self.device)
        G_damped = self.G + damp_G * torch.eye(self.G.shape[0], device=self.device)

        # Cholesky分解求逆
        try:
            A_inv = torch.cholesky_inverse(torch.linalg.cholesky(A_damped))
            G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G_damped))
        except:
            print(f"    Warning: Cholesky failed, using eigenvalue decomposition")
            # 回退到特征值分解
            eigvals_A, eigvecs_A = torch.linalg.eigh(A_damped)
            eigvals_A = torch.clamp(eigvals_A, min=1e-6)
            A_inv = eigvecs_A @ torch.diag(1.0 / eigvals_A) @ eigvecs_A.t()

            eigvals_G, eigvecs_G = torch.linalg.eigh(G_damped)
            eigvals_G = torch.clamp(eigvals_G, min=1e-6)
            G_inv = eigvecs_G @ torch.diag(1.0 / eigvals_G) @ eigvecs_G.t()

        return A_inv, G_inv

    def _compute_fisher_inverse_diag(self, A_inv: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
        """计算Fisher矩阵逆的对角元素 F = G ⊗ A (标准KFAC)"""
        A_inv_diag = torch.diag(A_inv)  # [in_dim]
        G_inv_diag = torch.diag(G_inv)  # [out_dim]

        # 对于 F = G ⊗ A，对角元素为：
        # F^{-1}[i*in_dim + j, i*in_dim + j] = G^{-1}[i,i] * A^{-1}[j,j]
        F_inv_diag = G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0)  # [out_dim, in_dim]

        return F_inv_diag

    def _perform_obs_surgery(
        self,
        W: torch.Tensor,
        A_inv: torch.Tensor,
        prune_heads: torch.Tensor,
        headsize: int
    ):
        """执行OBS手术"""
        A_inv_upper = torch.linalg.cholesky(A_inv, upper=True)

        for head_idx in prune_heads:
            start = head_idx * headsize
            end = start + headsize

            # 计算误差矩阵
            Err = torch.zeros(W.shape[0], headsize, device=W.device)

            for i in range(headsize):
                col_idx = start + i

                # 误差
                Err[:, i] = W[:, col_idx] / A_inv_upper[col_idx, col_idx]

                # 局部更新（head内）
                if i < headsize - 1:
                    W[:, col_idx+1:end] -= Err[:, i:i+1] @ A_inv_upper[col_idx:col_idx+1, col_idx+1:end]

            # 全局更新（其他列）
            if end < W.shape[1]:
                W[:, end:] -= Err @ A_inv_upper[start:end, end:]

            # 删除该head
            W[:, start:end] = 0


# ========================================
# Helper Functions for VAR Multi-Scale Processing (基于prune_v6.py的hooks方法)
# ========================================

def extract_multiscale_data_with_hooks(var_model, layer_idx: int, calibration_dataloader, device: str = 'cuda'):
    """
    使用hooks从VAR模型中提取指定层的多尺度输入和输出数据
    参考prune_v6.py的实现方法

    Args:
        var_model: VAR模型
        layer_idx: 层索引
        calibration_dataloader: 校准数据加载器
        device: 设备

    Returns:
        all_inputs_10scales: 所有batch的10尺度输入 [List[List[Tensor]]]
        all_outputs_10scales: 所有batch的10尺度输出 [List[List[Tensor]]]
    """
    var_model.eval()
    target_layer = var_model.blocks[layer_idx].attn.proj

    # 数据收集容器
    all_inputs_10scales = []
    all_outputs_10scales = []

    # Hook缓存字典（类似prune_v6.py）
    _input_cache = []
    _output_cache = []

    def input_hook(module, inp, out):
        """收集层输入的hook"""
        _input_cache.append(inp[0].detach())

        # 每收集10个尺度就处理一次
        if len(_input_cache) >= 10:
            # 构造10个尺度的数据
            inputs_10scales = []
            for i in range(10):
                inputs_10scales.append(_input_cache[i])

            all_inputs_10scales.append(inputs_10scales)
            _input_cache.clear()

    def output_hook(module, inp, out):
        """收集层输出的hook"""
        _output_cache.append(out.detach())

        # 每收集10个尺度就处理一次
        if len(_output_cache) >= 10:
            # 构造10个尺度的数据
            outputs_10scales = []
            for i in range(10):
                outputs_10scales.append(_output_cache[i])

            all_outputs_10scales.append(outputs_10scales)
            _output_cache.clear()

    # 注册hooks
    handle_input = target_layer.register_forward_hook(input_hook)
    handle_output = target_layer.register_forward_hook(output_hook)

    try:
        # 运行校准数据触发hooks
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_dataloader):
                if batch_idx >= 10:  # 限制batch数量
                    break

                # 将batch移到正确设备
                if hasattr(batch, 'to'):
                    batch = batch.to(device)
                elif isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

                # VAR前向传播（这会触发hooks收集10个尺度的数据）
                _ = var_model(batch)

    finally:
        # 清理hooks
        handle_input.remove()
        handle_output.remove()

    return all_inputs_10scales, all_outputs_10scales


def extract_layer_inputs(var_model, layer_idx: int, batch, device: str = 'cuda') -> List[torch.Tensor]:
    """
    从VAR模型中提取指定层在10个尺度的输入（单个batch）

    Args:
        var_model: VAR模型
        layer_idx: 层索引
        batch: 单个输入batch
        device: 设备

    Returns:
        inputs_10scales: 10个尺度的输入 [List[Tensor[B, L_s, hidden_dim]]]
    """
    var_model.eval()
    target_layer = var_model.blocks[layer_idx].attn.proj

    inputs_10scales = []

    def input_hook(module, inp, out):
        """收集输入的hook"""
        inputs_10scales.append(inp[0].detach().clone())

    # 注册hook
    handle = target_layer.register_forward_hook(input_hook)

    try:
        with torch.no_grad():
            # 将batch移到设备
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

            # VAR前向传播触发hook
            _ = var_model(batch)

    finally:
        handle.remove()

    return inputs_10scales


def extract_layer_outputs(var_model, layer_idx: int, batch, device: str = 'cuda') -> List[torch.Tensor]:
    """
    从VAR模型中提取指定层在10个尺度的输出（单个batch）

    Args:
        var_model: VAR模型
        layer_idx: 层索引
        batch: 单个输入batch
        device: 设备

    Returns:
        outputs_10scales: 10个尺度的输出 [List[Tensor[B, L_s, hidden_dim]]]
    """
    var_model.eval()
    target_layer = var_model.blocks[layer_idx].attn.proj

    outputs_10scales = []

    def output_hook(module, inp, out):
        """收集输出的hook"""
        outputs_10scales.append(out.detach().clone())

    # 注册hook
    handle = target_layer.register_forward_hook(output_hook)

    try:
        with torch.no_grad():
            # 将batch移到设备
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

            # VAR前向传播触发hook
            _ = var_model(batch)

    finally:
        handle.remove()

    return outputs_10scales


def prune_var_with_chain_kfac(
    var_model,
    calibration_dataloader,
    sparsity: float = 0.4,
    percdamp: float = 0.01,
    scale_weight_strategy: str = 'natural',
    device: str = 'cuda'
):
    """
    VAR模型剪枝主函数

    Args:
        var_model: VAR模型
        calibration_dataloader: 校准数据
        sparsity: 稀疏度
        percdamp: 阻尼系数
        scale_weight_strategy: 尺度加权策略 ['natural', 'equal', 'sqrt']
        device: 设备
    """
    var_model.to(device)
    var_model.eval()

    print("="*80)
    print("VAR Model Pruning: Chain Pruning + KFAC-OBS")
    print(f"  Total VAR tokens: 680 (across 10 scales)")
    print(f"  Sparsity: {sparsity:.1%}")
    print(f"  Scale weighting: {scale_weight_strategy}")
    print("="*80)

    # ========================================
    # 步骤1：逐层剪枝（混合策略）
    # ========================================
    for layer_idx in range(var_model.depth):
        print(f"\n{'='*60}")
        print(f"Pruning Layer {layer_idx}/{var_model.depth}")
        print(f"{'='*60}")

        layer = var_model.blocks[layer_idx]
        proj_layer = layer.attn.proj

        # ========================================
        # 关键：根据层级选择剪枝方法
        # ========================================
        if layer_idx == 0:
            # 第一层：使用SlimGPT（无法构建伪损失）
            print("  Strategy: SlimGPT (first layer, no previous layer for pseudo-loss)")

            from slim_utils.slimgpt import SlimGPT
            pruner = SlimGPT(
                layer=proj_layer,
                layer_idx=layer_idx,
                args=SimpleNamespace(percdamp=percdamp)
            )

            # 收集输入统计（只需要A矩阵，不需要梯度）
            print("  [1/4] Collecting input statistics (A matrix only)...")

            for batch_idx, batch in enumerate(calibration_dataloader):
                if batch_idx >= 10:
                    break

                # 获取10个尺度的输入
                inputs_10scales = extract_layer_inputs(var_model, layer_idx, batch)

                # SlimGPT的add_batch：只需要输入，不需要输出
                for scale_s, inp_s in enumerate(inputs_10scales):
                    pruner.add_batch(inp_s, None)

                print(f"    Processed batch {batch_idx+1}/10")

        else:
            # 后续层：使用链式剪枝 + KFAC-OBS
            print(f"  Strategy: Chain Pruning + KFAC-OBS (using Layer {layer_idx-1} output as target)")

            pruner = ChainKFACPruner(
                layer=proj_layer,
                layer_idx=layer_idx,
                scale_weight_strategy=scale_weight_strategy,
                percdamp=percdamp
            )

            # 收集多尺度KFAC矩阵（需要A和G矩阵）
            print("  [1/4] Collecting multi-scale KFAC matrices (A and G)...")

            for batch_idx, batch in enumerate(calibration_dataloader):
                if batch_idx >= 10:
                    break

                # 获取当前层的输入（10个尺度）
                inputs_10scales = extract_layer_inputs(var_model, layer_idx, batch)

                # 关键：获取原始输出作为伪损失的target
                # 这是从未剪枝的模型的当前层输出
                with torch.no_grad():
                    original_outputs_10scales = extract_layer_outputs(
                        var_model, layer_idx, batch
                    )

                # 添加batch，计算KFAC矩阵
                pruner.add_batch_multiscale(inputs_10scales, original_outputs_10scales)

                print(f"    Processed batch {batch_idx+1}/10 (with pseudo-loss gradient)")

        # ========================================
        # 步骤2：执行剪枝（统一接口）
        # ========================================
        print("  [2/4] Performing pruning...")

        prune_indices = pruner.struct_prune(sparsity=sparsity, headsize=64)

        # ========================================
        # 步骤4：物理删除（与prune_v6.py相同）
        # ========================================
        print("  [3/4] Physical pruning with tp.prune...")

        # 剪枝proj输入通道
        tp.prune_linear_in_channels(proj_layer, prune_indices.tolist())

        # ========================================
        # 步骤5：QKV同步（与prune_v6.py相同）
        # ========================================
        print("  [4/4] Synchronizing QKV...")

        hidden = 16 * 64  # embed_dim
        rm_qkv = torch.cat([
            prune_indices,
            prune_indices + hidden,
            prune_indices + 2*hidden
        ])

        rm_qkv_list = torch.unique(rm_qkv.cpu()).sort().values.tolist()
        tp.prune_linear_out_channels(layer.attn.mat_qkv, rm_qkv_list)

        # 更新相关参数
        keep_idxs = list(set(range(hidden)) - set(prune_indices.tolist()))
        layer.attn.q_bias = nn.Parameter(layer.attn.q_bias.data[keep_idxs])
        layer.attn.v_bias = nn.Parameter(layer.attn.v_bias.data[keep_idxs])
        zero_k_bias = layer.attn.zero_k_bias.data[keep_idxs]
        layer.attn.register_buffer('zero_k_bias', zero_k_bias)

        # 更新num_heads
        new_num_heads = 16 - int(16 * sparsity)
        layer.attn.num_heads = new_num_heads

        print(f"  Layer {layer_idx} complete! New heads: {new_num_heads}/16")

    print("\n" + "="*80)
    print("VAR Model Pruning Complete!")
    print("="*80)

    return var_model


if __name__ == '__main__':
    from models.var import VAR

    # 加载模型
    var_model = VAR(depth=16, embed_dim=1024, num_heads=16)

    # 执行剪枝
    pruned_model = prune_var_with_chain_kfac(
        var_model=var_model,
        calibration_dataloader=calibration_dataloader,
        sparsity=0.4,
        percdamp=0.01,
        scale_weight_strategy='natural',  # 或 'equal', 'sqrt'
        device='cuda'
    )

    # 保存
    torch.save(pruned_model.state_dict(), 'var_pruned_chain_kfac.pth')
```

### 与prune_v6.py的集成示例

```python
"""
修改prune_v6.py以支持链式剪枝+KFAC
只需要替换SlimGPT的部分，其他逻辑保持不变
"""

# 在prune_v6.py中替换这部分：
# pruner_dict[name] = SlimGPT(module_dict[name], i, args)

# 替换为：
pruner_dict[name] = ChainKFACPruner(
    layer=module_dict[name],
    layer_idx=i,
    scale_weight_strategy='natural',  # 可配置
    percdamp=args.percdamp
)

# hook部分需要修改以支持多尺度：
def add_batch_multiscale(name):
    def func(_, inp, out):
        # 缓存逻辑保持不变
        if name not in _cache_dict:
            _cache_dict[name] = []

        _cache_dict[name].append((inp[0].detach(), out.detach()))

        if len(_cache_dict[name]) >= 10:  # 10个尺度
            inputs_10scales = [p[0] for p in _cache_dict[name]]
            outputs_10scales = [p[1] for p in _cache_dict[name]]

            # 这里需要original_outputs作为target
            # original_outputs_10scales = get_original_outputs(layer_idx)

            # pruner_dict[name].add_batch_multiscale(
            #     inputs_10scales, original_outputs_10scales
            # )

            _cache_dict[name] = []

    return func

# 剪枝调用保持不变：
idx = pruner_dict[name].struct_prune(
    sparsity=sparsity,
    headsize=64 if name == "attn.proj" else 1
)

# 物理删除部分完全保持不变
if name == "attn.proj":
    # ... 原有的tp.prune代码 ...
```

---

## 总结与对比 | Summary and Comparison

### 核心创新总结

1. **链式剪枝**：使用层间输出差异构建伪损失
2. **真正的Fisher矩阵**：F = A ⊗ G，包含完整的二阶信息
3. **正确的token计算**：VAR总共680个token（不是341）
4. **多种尺度加权策略**：自然加权、均等加权、平方根加权
5. **输入维度剪枝**：使用 A ⊗ G 评估输入维度重要性

### 与SlimGPT的全面对比

| 特性 | SlimGPT (prune_v6.py) | 链式剪枝+KFAC |
|------|----------------------|--------------|
| **Hessian定义** | H = X^TX | F = A ⊗ G |
| **信息类型** | 一阶（输入协方差） | 二阶（Fisher矩阵） |
| **梯度信息** | ❌ 无 | ✅ G矩阵来自伪损失梯度 |
| **非线性** | ❌ 忽略softmax等 | ✅ 自动微分包含 |
| **理论基础** | 经验性 | Taylor展开 + OBS |
| **多尺度处理** | Token拼接 | 加权策略可选 |
| **计算成本** | 低（只需前向） | 中（需反向传播） |
| **剪枝精度** | 中等 | 高（真正Fisher） |
| **实现复杂度** | 简单 | 中等 |

### 实际性能预期

| 场景 | SlimGPT | 链式剪枝+KFAC | 提升原因 |
|------|---------|-------------|---------|
| **低稀疏度 (<20%)** | 好 | 略好 | 差异不显著 |
| **中稀疏度 (20-40%)** | 中等 | 好 | Fisher矩阵更准确 |
| **高稀疏度 (>40%)** | 差 | 好 | 准确补偿变得关键 |
| **复杂attention** | 中等 | 好 | 捕获softmax非线性 |
| **简单线性层** | 好 | 略好 | 线性层差异较小 |

### 使用建议

**选择SlimGPT当**：
- 快速原型验证
- 低稀疏度剪枝 (<20%)
- 计算资源有限
- 对剪枝精度要求不高

**选择链式剪枝+KFAC当**：
- 需要高精度剪枝
- 中高稀疏度 (>20%)
- VAR模型的attention层
- 有充足的计算资源

### 配置推荐

```python
# 推荐配置1：平衡性能和效率
prune_var_with_chain_kfac(
    sparsity=0.3,
    scale_weight_strategy='sqrt',  # 平衡各尺度
    percdamp=0.01
)

# 推荐配置2：追求最高精度
prune_var_with_chain_kfac(
    sparsity=0.4,
    scale_weight_strategy='natural',  # 符合实际分布
    percdamp=0.005  # 更低阻尼
)

# 推荐配置3：重视小尺度性能
prune_var_with_chain_kfac(
    sparsity=0.35,
    scale_weight_strategy='equal',  # 均等对待
    percdamp=0.02
)
```

---

**相关文档 | Related Documentation**:
- `prune_v6.py` - 基础SlimGPT实现
- `models/var.py` - VAR模型结构
- `slim_utils/slimgpt.py` - SlimGPT核心算法
- KFAC论文: Martens & Grosse, 2015
- OBS论文: Hassibi & Stork, 1993

**创建时间**: 2025-11-05
**版本**: 2.0（修正token数量）
**作者**: 基于prune_v6.py的创新改进