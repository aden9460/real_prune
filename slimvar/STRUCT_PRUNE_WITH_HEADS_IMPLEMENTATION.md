# struct_prune_with_heads函数实现方法详解

## 1. 核心概念与目标

### 1.1 设计目标

**核心需求**：创建一个新的剪枝函数，能够：
1. **接受预确定的head索引**：不通过误差度量选择，直接使用外部提供的head列表
2. **执行相同的补偿机制**：对剩余heads进行SlimGPT式的误差补偿
3. **集成差异化策略**：与scale_mul引导的层级化剪枝策略配合

### 1.2 与现有struct_prune的区别

| 方面 | 现有struct_prune | 新struct_prune_with_heads |
|------|------------------|---------------------------|
| Head选择 | 基于误差度量`sum(W²/Hinv_diag)`迭代选择 | 接受预确定的head索引列表 |
| 选择策略 | 贪心选择：每次选误差最小的head | 差异化策略：scale_mul + Hessian重要性 |
| 补偿机制 | ✅ 相同的局部+全局补偿 | ✅ 相同的局部+全局补偿 |
| 适用场景 | 通用结构化剪枝 | VAR差异化层级剪枝 |

## 2. 数学理论基础

### 2.1 SlimGPT补偿理论

**核心思想**：剪枝某些参数时，通过调整剩余参数来最小化输出变化。

给定权重矩阵 `W ∈ R^(d_out × d_in)`，要剪枝列索引集合 `P`：

```
目标：min ||ΔW||² subject to W[:, P] = 0
```

**二阶泰勒展开**：
```
Loss(W + ΔW) ≈ Loss(W) + g^T·ΔW + (1/2)·ΔW^T·H·ΔW
```

其中：
- `g`: 梯度（在最优点处g=0）
- `H`: Hessian矩阵（二阶导数）

**最优补偿解**：
```
ΔW*[:, j] = -W[:, P] · H[P, P]^(-1) · H[P, j]  for j ∉ P
```

### 2.2 Cholesky分解补偿算法

现有实现使用Cholesky分解实现高效补偿：

```python
# 1. Cholesky分解
Hinv = cholesky_inverse(cholesky(H))

# 2. 迭代补偿（局部更新）
for i in pruned_columns:
    Err[:, i] = W[:, i] / Hinv[i, i]
    W[:, i:] -= Err[:, i].unsqueeze(1) @ Hinv[i:i+1, i:]  # 局部补偿

# 3. 全局补偿
W[:, remaining] -= Err @ Hinv[:, remaining]  # 全局补偿
```

## 3. 实现方法分解

### 3.1 函数签名设计

```python
def struct_prune_with_heads(
    self,
    prune_head_indices,     # 预确定的head索引列表 [0, 3, 7, 11]
    headsize=64,           # 每个head的维度（VAR为64）
    percdamp=0.0,          # Hessian对角线阻尼系数
    layer_idx=None,        # 层索引（用于日志）
    return_compensation_info=False  # 是否返回补偿详情
):
    """
    使用预确定head索引进行结构化剪枝并补偿剩余heads
    """
```

### 3.2 实现步骤详解

#### Step 1: 输入验证与预处理

```python
# 1.1 验证输入合法性
assert self.columns % headsize == 0, "总列数必须能被头维度整除"
assert len(prune_head_indices) > 0, "必须指定至少一个要剪枝的head"

num_heads = self.columns // headsize
max_head_idx = max(prune_head_indices)
assert max_head_idx < num_heads, f"头索引{max_head_idx}超出范围[0, {num_heads-1}]"

# 1.2 转换head索引到列索引
prune_head_indices = torch.tensor(prune_head_indices, device=self.dev)
pruned_columns = []
for head_idx in prune_head_indices:
    start_col = head_idx * headsize
    end_col = start_col + headsize
    pruned_columns.extend(range(start_col, end_col))

pruned_columns = torch.tensor(pruned_columns, device=self.dev)
```

#### Step 2: 权重矩阵与Hessian准备

```python
# 2.1 获取权重矩阵
W = self.layer.weight.data.clone()
if isinstance(self.layer, nn.Conv2d):
    W = W.flatten(1)  # 卷积层展平
if isinstance(self.layer, transformers.Conv1D):
    W = W.t()  # GPT-style Conv1D转置
W = W.float()

# 2.2 处理Hessian矩阵
H = self.H.clone()
del self.H  # 释放原始Hessian内存

# 2.3 处理dead neurons（梯度为0的神经元）
dead = torch.diag(H) == 0
H[dead, dead] = 1  # 避免数值不稳定
W[:, dead] = 0     # 清零dead weights

# 2.4 可选：添加阻尼项（提高数值稳定性）
if percdamp > 0:
    damp = percdamp * torch.mean(torch.diag(H))
    diag_indices = torch.arange(H.size(0), device=self.dev)
    H[diag_indices, diag_indices] += damp
```

#### Step 3: 列重排序（关键优化）

```python
# 3.1 创建列排序：先放要剪枝的列，再放保留的列
remaining_columns = torch.tensor([i for i in range(self.columns)
                                if i not in pruned_columns], device=self.dev)

column_order = torch.cat([pruned_columns, remaining_columns])
cnt = len(pruned_columns)

# 3.2 按新顺序重排W和H
W_reordered = W[:, column_order]
H_reordered = H[column_order, :][:, column_order]

# 3.3 对重排序后的Hessian进行Cholesky分解
H_chol = torch.linalg.cholesky(H_reordered, upper=True)[:cnt]
```

#### Step 4: 误差计算与局部补偿

```python
# 4.1 提取要剪枝部分的权重和Hessian
W_prune = W_reordered[:, :cnt].clone()  # 要剪枝的列
H_prune = H_chol[:, :cnt]               # 对应的Hessian块

# 4.2 初始化误差矩阵
Err = torch.zeros_like(W_prune)

# 4.3 迭代补偿：逐列处理每个要剪枝的参数
for i in range(cnt):
    # 计算当前列的误差
    Err[:, i:i+1] = W_prune[:, i:i+1] / H_prune[i, i]

    if not self.no_compensate:
        # 局部补偿：更新当前列之后的所有列
        # 这确保了补偿的因果性（当前列只影响后续列）
        W_prune[:, i:] -= Err[:, i:i+1] @ H_prune[i:i+1, i:]
```

**局部补偿的数学意义**：
- 每次剪枝第`i`列时，其误差会分散到第`i+1`到最后的所有列上
- `H_prune[i, j]`表示第`i`列和第`j`列的二阶交互项
- 补偿量 = `误差 × 交互强度`

#### Step 5: 全局补偿

```python
# 5.1 清零已剪枝的列
W_reordered[:, :cnt] = 0

# 5.2 全局补偿：将误差分散到所有剩余列
if not self.no_compensate:
    remaining_start = cnt
    remaining_end = self.columns

    # 全局补偿公式：W_remaining -= Err @ H[pruned, remaining]
    W_reordered[:, remaining_start:remaining_end] -= \
        Err @ H_chol[:, remaining_start:remaining_end]
```

**全局补偿的作用**：
- 局部补偿只在剪枝列内部进行调整
- 全局补偿将剪枝误差传播到所有剩余参数
- 确保整体输出变化最小

#### Step 6: 恢复原始顺序并更新权重

```python
# 6.1 恢复原始列顺序
column_order_inv = torch.argsort(column_order)
W_final = W_reordered[:, column_order_inv]

# 6.2 根据层类型转换回原始形状
if isinstance(self.layer, transformers.Conv1D):
    W_final = W_final.t()

# 6.3 更新层权重
self.layer.weight.data = W_final.reshape(self.layer.weight.shape).to(
    self.layer.weight.data.dtype)
```

### 3.3 返回值设计

```python
return {
    'pruned_head_indices': prune_head_indices.cpu().tolist(),
    'pruned_column_indices': pruned_columns.cpu().tolist(),
    'compensation_norm': torch.norm(Err).item(),  # 补偿强度
    'sparsity_achieved': len(pruned_columns) / self.columns,
    'layer_idx': layer_idx
}
```

## 4. 与差异化剪枝策略的集成

### 4.1 调用流程

```python
# 步骤1: 差异化head选择
def differential_scale_guided_pruning(model, layer_idx, pruner, args):
    # 获取scale_mul和Hessian重要性
    scale_mul = model.blocks[layer_idx].attn.scale_mul_1H11.exp().squeeze()
    hessian_imp = pruner.compute_importance()  # 需要实现

    # 归一化重要性
    hess_norm = (hessian_imp - hessian_imp.min()) / \
                (hessian_imp.max() - hessian_imp.min() + 1e-8)

    # 根据层类型调整重要性
    q1_threshold = args.q1_threshold  # 14.15

    if layer_idx in range(0, 5):  # 早期通才层
        scale_bonus = torch.where(scale_mul < q1_threshold,
                                args.early_bonus, 1.0)  # 1.2
        adjusted_importance = hess_norm * scale_bonus
        sparsity = args.early_sparsity  # 0.30 for 40% config

    elif layer_idx in range(5, 14):  # 中间专家层
        scale_penalty = torch.where(scale_mul < q1_threshold,
                                  args.expert_penalty, 1.0)  # 0.5
        adjusted_importance = hess_norm * scale_penalty
        sparsity = args.expert_sparsity  # 0.50 for 40% config

    else:  # 末尾通才层
        scale_bonus = torch.where(scale_mul < q1_threshold,
                                args.late_bonus, 1.0)  # 1.2
        adjusted_importance = hess_norm * scale_bonus
        sparsity = args.late_sparsity  # 0.30 for 40% config

    # 选择要剪枝的heads
    num_heads = adjusted_importance.numel()
    num_prune = int(num_heads * sparsity)
    prune_head_indices = adjusted_importance.argsort()[:num_prune]

    return prune_head_indices.tolist(), sparsity

# 步骤2: 执行剪枝和补偿
prune_head_indices, sparsity = differential_scale_guided_pruning(
    model, layer_idx, pruner, args)

compensation_info = pruner.struct_prune_with_heads(
    prune_head_indices=prune_head_indices,
    headsize=64,  # VAR head dimension
    layer_idx=layer_idx
)
```

### 4.2 集成到prune_v6.py

```python
# 在prune_v6.py中添加新的剪枝模式
if args.differential_scale_pruning:
    for layer_idx in range(args.maxlayer):
        # 获取当前层的pruner
        pruner = pruner_dict[f'blocks.{layer_idx}.attn.qkv']

        # 差异化选择heads
        prune_heads, sparsity = differential_scale_guided_pruning(
            model, layer_idx, pruner, args)

        # 执行剪枝和补偿
        pruner.struct_prune_with_heads(
            prune_head_indices=prune_heads,
            headsize=64,
            layer_idx=layer_idx
        )

        print(f"Layer {layer_idx}: pruned {len(prune_heads)} heads "
              f"(sparsity={sparsity:.1%})")
```

## 5. 关键技术细节

### 5.1 数值稳定性保证

```python
# 1. Hessian条件数检查
cond_num = torch.linalg.cond(H)
if cond_num > 1e12:
    print(f"Warning: Hessian condition number {cond_num:.2e} is too large")

# 2. Cholesky分解失败处理
try:
    H_chol = torch.linalg.cholesky(H_reordered)
except RuntimeError:
    # 添加更大的阻尼项
    damp = 1e-3 * torch.mean(torch.diag(H_reordered))
    diag = torch.arange(H_reordered.size(0))
    H_reordered[diag, diag] += damp
    H_chol = torch.linalg.cholesky(H_reordered)

# 3. 除零保护
for i in range(cnt):
    if abs(H_prune[i, i]) < 1e-10:
        H_prune[i, i] = 1e-10  # 防止除零
    Err[:, i:i+1] = W_prune[:, i:i+1] / H_prune[i, i]
```

### 5.2 内存优化

```python
# 1. 原地操作减少内存
def struct_prune_with_heads_inplace(self, prune_head_indices, headsize=64):
    # 直接修改self.layer.weight，不创建副本
    W = self.layer.weight.data
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()

    # 使用视图操作而非复制
    # ...

# 2. 分块处理大矩阵
if self.columns > 4096:  # 对于大矩阵分块处理
    block_size = 1024
    for start in range(0, cnt, block_size):
        end = min(start + block_size, cnt)
        # 分块补偿
        # ...
```

### 5.3 并行化支持

```python
# 对于多head并行剪枝
def batch_struct_prune_with_heads(self, layer_head_indices_dict, headsize=64):
    """
    并行剪枝多层的多个heads

    Args:
        layer_head_indices_dict: {layer_idx: [head_indices]}
    """
    results = {}
    for layer_idx, head_indices in layer_head_indices_dict.items():
        pruner = pruner_dict[f'blocks.{layer_idx}.attn.qkv']
        results[layer_idx] = pruner.struct_prune_with_heads(
            head_indices, headsize, layer_idx=layer_idx)

    return results
```

## 6. 验证方法

### 6.1 单元测试

```python
def test_struct_prune_with_heads():
    # 1. 创建简单测试用例
    layer = nn.Linear(64, 32)  # 4 heads × 16 dims
    pruner = SlimGPT(layer)
    pruner.H = torch.eye(64) + 0.1 * torch.randn(64, 64)

    # 2. 剪枝head 0和head 2
    result = pruner.struct_prune_with_heads([0, 2], headsize=16)

    # 3. 验证结果
    assert result['sparsity_achieved'] == 0.5
    assert len(result['pruned_column_indices']) == 32

    # 4. 验证权重确实被置零
    W = layer.weight.data
    assert torch.allclose(W[:, 0:16], torch.zeros_like(W[:, 0:16]))
    assert torch.allclose(W[:, 32:48], torch.zeros_like(W[:, 32:48]))
```

### 6.2 输出一致性验证

```python
def test_output_consistency():
    # 1. 保存剪枝前的输出
    x = torch.randn(1, 256, 64)  # 测试输入
    output_before = model(x)

    # 2. 执行剪枝
    pruner.struct_prune_with_heads([0, 3, 7, 11], headsize=64)

    # 3. 比较剪枝后的输出
    output_after = model(x)
    mse_loss = torch.mean((output_before - output_after) ** 2)

    print(f"Output MSE after pruning: {mse_loss:.6f}")
    assert mse_loss < 1e-3, "输出变化过大，补偿可能不充分"
```

## 7. 预期效果分析

### 7.1 理论优势

1. **精确的head选择**：
   - 现有方法：基于单一Hessian误差度量
   - 新方法：结合scale_mul + Hessian + 层级策略

2. **差异化策略**：
   - 早期/末尾层：保护低scale heads（全局整合能力）
   - 中间层：剪除低scale heads（冗余通才功能）

3. **方差一致性改善**：
   - 目标：中间层方差降低40%以上
   - 指标：方差ratio从4.3降至<2.5

### 7.2 实验验证计划

```python
# 40%剪枝率实验
python prune_v6.py \
    --differential_scale_pruning \
    --q1_threshold 14.15 \
    --early_bonus 1.2 --early_sparsity 0.30 \
    --expert_penalty 0.5 --expert_sparsity 0.50 \
    --late_bonus 1.2 --late_sparsity 0.30 \
    --model_name var_d16_scale_guided_40percent

# 期望结果
Expected results:
- FID: 2.40 (vs 2.50 baseline, 改善4%)
- 中间层方差: ~17 (vs 33.96 baseline, 降低50%)
- 方差ratio: <2.5 (vs 4.3 baseline, 降低42%)
```

## 8. 总结

### 8.1 核心创新点

1. **预确定head剪枝**：突破传统greedy选择，支持复杂策略
2. **完整补偿机制**：保持SlimGPT的数学严谨性
3. **差异化层级策略**：首次针对VAR的三段式结构设计
4. **方差一致性目标**：新的结构化评价指标

### 8.2 实现复杂度

- **计算复杂度**：O(d²) (Cholesky分解主导)
- **空间复杂度**：O(d²) (Hessian存储)
- **与现有方法相当**，无额外开销

### 8.3 适用范围

- ✅ VAR模型的attention heads剪枝
- ✅ 任何具有scale_mul参数的Transformer
- ✅ 需要差异化策略的结构化剪枝
- ❌ 不适用于无结构化（weight-level）剪枝

这个实现为VAR模型的高效剪枝提供了数学严谨且实用的解决方案，将scale_mul的先验知识与SlimGPT的补偿理论完美结合。