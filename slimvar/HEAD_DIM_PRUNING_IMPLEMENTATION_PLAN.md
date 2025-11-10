# VAR Head_dim 剪枝实施方案

**日期**: 2025-11-10
**目标**: 在 model_slimming_basic.py 中添加 head_dim 剪枝功能

---

## 背景

### 当前实现
- **文件**: `model_slimming_basic.py`
- **方法**: Head 剪枝（删除整个 head）
- **效果**: 16 heads × 64 dim → 12 heads × 64 dim (sparsity=0.25)

### 目标实现
- **方法**: Head_dim 剪枝（减少每个 head 的维度）
- **效果**: 16 heads × 64 dim → 16 heads × 48 dim (sparsity=0.25)
- **优势**:
  1. 保留所有 head 的功能
  2. 每个 head 个性化优化（删除自己最不重要的维度）
  3. 保留 learned scale parameters (`scale_mul_1H11`)

---

## 核心问题分析

### 问题1: Global Update 的必要性

#### 分析结果：**需要 Global Update**

**原因**:
```python
# O 投影矩阵 (attn.proj) 的特性
output = O_matrix @ concat([head_0, head_1, ..., head_15])

# Hessian 矩阵结构
H = [H_00  H_01  H_02  ...  H_0,15]  # Head 间有相关性
    [H_10  H_11  H_12  ...  H_1,15]  # H_ij ≠ 0 (i≠j)
    [...                          ]
    [H_15,0 ...            H_15,15]
```

**结论**: O 矩阵的每一行混合所有 head 的信息，因此删除某个 head 内的维度会影响其他 head，**必须使用 Global Update 进行跨 head 补偿**。

#### 参考实现：struct_prune 的 Global Update

```python
# slimgpt.py Line 199-250
while pruned_columns < target_columns:
    # 1. 计算完整 Hinv
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    # 2. 选择要删除的列，重排到最前面
    W = W[:, column_sort_idx]
    Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]

    # 3. Local Update（待删除列内）
    for i in range(cnt):
        Err1[:, i] = W1[:, i] / Hinv1[i, i]
        W1[:, i:] -= Err1[:, i].matmul(Hinv1[i, i:])

    # 4. ⭐ Global Update（补偿所有剩余列）
    W[:, :cnt] = 0
    W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # 关键！

    # 5. 恢复原始顺序
    W = W[:, column_sort_idx_inv]

    # 6. 更新 H（清零已删除的行列）
    H[pruned_idx, :] = H[:, pruned_idx] = 0
    H[pruned_idx, pruned_idx] = 1
```

**关键点**:
1. **重排策略**: 将待删除列移到最前面
2. **Global Update 公式**: `W[:, cnt:end] -= Err1 @ Hinv[:, cnt:end]`
3. **每次迭代重新计算 Hinv**: 因为 H 在变化

---

### 问题2: head_dim 和维度调整

#### 分析结果：**需要修改 head_dim，Torch-Pruning 会自动处理**

**关键认知**:
```python
# Torch-Pruning 自动调整权重形状

# Head 剪枝：
# - 删除 idx_list = [0-63, 128-191, ...]（完整 head）
# - tp.prune_linear_in_channels(proj, idx_list)
# - proj.weight: (1024, 1024) → (1024, 768)
# - mat_qkv.weight: (1024, 3072) → (1024, 2304)

# Head_dim 剪枝：
# - 删除 idx_list = [3,7,15,..., 66,73,82,...]（分散的维度）
# - tp.prune_linear_in_channels(proj, idx_list)
# - proj.weight: (1024, 1024) → (1024, 768)  # 形状一样！
# - mat_qkv.weight: (1024, 3072) → (1024, 2304)  # 形状一样！
```

**Forward 计算**:
```python
# basic_var.py:117
C = self.num_heads * self.head_dim

# Head 剪枝后：
# num_heads = 12, head_dim = 64
# C = 12 * 64 = 768 ✓

# Head_dim 剪枝后：
# num_heads = 16, head_dim = 48
# C = 16 * 48 = 768 ✓
```

**结论**:
1. Torch-Pruning 逻辑**完全通用**（无需修改）
2. 只需正确设置 `head_dim`（而不是 `num_heads`）
3. scale_mul_1H11 **不需要修改**（所有 head 都保留）

---

## 实施方案

### 设计原则

⚠️ **向后兼容性保证**：
1. ✅ 默认行为**完全不变**（不影响原代码）
2. ✅ 只有显式添加 `--use_head_dim_prune` 参数才启用新功能
3. ✅ 原有测试用例和脚本无需修改
4. ✅ 新功能完全通过参数控制

---

### 修改 1: slim_utils/slimgpt.py - 添加 head_dim_prune 方法

**文件**: `/home/project/real_prune/slimvar/slim_utils/slimgpt.py`

**位置**: 在 `struct_prune` 方法后添加（约 Line 270）

**修改类型**: ✅ **新增方法**（不修改任何现有代码）

**实现要点**:
```python
def head_dim_prune(self, sparsity, headsize=64, percdamp=0.0, layer_idx=None):
    """
    Head维度剪枝：参考 struct_prune 的 Global Update 逻辑

    关键区别：
    - struct_prune: 逐次删除完整 head（while 循环）
    - head_dim_prune: 逐 head 处理，每个 head 内删除部分维度（for 循环）
    """
    num_heads = self.columns // headsize
    dims_to_remove_per_head = round(headsize * sparsity)

    # 准备
    W = self.layer.weight.data.clone().float()
    H = self.H
    del self.H

    # 处理死节点和阻尼
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    if percdamp > 0:
        damp = percdamp * torch.mean(torch.diag(H))
        H[torch.arange(H.size(0)), torch.arange(H.size(0))] += damp

    all_pruned_indices = []

    # ⭐ 逐 head 处理
    for head_idx in range(num_heads):
        start_col = head_idx * headsize
        end_col = start_col + headsize

        # 1. ⭐ 每次重新计算完整 Hinv（因为 H 在变化）
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

        # 2. 计算 head 内误差
        W_head = W[:, start_col:end_col]
        H_head = H[start_col:end_col, start_col:end_col]
        Hinv_head = Hinv[start_col:end_col, start_col:end_col]

        Hinv_diag_head = torch.diagonal(torch.linalg.cholesky(Hinv_head)) ** 2
        error_head = torch.sum(W_head ** 2 / Hinv_diag_head.unsqueeze(0), dim=0)

        # 3. 选择要删除的维度
        dim_sort_idx = error_head.argsort()
        dims_to_remove = dim_sort_idx[:dims_to_remove_per_head]
        global_pruned_idx = start_col + dims_to_remove
        all_pruned_indices.append(global_pruned_idx)

        # 4. ⭐ 重排：将待删除维度移到 head 最前面
        keep_dims = dim_sort_idx[dims_to_remove_per_head:]
        reorder_idx_head = torch.cat([dims_to_remove, keep_dims])

        reorder_idx_global = torch.arange(self.columns)
        reorder_idx_global[start_col:end_col] = start_col + reorder_idx_head

        W = W[:, reorder_idx_global]
        Hinv_reordered = Hinv[reorder_idx_global, :][:, reorder_idx_global]

        # 5. Cholesky 分解（上三角）
        Hinv_chol = torch.linalg.cholesky(Hinv_reordered, upper=True)[
            start_col:start_col+dims_to_remove_per_head
        ]

        # 6. Local Update（head 内）
        if not self.no_compensate:
            W1 = W[:, start_col:start_col+dims_to_remove_per_head].clone()
            Hinv1 = Hinv_chol[:, start_col:end_col]
            Err1 = torch.zeros_like(W1)

            for i in range(dims_to_remove_per_head):
                col_i = start_col + i
                Err1[:, i] = W1[:, i] / Hinv_chol[i, col_i]
                W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

            # 7. ⭐ Global Update（补偿所有其他列）
            W[:, start_col:start_col+dims_to_remove_per_head] = 0
            W[:, start_col+dims_to_remove_per_head:] -= Err1.matmul(
                Hinv_chol[:, start_col+dims_to_remove_per_head:]
            )

        # 8. 恢复原始顺序
        reorder_idx_inv = torch.argsort(reorder_idx_global)
        W = W[:, reorder_idx_inv]

        # 9. ⭐ 更新 H（清零已删除维度）
        for idx in global_pruned_idx:
            H[idx, :] = H[:, idx] = 0
            H[idx, idx] = 1

    # 写回
    self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
        self.layer.weight.data.dtype
    )

    if layer_idx is not None:
        print(f'Layer {layer_idx}: head_dim_prune completed, '
              f'removed {len(all_pruned_indices)} dims per head', flush=True)

    return torch.cat(all_pruned_indices) if all_pruned_indices else torch.tensor([])
```

**核心要点**:
1. **逐 head 迭代**: `for head_idx in range(num_heads)`
2. **每次重新计算完整 Hinv**: `Hinv = cholesky_inverse(cholesky(H))`
3. **重排策略**: 将待删除维度移到 head 最前面
4. **Local Update**: 只在 head 内的待删除维度之间
5. **Global Update**: 补偿 head 内剩余维度 + 所有其他 head
6. **恢复原序**: 确保维度顺序不变
7. **更新 H**: 每次处理完一个 head 后更新 H

---

### 修改 2: model_slimming_basic.py - 添加 head_dim 剪枝分支

**文件**: `/home/project/real_prune/slimvar/model_slimming_basic.py`

**修改类型**: ✅ **条件分支**（原代码在 else 分支，默认执行原逻辑）

#### 位置 1: Line 383-394（选择剪枝方法）

**修改前**:
```python
for name in prune_order:
    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
    print(f"  Layer {i}: {name} - pruning {sparsity*100:.1f}%")

    idx = pruner_dict[name].struct_prune(
        sparsity=sparsity,
        percdamp=args.percdamp,
        headsize=64 if name == "attn.proj" else 1,
        layer_idx=i,
    )
```

**修改后**:
```python
for name in prune_order:
    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
    print(f"  Layer {i}: {name} - pruning {sparsity*100:.1f}%")

    # ⭐ 根据参数选择剪枝方法
    # 默认：执行原始逻辑（else 分支）
    # 仅当显式指定 --use_head_dim_prune 且处理 attn.proj 时，才使用新方法
    if args.use_head_dim_prune and name == "attn.proj":
        # 新功能：Head_dim 剪枝（需要显式开启）
        idx = pruner_dict[name].head_dim_prune(
            sparsity=sparsity,
            headsize=64,
            percdamp=args.percdamp,
            layer_idx=i,
        )
    else:
        # 默认：原始 Head 剪枝或 FFN 剪枝（保持不变）
        idx = pruner_dict[name].struct_prune(
            sparsity=sparsity,
            percdamp=args.percdamp,
            headsize=64 if name == "attn.proj" else 1,
            layer_idx=i,
        )
```

#### 位置 2: Line 407-462（Torch-Pruning 处理）

**修改前**:
```python
elif name == "attn.proj":
    # Following tp_prune_reference.py exactly
    model.blocks[i].attn.num_heads = torch.round(torch.tensor(model.num_heads * (1 - sparsity))).int()

    idx_m = idx.to(dtype=torch.long)
    idx_list = idx.tolist()
    keep_idxs = list(set(range(target_layer.in_features)) - set(idx_list))

    # Update biases
    model.blocks[i].attn.q_bias = nn.Parameter(model.blocks[i].attn.q_bias.data[keep_idxs])
    zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
    model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
    model.blocks[i].attn.v_bias = nn.Parameter(model.blocks[i].attn.v_bias.data[keep_idxs])

    # Update scale parameter
    head_dim = 64
    old_num_heads = target_layer.in_features // head_dim
    removed_heads = set((idx_m // head_dim).tolist())
    all_heads = set(range(old_num_heads))
    keep_heads = sorted(list(all_heads - removed_heads))

    old_scale_mul = model.blocks[i].attn.scale_mul_1H11.data
    new_scale_mul = old_scale_mul[0, keep_heads, 0, 0].view(1, -1, 1, 1)

    model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
        new_scale_mul.clone().to(device),
        requires_grad=True
    )

    print(f"    ✓ Preserved scale_mul for heads {keep_heads}")
    print(f"      Removed heads: {sorted(removed_heads)}")

    # Prune proj
    tp.prune_linear_in_channels(target_layer, idx_list)

    # Prune mat_qkv
    target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
    hidden = 16 * 64

    rm_feat_q = idx_m
    rm_qkv = torch.cat([
        rm_feat_q,
        rm_feat_q + hidden,
        rm_feat_q + 2*hidden
    ], dim=0)

    rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
    tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

    print(f"    ✓ Pruned {len(idx_list)} channels ({len(idx_list)//64} heads)")
    print(f"    ✓ New head count: {model.blocks[i].attn.num_heads}")
```

**修改后**:
```python
elif name == "attn.proj":
    idx_m = idx.to(dtype=torch.long)
    idx_list = idx.tolist()
    keep_idxs = list(set(range(target_layer.in_features)) - set(idx_list))

    # ⭐ 根据参数选择不同的处理逻辑
    # 默认：执行原始 Head 剪枝逻辑（else 分支）
    if args.use_head_dim_prune:
        # ============ Head_dim 剪枝分支 ============

        # 1. ⭐ 更新 head_dim（不是 num_heads）
        old_head_dim = 64
        new_head_dim = old_head_dim - round(old_head_dim * sparsity)
        model.blocks[i].attn.head_dim = new_head_dim
        # num_heads 保持不变！

        # 2. 更新 biases
        model.blocks[i].attn.q_bias = nn.Parameter(
            model.blocks[i].attn.q_bias.data[keep_idxs]
        )
        zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
        model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
        model.blocks[i].attn.v_bias = nn.Parameter(
            model.blocks[i].attn.v_bias.data[keep_idxs]
        )

        # 3. ⭐ scale_mul 不需要修改（所有 head 都保留）
        # model.blocks[i].attn.scale_mul_1H11 保持不变

        # 4. Torch-Pruning（逻辑完全一样）
        tp.prune_linear_in_channels(target_layer, idx_list)

        target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
        hidden = 16 * 64

        rm_feat_q = idx_m
        rm_qkv = torch.cat([
            rm_feat_q,
            rm_feat_q + hidden,
            rm_feat_q + 2*hidden
        ], dim=0)

        rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
        tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

        print(f"    ✓ Pruned {len(idx_list)} dims ({len(idx_list)//16} dims per head)")
        print(f"    ✓ New head_dim: {new_head_dim} (num_heads={model.blocks[i].attn.num_heads})")

    else:
        # ============ 原始 Head 剪枝分支 ============

        # 1. 更新 num_heads（不是 head_dim）
        model.blocks[i].attn.num_heads = torch.round(
            torch.tensor(model.num_heads * (1 - sparsity))
        ).int()

        # 2. 更新 biases
        model.blocks[i].attn.q_bias = nn.Parameter(
            model.blocks[i].attn.q_bias.data[keep_idxs]
        )
        zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
        model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
        model.blocks[i].attn.v_bias = nn.Parameter(
            model.blocks[i].attn.v_bias.data[keep_idxs]
        )

        # 3. 更新 scale_mul（删除被移除 head 的 scale）
        head_dim = 64
        old_num_heads = target_layer.in_features // head_dim
        removed_heads = set((idx_m // head_dim).tolist())
        all_heads = set(range(old_num_heads))
        keep_heads = sorted(list(all_heads - removed_heads))

        old_scale_mul = model.blocks[i].attn.scale_mul_1H11.data
        new_scale_mul = old_scale_mul[0, keep_heads, 0, 0].view(1, -1, 1, 1)

        model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
            new_scale_mul.clone().to(device),
            requires_grad=True
        )

        print(f"    ✓ Preserved scale_mul for heads {keep_heads}")
        print(f"      Removed heads: {sorted(removed_heads)}")

        # 4. Torch-Pruning
        tp.prune_linear_in_channels(target_layer, idx_list)

        target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
        hidden = 16 * 64

        rm_feat_q = idx_m
        rm_qkv = torch.cat([
            rm_feat_q,
            rm_feat_q + hidden,
            rm_feat_q + 2*hidden
        ], dim=0)

        rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
        tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

        print(f"    ✓ Pruned {len(idx_list)} channels ({len(idx_list)//64} heads)")
        print(f"    ✓ New head count: {model.blocks[i].attn.num_heads}")
```

#### 位置 3: Line 680+（添加命令行参数）

**修改类型**: ✅ **新增参数**（默认值为 False，保持原行为）

**添加**:
```python
# Head_dim pruning option
parser.add_argument(
    "--use_head_dim_prune",
    action="store_true",  # ⭐ 默认 False，保证向后兼容
    help="Use head_dim pruning instead of head pruning for attention layers. "
         "Head_dim pruning reduces the dimension of each head while keeping all heads, "
         "whereas head pruning removes entire heads. "
         "Default: False (use original head pruning)"
)
```

**默认行为**:
```python
# 不加参数：使用原始 Head 剪枝
args.use_head_dim_prune = False  # 默认值

# 显式开启：使用新的 Head_dim 剪枝
args.use_head_dim_prune = True   # 需要显式指定 --use_head_dim_prune
```

---

## 实施步骤

### 步骤 1: 添加 head_dim_prune 方法（不影响原代码）
1. 打开 `/home/project/real_prune/slimvar/slim_utils/slimgpt.py`
2. 在 `struct_prune` 方法后（约 Line 270）添加 `head_dim_prune` 方法
3. **注意**: 不修改任何现有方法，只添加新方法
4. 参考上述实现要点编写代码

### 步骤 2: 修改 model_slimming_basic.py（通过条件分支）
1. **位置 1** (Line 383-394): 添加 `if args.use_head_dim_prune` 条件分支
   - ✅ 原代码移到 `else` 分支（默认执行）
   - ✅ 新代码在 `if` 分支（需要参数开启）
2. **位置 2** (Line 407-462): 添加 `if args.use_head_dim_prune` 条件分支
   - ✅ 原代码移到 `else` 分支（默认执行）
   - ✅ 新代码在 `if` 分支（需要参数开启）
3. **位置 3** (Line 680+): 添加命令行参数 `--use_head_dim_prune`
   - ✅ `action="store_true"` 确保默认为 False

### 步骤 3: 向后兼容性测试

#### 测试 1: 验证原代码不受影响（默认行为）
```bash
# ⭐ 不加任何新参数，应该和之前完全一样
python model_slimming_basic.py \
    --model_depth 16 \
    --sparsity 0.25 \
    --num_samples 256 \
    --save_dir ./pruned_models \
    --model_name var_original_0.25_256sample.pth

# 预期：
# - 使用 struct_prune（原方法）
# - 执行 Head 剪枝（删除整个 head）
# - num_heads 减少
# - 与之前的结果完全一致
```

#### 测试 2: 验证新功能正常工作
```bash
# ⭐ 显式添加 --use_head_dim_prune，启用新功能
python model_slimming_basic.py \
    --model_depth 16 \
    --sparsity 0.25 \
    --num_samples 256 \
    --use_head_dim_prune \
    --save_dir ./pruned_models \
    --model_name var_headdim_0.25_256sample.pth

# 预期：
# - 使用 head_dim_prune（新方法）
# - 执行 Head_dim 剪枝（减少每个 head 的维度）
# - num_heads 保持不变
# - head_dim 减少
```

#### 测试 3: 对比验证
```bash
# 检查两种方法的输出差异
python -c "
import torch

# 加载两个模型
model_head = torch.load('./pruned_models/var_original_0.25_256sample.pth')
model_dim = torch.load('./pruned_models/var_headdim_0.25_256sample.pth')

# 验证参数量相同
params_head = sum(p.numel() for p in model_head.values())
params_dim = sum(p.numel() for p in model_dim.values())

print(f'Head pruning params: {params_head}')
print(f'Head_dim pruning params: {params_dim}')
print(f'Params match: {params_head == params_dim}')

# 验证权重形状相同
for key in model_head.keys():
    if key in model_dim:
        if model_head[key].shape != model_dim[key].shape:
            print(f'Shape mismatch: {key}')
            print(f'  Head: {model_head[key].shape}')
            print(f'  Dim:  {model_dim[key].shape}')
"
```

---

## 关键区别对比

| 特性 | Head 剪枝 (struct_prune) | Head_dim 剪枝 (head_dim_prune) |
|------|-------------------------|------------------------------|
| **删除对象** | 完整 head | 每个 head 内部分维度 |
| **循环结构** | `while` 逐次删除 head | `for` 逐 head 处理 |
| **num_heads** | 减少 (16→12) | **保持不变 (16)** |
| **head_dim** | 保持不变 (64) | **减少 (64→48)** |
| **scale_mul** | 删除被移除 head 的 scale | **保持所有 head 的 scale** |
| **Hessian 计算** | 每次迭代重新计算 | 每个 head 处理时重新计算 |
| **Local Update** | 在待删除列内 | 在每个 head 内待删除列内 |
| **Global Update** | ✅ 补偿所有剩余列 | ✅ 补偿 head 内剩余 + 其他 head |
| **Torch-Pruning** | 删除完整 head 的通道 | 删除分散维度的通道（逻辑相同） |
| **最终维度** | 12×64 = 768 | 16×48 = 768 |

---

## 验证要点

### 1. 权重形状验证
```python
# 剪枝后应该相同
print(model.blocks[0].attn.proj.weight.shape)      # (1024, 768)
print(model.blocks[0].attn.mat_qkv.weight.shape)   # (1024, 2304)
```

### 2. 维度计算验证
```python
# Head 剪枝
C = model.blocks[0].attn.num_heads * 64  # 12 * 64 = 768

# Head_dim 剪枝
C = 16 * model.blocks[0].attn.head_dim  # 16 * 48 = 768
```

### 3. scale_mul 验证
```python
# Head 剪枝
print(model.blocks[0].attn.scale_mul_1H11.shape)  # (1, 12, 1, 1)

# Head_dim 剪枝
print(model.blocks[0].attn.scale_mul_1H11.shape)  # (1, 16, 1, 1)
```

### 4. Forward 验证
```python
# 测试 forward 能否正常运行
with torch.no_grad():
    output = model(test_labels, test_tokens)
    print(f"Output shape: {output.shape}")
```

---

## 注意事项

### ⚠️ 0. 向后兼容性（最重要！）
- **默认行为完全不变**: 不添加 `--use_head_dim_prune` 时，执行原代码
- **所有修改通过条件分支**: `if args.use_head_dim_prune` 控制
- **新方法独立添加**: 不修改任何现有方法
- **测试原代码**: 修改后先运行原测试用例，确保无影响

### 1. Global Update 的重要性
- **必须在每个 head 处理时重新计算完整 Hinv**
- 不能只在 head 内部做 Local Update
- 需要补偿所有其他列（包括其他 head）

### 2. 重排策略
- 将待删除维度移到 head 最前面
- 便于 Local Update 和 Global Update 的计算
- 处理完成后必须恢复原始顺序

### 3. H 的更新
- 每处理完一个 head，必须更新 H（清零已删除维度的行列）
- 这会影响下一个 head 的 Hinv 计算

### 4. Torch-Pruning 的通用性
- idx_list 只是要删除的列索引（无论是完整 head 还是分散维度）
- `tp.prune_linear_in_channels` 和 `tp.prune_linear_out_channels` 逻辑完全通用
- 不需要为 head_dim 剪枝修改 Torch-Pruning 代码

### 5. 参数控制检查清单
```python
# 检查所有条件判断都使用 args.use_head_dim_prune
✅ Line 383: if args.use_head_dim_prune and name == "attn.proj"
✅ Line 407: if args.use_head_dim_prune
✅ Line 680: parser.add_argument("--use_head_dim_prune", action="store_true")

# 确保默认值为 False
✅ action="store_true" 默认为 False

# 确保原代码在 else 分支
✅ else: 原有的 struct_prune 逻辑
```

---

## 预期效果

### 参数量
- **相同**: 两种方法最终参数量相同（sparsity 相同时）

### 性能
- **Head_dim 剪枝可能更好**:
  1. 保留所有 head 的功能
  2. 每个 head 个性化优化
  3. 保留所有 learned scale parameters

### 计算效率
- **训练时**: Head_dim 剪枝可能稍慢（需要逐 head 计算 Hinv）
- **推理时**: 相同（参数量和计算量相同）

---

## 向后兼容性保证总结

### ✅ 修改清单

| 文件 | 修改类型 | 影响原代码 | 默认行为 |
|------|---------|----------|---------|
| `slim_utils/slimgpt.py` | 新增方法 | ❌ 无影响 | 不调用新方法 |
| `model_slimming_basic.py` Line 383 | 条件分支 | ❌ 无影响 | 执行 else 分支（原代码） |
| `model_slimming_basic.py` Line 407 | 条件分支 | ❌ 无影响 | 执行 else 分支（原代码） |
| `model_slimming_basic.py` Line 680 | 新增参数 | ❌ 无影响 | False（原行为） |

### ✅ 测试验证

```bash
# 步骤 1: 测试原代码（不加新参数）
python model_slimming_basic.py --model_depth 16 --sparsity 0.25 --num_samples 256

# 预期：完全和之前一样，使用 Head 剪枝

# 步骤 2: 测试新功能（添加新参数）
python model_slimming_basic.py --model_depth 16 --sparsity 0.25 --num_samples 256 --use_head_dim_prune

# 预期：使用 Head_dim 剪枝，num_heads 不变，head_dim 减少

# 步骤 3: 对比结果
# - 参数量应该相同
# - 权重形状应该相同
# - 性能可能不同（需要实验验证）
```

### ✅ 回滚方案

如果新功能有问题，回滚非常简单：

**方案 1: 不使用新参数**
```bash
# 直接去掉 --use_head_dim_prune 参数即可
python model_slimming_basic.py ...  # 不加 --use_head_dim_prune
```

**方案 2: 删除新代码**
```bash
# 1. 删除 slimgpt.py 中的 head_dim_prune 方法
# 2. 删除 model_slimming_basic.py 中的 if args.use_head_dim_prune 分支
# 3. 删除命令行参数定义
# 原代码完全不受影响
```

---

## 参考文档

1. `/home/project/real_prune/slimgpt_pub_prune/slim_utils/HEAD_DIM_PRUNE_README.md`
2. `/home/project/real_prune/slimgpt_pub_prune/slim_utils/head_dim_prune_example.py`
3. `/home/project/real_prune/slimgpt_pub_prune/OBS_PRUNING_EXPLAINED.md`

---

## 作者

**日期**: 2025-11-10
**版本**: v1.0
