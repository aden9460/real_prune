# Head维度剪枝 (head_dim_prune) 功能说明

## 概述

在原有的`struct_prune`基础上，新增了`head_dim_prune`方法，用于对attention层进行**个性化head维度剪枝**。

## 主要区别

| 特性 | struct_prune（原有） | head_dim_prune（新增） |
|------|---------------------|----------------------|
| **删除对象** | 完整的head | 每个head内的部分维度 |
| **head数量** | 减少（12→9） | 保持不变（12） |
| **head维度** | 保持不变（64） | 减少（64→48） |
| **个性化** | ❌ 删除整个head | ✅ 每个head独立评估 |
| **Global Update** | ✅ 需要 | ❌ 不需要 |
| **计算复杂度** | 高（192×192矩阵） | 低（12个64×64矩阵） |
| **并行化** | 困难 | ✅ 容易 |
| **适用场景** | 减少head数量 | 减少每个head的复杂度 |

## 使用方法

### 基本调用

```python
from slim_utils.slimgpt import SlimGPT

# 创建pruner
pruner = SlimGPT(layer, layer_idx=0, args=args)

# 统计Hessian
for batch in dataloader:
    inp = ...
    out = layer(inp)
    pruner.add_batch(inp, out)

# 执行head_dim剪枝（每个head删除25%维度）
pruned_indices = pruner.head_dim_prune(
    sparsity=0.25,      # 稀疏度：每个head删除25%维度
    headsize=64,        # 每个head的维度
    percdamp=0.01,      # Hessian阻尼系数
    layer_idx=0         # 层索引（用于日志）
)
```

### 参数说明

- **sparsity**: 稀疏度（0-1之间），表示每个head删除的维度比例
  - 0.25 → 每个head删除25%维度（64→48）
  - 0.5 → 每个head删除50%维度（64→32）

- **headsize**: 每个head的维度，默认64
  - 确保 `layer.weight.shape[1] % headsize == 0`

- **percdamp**: Hessian对角线阻尼系数，默认0.0
  - 增加数值稳定性，建议0.01

- **layer_idx**: 层索引，用于打印日志

### 返回值

- **pruned_indices**: torch.Tensor
  - 被删除的列索引（全局索引）
  - 可用于追踪哪些维度被删除

## 核心优势

### 1. 个性化剪枝

每个head根据自己的特征分布独立决定删除哪些维度：

```python
Head 0: 删除 [dim3, dim17, dim25, ...]  # Head 0最不重要的维度
Head 1: 删除 [dim7, dim42, dim55, ...]  # Head 1最不重要的维度
Head 2: 删除 [dim1, dim31, dim45, ...]  # Head 2最不重要的维度
```

### 2. 计算效率高

- 无需Global Update（只在head内部做Local Update）
- 每个head独立处理小矩阵（64×64）
- 可并行化处理

### 3. 保持结构规整

```python
原始: [num_heads=12, head_dim=64] = 768维
剪枝: [num_heads=12, head_dim=48] = 576维

# reshape仍然有效
Q = Q.view(batch, seq_len, 12, 48)  ✓
```

## 实现细节

### 算法流程

1. **逐head处理**：对每个head（64维）独立评估
2. **计算head内误差**：使用OBS公式在head内部计算
3. **选择删除维度**：选择该head内误差最小的维度
4. **Local Update**：只在head内部进行权重补偿
5. **清零并更新**：清零被删除维度，更新Hessian

### 核心代码片段

```python
# 对每个head独立处理
for head_idx in range(num_heads):
    # 1. 提取head子空间
    W_head = W[:, start:end]
    H_head = H[start:end, start:end]

    # 2. 计算head内误差
    error_head = compute_obs_error(W_head, H_head)

    # 3. 选择删除维度
    dims_to_remove = error_head.argsort()[:k]

    # 4. Local Update（只在head内）
    apply_local_update(W_head, H_head, dims_to_remove)

    # 5. 清零并更新
    W_head[:, dims_to_remove] = 0
    H_head[dims_to_remove, :] = 0
```

## 使用示例

详细示例请参考：`slim_utils/head_dim_prune_example.py`

示例包括：
1. 基本使用
2. 与struct_prune对比
3. 不同稀疏度效果
4. 实际应用场景（Transformer QKV投影）

运行示例：
```bash
cd /home/project/real_prune/slimgpt_pub_prune/slim_utils
python head_dim_prune_example.py
```

## 兼容性

### 向后兼容

- ✅ 完全兼容原有的`struct_prune`方法
- ✅ 不修改任何原有代码
- ✅ 新增方法，原有调用不受影响

### 调用方式

```python
# 原有方法（仍然可用）
pruner.struct_prune(sparsity=0.25, headsize=64)

# 新方法
pruner.head_dim_prune(sparsity=0.25, headsize=64)
```

## 理论基础

### OBS在子空间中的应用

Head_dim剪枝是将OBS算法应用在独立子空间：

```
完整空间: R^192

分解为独立子空间:
  Head 0: R^64
  Head 1: R^64  ⊕  独立
  Head 2: R^64

在每个子空间独立应用OBS:
  Head 0: R^64 → R^48
  Head 1: R^64 → R^48
  Head 2: R^64 → R^48
```

### 关键假设

**Head间影响较小**：

```python
完整Hessian:
        Head0    Head1    Head2
      ┌──────┬──────┬──────┐
Head0 │ H_00 │ H_01 │ H_02 │  H_00 强相关
      │强关联│弱关联│弱关联│  H_01 弱相关
      ├──────┼──────┼──────┤
Head1 │      │ H_11 │ H_12 │
      ├──────┼──────┼──────┤
Head2 │      │      │ H_22 │
      └──────┴──────┴──────┘
```

因此只需在head内做Local Update，无需Global Update。

## 性能对比

### 计算复杂度

| 操作 | struct_prune | head_dim_prune |
|------|-------------|----------------|
| **Hessian逆** | O(192³) | 12 × O(64³) |
| **误差计算** | O(192) | 12 × O(64) |
| **Local Update** | O(N × 64) | 12 × O(N × k) |
| **Global Update** | O(N × 192) | ❌ 无需 |

其中 N 是rows（通常很大），k 是每个head删除的维度数。

### 内存占用

- struct_prune: 需要完整的192×192 Hessian逆
- head_dim_prune: 只需要64×64子矩阵（逐个处理）

## 适用场景

### 推荐使用head_dim_prune的场景

1. ✅ 需要保持多头结构
2. ✅ 希望每个head个性化优化
3. ✅ 计算资源受限（内存/时间）
4. ✅ 模型的head间相关性较弱

### 推荐使用struct_prune的场景

1. ✅ 需要大幅减少head数量
2. ✅ 某些head整体不重要
3. ✅ 需要最高精度的跨head补偿

## 文档

详细的算法原理和数学推导请参考：
- **完整文档**: `/home/project/real_prune/slimgpt_pub_prune/OBS_PRUNING_EXPLAINED.md`
- **第七部分**: 扩展应用 - Head维度剪枝

## 更新日志

### 2025-11-03
- ✅ 新增 `head_dim_prune` 方法
- ✅ 添加详细的文档和示例
- ✅ 保持向后兼容性
- ✅ 完整的错误处理和日志

## 作者

Claude (Anthropic) - 2025-11-03

## 许可

遵循项目原有许可证
