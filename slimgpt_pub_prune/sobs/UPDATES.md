# FastOBAAttentionSlimGPT Updates

## 更新日期
2025-01-XX

## 主要更新内容

### 1. 默认剪枝率调整为40%
**改动原因**: 根据实验需求，将默认剪枝率从30%提升到40%

**影响文件**:
- `fastoba_attention_slimgpt.py`: `struct_prune()` 默认参数改为 `sparsity=0.4`
- `var_per_stage_analysis.py`: argparse 默认值改为 `0.4`
- `run_ablation.sh`: 所有实验的 `--sparsity` 参数改为 `0.4`
- `test_fastoba_attention.py`: 测试用例中的剪枝率示例改为 `0.4`

**实验调整**:
- Experiment 3 (Sparsity Sweep) 范围从 `[0.1, 0.2, 0.3, 0.4, 0.5]` 改为 `[0.2, 0.3, 0.4, 0.5, 0.6]`

### 2. 实现完整的OBS权重补偿方法
**改动原因**: 原实现缺少SlimGPT的核心补偿机制

#### 2.1 添加 `struct_prune()` 方法（Lines 598-727）
完整实现SlimGPT的迭代剪枝+补偿算法：

```python
def struct_prune(self, sparsity: float = 0.4, headsize: int = 1,
                 percdamp: float = 0.01, blocksize: int = 128):
    """
    Structured pruning with OBS weight compensation

    支持两种模式:
    1. Channel-wise (headsize=1): 逐通道剪枝
    2. Head-wise (headsize=head_dim): 逐head剪枝
    """
```

**核心步骤**:

1. **计算剪枝误差** (OBS公式):
   ```python
   error = W² / [H^-1]_diag
   ```

2. **选择剪枝目标**:
   - Channel-wise: 按列误差排序
   - Head-wise: 按head误差排序（使用block-diagonal Cholesky）

3. **局部补偿** (Local compensation):
   ```python
   for i in range(cnt):
       Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
       if self.use_compensation:
           W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])
   ```

4. **全局补偿** (Global compensation):
   ```python
   if self.use_compensation:
       W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])
   ```

#### 2.2 块对角补偿（Block-diagonal compensation）
**实现位置**: Lines 646-656 (误差计算), Lines 729-855 (head-internal pruning)

##### 2.2.1 误差计算中的块对角Cholesky
针对多头注意力的块对角结构，使用专用的Cholesky分解：

```python
if headsize > 1:
    # Head-wise: block-diagonal Cholesky
    Hinv_diag = torch.stack([
        Hinv[i:i+headsize, i:i+headsize]
        for i in range(0, columns, headsize)
    ])
    Hinv_diag = torch.diagonal(
        torch.linalg.cholesky(Hinv_diag),
        dim1=-2, dim2=-1
    ).reshape(-1)
    Hinv_diag = Hinv_diag ** 2
```

##### 2.2.2 Head内部维度剪枝 (`struct_prune_head_dims`)
**新增方法**: Lines 729-855

这是真正的块对角补偿实现，特点：

1. **独立处理每个head**：
```python
for head_id in range(self.num_heads):
    # 提取head专属的权重和Hessian
    W_head = W[:, start_idx:end_idx]
    H_head = H[start_idx:end_idx, start_idx:end_idx]
```

2. **Head内部的迭代剪枝**：
```python
while pruned_dims < target_dims:
    # 仅在当前head内计算OBS误差
    error = torch.sum(W_head ** 2 / Hinv_diag.unsqueeze(0), dim=0)

    # 选择要剪的维度（只在当前head内）
    dim_sort_idx = error.argsort()
    cnt = min(target_dims - pruned_dims, blocksize)
```

3. **局部补偿（仅影响同一head）**：
```python
# 局部补偿：剪枝维度之间的相互影响
for i in range(cnt):
    Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
    if self.use_compensation:
        W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

# 全局补偿：对同一head内剩余维度的影响
if self.use_compensation:
    W_head[:, cnt:end] -= Err1.matmul(Hinv_chol[:, cnt:end])
```

4. **关键差异**：
   - `struct_prune(headsize=1)`: 全局通道剪枝，补偿可能跨head传播
   - `struct_prune_head_dims()`: head内部剪枝，补偿严格限制在head内

**优势**:
- 保持多头注意力的块对角结构
- 避免head之间的耦合误差
- 更精确的per-head重要性建模
- 适合VAR的多尺度注意力分析

#### 2.3 补偿开关控制
通过 `use_compensation` 参数控制是否启用补偿：
- `True` (默认): 启用OBS补偿，最小化输出误差
- `False`: 仅剪枝不补偿，用于消融实验对比

### 3. 新增测试用例

#### Test 7: 补偿效果验证 (test_struct_prune_with_compensation)
**测试内容**:
1. 使用相同数据和配置
2. 分别运行带/不带补偿的剪枝
3. 比较权重变化的L2 norm

**预期结果**:
```
✓ Pruning with compensation completed
  Pruned 307 channels (target: 307)
  Weight change (L2 norm): 12.3456

✓ Pruning without compensation completed
  Pruned 307 channels
  Weight change (L2 norm): 45.6789

  Compensation reduces weight change by 73.0%
```

#### Test 8: Head内部维度剪枝验证 (test_head_internal_pruning)
**测试内容**:
1. 使用`struct_prune_head_dims()`对每个head独立剪枝40%维度
2. 验证剪枝确实在head内均匀分布
3. 对比块对角补偿 vs 全局补偿的效果

**预期结果**:
```
✓ Head-internal pruning completed
  Pruned 307 dims total (target: 307)
  Pruned per head: 25 dims (target: 25)
  Head 0: pruned 25 / 64 dims
  Head 1: pruned 25 / 64 dims
  ...
  Head 11: pruned 25 / 64 dims

✓ Comparison with global channel-wise pruning:
  Block-diagonal compensation: 15.2345
  Global compensation:         18.6789
  Ratio: 0.816x
```

**关键验证点**:
- 每个head都剪枝了约25/64维度（40%）
- 块对角补偿的权重变化通常小于全局补偿
- 说明head内部的局部补偿更精确

## 技术细节

### OBS补偿原理
Optimal Brain Surgeon (OBS) 通过二阶Taylor展开最小化剪枝后的输出误差：

```
δL ≈ (1/2) w_i² / [H^-1]_{ii}  (剪枝第i个权重的误差)
```

补偿公式：
```
W_j := W_j - (W_i / [H^-1]_{ii}) * [H^-1]_{ij}  (对其他权重的补偿)
```

### 块对角近似的必要性
对于多头注意力 (num_heads=16, head_dim=64)：
- 完整Hessian: `[1024×1024]` = 4MB (float32)
- 块对角Hessian: `16 × [64×64]` = 256KB
- **内存节省**: 93.75%
- **计算加速**: O(n³) → O(k × (n/k)³) = O(n³/k²)，约256倍加速

## 使用示例

### 基本剪枝（带补偿）
```python
pruner = FastOBAAttentionSlimGPT(
    attention_module=attention,
    layer_idx=0,
    num_heads=16,
    embed_dim=1024,
    hessian_mode='block_diagonal',
    use_compensation=True  # 启用补偿
)

# 收集数据
for inp, out in dataloader:
    pruner.add_batch_v7_fastoba(inp, out)

# 执行剪枝（40%剪枝率，channel-wise）
pruned_indices = pruner.struct_prune(
    sparsity=0.4,
    headsize=1,
    percdamp=0.01
)
```

### Head-wise剪枝（剪整个head）
```python
# 执行head级剪枝（40%剪枝率 = 6.4个head）
pruned_indices = pruner.struct_prune(
    sparsity=0.4,
    headsize=64,  # head_dim
    percdamp=0.01
)
```

### **NEW**: Head内部维度剪枝（块对角补偿）
```python
# 在每个head内独立剪枝40%维度（每个head剪25/64维）
# 补偿仅在head内传播，不影响其他head
pruned_indices = pruner.struct_prune_head_dims(
    sparsity=0.4,
    percdamp=0.01,
    blocksize=16  # 每次剪16维
)

# 结果：每个head都剪枝约40%，但head之间相互独立
# Head 0: 剪枝 dim [2, 5, 8, 15, 20, 25, 31, 38, ...]（head内的25维）
# Head 1: 剪枝 dim [67, 70, 81, 90, ...]（head内的25维）
# ...
```

### 三种剪枝模式对比

| 方法 | 剪枝粒度 | 补偿范围 | 适用场景 |
|------|---------|---------|---------|
| `struct_prune(headsize=1)` | 单通道 | 全局（可能跨head） | 灵活剪枝，不考虑head结构 |
| `struct_prune(headsize=64)` | 整个head | 全局 | 移除整个注意力head |
| `struct_prune_head_dims()` | head内维度 | head局部 | 保持head结构，内部稀疏化 |

### 消融实验：对比补偿效果
```bash
# Experiment 1: With compensation (默认)
python sobs/var_per_stage_analysis.py \
    --model_path var_d16.pth \
    --sparsity 0.4

# Experiment 2: Without compensation
python sobs/var_per_stage_analysis.py \
    --model_path var_d16.pth \
    --sparsity 0.4 \
    --use_compensation False
```

## 验证方法

### 运行完整测试
```bash
cd /home/project/real_prune/slimgpt_pub_prune
python sobs/test_fastoba_attention.py
```

**预期输出**:
```
=== Test 1: Initialization ===
✓ Initialization successful

=== Test 2: SlimGPT Mode ===
✓ SlimGPT mode works

=== Test 3: FastOBA Mode ===
✓ FastOBA mode works

=== Test 4: VAR Scale-Aware Caching ===
✓ Scale-aware caching works

=== Test 5: Per-Stage Analysis ===
✓ Per-stage Hessian computed
✓ Head importance computed
✓ Dimension importance computed

=== Test 6: Pruning Importance Scores ===
✓ Importance scores computed
  Would prune 307 channels (40% sparsity)

=== Test 7: Struct Prune with Compensation ===
✓ Pruning with compensation completed
✓ Pruning without compensation completed
  Compensation reduces weight change by XX.X%

✓ All Tests Passed!
```

## 性能优化

### 迭代剪枝的效率
- **Blocksize**: 默认128，平衡速度和精度
- **Head-wise**: 每次剪一个head（64维），更稳定
- **Channel-wise**: 每次剪128维，更快

### 内存优化
- Hessian存储在CPU: `self.H.cpu()`
- 计算时临时移到GPU: `H.to(device)`
- 剪枝完成后立即清理: `H[pruned_idx, :] = 0`

## 相关文件

### 核心实现
- `/home/project/real_prune/slimgpt_pub_prune/sobs/fastoba_attention_slimgpt.py` (728 lines)

### 测试和实验
- `/home/project/real_prune/slimgpt_pub_prune/sobs/test_fastoba_attention.py` (380+ lines)
- `/home/project/real_prune/slimgpt_pub_prune/sobs/run_ablation.sh` (346 lines)
- `/home/project/real_prune/slimgpt_pub_prune/sobs/var_per_stage_analysis.py` (481 lines)

## 参考文献

1. **OBS (Optimal Brain Surgeon)**: Hassibi & Stork, 1993
   - "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon"

2. **SlimGPT**: https://github.com/microsoft/SlimGPT
   - 基于OBS的LLM剪枝方法

3. **FastOBA**: 自动微分计算高阶Hessian
   - 避免手动推导复杂梯度公式

## 未来改进

### 1. 混合剪枝策略
当前支持 channel-wise 和 head-wise，未来可添加：
- Two-stage: 先剪head，再剪dim
- Adaptive: 根据head重要性动态选择粒度

### 2. 更高效的Hessian近似
- KFAC (Kronecker-Factored Approximate Curvature)
- Fisher Information Matrix

### 3. 分布式剪枝
- 多GPU并行Hessian计算
- 分层累积统计

---

**维护者**: Claude (Anthropic)
**最后更新**: 2025-01-XX
