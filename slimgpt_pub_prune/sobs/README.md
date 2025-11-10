# FastOBA + Attention + SlimGPT for VAR Models

快速开始指南和完整功能概述

## 🎯 核心功能

### 1. FastOBA Hessian 计算
使用自动微分计算二阶Hessian，支持：
- **SlimGPT模式**: `H = X @ X^T` (一阶统计)
- **FastOBA模式**: 块对角二阶Hessian (更精确)

### 2. 多尺度VAR支持
- 自动识别VAR的10个尺度（1×1到16×16）
- 逐尺度独立统计和分析
- Per-stage Hessian和importance计算

### 3. 三种剪枝模式

#### Mode 1: 全局通道剪枝
```python
pruned_indices = pruner.struct_prune(sparsity=0.4, headsize=1)
```
- 粒度：单通道
- 补偿：全局（可能跨head）
- 用途：灵活剪枝，不考虑head结构

#### Mode 2: 整头剪枝
```python
pruned_indices = pruner.struct_prune(sparsity=0.4, headsize=64)
```
- 粒度：整个head
- 补偿：全局
- 用途：移除完整的注意力head

#### Mode 3: Head内部维度剪枝 ⭐ NEW
```python
pruned_indices = pruner.struct_prune_head_dims(sparsity=0.4)
```
- 粒度：head内维度
- 补偿：head局部（块对角）
- 用途：保持head结构，内部稀疏化
- **关键特性**：补偿仅在head内传播，不影响其他head

### 4. OBS权重补偿
- 局部补偿：剪枝维度之间
- 全局补偿：对剩余权重
- 块对角补偿：head内部独立补偿

### 5. 分尺度对比分析
- Head importance ranking by scale (Kendall's τ)
- Dimension selection overlap (Jaccard similarity)
- Scale-specific head identification

## 📦 文件结构

```
sobs/
├── fastoba_attention_slimgpt.py    # 核心实现（855行）
├── var_per_stage_analysis.py       # 分尺度分析实验
├── run_ablation.sh                 # 完整消融实验脚本
├── test_fastoba_attention.py       # 单元测试（8个测试）
├── UPDATES.md                      # 详细更新日志
└── README.md                       # 本文件
```

## 🚀 快速开始

### 1. 运行单元测试
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

=== Test 7: Struct Prune with Compensation ===
✓ Pruning with compensation completed
✓ Pruning without compensation completed
  Compensation reduces weight change by XX.X%

=== Test 8: Head-Internal Dimension Pruning ===
✓ Head-internal pruning completed
  Pruned 307 dims total (target: 307)
  Pruned per head: 25 dims (target: 25)
  Head 0: pruned 25 / 64 dims
  ...

✓ All Tests Passed!
```

### 2. VAR-d16 分尺度分析
```bash
# 更新模型路径
vim sobs/var_per_stage_analysis.py  # 或直接用命令行参数

# 运行分析
python sobs/var_per_stage_analysis.py \
    --model_path /path/to/var_d16.pth \
    --data_dir ./data/imagenet \
    --layer_idx 0 \
    --num_samples 128 \
    --output_dir results/per_stage_analysis \
    --hessian_mode block_diagonal \
    --head_importance_mode block_mean \
    --sparsity 0.4
```

**输出**:
- `results.json`: 原始数值结果
- `head_importance_by_stage.png`: 热力图可视化
- `stage_head_ranking_correlation.png`: 相关性矩阵
- `scale_specific_heads.json`: 尺度特定的head

### 3. 完整消融实验
```bash
# 更新配置
vim sobs/run_ablation.sh
# 修改 MODEL_PATH 和 DATA_DIR

# 运行所有实验（5个实验组）
bash sobs/run_ablation.sh
```

**实验内容**:
1. **Hessian模式对比**: SlimGPT vs Block-diagonal
2. **Head importance评估**: block_mean vs slimgpt_mean
3. **剪枝率扫描**: 0.2, 0.3, 0.4, 0.5, 0.6
4. **层级分析**: Layers 0, 4, 8, 12, 15
5. **详细分尺度分析**: 256样本

## 💡 使用示例

### 基础用法
```python
import torch
from fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

# 初始化
pruner = FastOBAAttentionSlimGPT(
    attention_module=model.blocks[0].attn,
    layer_idx=0,
    num_heads=16,
    embed_dim=1024,
    hessian_mode='block_diagonal',
    head_importance_mode='block_mean',
    use_compensation=True,
    fastoba_order=2,
    fastoba_delta=1.0,
    hessian_accumulate_freq=10
)

# 收集数据（VAR-aware）
for batch in dataloader:
    inp, out = batch
    pruner.add_batch_v7_fastoba(inp, out)
    # stage_id 会自动从 inp.shape[1] 推断
    # 1×1=1, 2×2=4, 3×3=9, ..., 16×16=256

# 执行剪枝（选择以下三种之一）

# Option 1: 全局通道剪枝
pruned_indices = pruner.struct_prune(
    sparsity=0.4,
    headsize=1,
    percdamp=0.01
)

# Option 2: 整头剪枝
pruned_indices = pruner.struct_prune(
    sparsity=0.4,
    headsize=64,  # head_dim
    percdamp=0.01
)

# Option 3: Head内部维度剪枝（块对角补偿）
pruned_indices = pruner.struct_prune_head_dims(
    sparsity=0.4,
    percdamp=0.01,
    blocksize=16
)
```

### 分尺度重要性分析
```python
# 计算每个尺度的Hessian
for stage_id in range(10):
    H_stage = pruner.compute_per_stage_hessian(stage_id)
    head_imp = pruner.compute_per_stage_head_importance(stage_id)
    dim_imp = pruner.compute_per_stage_head_dim_importance(stage_id)

    print(f"Stage {stage_id}: Head importance = {head_imp}")
```

### 消融实验：补偿效果
```python
# 带补偿
pruner1 = FastOBAAttentionSlimGPT(..., use_compensation=True)
pruner1.add_batch(inp, out)
W_original = model.layer.weight.data.clone()
pruned_idx1 = pruner1.struct_prune(sparsity=0.4)
W_comp = model.layer.weight.data.clone()

# 不带补偿
model.layer.weight.data = W_original.clone()
pruner2 = FastOBAAttentionSlimGPT(..., use_compensation=False)
pruner2.add_batch(inp, out)
pruned_idx2 = pruner2.struct_prune(sparsity=0.4)
W_no_comp = model.layer.weight.data.clone()

# 对比
print(f"With compensation: {(W_comp - W_original).norm():.4f}")
print(f"Without compensation: {(W_no_comp - W_original).norm():.4f}")
```

## 📊 关键参数

### 初始化参数
| 参数 | 默认值 | 说明 |
|------|-------|------|
| `hessian_mode` | `'block_diagonal'` | `'slimgpt'` 或 `'block_diagonal'` |
| `head_importance_mode` | `'block_mean'` | `'block_mean'` 或 `'slimgpt_mean'` |
| `use_compensation` | `True` | 是否启用OBS补偿 |
| `fastoba_order` | `2` | Taylor展开阶数（2=Hessian） |
| `fastoba_delta` | `1.0` | FastOBA的delta参数 |
| `hessian_accumulate_freq` | `10` | 每N个batch计算一次Hessian |

### 剪枝参数
| 参数 | 默认值 | 说明 |
|------|-------|------|
| `sparsity` | `0.4` | 目标剪枝率（40%） |
| `percdamp` | `0.01` | 数值稳定性的dampening |
| `blocksize` | `128` / `16` | 迭代剪枝的块大小 |
| `headsize` | `1` 或 `64` | head粒度（仅`struct_prune`） |

## 🔬 技术细节

### OBS补偿公式
```
剪枝误差: δL ≈ (1/2) W_i² / [H^-1]_{ii}

补偿公式: W_j := W_j - (W_i / [H^-1]_{ii}) * [H^-1]_{ij}
```

### 块对角Hessian结构
```
H = [H_0   0    0    ...  0   ]  ← Head 0 (64×64)
    [0     H_1  0    ...  0   ]  ← Head 1 (64×64)
    [0     0    H_2  ...  0   ]
    [...   ...  ...  ...  ... ]
    [0     0    0    ...  H_15]  ← Head 15 (64×64)
```

### VAR的10个尺度
```
Stage 0:  1×1   =   1 token
Stage 1:  2×2   =   4 tokens
Stage 2:  3×3   =   9 tokens
Stage 3:  4×4   =  16 tokens
Stage 4:  5×5   =  25 tokens
Stage 5:  6×6   =  36 tokens
Stage 6:  8×8   =  64 tokens
Stage 7: 10×10  = 100 tokens
Stage 8: 13×13  = 169 tokens
Stage 9: 16×16  = 256 tokens
Total:          = 680 tokens
```

## 📈 性能优化

### 内存优化
- Hessian存储在CPU: `self.H.cpu()`
- 计算时临时移到GPU
- 块对角结构节省93.75%内存（1024维）

### 计算优化
- 块对角Cholesky: 约256倍加速
- 迭代剪枝：每次处理blocksize维
- VAR scale-aware: 按序列长度归一化

## 🔍 调试和验证

### 启用调试模式
```python
pruner = FastOBAAttentionSlimGPT(..., debug=True)
```

**调试输出**:
```
Starting pruning: target=307/768 (40.0%)
  Iteration: pruned 128/307
  Iteration: pruned 256/307
  Iteration: pruned 307/307
Pruning complete: 307 channels pruned
```

### 验证块对角结构
```python
H = pruner.H
head_dim = 64

# 检查head 0的块
block_0 = H[:head_dim, :head_dim]
off_diag = H[:head_dim, head_dim:2*head_dim]

print(f"Block norm: {block_0.norm():.4f}")
print(f"Off-diagonal norm: {off_diag.norm():.4f}")
print(f"Ratio: {block_0.norm() / (off_diag.norm() + 1e-8):.2f}x")
# 预期: Ratio > 10x （块对角占优）
```

## 🐛 常见问题

### Q1: Cholesky分解失败
**问题**: `RuntimeError: Cholesky decomposition failed`
**解决**:
- 增大`percdamp`（如0.01 → 0.05）
- 检查Hessian是否包含NaN/Inf
- 使用对角近似作为fallback（代码已内置）

### Q2: 内存不足
**问题**: CUDA out of memory
**解决**:
- 减小`hessian_accumulate_freq`（如10 → 5）
- 使用更小的`blocksize`
- 逐head处理（使用`struct_prune_head_dims`）

### Q3: 剪枝效果不佳
**问题**: 剪枝后性能下降严重
**解决**:
- 确保`use_compensation=True`
- 增加数据量（`num_samples`）
- 尝试`hessian_mode='block_diagonal'`
- 使用更小的`sparsity`（如0.3 → 0.2）

### Q4: 不同尺度的重要性差异大
**问题**: Stage 0 vs Stage 9 的head ranking完全不同
**解答**: 这是正常的！VAR的不同尺度关注不同特征：
- 早期尺度（1×1, 2×2）：全局语义
- 中期尺度（5×5, 8×8）：结构特征
- 后期尺度（13×13, 16×16）：细节纹理

使用`scale_specific_heads.json`识别尺度特定的head。

## 📚 参考资料

### 论文
1. **OBS**: Hassibi & Stork (1993) - "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon"
2. **SlimGPT**: Microsoft - Layer-wise pruning with OBS
3. **FastOBA**: Automatic differentiation for high-order derivatives

### 相关代码
- SlimGPT: https://github.com/microsoft/SlimGPT
- VAR: Visual AutoRegressive modeling
- torch.autograd: PyTorch automatic differentiation

## 📝 更新日志

详见 [UPDATES.md](UPDATES.md)

**最新更新** (2025-01-XX):
- ✅ 默认剪枝率改为40%
- ✅ 实现完整OBS补偿（局部+全局）
- ✅ 新增块对角补偿（head内部维度剪枝）
- ✅ 新增Test 7和Test 8验证补偿效果
- ✅ 支持三种剪枝模式对比

## 🤝 贡献者

**维护者**: Claude (Anthropic)
**项目**: FastOBA + Attention + SlimGPT Integration for VAR Models

---

**License**: MIT (if applicable)

**联系**: 通过GitHub Issues报告问题
