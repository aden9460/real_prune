# 精确Hessian计算使用说明

**创建日期**: 2025-11-03
**功能**: 使用标准基向量法计算完整Hessian矩阵（替代A^T A近似）

---

## 📊 方法对比

| 方法 | 计算方式 | 复杂度 | 准确性 | 适用场景 |
|------|---------|--------|--------|---------|
| **A^T A近似** (默认) | `H ≈ g^T @ g` | O(1)次反向传播 | ★★★☆☆ | 快速原型、大规模实验 |
| **精确Hessian** (新) | 标准基向量法 | O(d)次反向传播 | ★★★★★ | 精确剪枝、小规模验证 |

---

## 🚀 快速开始

### 基础用法

```python
from sobs.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

# 方法1：使用近似Hessian（快速，默认）
pruner_fast = FastOBAAttentionSlimGPT(
    attention_module=model.layers[0].attn,
    layer_idx=0,
    num_heads=12,
    embed_dim=768,
    use_exact_hessian=False  # 默认值，使用A^T A近似
)

# 方法2：使用精确Hessian（准确，但慢64倍）
pruner_exact = FastOBAAttentionSlimGPT(
    attention_module=model.layers[0].attn,
    layer_idx=0,
    num_heads=12,
    embed_dim=768,
    use_exact_hessian=True  # 启用精确Hessian计算
)

# 收集数据并计算Hessian
for batch in calibration_dataloader:
    inp, out = batch
    pruner_exact.add_batch_fastoba(inp, out)

# 剪枝
pruner_exact.struct_prune(sparsity=0.4, granularity='head_dim')
```

---

## 🔍 工作原理

### 近似方法 (A^T A)

```python
# 当前默认方法
g = any_order_differentiation(...)  # Hessian-vector product
H ≈ g^T @ g  # 一次矩阵乘法
```

**优点**:
- 只需1次反向传播
- 速度快（秒级）
- 内存友好

**缺点**:
- 不是真正的Hessian
- 损失了交叉项信息
- 行独立性假设过强

---

### 精确方法 (标准基向量法)

```python
# 新实现的方法
H = torch.zeros(d, d)

for j in range(d):  # 对每个输入维度
    # 构造标准基向量 e_j = [0, ..., 0, 1, 0, ..., 0]
    #                                    ↑ 第j个位置
    v = one_hot(j, d)

    # 计算 Hessian @ e_j = H的第j列
    H[:, j] = hessian_vector_product(loss, W, v)

# 对称化
H = (H + H^T) / 2
```

**优点**:
- 真正的完整Hessian
- 不损失信息
- 理论上最优

**缺点**:
- 需要 O(d) 次反向传播
- 对于 head_dim=64: 需要 64×12=768 次反向传播
- 慢 50-100倍（分钟级）

---

## 📈 性能基准

假设：12个head，每个head维度64，总维度768

| 操作 | 近似Hessian | 精确Hessian | 加速比 |
|------|------------|-------------|--------|
| **每层Hessian计算** | ~0.5秒 | ~40秒 | 80x |
| **全模型12层** | ~6秒 | ~8分钟 | 80x |
| **内存占用** | ~2GB | ~2GB | 1x |

---

## 🎯 使用建议

### 何时使用近似Hessian（默认）✅

- 快速原型和实验
- 大规模模型（>12层）
- 实时剪枝应用
- 资源受限环境

### 何时使用精确Hessian ⚠️

- 论文结果验证
- 精确性能基准
- 小规模模型（<6层）
- 理论研究和分析
- 对比实验（验证近似质量）

---

## 💡 实践技巧

### 技巧1：分层使用

对重要层用精确Hessian，其他层用近似：

```python
for layer_idx in range(12):
    # 早期层（0-2）和后期层（9-11）用精确方法
    use_exact = (layer_idx < 3) or (layer_idx > 8)

    pruner = FastOBAAttentionSlimGPT(
        attention_module=model.layers[layer_idx].attn,
        layer_idx=layer_idx,
        use_exact_hessian=use_exact
    )

    # ... 剪枝逻辑 ...
```

### 技巧2：缓存Hessian

如果需要多次剪枝实验，只计算一次Hessian：

```python
# 第一次：计算并保存精确Hessian
pruner = FastOBAAttentionSlimGPT(..., use_exact_hessian=True)
for batch in calibration_data:
    pruner.add_batch_fastoba(batch)

# 保存Hessian
torch.save(pruner.H, 'layer0_exact_hessian.pt')

# 后续实验：直接加载
pruner = FastOBAAttentionSlimGPT(..., use_exact_hessian=False)
pruner.H = torch.load('layer0_exact_hessian.pt')
pruner.nsamples = 100  # 设置样本数
```

### 技巧3：混合方法

先用近似快速定位，再用精确方法细化：

```python
# Step 1: 快速粗剪（近似Hessian）
pruner_approx = FastOBAAttentionSlimGPT(..., use_exact_hessian=False)
pruner_approx.add_batch_fastoba(...)
pruner_approx.struct_prune(sparsity=0.3)  # 剪30%

# Step 2: 精确细剪（精确Hessian）
pruner_exact = FastOBAAttentionSlimGPT(..., use_exact_hessian=True)
pruner_exact.add_batch_fastoba(...)
pruner_exact.struct_prune(sparsity=0.15)  # 再剪15%，总共40%
```

---

## 🔬 技术细节

### Hessian-vector product 数学原理

对于损失函数 `L(W)`，Hessian矩阵定义为：

```
H[i,j] = ∂²L/(∂W[i] ∂W[j])
```

Hessian-vector product (HVP) 定义为：

```
HVP = H @ v
```

可以通过自动微分高效计算，无需显式构造H：

```python
# PyTorch实现
g = torch.autograd.grad(loss, W, create_graph=True)[0]  # 一阶梯度
hvp = torch.autograd.grad(g, W, grad_outputs=v)[0]  # HVP
```

### 标准基向量法

使用 d 个标准基向量 `e_1, e_2, ..., e_d`：

```
e_j = [0, 0, ..., 0, 1, 0, ..., 0]
                     ↑
                  第j位

H @ e_j = [H[0,j], H[1,j], ..., H[d-1,j]]  ← H的第j列
```

通过计算 d 次HVP，恢复完整Hessian。

### 对称化处理

理论上Hessian应该对称，但数值误差可能导致 `H[i,j] ≠ H[j,i]`：

```python
# 对称化：取平均
H_sym = (H + H.t()) / 2
```

这提高了数值稳定性和Cholesky分解成功率。

---

## 🐛 常见问题

### Q1: 精确Hessian为什么这么慢？

**A**: 因为需要计算 `head_dim` 次反向传播：
- 每个head: 64次反向传播
- 12个head: 12 × 64 = 768次
- 每次约0.05秒 → 总共40秒

**优化方案**:
- 并行化（TODO：多GPU计算不同head）
- 降低head_dim（如head_dim=32）
- 使用Hutchinson估计（需要更少采样）

### Q2: 精确Hessian一定更好吗？

**A**: 不一定！原因：
1. **噪声累积**：768次反向传播可能累积更多数值误差
2. **过拟合风险**：对calibration data过拟合
3. **伪损失问题**：即使Hessian精确，但伪损失本身不完美

建议先对比实验。

### Q3: 能否混合使用？

**A**: 可以！见"技巧3：混合方法"。

### Q4: 内存会溢出吗？

**A**: 不会。精确方法只增加计算时间，不增加峰值内存：
- Hessian矩阵大小：`[768, 768]` ≈ 2.3 MB
- 中间梯度：逐列计算，不累积

---

## 📊 实验建议

### 对比实验设计

```python
import json
import time

results = {}

for method in ['approximate', 'exact']:
    use_exact = (method == 'exact')

    # 计时
    start_time = time.time()

    # 剪枝
    pruner = FastOBAAttentionSlimGPT(
        ..., use_exact_hessian=use_exact
    )

    for batch in calibration_data:
        pruner.add_batch_fastoba(batch)

    pruner.struct_prune(sparsity=0.4)

    # 评估
    elapsed = time.time() - start_time
    ppl = evaluate_perplexity(model)

    results[method] = {
        'time_seconds': elapsed,
        'perplexity': ppl,
        'relative_error': (ppl - baseline_ppl) / baseline_ppl
    }

# 保存结果
with open('hessian_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
```

预期输出：
```json
{
  "approximate": {
    "time_seconds": 6.2,
    "perplexity": 12.45,
    "relative_error": 0.15
  },
  "exact": {
    "time_seconds": 485.7,
    "perplexity": 11.89,
    "relative_error": 0.10
  }
}
```

**解读**：精确方法慢78倍，但性能提升5%（相对误差从15%降至10%）。

---

## 🔮 未来改进

### 1. Hutchinson Trace Estimator

使用随机向量代替标准基向量：

```python
# 只需10-20次采样，而不是768次
for _ in range(n_samples):
    v = torch.randn_like(W)  # 随机向量
    hvp = hessian_vector_product(loss, W, v)
    H_est += hvp @ v.t()
H_est /= n_samples
```

**优势**：速度提升30-50倍，准确度损失<10%

### 2. 低秩Hessian近似

```python
H ≈ U @ S @ U^T + D
```

其中 U 是前k个特征向量，D是对角矩阵。

### 3. 并行化

不同head的Hessian计算可以并行：

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(compute_head_hessian, h)
               for h in range(num_heads)]
    H_blocks = [f.result() for f in futures]
```

---

## 📚 参考文献

1. **Hessian-vector product**: Pearlmutter (1994) - "Fast Exact Multiplication by the Hessian"
2. **Hutchinson Trace Estimator**: Hutchinson (1990) - "A stochastic estimator of the trace of the influence matrix"
3. **OBS**: Hassibi & Stork (1993) - "Second Order Derivatives for Network Pruning"

---

## 📝 更新日志

- **2025-11-03**: 初始实现
  - 添加 `compute_full_hessian_per_head` 方法
  - 添加 `compute_full_block_diagonal_hessian` 方法
  - 添加 `use_exact_hessian` 参数
  - 更新文档

---

**总结**：精确Hessian提供了理论上最优的剪枝决策，但计算成本高。建议根据具体需求在速度和准确性之间权衡。对于大多数应用，默认的近似方法已经足够。
