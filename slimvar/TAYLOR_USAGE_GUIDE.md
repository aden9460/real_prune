# LLM-Pruner Taylor方法使用指南

## 快速开始

### 1. Taylor一阶方法（最快）

```bash
python model_slimming_basic_v1.py \
    --model_depth 16 \
    --sparsity 0.2 \
    --prune_method taylor \
    --taylor_type param_first \
    --num_taylor_samples 10 \
    --num_samples 256
```

**特点**：
- 只需要一阶梯度，最快
- 公式: `I = |w · ∂L/∂w|`
- 适合快速测试

### 2. Taylor二阶方法

```bash
python model_slimming_basic_v1.py \
    --model_depth 16 \
    --sparsity 0.2 \
    --prune_method taylor \
    --taylor_type param_second \
    --num_taylor_samples 20 \
    --num_samples 256
```

**特点**：
- 使用二阶梯度（Hessian对角线近似）
- 公式: `I = |w · H_ii · w|` 其中 `H_ii ≈ (∂L/∂w)²`
- 慢于一阶，快于混合

### 3. Taylor混合方法（推荐⭐⭐⭐）

```bash
python model_slimming_basic_v1.py \
    --model_depth 16 \
    --sparsity 0.4 \
    --prune_method taylor \
    --taylor_type param_mix \
    --num_taylor_samples 20 \
    --num_samples 256 \
    --use_images \
    --imagenet_dir /path/to/imagenet
```

**特点**：
- 结合一阶和二阶信息
- 公式: `I = |w · ∂L/∂w - 0.5 · w · (∂L/∂w)² · w|`
- 精度最高，速度适中

## 对比实验

### SlimGPT vs Taylor对比

```bash
# 方案1: SlimGPT (baseline, 精度最高但最慢)
python model_slimming_basic_v1.py \
    --prune_method slimgpt \
    --sparsity 0.2 \
    --num_samples 256 \
    --percdamp 0.01

# 方案2: Taylor param_mix (推荐)
python model_slimming_basic_v1.py \
    --prune_method taylor \
    --taylor_type param_mix \
    --sparsity 0.2 \
    --num_taylor_samples 20

# 方案3: Magnitude (最快但精度最低)
python model_slimming_basic_v1.py \
    --prune_method magnitude \
    --sparsity 0.2
```

### 预期性能对比（20%剪枝率）

| 方法 | 剪枝时间 | 预期FID增加 | 校准样本 |
|------|---------|------------|---------|
| SlimGPT | ~15分钟 | +12% | 256 |
| Taylor param_mix | ~3-5分钟 | +15-18% | 20 |
| Taylor param_first | ~2-3分钟 | +18-22% | 10 |
| Magnitude | ~1分钟 | +25-35% | 0 |

## 参数说明

### Taylor特定参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--prune_method` | `slimgpt` | 选择`taylor`启用Taylor方法 |
| `--taylor_type` | `param_mix` | `param_first`(一阶), `param_second`(二阶), `param_mix`(混合) |
| `--num_taylor_samples` | `20` | Taylor梯度收集样本数，推荐10-50 |

### 通用参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model_depth` | `16` | VAR模型深度 (16/20/24/30) |
| `--sparsity` | `0.2` | 剪枝率 (0-1) |
| `--num_samples` | `256` | 激活传播的总样本数 |
| `--use_images` | `False` | 是否使用真实ImageNet图像 |
| `--imagenet_dir` | - | ImageNet数据集路径 |

## 技术细节

### Taylor方法工作原理

#### Phase 1: 收集二阶梯度（param_second和param_mix需要）

```python
for sample in calibration_data:
    loss = model(sample)
    loss.backward()

    # 累积梯度平方
    for param in model.parameters():
        param.acc_grad += param.grad ** 2 / num_samples

    model.zero_grad()
```

#### Phase 2: 收集一阶梯度（param_first和param_mix需要）

```python
total_loss = 0
for sample in calibration_data:
    total_loss += model(sample)

total_loss /= num_samples
total_loss.backward()

# param.grad 现在包含一阶梯度
```

#### Phase 3: 计算重要性并剪枝

```python
if taylor_type == 'param_mix':
    importance = |W * grad - 0.5 * W * acc_grad * W|
elif taylor_type == 'param_first':
    importance = |W * grad|
elif taylor_type == 'param_second':
    importance = |W * acc_grad * W|

# 剪除重要性最低的channels/heads
```

### 为什么Taylor比SlimGPT快？

| 维度 | SlimGPT | Taylor |
|------|---------|--------|
| **内存** | 存储完整Hessian `[d×d]` | 只存储梯度 `[d]` |
| **计算** | Cholesky分解 O(d³) | 矩阵乘法 O(d²) |
| **校准样本** | 需要256个准确估计完整Hessian | 只需10-20个估计对角线 |
| **补偿机制** | 全局最优补偿（慢但精确） | 无补偿（快但近似） |

### 何时使用Taylor？

**推荐使用Taylor的场景**：
- ✅ 快速实验和参数搜索
- ✅ 大模型（VAR-d30, 600M+参数）
- ✅ 内存受限环境
- ✅ 校准数据稀缺（<50样本）
- ✅ 高剪枝率（>40%），此时精度差距缩小

**推荐使用SlimGPT的场景**：
- ✅ 追求最高精度（论文主实验）
- ✅ 低剪枝率（<30%）
- ✅ 有充足计算资源和时间
- ✅ 校准数据充足（>256样本）

## 常见问题

### Q1: Taylor需要的num_taylor_samples和num_samples有什么区别？

**A**:
- `--num_taylor_samples`: Taylor方法收集梯度的样本数（10-50即可，影响剪枝精度）
- `--num_samples`: 激活传播到下一层的样本数（建议256，影响下层剪枝精度）

### Q2: 为什么Taylor不需要@torch.no_grad()？

**A**:
Taylor需要计算梯度，所以必须移除`@torch.no_grad()`装饰器。我们已经在代码中处理了：
- SlimGPT/Magnitude: 自动使用`torch.no_grad()`（不需要梯度）
- Taylor: 允许梯度计算

### Q3: param_mix为什么比param_first和param_second都好？

**A**:
`param_mix`结合了一阶项（梯度方向）和二阶项（曲率信息）：
- 一阶项: 捕获参数对loss的线性影响
- 二阶项: 捕获loss landscape的弯曲程度
- 混合: 更准确地估计剪枝后的loss变化

### Q4: 可以在Taylor中添加补偿机制吗？

**A**:
理论上可以，但会大幅降低Taylor的速度优势。建议：
- 如果需要补偿，直接使用SlimGPT
- 或者使用Taylor快速筛选 + SlimGPT精排的混合方案（见高级用法）

## 高级用法

### 混合方案：Taylor筛选 + SlimGPT精排

对于追求精度和速度平衡的场景，可以修改代码实现两阶段剪枝：

```python
# 伪代码（需要自己实现）
def hybrid_prune(layer, sparsity):
    # 阶段1: Taylor快速筛选（保留20%候选）
    taylor_imp = compute_taylor_importance(layer)
    candidates = taylor_imp.argsort()[:int(len(taylor_imp) * 0.2)]

    # 阶段2: SlimGPT精确排序候选
    slimgpt_imp = compute_slimgpt_importance(layer, candidates_only=True)
    final_prune = candidates[slimgpt_imp.argsort()[:num_prune]]

    return final_prune
```

预期效果：速度~5分钟，FID+13%（介于两者之间）

## 调试和验证

### 验证剪枝正确性

```bash
# 剪枝后检查模型
python -c "
import torch
model = torch.load('pruned_models/var_taylor_0.2.pth')
print('Pruned model loaded successfully')
"

# 测试前向传播
python model_slimming_basic_v1.py \
    --prune_method taylor \
    --sparsity 0.0  # 不剪枝，仅测试
```

### 查看梯度统计

在代码中添加日志（调试用）：

```python
# 在accumulate_hessian_diag()中添加
print(f"Grad mean: {grad.mean():.6f}, std: {grad.std():.6f}")
print(f"Grad² mean: {grad_squared.mean():.6f}")
```

## 更新日志

- **2025-11-12**: 初始版本，集成LLM-Pruner Taylor方法到VAR剪枝代码
- 支持param_first、param_second、param_mix三种Taylor变体
- 向后兼容原有的SlimGPT和magnitude方法
