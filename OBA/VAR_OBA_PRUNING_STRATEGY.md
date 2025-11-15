# VAR模型OBA剪枝实现策略文档

**最后更新**: 2025-11-12
**状态**: 初始版本 - 待验证和更新

---

## 1. 实现目标

### 1.1 核心目标
- **模型**: VAR (Visual Autoregressive Transformer) - depth 16
- **剪枝方法**: OBA (Optimal Brain Apoptosis) - 基于完整Hessian-vector积的结构化剪枝
- **目标稀疏度**: 40% (剪枝至60% FLOPs)
- **剪枝策略**: 一次性剪枝（One-shot pruning）

### 1.2 目标层
根据VAR架构和参考实现 (`slimvar/model_slimming_basic_v1.py`)，剪枝以下层：

1. **注意力机制输出投影输入** (`attn.proj` 输入通道)
   - 耦合层: `attn.mat_qkv` 输出通道
   - 剪枝粒度: 整个attention head (head_dim=64固定)
   - 层数: 16层 × 16 heads = 256 heads total

2. **QKV投影输出** (`attn.mat_qkv` 输出通道)
   - 直接耦合: `attn.proj` 输入通道
   - 输出维度: embed_dim × 3 (Q, K, V concatenated)

3. **FFN第二层输入** (`ffn.fc2` 输入通道)
   - 耦合层: `ffn.fc1` 输出通道
   - 默认维度: embed_dim × mlp_ratio (1024 × 4 = 4096)

4. **FFN第一层输出** (`ffn.fc1` 输出通道)
   - 直接耦合: `ffn.fc2` 输入通道

### 1.3 与SlimGPT的对比

| 维度 | SlimGPT (现有) | OBA (目标) |
|------|----------------|------------|
| 理论基础 | Fisher信息矩阵对角近似 | 完整Hessian-vector积 |
| 连接性建模 | 逐层独立 | 上行/下行/并行三种连接性 |
| 注意力处理 | 基于权重+激活统计 | 显式Q-K-V交互建模 |
| 全局优化 | 逐层贪心 | 全局联合优化 |
| 补偿机制 | Cholesky-based error compensation | Hessian-guided importance redistribution |

**预期优势**: OBA应该在跨层依赖较强的情况下表现更好，特别是对于自回归模型如VAR。

---

## 2. 技术路线

### 2.1 整体流程

```
加载预训练VAR模型
    ↓
准备校准数据 (ImageNet → VAE tokens)
    ↓
初始化OBAPruner + 注册hooks
    ↓
前向传播收集激活
    ↓
反向传播计算Hessian importance
    ↓
    ├── Self importance (first-order Taylor)
    ├── Upward connectivity (direct output impact)
    ├── Downward connectivity (upstream propagation)
    └── Parallel connectivity (Q-K-V interactions)
    ↓
全局排序 + 执行剪枝
    ↓
微调pruned模型
    ↓
评估FID on ImageNet validation
```

### 2.2 OBA算法核心

**重要性计算公式** (来自 `torch_pruning/pruner/oba_importance.py`):

```python
# 1. Self importance (first-order)
I_self = delta * (δw ⊙ ∂L/∂w)

# 2. Upward connectivity (direct layer output impact)
# Perturb weight: w' = w + δw
# Forward: y' = f(x, w')
# Backward: y'.backward(grad_output)
I_upward = upward_delta * (∂²L/∂w∂y ⊙ δw)

# 3. Downward connectivity (upstream layer impact)
# Use JVP to propagate through computational graph
I_downward = downward_delta * JVP(upstream_grads, δw)

# 4. Parallel connectivity (for attention: Q-K-V)
# Compute cross-derivatives between coupled parameters
I_parallel = parallel_delta * ∂²L/(∂w_q ∂w_k)

# Total importance
I_total = |I_self + I_upward + I_downward + I_parallel|
```

### 2.3 VAR特殊处理

#### 架构特点
- **自回归性**: 使用causal masking，剪枝不能破坏因果结构
- **多尺度token**: 10个scales，总序列长度680 tokens
- **条件生成**: 使用AdaLN with class conditioning
- **固定head维度**: head_dim=64，只能剪枝整个head

#### 需要特别注意的点
1. **mat_qkv层**: 输出3倍embed_dim，需要确保Q/K/V同步剪枝
2. **AdaLN**: 条件归一化层不剪枝，保留完整维度
3. **位置编码**: pos_1LC包含多尺度位置信息，不剪枝
4. **自回归mask**: attn_bias_for_masking必须保持有效

---

## 3. 文件清单

### 3.1 需要创建的文件

#### A. `OBA/var_prune_oba.py` (主剪枝脚本)
**功能**:
- 加载预训练VAR模型和VAE
- 准备ImageNet校准数据
- 配置OBAPruner参数
- 执行剪枝和微调
- 保存pruned模型和统计信息

**主要参数**:
```python
--model_depth: 16/20/30 (VAR depth)
--var_ckpt: 预训练VAR checkpoint路径
--vae_ckpt: VAE checkpoint路径
--imagenet_dir: ImageNet数据集路径
--importance_type: OBA
--ops_ratios: 0.6 (目标60% FLOPs)
--num_samples: 256 (校准样本数)
--delta, --upward_delta, --downward_delta, --parallel_delta: 均为1.0
--normalizer: max
--sl_num_epochs: 150 (微调轮数)
--sl_lr: 0.001 (微调学习率)
--batch_size: 8-16 (内存受限)
--save_dir: 保存路径
```

#### B. `OBA/engine/var_dataset.py` (VAR数据加载)
**功能**:
- 实现VARCalibrationDataset类
- 加载ImageNet图像 → VAE编码 → VAR tokens
- 处理class labels
- 支持多尺度token生成

**核心方法**:
```python
class VARCalibrationDataset(Dataset):
    def __init__(self, imagenet_path, vae_model, num_samples, transform):
        # 从ImageNet验证集采样
        # 加载VAE模型用于编码

    def __getitem__(self, idx):
        # 加载图像 (H, W, 3)
        # VAE编码: img → tokens (680, 32)
        # 返回: (label, tokens[:679])  # VAR输入去掉最后一个token
```

#### C. `OBA/scripts/run_var_prune_oba.bash` (运行脚本)
**功能**:
- 简化命令行调用
- 配置默认参数
- 支持多个depth和稀疏度实验

### 3.2 需要修改的文件

#### A. `OBA/registry.py`
**修改内容**:
```python
# 添加VAR模型注册
def get_model(name, depth=16, num_classes=1000, **kwargs):
    if name.startswith('var_d'):
        from engine.models.var import VAR
        depth = int(name.split('_d')[1])
        return VAR(
            depth=depth,
            embed_dim=1024,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=num_classes,
            **kwargs
        )
```

#### B. `OBA/torch_pruning/pruner/algorithms/oba_pruner.py` (可能需要)
**修改内容**:
- 如果现有实现无法正确处理VAR的SelfAttention，添加特殊case
- 确保mat_qkv的3倍输出维度正确分组剪枝
- 可能需要添加：
```python
# 特殊处理VAR的mat_qkv层
if isinstance(module, SelfAttention) and hasattr(module, 'mat_qkv'):
    # mat_qkv输出: [B, L, 3*embed_dim]
    # 需要确保每个head的Q/K/V同步剪枝
    qkv_importance = importance.reshape(3, num_heads, head_dim)
    # 对每个head求和: [num_heads]
    head_importance = qkv_importance.sum(dim=(0, 2))
```

#### C. `OBA/engine/models/var.py` (可能需要复制并修改)
**修改内容**:
- 从 `/home/project/real_prune/VAR_FIDtest/models/basic_var.py` 复制
- 可能需要添加hook支持（如果需要自定义）
- 确保与OBA框架兼容

---

## 4. 核心实现细节

### 4.1 模型初始化

```python
import torch
from engine.models.var import VAR
from engine.models.vae import VQVAE

# 加载VAR模型
var = VAR(depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4.0)
var_ckpt = torch.load(args.var_ckpt, map_location='cpu')
var.load_state_dict(var_ckpt['model_state_dict'])
var = var.cuda().eval()

# 加载VAE用于图像编码
vae = VQVAE(...)
vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')
vae.load_state_dict(vae_ckpt['state_dict'])
vae = vae.cuda().eval()
```

### 4.2 校准数据准备

```python
from engine.var_dataset import VARCalibrationDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = VARCalibrationDataset(
    imagenet_path=args.imagenet_dir,
    vae_model=vae,
    num_samples=256,
    transform=standard_imagenet_transform()
)

# DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=8,  # 受内存限制
    shuffle=True,
    num_workers=4
)
```

### 4.3 OBAPruner配置

```python
from torch_pruning import WrappedPruner
from torch_pruning.pruner.oba_importance import HessianImportance

# Example inputs for dependency graph
example_label = torch.zeros(8, dtype=torch.long).cuda()
example_tokens = torch.randn(8, 679, 32).cuda()
example_inputs = (example_label, example_tokens)

# Importance function
importance = HessianImportance(normalizer="max")

# Ignored layers (不剪枝的层)
ignored_layers = [
    var.class_emb,      # Class embedding
    var.pos_1LC,        # Position embedding
    var.lvl_embed,      # Level embedding
    var.word_embed,     # Token embedding (如果存在)
    var.head,           # Output head
    var.head_nm,        # Head normalization
]

# 为每个attention层指定num_heads
num_heads_dict = {}
for i, block in enumerate(var.blocks):
    num_heads_dict[block.attn.mat_qkv] = 16
    num_heads_dict[block.attn.proj] = 16

# 创建pruner
pruner = WrappedPruner(
    model=var,
    example_inputs=example_inputs,
    importance_type='OBA',
    importance=importance,
    global_pruning=True,        # 全局剪枝
    ops_ratios=0.6,             # 目标60% FLOPs
    delta=1.0,                  # Self importance权重
    upward_delta=1.0,           # Upward connectivity权重
    downward_delta=1.0,         # Downward connectivity权重
    parallel_delta=1.0,         # Parallel connectivity权重 (attention)
    max_pruning_ratio=0.95,     # 单层最大剪枝比例
    normalizer='max',           # Importance归一化方式
    ignored_layers=ignored_layers,
    num_heads=num_heads_dict,   # Attention head配置
    prune_num_heads=True,       # 按head粒度剪枝
    prune_head_dims=False,      # 不剪枝head内部维度
)

# 注册hooks
pruner.register_hooks()
```

### 4.4 剪枝执行

```python
import torch.nn as nn

# 损失函数
criterion = nn.CrossEntropyLoss()

# 收集importance
print("Collecting importance scores...")
for batch_idx, (labels, tokens) in enumerate(train_loader):
    labels = labels.cuda()
    tokens = tokens.cuda()

    # Forward
    logits = var(labels, tokens)  # [B, 680, vocab_size]

    # Compute loss (predict next token)
    # VAR自回归：predict token[i+1] from token[:i]
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tokens_target.reshape(-1)  # 需要准备target tokens
    )

    # 计算Hessian importance
    pruner.obtain_importance(loss)

    # Backward
    loss.backward()

    # 清理中间变量
    if batch_idx % 10 == 0:
        torch.cuda.empty_cache()

    print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

# 执行剪枝
print("Executing pruning...")
pruner.step()

print("Pruning completed!")
print(f"Original FLOPs: {original_flops}")
print(f"Pruned FLOPs: {pruned_flops}")
print(f"Compression ratio: {pruned_flops / original_flops:.2%}")
```

### 4.5 微调训练

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 移除hooks（微调时不需要）
pruner.remove_hooks()

# 优化器
optimizer = AdamW(var.parameters(), lr=0.001, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=150)

# 微调循环
var.train()
for epoch in range(150):
    for labels, tokens in train_loader:
        labels, tokens = labels.cuda(), tokens.cuda()

        optimizer.zero_grad()
        logits = var(labels, tokens)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    scheduler.step()

    # 定期评估
    if (epoch + 1) % 10 == 0:
        fid = evaluate_fid(var, vae, imagenet_val_loader)
        print(f"Epoch {epoch+1}, FID: {fid:.2f}")

# 保存pruned模型
torch.save({
    'model_state_dict': var.state_dict(),
    'pruning_config': pruner.get_pruning_plan(),
    'final_fid': fid,
}, args.save_dir + '/pruned_var_d16_oba_0.6flops.pth')
```

---

## 5. 代码模板与示例

### 5.1 完整main函数框架

```python
def main(args):
    # 1. 加载模型
    var = load_var_model(args.var_ckpt, depth=args.depth)
    vae = load_vae_model(args.vae_ckpt)

    # 2. 准备数据
    train_loader = get_calibration_loader(args.imagenet_dir, vae, args.num_samples)
    val_loader = get_validation_loader(args.imagenet_dir, vae)

    # 3. 初始化pruner
    pruner = initialize_oba_pruner(var, args)

    # 4. 执行剪枝
    execute_pruning(var, pruner, train_loader)

    # 5. 微调
    finetune(var, train_loader, val_loader, args)

    # 6. 评估
    fid = evaluate_fid(var, vae, val_loader)
    print(f"Final FID: {fid:.2f}")

    # 7. 保存
    save_pruned_model(var, pruner, fid, args.save_dir)
```

### 5.2 VAE编码示例

```python
@torch.no_grad()
def encode_image_to_tokens(image, vae):
    """
    将图像编码为VAR token序列

    Args:
        image: [B, 3, 256, 256] ImageNet图像
        vae: 预训练VAE模型

    Returns:
        tokens: [B, 680, 32] VAR输入token序列
    """
    # VAE编码
    z = vae.encode(image)  # [B, C, H, W]

    # 量化为离散tokens
    # VAR使用多尺度：10个scales
    # patch_nums: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    # 总tokens: 1² + 2² + ... + 16² = 680

    tokens = vae.quantize(z)  # [B, 680, codebook_dim=32]

    return tokens
```

### 5.3 FID评估示例

```python
@torch.no_grad()
def evaluate_fid(var, vae, val_loader, num_samples=50000):
    """
    在ImageNet验证集上评估FID

    Args:
        var: Pruned VAR model
        vae: VAE decoder
        val_loader: ImageNet validation loader
        num_samples: 用于FID计算的样本数

    Returns:
        fid_score: float
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048).cuda()
    var.eval()

    for i, (labels, real_images) in enumerate(val_loader):
        if i * val_loader.batch_size >= num_samples:
            break

        labels = labels.cuda()

        # VAR生成
        generated_tokens = var.autoregressive_generate(
            labels,
            top_k=900,
            top_p=0.96
        )  # [B, 680, 32]

        # VAE解码
        generated_images = vae.decode(generated_tokens)  # [B, 3, 256, 256]

        # 更新FID统计
        fid.update(real_images.cuda(), real=True)
        fid.update(generated_images, real=False)

    fid_score = fid.compute().item()
    return fid_score
```

---

## 6. 实验配置

### 6.1 基础配置

```bash
# 命令行示例
python var_prune_oba.py \
    --model_depth 16 \
    --var_ckpt /path/to/var_d16.pth \
    --vae_ckpt /path/to/vae_ch160v4096z32.pth \
    --imagenet_dir /path/to/imagenet \
    --importance_type OBA \
    --ops_ratios 0.6 \
    --num_samples 256 \
    --batch_size 8 \
    --delta 1.0 \
    --upward_delta 1.0 \
    --downward_delta 1.0 \
    --parallel_delta 1.0 \
    --normalizer max \
    --sl_num_epochs 150 \
    --sl_lr 0.001 \
    --save_dir ./checkpoints/var_d16_oba_0.6flops
```

### 6.2 硬件要求估算

- **GPU**: A100 80GB (推荐) 或 4×RTX 3090 24GB
- **内存**: 64GB+ RAM
- **存储**: ~500GB (ImageNet + checkpoints + 中间结果)
- **训练时间估算**:
  - 校准数据收集: ~30分钟
  - Importance计算: ~2-3小时
  - 剪枝执行: <5分钟
  - 微调150 epochs: ~20-30小时

### 6.3 超参数调优建议

如果初始结果不理想，可尝试：

| 参数 | 默认值 | 调优方向 |
|------|--------|----------|
| `delta` | 1.0 | 如果过度依赖权重大小，降低至0.5 |
| `upward_delta` | 1.0 | 如果上层过度剪枝，增加至1.5-2.0 |
| `parallel_delta` | 1.0 | 对attention敏感，可尝试1.5-2.0 |
| `normalizer` | max | 尝试 'mean', 'sum', 'gaussian' |
| `num_samples` | 256 | 增加至512-1024提高稳定性 |
| `sl_lr` | 0.001 | 如果微调不稳定，降低至0.0005 |

---

## 7. 验证方法

### 7.1 剪枝正确性验证

**检查项**:
1. ✅ 模型FLOPs是否达到目标60%
2. ✅ 参数量是否合理减少
3. ✅ 每层剪枝比例是否在max_pruning_ratio内
4. ✅ Attention head数是否为整数（不能有半个head）
5. ✅ 模型forward是否无错误
6. ✅ 输出shape是否正确

```python
# 验证脚本
from thop import profile

original_flops, _ = profile(var_original, inputs=(example_label, example_tokens))
pruned_flops, _ = profile(var_pruned, inputs=(example_label, example_tokens))

print(f"Original FLOPs: {original_flops / 1e9:.2f}G")
print(f"Pruned FLOPs: {pruned_flops / 1e9:.2f}G")
print(f"Ratio: {pruned_flops / original_flops:.2%}")

# 逐层统计
for name, module in var_pruned.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name}: {module.weight.shape}")
```

### 7.2 质量评估

**指标**:
- **主要指标**: FID (Fréchet Inception Distance) on ImageNet 50k validation
- **次要指标**: Inception Score (IS)
- **对比基准**:
  - 原始VAR-d16: FID ~2.5-3.0 (未剪枝)
  - SlimGPT 40%稀疏度: FID ~X.X (需要运行baseline获得)
  - 目标: FID < SlimGPT或相当

```python
# 评估命令
python evaluate_var_fid.py \
    --model_path ./checkpoints/var_d16_oba_0.6flops/best.pth \
    --vae_ckpt /path/to/vae.pth \
    --imagenet_dir /path/to/imagenet \
    --num_samples 50000 \
    --batch_size 50
```

### 7.3 分析工具

**需要记录的统计数据**:
1. 每层剪枝比例分布
2. Attention head保留数量（16层×16 heads → 16层×? heads）
3. FFN hidden_dim变化（默认4096 → ?）
4. Importance score分布（per layer）
5. 训练曲线（loss, FID over epochs）

```python
# 统计脚本示例
def analyze_pruning_result(var_original, var_pruned):
    stats = {}

    for i, (orig_block, pruned_block) in enumerate(zip(var_original.blocks, var_pruned.blocks)):
        # Attention
        orig_qkv_dim = orig_block.attn.mat_qkv.weight.shape[0]
        pruned_qkv_dim = pruned_block.attn.mat_qkv.weight.shape[0]
        attn_ratio = pruned_qkv_dim / orig_qkv_dim

        # FFN
        orig_fc1_dim = orig_block.ffn.fc1.weight.shape[0]
        pruned_fc1_dim = pruned_block.ffn.fc1.weight.shape[0]
        ffn_ratio = pruned_fc1_dim / orig_fc1_dim

        stats[f'layer_{i}'] = {
            'attn_heads_remaining': pruned_qkv_dim // (3 * 64),
            'attn_pruning_ratio': 1 - attn_ratio,
            'ffn_hidden_dim': pruned_fc1_dim,
            'ffn_pruning_ratio': 1 - ffn_ratio,
        }

    return stats
```

---

## 8. 待解决问题与风险

### 8.1 技术问题

**Q1**: VAR的mat_qkv输出3倍维度，OBA是否能正确处理？
- **状态**: 待验证
- **风险**: 可能需要修改oba_pruner.py
- **解决方案**: 如果默认实现不支持，添加特殊case处理

**Q2**: VAR使用AdaLN conditioning，剪枝后AdaLN参数是否需要调整？
- **状态**: 待验证
- **风险**: 如果AdaLN的scale/shift维度不匹配，会导致错误
- **解决方案**: 确保AdaLN层的输入维度与剪枝后的embed_dim一致

**Q3**: 自回归生成时的KV cache是否受剪枝影响？
- **状态**: 待验证
- **风险**: 推理时cache维度不匹配
- **解决方案**: 更新cache相关代码以适应新的head数量

**Q4**: 校准数据256样本是否足够？
- **状态**: 待实验验证
- **风险**: 样本不足可能导致importance估计不准
- **解决方案**: 如果结果不佳，增加至512-1024样本

**Q5**: 内存是否足够？VAR序列长度680，batch_size能否>8？
- **状态**: 待测试
- **风险**: OOM导致无法运行
- **解决方案**: 使用gradient checkpointing或减少batch_size

### 8.2 对比实验问题

**Q6**: SlimGPT baseline结果是什么？
- **状态**: 需要运行 `slimvar/model_slimming_basic_v1.py` 获取
- **重要性**: 必须有baseline才能判断OBA是否更优
- **行动**: 先运行SlimGPT获得FID baseline

**Q7**: 不同delta权重组合的影响？
- **状态**: 需要ablation study
- **实验**: 测试 (1,1,1,1), (1,2,1,1), (1,1,1,2) 等配置
- **目标**: 找到最优权重组合

### 8.3 工程问题

**Q8**: VAR模型文件路径在哪？
- ✅ **已确认**:
  - var_d16.pth: `/home/project/daily/AR/model_zoo/var_d16.pth` (1.2G)
  - vae_ch160v4096z32.pth: `/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth` (416M)
  - 预训练权重已存在

**Q9**: ImageNet数据集在哪？
- ✅ **已确认**: `/home/project/ImageNet-1K/`

**Q10**: 是否需要wandb日志？
- **状态**: 待确认
- **建议**: 强烈推荐，方便跟踪实验

---

## 9. 实施时间表（初步估计）

### Phase 1: 代码准备 (1-2天)
- [ ] 创建 `var_prune_oba.py`
- [ ] 创建 `engine/var_dataset.py`
- [ ] 修改 `registry.py`
- [ ] 测试模型加载和数据准备

### Phase 2: 剪枝验证 (1-2天)
- [ ] 运行OBA剪枝（无微调）
- [ ] 验证剪枝正确性（FLOPs、shape、forward）
- [ ] Debug任何出现的问题
- [ ] 分析importance分布

### Phase 3: 微调训练 (2-3天)
- [ ] 微调pruned模型
- [ ] 监控训练曲线
- [ ] 定期评估FID

### Phase 4: 对比与分析 (1-2天)
- [ ] 运行SlimGPT baseline（如果还没有）
- [ ] 对比OBA vs SlimGPT
- [ ] 可视化剪枝分布
- [ ] 撰写实验报告

### Phase 5: 优化与调优 (可选，1-3天)
- [ ] 根据初步结果调整超参数
- [ ] 尝试不同delta组合
- [ ] 测试不同稀疏度（30%, 50%）

**总计**: 5-12天（取决于问题复杂度和调优需求）

---

## 10. 参考资料

### 相关文件
- `/home/project/real_prune/slimvar/model_slimming_basic_v1.py` - SlimGPT baseline实现
- `/home/project/real_prune/VAR_FIDtest/models/basic_var.py` - VAR模型定义
- `/home/project/real_prune/OBA/train_prune.py` - OBA CIFAR示例
- `/home/project/real_prune/OBA/torch_pruning/pruner/algorithms/oba_pruner.py` - OBA实现

### OBA论文
- 标题: Optimal Brain Apoptosis (假设)
- 核心思想: 使用完整Hessian-vector积进行神经网络剪枝，建模上行/下行/并行连接性

### VAR论文
- 标题: Visual Autoregressive Modeling
- 核心思想: 自回归生成图像，使用transformer和多尺度token表示

---

## 11. 更新日志

**2025-11-12 - v1.0 初始版本**
- 创建策略文档
- 定义实现目标和技术路线
- 规划文件结构和核心代码
- 列出待解决问题

**待更新内容**:
- [ ] 实际checkpoint路径
- [ ] SlimGPT baseline FID结果
- [ ] 第一次运行的实际问题和解决方案
- [ ] 最终超参数配置
- [ ] 最终FID结果和对比分析

---

## 12. 联系与问题反馈

如有任何问题或需要澄清，请随时提出：
- VAR架构理解相关
- OBA算法实现细节
- 工程实施问题
- 实验结果分析

**当前最需要确认的事项**:
1. ✅ VAR和VAE checkpoint路径
2. ✅ ImageNet数据集路径
3. ⚠️ SlimGPT baseline FID结果（用于对比）
4. ⚠️ 可用GPU资源和内存

---

**文档状态**: 🔶 初始版本 - 等待实施验证和更新
