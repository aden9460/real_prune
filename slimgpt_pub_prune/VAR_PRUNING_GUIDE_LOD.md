# VAR模型剪枝应用指南

**创建日期**: 2025-11-04
**目标**: 将SlimGPT/FastOBA剪枝方法应用到VAR (Visual Auto-Regressive)模型

---

## 📋 VAR模型结构分析

### 模型架构

VAR模型（`models/var.py`）包含：

```python
VAR模型结构：
├── word_embed: nn.Linear(Cvae, C)  # 输入嵌入
├── class_emb: nn.Embedding         # 类别嵌入
├── pos_1LC: Parameter               # 位置嵌入
├── lvl_embed: nn.Embedding          # 层级嵌入
├── blocks: ModuleList               # 核心Transformer块
│   └── AdaLNSelfAttn × depth (默认16层)
│       ├── attn: SelfAttention     # ← 剪枝目标！
│       │   ├── mat_qkv: Linear(C, 3C)  # QKV投影
│       │   └── proj: Linear(C, C)       # 输出投影（O矩阵）
│       ├── ffn: FFN                # FFN层
│       └── ada_ln: AdaLN           # 自适应LayerNorm
└── head: AdaLNBeforeHead           # 输出头
```

**关键参数**（默认配置）：
- `depth=16`: 16层Transformer块
- `embed_dim=1024` (C): 嵌入维度
- `num_heads=16`: 注意力头数
- `head_dim=64`: 每个头的维度 (1024/16)
- `patch_nums=(1,2,3,4,5,6,8,10,13,16)`: 10个尺度

---

## 🎯 剪枝目标层

### 目标：SelfAttention层的输出投影（proj）

```python
# 位置：models/basic_var.py:78
class SelfAttention(nn.Module):
    def __init__(self, ...):
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)  # ← 剪枝这一层！
```

**为什么剪枝proj（而不是mat_qkv）？**
1. `proj`是O矩阵，对应SlimGPT论文中的剪枝位置
2. `proj`的输入维度对应attention head的分组结构
3. 可以进行head-wise或head-dim剪枝

---

## 🔧 实现方案

### 方案1：使用SlimGPT剪枝（推荐）

基于之前的实验发现，SlimGPT的Fisher方法效果最好。

#### 步骤1：创建VAR剪枝脚本

```python
# 文件：prune_var.py
import torch
import sys
import os
sys.path.insert(0, 'slim_utils')

from slimgpt import SlimGPT
from models.var import VAR
from models.vqvae import VQVAE

def prune_var_model(
    var_model: VAR,
    calibration_data,  # 校准数据：图像batch
    sparsity=0.4,      # 稀疏度
    granularity='head',  # 'head' 或 'head_dim'
    percdamp=0.01,
    device='cuda'
):
    """
    剪枝VAR模型的所有attention层

    Args:
        var_model: VAR模型实例
        calibration_data: 用于统计Hessian的数据 [(images, labels), ...]
        sparsity: 稀疏度（删除比例）
        granularity: 'head' 或 'head_dim'
        percdamp: 阻尼系数
        device: 设备
    """
    var_model.to(device)
    var_model.eval()

    # Args配置
    class Args:
        no_compensate = False
    args = Args()

    print("=" * 80)
    print(f"开始剪枝VAR模型 (depth={var_model.depth}, embed_dim={var_model.C})")
    print(f"  稀疏度: {sparsity:.1%}")
    print(f"  粒度: {granularity}")
    print(f"  校准数据: {len(calibration_data)} batches")
    print("=" * 80)
    print()

    # 遍历所有AdaLNSelfAttn块
    for layer_idx in range(var_model.depth):
        block = var_model.blocks[layer_idx]
        attn_layer = block.attn
        proj_layer = attn_layer.proj  # 输出投影层

        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}/{var_model.depth}: Pruning SelfAttention.proj")
        print(f"{'='*60}")

        # 创建pruner
        pruner = SlimGPT(proj_layer, layer_idx=layer_idx, args=args)

        # 步骤1: 收集Hessian统计量
        print(f"  [1/3] 收集Hessian统计量...")
        for batch_idx, (images, labels) in enumerate(calibration_data):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播到该层
            with torch.no_grad():
                # VAR的前向传播（简化版）
                # 获取attention层的输入和输出
                x = var_model.word_embed(var_model.vae_proxy[0].encode(images))

                # 传播到当前层
                for l in range(layer_idx + 1):
                    if l < layer_idx:
                        x = var_model.blocks[l](x, cond=None)
                    else:
                        # 当前层：获取attention的中间结果
                        attn_input = x
                        # ... 需要hook来获取proj的输入输出

            # 这部分需要根据VAR的实际前向传播调整
            # 见下方完整实现

        print(f"    完成！nsamples={pruner.nsamples}")

        # 步骤2: 执行剪枝
        print(f"  [2/3] 执行剪枝...")
        head_dim = var_model.C // var_model.num_heads

        if granularity == 'head':
            pruned_indices = pruner.struct_prune(
                sparsity=sparsity,
                headsize=head_dim,
                percdamp=percdamp,
                layer_idx=layer_idx
            )
            num_heads_pruned = len(pruned_indices) // head_dim
            print(f"    删除了 {num_heads_pruned} 个head")
        elif granularity == 'head_dim':
            pruned_indices = pruner.head_dim_prune(
                sparsity=sparsity,
                headsize=head_dim,
                percdamp=percdamp,
                layer_idx=layer_idx
            )
            dims_per_head = len(pruned_indices) // var_model.num_heads
            print(f"    每个head删除了 {dims_per_head} 维")
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # 步骤3: 同步剪枝QKV
        print(f"  [3/3] 同步剪枝mat_qkv...")
        sync_prune_qkv(attn_layer, pruned_indices)

        print(f"  Layer {layer_idx} 剪枝完成！")

    print("\n" + "=" * 80)
    print("VAR模型剪枝完成！")
    print("=" * 80)

    return var_model


def sync_prune_qkv(attn_layer, pruned_indices):
    """
    同步剪枝mat_qkv的输出维度

    proj已经被剪枝（包括权重补偿），现在需要：
    1. 清零mat_qkv对应的输出行
    2. 清零q_bias和v_bias对应的元素
    """
    with torch.no_grad():
        # mat_qkv的权重形状：[3*embed_dim, embed_dim]
        # 前1/3是Q，中间1/3是K，后1/3是V
        embed_dim = attn_layer.proj.in_features

        # 清零Q的输出
        attn_layer.mat_qkv.weight.data[pruned_indices, :] = 0

        # 清零K的输出（中间1/3）
        k_indices = [idx + embed_dim for idx in pruned_indices]
        attn_layer.mat_qkv.weight.data[k_indices, :] = 0

        # 清零V的输出（后1/3）
        v_indices = [idx + 2*embed_dim for idx in pruned_indices]
        attn_layer.mat_qkv.weight.data[v_indices, :] = 0

        # 清零bias
        attn_layer.q_bias.data[pruned_indices] = 0
        attn_layer.v_bias.data[v_indices] = 0


def collect_hessian_for_layer(var_model, layer_idx, calibration_data, pruner, device):
    """
    为特定层收集Hessian统计量

    使用hook来截获proj层的输入和输出
    """
    proj_layer = var_model.blocks[layer_idx].attn.proj

    # 用于存储proj的输入
    proj_inputs = []
    proj_outputs = []

    def hook_fn(module, input, output):
        # input是tuple，取第一个元素
        proj_inputs.append(input[0].detach().cpu())
        proj_outputs.append(output.detach().cpu())

    # 注册hook
    handle = proj_layer.register_forward_hook(hook_fn)

    try:
        # 前向传播
        with torch.no_grad():
            for images, labels in calibration_data:
                images = images.to(device)
                labels = labels.to(device)

                # VAR的完整前向传播
                # 注意：需要传入正确的参数
                var_model(images, labels)

        # 使用收集的数据计算Hessian
        print(f"    收集到 {len(proj_inputs)} 个batch的数据")
        for inp, out in zip(proj_inputs, proj_outputs):
            inp = inp.to(device)
            out = out.to(device)

            # 展平batch和seq维度
            inp_flat = inp.reshape(-1, inp.shape[-1])
            out_flat = out.reshape(-1, out.shape[-1])

            # 添加到SlimGPT的统计
            pruner.add_batch(inp_flat, out_flat)

    finally:
        # 移除hook
        handle.remove()


# 完整版本（带hook）
def prune_var_model_with_hooks(
    var_model: VAR,
    calibration_data,
    sparsity=0.4,
    granularity='head',
    percdamp=0.01,
    device='cuda'
):
    """完整版本：使用hook收集数据"""
    var_model.to(device)
    var_model.eval()

    class Args:
        no_compensate = False
    args = Args()

    print("=" * 80)
    print(f"开始剪枝VAR模型")
    print(f"  depth={var_model.depth}, embed_dim={var_model.C}, num_heads={var_model.num_heads}")
    print(f"  稀疏度: {sparsity:.1%}, 粒度: {granularity}")
    print("=" * 80)

    for layer_idx in range(var_model.depth):
        print(f"\n{'='*60}")
        print(f"剪枝 Layer {layer_idx}/{var_model.depth}")
        print(f"{'='*60}")

        proj_layer = var_model.blocks[layer_idx].attn.proj
        pruner = SlimGPT(proj_layer, layer_idx=layer_idx, args=args)

        # 使用hook收集Hessian
        print(f"  [1/3] 收集Hessian (使用hook)...")
        collect_hessian_for_layer(var_model, layer_idx, calibration_data, pruner, device)
        print(f"    nsamples={pruner.nsamples}")

        # 执行剪枝
        print(f"  [2/3] 执行剪枝...")
        head_dim = var_model.C // var_model.num_heads

        if granularity == 'head':
            pruned_indices = pruner.struct_prune(
                sparsity=sparsity, headsize=head_dim,
                percdamp=percdamp, layer_idx=layer_idx
            )
        else:
            pruned_indices = pruner.head_dim_prune(
                sparsity=sparsity, headsize=head_dim,
                percdamp=percdamp, layer_idx=layer_idx
            )

        # 同步剪枝QKV
        print(f"  [3/3] 同步剪枝mat_qkv...")
        sync_prune_qkv(var_model.blocks[layer_idx].attn, pruned_indices)

        print(f"  Layer {layer_idx} 完成！")

    print("\n" + "=" * 80)
    print("VAR模型剪枝完成！")
    print("=" * 80)

    return var_model
```

---

#### 步骤2：准备校准数据

```python
# 文件：prepare_calibration_data.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def prepare_imagenet_calibration_data(
    data_path='/path/to/imagenet/val',
    num_batches=20,
    batch_size=32,
    image_size=256
):
    """
    准备ImageNet校准数据

    Args:
        data_path: ImageNet验证集路径
        num_batches: 使用多少个batch
        batch_size: batch大小
        image_size: 图像尺寸

    Returns:
        calibration_data: [(images, labels), ...]
    """
    # 数据预处理（与VAR训练时相同）
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 收集指定数量的batch
    calibration_data = []
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        calibration_data.append((images, labels))

    print(f"收集了 {len(calibration_data)} 个batch的校准数据")
    print(f"  总样本数: {len(calibration_data) * batch_size}")
    print(f"  图像形状: {calibration_data[0][0].shape}")

    return calibration_data
```

---

#### 步骤3：主剪枝脚本

```python
# 文件：run_var_pruning.py
import torch
import argparse
from prune_var import prune_var_model_with_hooks
from prepare_calibration_data import prepare_imagenet_calibration_data
from models.var import VAR
from models.vqvae import VQVAE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--var_ckpt', type=str, required=True,
                        help='VAR checkpoint path')
    parser.add_argument('--vae_ckpt', type=str, required=True,
                        help='VAE checkpoint path')
    parser.add_argument('--data_path', type=str, required=True,
                        help='ImageNet validation data path')
    parser.add_argument('--sparsity', type=float, default=0.4,
                        help='Sparsity ratio')
    parser.add_argument('--granularity', type=str, default='head',
                        choices=['head', 'head_dim'])
    parser.add_argument('--num_batches', type=int, default=20,
                        help='Number of calibration batches')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default='var_pruned.pth',
                        help='Output checkpoint path')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载VAR模型
    print("加载VAR模型...")
    vae = VQVAE.from_pretrained(args.vae_ckpt).to(device).eval()
    var_model = VAR(
        vae_local=vae,
        depth=16,
        embed_dim=1024,
        num_heads=16
    ).to(device)

    checkpoint = torch.load(args.var_ckpt, map_location=device)
    var_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  模型加载完成！")

    # 2. 准备校准数据
    print("\n准备校准数据...")
    calibration_data = prepare_imagenet_calibration_data(
        data_path=args.data_path,
        num_batches=args.num_batches,
        batch_size=args.batch_size
    )

    # 3. 剪枝
    print("\n开始剪枝...")
    var_model_pruned = prune_var_model_with_hooks(
        var_model=var_model,
        calibration_data=calibration_data,
        sparsity=args.sparsity,
        granularity=args.granularity,
        device=device
    )

    # 4. 保存
    print(f"\n保存剪枝后的模型到 {args.output}...")
    torch.save({
        'model_state_dict': var_model_pruned.state_dict(),
        'sparsity': args.sparsity,
        'granularity': args.granularity,
    }, args.output)

    print("完成！")


if __name__ == '__main__':
    main()
```

---

#### 步骤4：运行剪枝

```bash
# 示例：剪枝40%，head-wise
python run_var_pruning.py \
    --var_ckpt /path/to/var_checkpoint.pth \
    --vae_ckpt /path/to/vae_checkpoint.pth \
    --data_path /path/to/imagenet/val \
    --sparsity 0.4 \
    --granularity head \
    --num_batches 20 \
    --batch_size 32 \
    --output var_pruned_40pct_head.pth

# 示例：剪枝40%，head-dim-wise
python run_var_pruning.py \
    --var_ckpt /path/to/var_checkpoint.pth \
    --vae_ckpt /path/to/vae_checkpoint.pth \
    --data_path /path/to/imagenet/val \
    --sparsity 0.4 \
    --granularity head_dim \
    --output var_pruned_40pct_headdim.pth
```

---

### 方案2：使用FastOBA（实验对比）

如果你想对比FastOBA和SlimGPT的效果：

```python
from sobs.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT

# 创建FastOBA pruner（替代SlimGPT）
pruner_fastoba = FastOBAAttentionSlimGPT(
    attention_module=var_model.blocks[layer_idx].attn,
    layer_idx=layer_idx,
    num_heads=var_model.num_heads,
    embed_dim=var_model.C,
    use_exact_hessian=False,  # 或True（更慢但更精确）
    debug=False
)

# 收集数据（FastOBA需要完整的attention模块）
for images, labels in calibration_data:
    images = images.to(device)
    # 获取attention层的输入（需要hook）
    inp = ...
    out = var_model.blocks[layer_idx].attn(inp, attn_bias=None)
    pruner_fastoba.add_batch(inp, out)

# 剪枝
pruned_indices = pruner_fastoba.struct_prune(sparsity=0.4, headsize=head_dim)
```

---

## 🔍 VAR特殊考虑

### 1. 多尺度结构

VAR有10个尺度（patch_nums），需要考虑：

```python
# VAR的10个尺度：1×1, 2×2, 3×3, ..., 16×16
# 不同尺度的序列长度不同

# 选项A：所有尺度使用相同稀疏度
for layer_idx in range(depth):
    prune_layer(layer_idx, sparsity=0.4)

# 选项B：不同尺度使用不同稀疏度
# 早期层（小尺度）：低稀疏度（更重要）
# 后期层（大尺度）：高稀疏度（可以更激进）
sparsity_schedule = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, ...]
for layer_idx in range(depth):
    prune_layer(layer_idx, sparsity=sparsity_schedule[layer_idx])
```

### 2. 自适应LayerNorm

VAR使用AdaLN（自适应LayerNorm），不影响剪枝逻辑，但需要注意：
- AdaLN不需要剪枝
- 只剪枝SelfAttention.proj

### 3. Flash Attention兼容性

VAR可能使用Flash Attention优化：

```python
# 检查是否使用Flash Attention
if attn_layer.using_flash:
    print("  注意：该层使用Flash Attention")
    # Flash Attention不影响剪枝逻辑
    # 剪枝后模型仍可使用Flash Attention
```

---

## 📊 评估剪枝效果

### 1. FID评估

```python
# 使用VAR的evaluate脚本
python VAR_train/eval.py \
    --var_ckpt var_pruned_40pct.pth \
    --vae_ckpt /path/to/vae.pth \
    --data_path /path/to/imagenet/val \
    --metric fid

# 对比原始模型和剪枝后模型的FID
```

### 2. 生成质量对比

```python
# 生成图像并可视化
from generate import generate_images

# 原始模型
images_original = generate_images(var_model_original, num_images=100)

# 剪枝模型
images_pruned = generate_images(var_model_pruned, num_images=100)

# 计算FID、IS等指标
fid_original = calculate_fid(images_original, real_images)
fid_pruned = calculate_fid(images_pruned, real_images)

print(f"FID degradation: {fid_pruned - fid_original:.2f}")
```

### 3. 模型大小和速度

```python
def evaluate_efficiency(model):
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    sparsity_actual = 1 - non_zero_params / total_params

    # 推理速度
    import time
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 256, 256, 3).cuda()

        # 预热
        for _ in range(10):
            _ = model.generate(dummy_input)

        # 计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = model.generate(dummy_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    print(f"参数量: {total_params:,} → {non_zero_params:,}")
    print(f"实际稀疏度: {sparsity_actual:.2%}")
    print(f"平均推理时间: {elapsed/100:.3f}s")
```

---

## ⚠️ 常见问题

### Q1: VAR的前向传播很复杂，如何获取中间结果？

**A**: 使用PyTorch的hook机制（见上方`collect_hessian_for_layer`函数）

### Q2: 剪枝后FID显著下降怎么办？

**A**:
1. 降低稀疏度（0.4 → 0.3 → 0.2）
2. 使用渐进式剪枝（先剪20%，微调，再剪20%）
3. 使用知识蒸馏恢复性能

### Q3: 不同尺度应该用不同稀疏度吗？

**A**: 建议实验对比：
- 统一稀疏度（简单）
- 分层稀疏度（更优，但需要调优）

### Q4: 剪枝后需要微调吗？

**A**:
- OBS方法包含权重补偿，理论上不需要微调
- 但实践中，轻微微调（1-2 epoch）通常能进一步改善
- 大稀疏度（>50%）强烈建议微调

---

## 🚀 快速开始

### 最简单的使用方式

```bash
# 1. 准备环境
cd /home/project/real_prune/slimgpt_pub_prune

# 2. 复制上面的脚本到对应文件

# 3. 运行剪枝
python run_var_pruning.py \
    --var_ckpt your_var_model.pth \
    --vae_ckpt your_vae_model.pth \
    --data_path /path/to/imagenet \
    --sparsity 0.4 \
    --granularity head \
    --output var_pruned.pth

# 4. 评估
python evaluate_var.py \
    --var_ckpt var_pruned.pth \
    --vae_ckpt your_vae_model.pth
```

---

## 📚 推荐阅读

1. **OBS原理**: `OBS_PRUNING_EXPLAINED.md`
2. **SlimGPT vs FastOBA对比**: `FASTOBA_IMPROVEMENTS.md`
3. **精确Hessian**: `EXACT_HESSIAN_USAGE.md`

---

**总结**：
- ✅ 使用SlimGPT方法（H = X^T X），效果最好
- ✅ 剪枝SelfAttention.proj层
- ✅ 使用hook收集中间结果
- ✅ 注意VAR的多尺度特性
- ✅ 剪枝后评估FID和生成质量

祝剪枝顺利！🎉
