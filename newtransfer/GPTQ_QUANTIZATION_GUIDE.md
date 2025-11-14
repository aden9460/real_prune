# VAR模型GPTQ量化完整指南

## 📌 核心问题回答

**问：如果我想用GPTQ量化该怎么做呢？是不是先用GPTQ将PyTorch量化到相应值，然后再用CoreML提供的线性量化函数？**

**答案：**
- ✅ **前半部分正确**：先用GPTQ将PyTorch模型量化
- ❌ **后半部分错误**：不需要再用CoreML的线性量化函数

**正确工作流程：**
```
ImageNet图像 → VQVAE编码成tokens → GPTQ量化 → 转换CoreML → 完成
     ↑              ↑                                      ↑
  校准数据    形成真实输入分布(679,32)          量化信息自动保留
```

**关键点：**
1. GPTQ在PyTorch阶段完成，使用VQVAE编码的tokens作为校准数据
2. 转换到CoreML时量化信息自动保留
3. ❌ **不要**再应用CoreML的线性量化（会导致重复量化，精度下降）

---

## 🚀 快速开始（3步）

### 步骤1：GPTQ量化（PyTorch阶段）

```bash
cd /home/project/real_prune/newtransfer

# 使用int4量化（最小模型）
bash run_gptq_quantization.bash int4

# 或使用int8量化（更好质量）
bash run_gptq_quantization.bash int8
```

### 步骤2：转换到CoreML

```bash
python transfer.py \
    --depth 16 \
    --var_model pytorch_quantized_models/var_d16_gptq_int4.pth
```

### 步骤3：完成

不需要额外步骤，模型已经是量化的。

---

## 📊 GPTQ vs 线性量化对比

### 方法对比

| 特性 | GPTQ量化（新方法） | 线性量化（当前方法） |
|------|------------------|---------------------|
| **应用时机** | PyTorch模型，转换前 | CoreML模型，转换后 |
| **校准数据** | 需要（~128个样本） | 不需要 |
| **数据格式** | ImageNet图像→VQVAE tokens (679,32) | N/A |
| **精度** | 更高（基于真实数据分布） | 较低（简单线性缩放） |
| **复杂度** | 高 | 低 |
| **推荐场景** | 追求最佳精度+压缩比 | 快速量化、无数据 |

### 工作流程对比

**GPTQ方案（推荐）：**
```bash
# 1. PyTorch阶段GPTQ量化
bash run_gptq_quantization.bash int4

# 2. 转换到CoreML（量化自动保留）
python transfer.py --depth 16 --var_model pytorch_quantized_models/var_d16_gptq_int4.pth

# 完成！不需要再量化
```

**线性量化方案（当前）：**
```bash
# 1. 转换到CoreML
python transfer.py --depth 16 --var_model model.pth

# 2. CoreML阶段线性量化
python quantize.py  # dtype="int4"
```

### ❌ 常见错误

**错误做法：** 两种方法叠加
```bash
bash run_gptq_quantization.bash int4  # GPTQ量化
python quantize.py                     # 又做线性量化 ← 错误！重复量化
```

**正确做法：** 选择其中一种
```bash
# 方案A：仅GPTQ
bash run_gptq_quantization.bash int4
python transfer.py ...

# 方案B：仅线性量化
python transfer.py ...
python quantize.py
```

---

## 🔧 Calibration Data详解

### 为什么需要VQVAE Tokens？

**错误做法：** 只使用class labels
```python
calibration_data = labels  # (N,) - 不能代表VAR真实输入
```

**正确做法：** 使用VQVAE编码的tokens
```python
# 1. 加载ImageNet图像
images = load_images()  # (B, 3, 256, 256)

# 2. VQVAE编码（与VAR训练流程一致）
with torch.no_grad():
    gt_idx_Bl = vae.img_to_idxBl(images)  # Multi-scale VQ indices
    tokens = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # (B, 679, 32)

# 3. 用于GPTQ calibration
calibration_data = (labels, tokens)
```

### Token形状说明

VAR使用多尺度VQ编码：
```
Scale 1: 1×1 = 1 token
Scale 2: 2×2 = 4 tokens
Scale 3: 3×3 = 9 tokens
...
Scale 10: 16×16 = 256 tokens
──────────────────────────
总计: 680 tokens

VAR输入: 679 tokens (去掉第一个作为条件)
每个token: 32维embedding
最终形状: (N, 679, 32)
```

### 数据流程

```
ImageNet验证集
    ↓ 采样128张图像
(128, 3, 256, 256)
    ↓ vae.img_to_idxBl()
List[10个尺度的VQ indices]
    ↓ vae.quantize.idxBl_to_var_input()
(128, 679, 32) tokens
    ↓ 用于GPTQ calibration
优化的量化参数
    ↓ 应用量化
GPTQ量化模型
```

---

## ⚙️ 详细配置

### 基本使用

```bash
bash run_gptq_quantization.bash [量化位数] [模型深度] [模型路径] [数据路径]

# 示例
bash run_gptq_quantization.bash int4 16 /path/to/var_model.pth /path/to/imagenet
```

### 参数说明

| 参数 | 可选值 | 说明 |
|------|--------|------|
| 量化位数 | `int4`, `int8` | int4更小(~12.5%)，int8更高质量(~25%) |
| 模型深度 | 12, 16, 20, 24, 30 | VAR模型深度 |
| 模型路径 | 文件路径 | VAR PyTorch模型 |
| 数据路径 | 目录路径（可选） | ImageNet根目录，包含val子目录 |

### 修改脚本配置

编辑 `quantize_gptq_pytorch.py` (第380行左右)：

```python
def main():
    # Configuration
    USE_GPTQ = True              # True=GPTQ, False=LinearQuantizer
    WEIGHT_DTYPE = "int4"        # "int4" 或 "int8"
    CALIBRATION_SAMPLES = 128    # 校准样本数量
```

### ImageNet数据路径

自动检测优先级：
1. 命令行参数 `--data_path`
2. `/home/project/daily/AR/imagenet`
3. `/imagenet`
4. `./imagenet`
5. 如果都没找到 → 使用合成标签（精度较低）

---

## 📈 量化位数选择

### int8 vs int4

| 量化位数 | 模型大小 | 质量损失 | 推荐场景 |
|---------|---------|---------|---------|
| **int8** | ~25%原始 | 很小 | 生产环境，追求质量 |
| **int4** | ~12.5%原始 | 中等 | 资源受限设备，追求压缩 |

### 推荐策略

1. **先试int8**：质量损失小，模型仍减少75%
2. **如果太大**：再试int4
3. **对比FID**：测试实际效果决定

---

## 🐛 故障排查

### 1. LayerwiseCompressor失败

**错误信息：**
```
✗ GPTQ quantization failed: ...
Note: VAR model may not be compatible with LayerwiseCompressor.
Falling back to LinearQuantizer...
```

**原因：** LayerwiseCompressor要求nn.Sequential架构，VAR使用nn.ModuleList

**解决：** 脚本自动fallback到LinearQuantizer（仍在PyTorch阶段，比CoreML线性量化好）

### 2. ImageNet数据未找到

**警告信息：**
```
Warning: ImageNet validation set not found
Using synthetic labels for calibration (less accurate)
```

**影响：** GPTQ精度降低，但仍比后处理线性量化好

**解决方案：**
1. 提供正确路径：`--data_path /path/to/imagenet`
2. 或接受合成数据

### 3. 内存不足

**症状：** CUDA out of memory

**解决：**
```python
# 编辑 quantize_gptq_pytorch.py
CALIBRATION_SAMPLES = 64  # 减少样本数（从128降到64）

# 或在脚本中修改batch_size
process_batch_size = 4  # 从8降到4
```

### 4. 转换CoreML失败

**问题：** 量化模型无法转换

**解决：**
```python
# 加载量化模型时
checkpoint = torch.load(quantized_model_path, map_location='cpu')
if 'model_state_dict' in checkpoint:
    var.load_state_dict(checkpoint['model_state_dict'], strict=True)
else:
    var.load_state_dict(checkpoint, strict=True)
```

---

## 📁 文件结构

```
newtransfer/
├── quantize_gptq_pytorch.py      # GPTQ量化主脚本
├── run_gptq_quantization.bash    # 便捷运行脚本
├── transfer.py                   # CoreML转换脚本
├── quantize.py                   # CoreML线性量化（现有方法）
├── pytorch_quantized_models/     # GPTQ量化后的PyTorch模型
│   ├── var_d16_gptq_int4.pth
│   └── var_d16_gptq_int8.pth
└── coreml_models/                # CoreML模型
    └── d16_0.4_distill.mlpackage
```

---

## 🔬 实现细节

### CalibrationDataLoader

```python
class CalibrationDataLoader:
    """
    准备GPTQ校准数据：
    1. 加载ImageNet图像
    2. VQVAE编码成tokens (N, 679, 32)
    3. 返回(labels, tokens)对
    """
    def __init__(self, vae, data_path, num_samples=128):
        # 加载并编码图像
        self._load_and_encode_images(data_path)

    def _load_and_encode_images(self, data_path):
        # 1. 使用VAR的build_dataset加载图像
        dataset = build_dataset(data_path, final_reso=256)

        # 2. 批量处理
        for batch in batches:
            images = load_images(batch)  # (B, 3, 256, 256)

            # 3. VQVAE编码
            with torch.no_grad():
                gt_idx_Bl = self.vae.img_to_idxBl(images)
                tokens = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)

            # 4. 存储
            self.calibration_tokens.append(tokens.cpu())
```

### API对比

**GPTQ API (coremltools.optimize.torch)：**
```python
from coremltools.optimize.torch.quantization import LayerwiseCompressor

config = LayerwiseCompressorConfig.from_dict({
    "global_config": {
        "algorithm": "gptq",
        "weight_dtype": "int4",
        "granularity": "per_channel",
    },
    "calibration_nsamples": 128,
})

compressor = LayerwiseCompressor(pytorch_model, config)
quantized_model = compressor.compress(calibration_loader)
```

**线性量化API (coremltools.optimize.coreml)：**
```python
import coremltools.optimize.coreml as cto

config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4"
    )
)

compressed_model = cto.linear_quantize_weights(coreml_model, config)
```

---

## 💡 最佳实践

### 1. 对比实验

```bash
# 实验A：GPTQ int4
bash run_gptq_quantization.bash int4
python transfer.py --depth 16 --var_model pytorch_quantized_models/var_d16_gptq_int4.pth
# → 测试FID

# 实验B：GPTQ int8
bash run_gptq_quantization.bash int8
python transfer.py --depth 16 --var_model pytorch_quantized_models/var_d16_gptq_int8.pth
# → 测试FID

# 实验C：线性量化int4（现有方法）
python transfer.py --depth 16 --var_model original_model.pth
python quantize.py  # dtype="int4"
# → 测试FID

# 对比：模型大小、FID分数、推理速度
```

### 2. 选择建议

**选择GPTQ的条件：**
- ✅ 有高质量ImageNet数据
- ✅ 追求最佳精度和压缩比
- ✅ 有时间调优和测试

**选择线性量化的条件：**
- ✅ 需要快速验证
- ✅ 没有校准数据
- ✅ 当前方法已满足需求
- ✅ GPTQ遇到兼容性问题

---

## ❓ 常见问题

**Q: GPTQ一定比线性量化好吗？**

A: 理论上是，但需要满足：
- 有足够的高质量校准数据
- 模型架构兼容
- 正确配置参数

如果条件不满足，差距可能不大。

**Q: 我应该用int4还是int8？**

A: 推荐先试int8（质量损失小），如果模型还是太大再试int4。

**Q: 可以在GPTQ后再做CoreML线性量化吗？**

A: **不推荐！** 这会导致重复量化，精度大幅下降。GPTQ量化后转换到CoreML时量化信息已经保留。

**Q: 没有ImageNet数据怎么办？**

A: 脚本会自动使用合成标签（精度降低但仍可用），或者可以提供其他图像数据集。

**Q: LayerwiseCompressor失败怎么办？**

A: 脚本会自动fallback到LinearQuantizer，这仍是PyTorch阶段的量化，比CoreML后处理好。

---

## 🎯 总结

### 核心要点

1. ✅ GPTQ在PyTorch阶段完成
2. ✅ 使用VQVAE编码的tokens (679, 32)作为校准数据
3. ✅ 转换到CoreML时量化自动保留
4. ❌ 不需要（也不应该）再做线性量化
5. 💡 运行：`bash run_gptq_quantization.bash int4`

### 立即开始

```bash
cd /home/project/real_prune/newtransfer
bash run_gptq_quantization.bash int4
```

### 相关文件

- `quantize_gptq_pytorch.py` - GPTQ量化主脚本（含详细注释）
- `run_gptq_quantization.bash` - 便捷运行脚本
- `quantize.py` - CoreML线性量化脚本（现有方法）
- `transfer.py` - CoreML转换脚本
