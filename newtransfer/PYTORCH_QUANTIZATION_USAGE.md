# PyTorch量化使用说明

## 🚀 快速开始

### 使用Linear量化（推荐，默认）

```bash
cd /home/project/real_prune/newtransfer

# 方式1：使用默认参数（linear, int8）
bash run_gptq_quantization.bash

# 方式2：指定int4
bash run_gptq_quantization.bash int4

# 方式3：完整参数
bash run_gptq_quantization.bash int4 16 0.4 linear
```

### 使用GPTQ量化（会自动fallback到Linear）

```bash
# 尝试GPTQ（如果不可用会自动使用Linear）
bash run_gptq_quantization.bash int4 16 0.4 gptq

# GPTQ with ImageNet数据
bash run_gptq_quantization.bash int4 16 0.4 gptq /path/to/model.pth /path/to/imagenet
```

## 📋 参数说明

```bash
bash run_gptq_quantization.bash [dtype] [depth] [sparsity] [method] [model_path] [data_path]
```

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| dtype | int4, int8 | int8 | 量化位数 |
| depth | 12,16,20,24,30 | 16 | 模型深度 |
| sparsity | 0-1的浮点数 | 0.4 | 剪枝率 |
| method | linear, gptq | **linear** | 量化方法 |
| model_path | 文件路径 | 默认路径 | VAR模型路径 |
| data_path | 目录路径 | 默认路径 | ImageNet路径（仅GPTQ需要） |

## 💡 推荐配置

### 方案1：Linear量化（推荐）

**特点：**
- ✅ 在Linux上稳定可用
- ✅ 不需要calibration data
- ✅ 速度快
- ✅ PyTorch阶段完成，转CoreML时保留量化

**使用：**
```bash
# int4量化（最小模型）
bash run_gptq_quantization.bash int4 16 0.4 linear

# int8量化（更好质量）
bash run_gptq_quantization.bash int8 16 0.4 linear
```

**输出：**
```
pytorch_quantized_models/var_d16_s0.4_linear_int4.pth
```

### 方案2：GPTQ量化（实验性）

**特点：**
- ⚠️ 在Linux上可能不可用（会自动fallback到Linear）
- 📊 理论上质量更好（如果可用）
- 📁 需要calibration data

**使用：**
```bash
bash run_gptq_quantization.bash int4 16 0.4 gptq
```

**注意：** 如果GPTQ不可用，会自动使用LinearQuantizer作为fallback

## 📊 输出文件

### 文件命名

```
var_d{depth}_s{sparsity}_{method}_{dtype}.pth

示例：
- var_d16_s0.4_linear_int4.pth    # Linear量化, int4
- var_d16_s0.4_linear_int8.pth    # Linear量化, int8
- var_d16_s0.4_gptq_int4.pth      # GPTQ量化, int4（或fallback到linear）
```

### 文件内容

```python
checkpoint = {
    'model_state_dict': ...,      # 量化后的模型权重
    'quantization_config': {
        'method': 'linear',       # 或 'gptq'
        'weight_dtype': 'int4',
        'sparsity': 0.4,
        'model_depth': 16,
        ...
    }
}
```

## 🔧 使用示例

### 完整工作流程

```bash
# 1. Linear量化（推荐）
bash run_gptq_quantization.bash int4 16 0.4 linear

# 2. 在Linux上测试质量
python test_quantized.py \
    --depth 16 \
    --sparsity 0.4 \
    --var_model pytorch_quantized_models/var_d16_s0.4_linear_int4.pth

# 3. 转换到CoreML
python transfer.py \
    --depth 16 \
    --sparsity 0.4 \
    --var_model pytorch_quantized_models/var_d16_s0.4_linear_int4.pth
```

### 对比不同配置

```bash
# 量化不同配置
bash run_gptq_quantization.bash int4 16 0.4 linear
bash run_gptq_quantization.bash int8 16 0.4 linear
bash run_gptq_quantization.bash int4 16 0.4 gptq

# 测试对比
for model in pytorch_quantized_models/var_d16_s0.4_*.pth; do
    echo "Testing $model"
    python test_quantized.py --var_model $model
done
```

## 🐛 故障排查

### GPTQ不可用

**现象：**
```
✗ GPTQ quantization failed: ...
Falling back to LinearQuantizer...
```

**原因：** LayerwiseCompressor在当前环境不可用（可能仅限macOS）

**解决：** 使用linear方法（默认），效果也很好
```bash
bash run_gptq_quantization.bash int4 16 0.4 linear
```

### 内存不足

**解决：** 修改脚本中的CALIBRATION_SAMPLES（仅影响GPTQ）
```python
# 在 quantize_gptq_pytorch.py 中
CALIBRATION_SAMPLES = 64  # 从128减少到64
```

## 📚 相关文档

- `COREML_GPTQ_LIMITATION.md` - GPTQ限制说明
- `GPTQ_USAGE_SIMPLE.md` - 使用指南
- `quantize_gptq_pytorch.py` - 主脚本（含详细注释）

## 🎯 推荐

**对于VAR模型，推荐使用Linear量化：**
```bash
bash run_gptq_quantization.bash int4 16 0.4 linear
```

原因：
- ✅ 稳定可靠，在所有平台工作
- ✅ 不需要复杂的calibration data
- ✅ 对于已剪枝40%的模型，Linear量化足够好
- ✅ 速度快，易于调试
