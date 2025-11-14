# CoreML Tools GPTQ支持说明

## 问题

在当前环境中，`LayerwiseCompressor`不可用：

```python
from coremltools.optimize.torch.quantization import LayerwiseCompressor
# ImportError: cannot import name 'LayerwiseCompressor'
```

当前可用的量化工具：
- `LinearQuantizer` ✅
- `PostTrainingQuantizer` ✅
- `LinearQuantizerConfig` ✅
- `PostTrainingQuantizerConfig` ✅

## 原因

`LayerwiseCompressor`可能存在以下限制之一：
1. **仅macOS支持**：可能只在macOS上编译可用
2. **需要特定版本**：虽然coremltools 8.3.0应该支持，但可能需要特定的编译版本
3. **平台限制**：Linux环境可能缺少必要的底层库

## 解决方案

### 方案1：使用PostTrainingQuantizer（推荐）

`PostTrainingQuantizer`也支持类似GPTQ的后训练量化：

```python
from coremltools.optimize.torch.quantization import (
    PostTrainingQuantizer,
    PostTrainingQuantizerConfig
)

config = PostTrainingQuantizerConfig.from_dict({
    "global_config": {
        "quantization_scheme": "symmetric",  # or "affine"
        "milestones": [0, 0, 0, 0],
    }
})

quantizer = PostTrainingQuantizer(model, config)
prepared_model = quantizer.prepare()
quantized_model = quantizer.finalize(calibration_loader)
```

### 方案2：使用LinearQuantizer（当前实现）

```python
from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig
)

config = LinearQuantizerConfig.from_dict({
    "global_config": {
        "quantization_scheme": "symmetric",
        "milestones": [0, 0, 10, 10],
    }
})

quantizer = LinearQuantizer(model, config)
prepared_model = quantizer.prepare()
quantized_model = quantizer.finalize()
```

### 方案3：直接使用CoreML后处理量化

在PyTorch阶段不做量化，转换到CoreML后再量化（当前的quantize.py方式）。

## 推荐方案

**对于VAR模型：使用LinearQuantizer（方案2）**

原因：
1. ✅ 在Linux上可用
2. ✅ 不需要复杂的calibration data处理
3. ✅ 代码简单，容易调试
4. ✅ 可以在PyTorch阶段完成，转换到CoreML时保留量化

虽然没有GPTQ的calibration-based优化，但对于VAR这种已经剪枝过的模型，LinearQuantizer应该足够了。

## 更新策略

将`quantize_gptq_pytorch.py`重命名为`quantize_pytorch.py`，去掉GPTQ相关内容，使用LinearQuantizer实现PyTorch阶段的量化。

这样既能：
- ✅ 在PyTorch阶段完成量化
- ✅ 在Linux上测试量化质量
- ✅ 转换到CoreML时保留量化信息
- ✅ 避免依赖不可用的LayerwiseCompressor
