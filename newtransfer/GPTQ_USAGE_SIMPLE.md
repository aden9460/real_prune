# GPTQ量化使用说明（更新版）

## ✅ 简化说明

现在只保存一个模型文件，既可用于CoreML转换，也可用于Linux测试。

## 🚀 快速开始

### 1. GPTQ量化

```bash
cd /home/project/real_prune/newtransfer

# 量化剪枝后的模型（sparsity=0.4）
bash run_gptq_quantization.bash int4 16 0.4
```

**输出：**
```
pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth
```

### 2. 在Linux上测试

```python
import torch
from models import build_vae_var

# 构建模型架构
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device='cuda', patch_nums=(1,2,3,4,5,6,8,10,13,16),
    num_classes=1000, depth=16, shared_aln=False, args=args
)

# 加载量化权重
checkpoint = torch.load('pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth')
var.load_state_dict(checkpoint['model_state_dict'])

# 可选：查看量化配置
print(checkpoint['quantization_config'])
# {'method': 'gptq', 'weight_dtype': 'int4', 'sparsity': 0.4, ...}

# 测试
var.eval()
test_fid(var, vae)
```

### 3. 转换到CoreML

```bash
python transfer.py \
    --depth 16 \
    --sparsity 0.4 \
    --var_model pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth
```

## 📦 模型文件内容

```python
checkpoint = torch.load('var_d16_s0.4_gptq_int4.pth')

# 内容：
{
    'model_state_dict': OrderedDict({...}),  # 量化后的模型权重
    'quantization_config': {                  # 量化配置信息
        'method': 'gptq',
        'weight_dtype': 'int4',
        'sparsity': 0.4,
        'model_depth': 16,
        'calibration_samples': 128,
        ...
    }
}
```

## 🎯 参数说明

### Bash脚本参数

```bash
bash run_gptq_quantization.bash [dtype] [depth] [sparsity] [model_path] [data_path]

# 参数：
#   dtype:      int4 或 int8
#   depth:      模型深度 (12/16/20/24/30)
#   sparsity:   剪枝率 (例如 0.4 表示40%剪枝)
#   model_path: PyTorch模型路径
#   data_path:  ImageNet路径（可选）

# 示例：
bash run_gptq_quantization.bash int4 16 0.4
bash run_gptq_quantization.bash int8 16 0.4 /path/to/model.pth /path/to/imagenet
```

### Python脚本参数

```bash
python quantize_gptq_pytorch.py \
    --depth 16 \
    --sparsity 0.4 \
    --var_model /path/to/model.pth \
    --data_path /path/to/imagenet  # 可选
```

## 📝 使用示例

### 完整工作流程

```bash
# 1. GPTQ量化
bash run_gptq_quantization.bash int4 16 0.4

# 2. （推荐）在Linux上验证质量
python -c "
import torch
from models import build_vae_var
from utils import arg_util

args = arg_util.init_dist_and_get_args()
vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device='cuda', patch_nums=(1,2,3,4,5,6,8,10,13,16),
    num_classes=1000, depth=16, shared_aln=False, args=args)

checkpoint = torch.load('pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth')
var.load_state_dict(checkpoint['model_state_dict'])
var.eval()
print('✓ Model loaded successfully')
"

# 3. 如果质量满意，转换到CoreML
python transfer.py --depth 16 --sparsity 0.4 \
    --var_model pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth
```

### 测试脚本模板

创建 `test_quantized.py`：

```python
#!/usr/bin/env python3
import torch
from models import build_vae_var
from utils import arg_util
import sys

# 解析参数
args = arg_util.init_dist_and_get_args()

# 构建模型
print("Building model...")
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    device='cuda', patch_nums=(1,2,3,4,5,6,8,10,13,16),
    num_classes=1000, depth=args.depth, shared_aln=False, args=args
)

# 加载量化模型
print(f"Loading quantized model: {args.var_model}")
checkpoint = torch.load(args.var_model, map_location='cuda')

# 显示量化配置
print("\nQuantization config:")
for key, value in checkpoint['quantization_config'].items():
    print(f"  {key}: {value}")

# 加载权重
var.load_state_dict(checkpoint['model_state_dict'])
var.eval()

# 测试推理
print("\nTesting inference...")
with torch.no_grad():
    label = torch.randint(0, 1000, (1,)).cuda()
    output = var.autoregressive_infer_cfg(
        B=1, label_B=label, cfg=5.0,
        top_k=900, top_p=0.96, g_seed=0
    )
    print(f"✓ Output shape: {output.shape}")

print("\n✓ Quantized model works correctly!")
```

使用：
```bash
python test_quantized.py --depth 16 --sparsity 0.4 \
    --var_model pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth
```

## 🔍 文件命名规则

```
var_d{depth}_s{sparsity}_{method}_{dtype}.pth

示例：
- var_d16_s0.4_gptq_int4.pth   # depth=16, sparsity=0.4, GPTQ, int4
- var_d16_s0.4_gptq_int8.pth   # depth=16, sparsity=0.4, GPTQ, int8
- var_d16_s0_gptq_int4.pth     # depth=16, no pruning, GPTQ, int4
```

## 💡 关键要点

1. **一个文件两用途**
   - ✅ CoreML转换：transfer.py直接使用
   - ✅ Linux测试：加载`checkpoint['model_state_dict']`

2. **包含完整信息**
   - 模型权重（量化后的）
   - 量化配置（方法、位数、剪枝率等）

3. **加载方式简单**
   ```python
   # 只需两行
   checkpoint = torch.load(path)
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

## 📊 对比不同量化方式

```bash
# 量化int4和int8，对比质量
bash run_gptq_quantization.bash int4 16 0.4
bash run_gptq_quantization.bash int8 16 0.4

# 测试两个模型
python test_quantized.py --var_model pytorch_quantized_models/var_d16_s0.4_gptq_int4.pth
python test_quantized.py --var_model pytorch_quantized_models/var_d16_s0.4_gptq_int8.pth

# 对比FID分数，选择最优方案
```
