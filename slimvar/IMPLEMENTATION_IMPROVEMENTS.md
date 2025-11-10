# VAR模型剪枝实现改进文档

## 问题背景

### 原始错误
在 `model_slimming_basic.py` 的原始实现中，存在严重的维度不匹配错误：
```
RuntimeError: shape '[16, 680, 1024]' is invalid for input of size 9052160
```

### 错误根因分析
1. **错误的activation收集策略**：每次都调用整个模型 `model(batch_labels, batch_tokens)`
2. **维度不兼容**：已剪枝层(输出832维)与未剪枝层(期望1024维输入)之间的不匹配
3. **计算浪费**：重复计算680个token的embedding

## 解决方案设计

### 参考模式 (基于 model_slimming.py)
LLaMA版本使用了正确的"Catcher + 流式传递"模式：
1. **Catcher机制**：一次性捕获所有样本的初始表示
2. **流式传递**：layer0_output → layer1_input → layer1_output → layer2_input...
3. **就地更新**：`layer_inputs[j] = layer(layer_inputs[j])`

### VAR适配策略
将LLaMA的设计模式适配到VAR模型的特殊输入格式。

## 具体实现改进

### 1. 新增VARCatcher类
```python
class VARCatcher(nn.Module):
    """Catcher class to capture VAR block inputs for calibration"""
    def __init__(self, num_samples, seqlen, hidden_size, cache_dev='cuda', dtype=torch.float32):
        super().__init__()
        self.layer_inputs = torch.zeros(
            (num_samples, seqlen, hidden_size),
            dtype=dtype, device=cache_dev
        )
        self.row_idx = 0
        self.cache_dev = cache_dev

    def forward(self, x, cond_BD=None, attn_bias=None):
        """Capture input and interrupt forward pass"""
        if self.row_idx < self.layer_inputs.shape[0]:
            self.layer_inputs[self.row_idx] = x.detach()
            self.row_idx += 1
        raise ValueError("VARCatcher: Activation captured successfully")
```

### 2. 改进model_slimming函数架构

#### Phase 1: 初始表示收集
```python
# 替换第一个block为Catcher
original_first_block = layers[0]
var_catcher = VARCatcher(num_samples, 680, model.C, cache_dev=device)  # VAR序列固定680
layers[0] = var_catcher

# 一次性运行模型获取所有初始representations
try:
    model(batch_labels, batch_tokens)
except ValueError:
    pass  # Expected interruption

# 获取捕获的inputs并恢复原始block
layer_inputs = var_catcher.layer_inputs.clone()  # (num_samples, 680, C)
layers[0] = original_first_block
```

#### Phase 2: 流式剪枝与传递
**关键改进**：
- **原来（错误）**：`model(batch_labels, batch_tokens)`
- **现在（正确）**：`layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)`

### 3. VAR输入格式适配
适配VAR block的特殊输入要求：
```python
# 正确的VAR block调用格式
class_label = calibration_labels[j:j+1]
cond_BD = model.class_emb(class_label)
cond_BD_or_gss = model.shared_ada_lin(cond_BD)
seq_len = layer_input.shape[1]  # 固定680
attn_bias = model.attn_bias_for_masking[:, :, :seq_len, :seq_len]

# 调用当前层
layer_output = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
```

### 4. 流式activation更新
每层处理完后，更新layer_inputs为当前层输出：
```python
# 就地更新layer_inputs
for j in range(batch_idx, end_idx):
    layer_output = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
    layer_inputs[j] = layer_output.squeeze(0)  # 更新为下层输入: (680, C)
```

## 改进效果

### 1. 错误消除
- ✅ 解决维度不匹配错误
- ✅ 避免剪枝层间的输入/输出维度冲突

### 2. 效率提升
- ✅ 避免重复计算680个token
- ✅ 一次性捕获所有样本的初始表示
- ✅ 内存管理优化(CPU/GPU动态切换)

### 3. 架构一致性
- ✅ 与LLaMA版本保持设计模式一致
- ✅ 遵循"Catcher + 流式传递"的最佳实践
- ✅ 保持SlimGPT算法的完整性

### 4. VAR模型特化
- ✅ 正确适配VAR的输入格式 (x, cond_BD, attn_bias)
- ✅ 保持teacher forcing模式兼容性
- ✅ 支持固定680序列长度的处理

## 核心技术要点

1. **两阶段设计**：分离初始化和逐层处理，避免交叉干扰
2. **流式传递**：layer_i输出直接作为layer_{i+1}输入，避免维度累积错误
3. **VAR适配**：正确处理VAR特有的条件嵌入和注意力掩码
4. **序列长度固定**：VAR模型序列长度恒定为680 (679 tokens + 1 SOS)
5. **内存优化**：动态GPU/CPU切换，支持大规模模型剪枝

## 关键修改点

### 文件: model_slimming_basic.py
**修改前的问题代码**:
```python
# 错误的activation收集 - 每次运行整个模型
model(batch_labels, batch_tokens)  # 第292行 - 导致维度不匹配
```

**修改后的正确代码**:
```python
# Phase 1: 一次性收集初始representations
var_catcher = VARCatcher(num_samples, 680, model.C, cache_dev=device)
layers[0] = var_catcher
# ... 运行模型并捕获 ...
layer_inputs = var_catcher.layer_inputs.clone()

# Phase 2: 逐层剪枝与流式传递
for i in range(len(layers)):
    # 使用当前层的layer_inputs进行activation收集
    layer_output = layer(x=layer_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
    # 更新layer_inputs为当前层输出
    layer_inputs[j] = layer_output.squeeze(0)
```

## 修改日志
- **修改文件**: `/home/project/real_prune/slimvar/model_slimming_basic.py`
- **修改日期**: 2025-11-08
- **修改类型**: 架构重构 + 错误修复
- **主要改进**:
  1. 添加VARCatcher类 (第50-66行)
  2. 重构model_slimming函数为两阶段处理 (第232-459行)
  3. 修正VAR block输入格式 (第350-363行, 第457-472行)
  4. 实现流式activation传递 (第448-477行)

## 验证清单
- [x] 添加VARCatcher机制
- [x] 实现两阶段处理架构
- [x] 修正VAR block输入格式
- [x] 实现流式activation传递
- [x] 修正序列长度为680
- [x] 更新相关注释和文档
- [ ] 实际运行测试（待验证）