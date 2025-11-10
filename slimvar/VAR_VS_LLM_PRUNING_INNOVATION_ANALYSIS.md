# VAR vs LLM: 深度剪枝创新分析

## 目录
1. [架构差异对比](#1-架构差异对比)
2. [Token获取策略创新](#2-token获取策略创新)
3. [scale_mul_1H11深度分析](#3-scale_mul_1h11深度分析)
4. [多尺度Token金字塔](#4-多尺度token金字塔)
5. [创新点总结](#5-创新点总结)

---

## 1. 架构差异对比

### 1.1 核心差异表

| 维度 | LLM (LLaMA) | VAR | 影响 |
|------|-------------|-----|------|
| **任务类型** | 文本生成 (因果) | 图像生成 (多尺度) | 数据流完全不同 |
| **Token性质** | 离散语义单元 | 连续视觉特征 (VQVAE编码) | VAR token更连续 |
| **序列结构** | 单一线性序列 | 10层金字塔 (1²→16²) | VAR有层次结构 |
| **序列长度** | 2048-8192 tokens | 680 tokens (固定) | VAR更短但更密集 |
| **注意力机制** | 因果掩码 (下三角) | 金字塔掩码 (层级) | VAR掩码更复杂 |
| **条件输入** | 无/简单prompt | 强类别条件 (class_emb) | VAR条件更强 |
| **生成模式** | 逐token生成 | 逐尺度生成 (10阶段) | VAR有明确阶段 |

### 1.2 Attention机制对比

#### LLM Attention (标准多头注意力)
```python
# 固定scale，标准softmax
scale = 1 / sqrt(head_dim)  # 固定值
attn = softmax(Q @ K^T / scale) @ V
```

#### VAR Attention (L2归一化 + 可学习scale)
```python
# Line 101-105 in basic_var.py
if self.attn_l2_norm:
    scale_mul = self.scale_mul_1H11.clamp_max(max_scale_mul).exp()  # (1, H, 1, 1)
    q = F.normalize(q, dim=-1).mul(scale_mul)  # L2归一化 + 可学习缩放
    k = F.normalize(k, dim=-1)
    attn = softmax(q @ k^T) @ v  # scale=1
```

**关键差异**:
1. **L2归一化**: VAR对Q/K进行L2归一化，消除幅值差异
2. **可学习scale**: 每个head有独立的 `scale_mul` 参数（初始值=e^4≈54.6）
3. **Head级适应**: 不同head可学习不同的注意力锐度

---

## 2. Token获取策略创新

### 2.1 当前实现 (Teacher Forcing)

**model_slimming_basic.py Line 287-290**:
```python
if calibration_tokens is not None:
    # 使用预编码的真实tokens（一次性获得所有680 tokens）
    batch_tokens = calibration_tokens[batch_idx:end_idx].to(device)
    model(batch_labels, batch_tokens)
```

**特点**:
- ✅ **真实分布**: 使用VQVAE编码的真实ImageNet图像
- ✅ **并行高效**: 一次forward获得所有层的激活
- ✅ **分布准确**: 激活值完全匹配训练分布
- ❌ **分布偏差**: 未考虑推理时的自回归误差累积

### 2.2 创新方案：Autoregressive Token Generation

#### 方案A：逐尺度渐进剪枝

```python
def model_slimming_progressive(model, calibration_labels, args):
    """
    创新点：模拟VAR推理过程，逐尺度生成token并剪枝
    """
    device = 'cuda'
    layers = model.blocks
    num_samples = len(calibration_labels)

    # Stage 1: 剪枝前4层（处理scale 1-3）
    # 使用真实tokens for scale 1-3
    calibration_tokens_s1 = get_tokens_for_scale(1, 3)  # (N, 14, 32)
    layer_inputs_s1 = capture_layer_inputs(model, calibration_tokens_s1)
    prune_layers(model, layers[0:4], layer_inputs_s1, sparsity=0.1)

    # Stage 2: 剪枝layer 4-8（处理scale 4-6）
    # 使用AR生成的tokens（累积前期剪枝误差）
    for si in range(1, 4):
        generated_tokens = model.autoregressive_infer(
            calibration_labels,
            max_scale=si,  # 只生成到si尺度
            kv_cache=True
        )
        layer_inputs_si = capture_layer_inputs(model, generated_tokens)
    prune_layers(model, layers[4:8], layer_inputs_si, sparsity=0.15)

    # Stage 3-4: 后续层使用完全AR生成的tokens
    # ...逐渐增加剪枝率
```

**理论优势**:
1. **误差累积建模**: 后层剪枝使用前层剪枝后生成的tokens
2. **推理对齐**: 校准数据分布更接近实际推理
3. **渐进适应**: 逐尺度调整剪枝策略

#### 方案B：混合Token策略

```python
def prepare_calibration_data_mixed(vae, num_samples, mixing_ratio=0.5):
    """
    创新点：混合真实tokens和AR生成tokens
    """
    # 50% 真实tokens (teacher forcing)
    real_tokens = encode_images_to_tokens(vae, num_samples // 2)

    # 50% AR生成tokens（模拟推理）
    ar_tokens = []
    for label in labels:
        ar_token = model.autoregressive_infer(label, temperature=1.0)
        ar_tokens.append(ar_token)

    mixed_tokens = torch.cat([real_tokens, ar_tokens], dim=0)
    return mixed_tokens
```

### 2.3 实验对比方案

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **全真实Tokens** | 分布最准确，训练最接近 | 忽略推理误差累积 | 浅层剪枝 |
| **全AR生成** | 推理最接近 | 前期无剪枝时开销大 | 深层剪枝 |
| **渐进混合** | 平衡准确性和推理 | 实现复杂 | 全模型剪枝 |
| **逐尺度AR** | 误差累积建模好 | 需要10次生成 | 高剪枝率 |

**实验假设**:
> **猜想1**: VAR使用逐层生成的tokens可能比全真实tokens效果更好！
>
> **原因**:
> 1. VAR推理时存在**10个尺度的自回归累积**
> 2. 剪枝后的模型输出分布会**偏离原始分布**
> 3. 后层需要适应**前层剪枝后的激活分布**
> 4. 使用AR生成的tokens能让后层看到**更真实的推理时激活**

---

## 3. scale_mul_1H11深度分析

### 3.1 参数详解

**定义** (basic_var.py Line 69):
```python
self.scale_mul_1H11 = nn.Parameter(
    torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
    requires_grad=True
)
```

**形状**: `(1, num_heads, 1, 1)`
**初始值**: `log(4.0) ≈ 1.386` (exp后为4.0)
**取值范围**: `[0, log(100)] → [1, 100]` (经过exp)

### 3.2 作用机制

```python
# Forward过程 (Line 102-104)
scale_mul = self.scale_mul_1H11.clamp_max(max_scale_mul).exp()  # (1, H, 1, 1)
q = F.normalize(q, dim=-1)  # ||q|| = 1
q = q.mul(scale_mul)        # q_scaled = q * scale_mul

# 注意力计算
attn = softmax(q_scaled @ k^T) @ v
     = softmax(scale_mul * (q/||q||) @ (k/||k||)^T) @ v
```

**数学意义**:
- `scale_mul` 控制**注意力锐度** (sharpness)
- 较大的 `scale_mul` → softmax更sharp → 注意力更集中
- 较小的 `scale_mul` → softmax更smooth → 注意力更分散

### 3.3 Head重要性指示器

**观察1**: 不同head的scale_mul值反映head重要性

```python
# 剪枝后保留的scale_mul (model_slimming_basic.py Line 431-432)
keep_heads = sorted(all_heads - removed_heads)
new_scale_mul = old_scale_mul[0, keep_heads, 0, 0]
```

**假设**: scale_mul值 ↔️ Head重要性

| scale_mul取值 | 注意力特性 | 可能重要性 |
|--------------|-----------|-----------|
| **接近100** (max) | 极度锐化，几乎one-hot | **高重要性**：需要精确定位 |
| **4-20** (中等) | 适度集中 | **中等重要性** |
| **接近1** (min) | 高度分散，接近均匀 | **低重要性**：可能冗余 |

### 3.4 创新：scale_mul引导剪枝

#### 方案1：Scale-Aware Head Pruning

```python
def scale_aware_head_pruning(model, layer_idx, target_sparsity):
    """
    创新点：使用scale_mul值辅助head重要性评估
    """
    attn = model.blocks[layer_idx].attn
    scale_mul = attn.scale_mul_1H11.exp().squeeze()  # (num_heads,)

    # 计算综合重要性
    hessian_importance = compute_hessian_importance(attn.proj)  # SlimGPT
    scale_importance = scale_mul.cpu()  # scale_mul作为重要性先验

    # 融合两种信息
    alpha = 0.3  # 超参数
    combined_importance = (1 - alpha) * hessian_importance + alpha * scale_importance

    # 按综合重要性剪枝
    num_prune = int(attn.num_heads * target_sparsity)
    prune_heads = combined_importance.argsort()[:num_prune]

    return prune_heads
```

**优势**:
1. **先验知识**: scale_mul是训练学到的head重要性信号
2. **稳定性**: 减少Hessian估计的噪声影响
3. **解释性**: scale_mul提供可解释的剪枝依据

#### 方案2：Non-uniform Sparsity by Scale Distribution

```python
def compute_layer_sparsity_by_scale(model, base_sparsity=0.2):
    """
    创新点：根据每层的scale_mul分布动态调整剪枝率
    """
    layer_sparsities = []

    for i, block in enumerate(model.blocks):
        scale_mul = block.attn.scale_mul_1H11.exp()

        # 指标1：scale均值（高均值=高重要性层）
        mean_scale = scale_mul.mean()

        # 指标2：scale方差（高方差=head分化明显）
        var_scale = scale_mul.var()

        # 指标3：接近max的head比例（高比例=关键层）
        critical_heads_ratio = (scale_mul > 50).float().mean()

        # 动态调整sparsity
        if critical_heads_ratio > 0.3:
            sparsity = base_sparsity * 0.5  # 关键层：减少剪枝
        elif var_scale < 10:
            sparsity = base_sparsity * 1.5  # 均匀层：增加剪枝
        else:
            sparsity = base_sparsity

        layer_sparsities.append(sparsity)

    return layer_sparsities
```

#### 方案3：Scale-Mul Distillation

```python
def scale_mul_distillation(teacher_model, student_model):
    """
    创新点：剪枝后蒸馏scale_mul分布
    """
    for i in range(len(teacher_model.blocks)):
        teacher_scale = teacher_model.blocks[i].attn.scale_mul_1H11
        student_scale = student_model.blocks[i].attn.scale_mul_1H11

        # 保留head的scale_mul应接近teacher
        loss_scale = F.mse_loss(student_scale, teacher_scale[keep_heads])

    return loss_scale
```

### 3.5 实验设计

**实验1: Scale-Mul与Head重要性相关性**
```python
# 测试scale_mul是否真的反映重要性
for layer_idx in range(depth):
    scale_values = get_scale_mul(layer_idx)
    hessian_importance = compute_hessian_importance(layer_idx)

    correlation = np.corrcoef(scale_values, hessian_importance)[0, 1]
    print(f"Layer {layer_idx}: correlation = {correlation:.3f}")
```

**实验2: Scale-Guided vs Pure-Hessian Pruning**
```bash
# Baseline: 纯SlimGPT
python model_slimming_basic.py --sparsity 0.3

# 创新: Scale-Mul引导
python model_slimming_scale_guided.py --sparsity 0.3 --scale_weight 0.3

# 对比FID/IS
```

**预期结果**:
- 假设1: scale_mul与Hessian重要性正相关（r > 0.5）
- 假设2: scale引导剪枝比纯Hessian剪枝FID降低5-10%

---

## 4. 多尺度Token金字塔

### 4.1 VAR的10尺度结构

```python
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # 10 scales
# 每个尺度的token数量:
# Scale 0: 1²  = 1    tokens  (累积: 1)
# Scale 1: 2²  = 4    tokens  (累积: 5)
# Scale 2: 3²  = 9    tokens  (累积: 14)
# Scale 3: 4²  = 16   tokens  (累积: 30)
# Scale 4: 5²  = 25   tokens  (累积: 55)
# Scale 5: 6²  = 36   tokens  (累积: 91)
# Scale 6: 8²  = 64   tokens  (累积: 155)
# Scale 7: 10² = 100  tokens  (累积: 255)
# Scale 8: 13² = 169  tokens  (累积: 424)
# Scale 9: 16² = 256  tokens  (累积: 680)
```

### 4.2 尺度不均对剪枝的影响

**问题1: 不同尺度token的激活分布差异**

```python
# 分析不同尺度的激活统计
def analyze_scale_activation_stats(model, calibration_tokens):
    stats = {}

    for si, (bg, ed) in enumerate(model.begin_ends):
        scale_tokens = calibration_tokens[:, bg:ed, :]  # 该尺度的tokens

        stats[f'scale_{si}'] = {
            'mean': scale_tokens.mean().item(),
            'std': scale_tokens.std().item(),
            'l2_norm': scale_tokens.norm(dim=-1).mean().item(),
        }

    return stats
```

**观察**:
- **早期尺度** (1x1 - 4x4): 语义信息（类别、整体布局）
- **中期尺度** (5x5 - 8x8): 结构信息（对象形状）
- **后期尺度** (10x10 - 16x16): 细节信息（纹理、边缘）

**假设**: 不同尺度对剪枝的敏感度不同
- 早期尺度: 高敏感（语义关键）
- 后期尺度: 低敏感（细节冗余）

### 4.3 创新：尺度感知校准集

#### 方案1: Scale-Balanced Sampling

```python
def scale_balanced_calibration(vae, num_samples_per_scale=25):
    """
    创新点：每个尺度独立采样，确保覆盖所有尺度的数据分布
    """
    calibration_data = []

    for si in range(10):  # 10个尺度
        # 针对该尺度采样特定类型的图像
        if si < 3:  # 早期尺度：采样类别多样性高的图像
            images = sample_diverse_classes(num_samples_per_scale)
        elif si < 7:  # 中期尺度：采样结构复杂的图像
            images = sample_complex_structure(num_samples_per_scale)
        else:  # 后期尺度：采样纹理丰富的图像
            images = sample_rich_texture(num_samples_per_scale)

        calibration_data.append(images)

    # 总计250个样本，但尺度覆盖更均衡
    return calibration_data
```

#### 方案2: Scale-Progressive Calibration

```python
def scale_progressive_pruning(model, calibration_labels, args):
    """
    创新点：针对不同尺度使用不同的校准数据量
    """
    for i, layer in enumerate(model.blocks):
        # 确定该层主要处理的尺度范围
        dominant_scales = get_layer_dominant_scales(i, depth=16)
        # Layer 0-3  → Scale 0-2  (早期)
        # Layer 4-8  → Scale 3-6  (中期)
        # Layer 9-15 → Scale 7-9  (后期)

        if dominant_scales[0] < 3:  # 早期尺度层
            num_calib = 512  # 使用更多校准数据
            sparsity = 0.15  # 保守剪枝
        elif dominant_scales[0] < 7:  # 中期尺度层
            num_calib = 256
            sparsity = 0.25
        else:  # 后期尺度层
            num_calib = 128
            sparsity = 0.35  # 激进剪枝

        prune_layer(layer, num_calib, sparsity)
```

### 4.4 尺度特定的剪枝策略

```python
def scale_aware_pruning_strategy(model, args):
    """
    创新点：根据尺度特性设计剪枝策略
    """
    pruning_config = []

    for i in range(args.depth):
        # 计算该层的尺度覆盖
        scale_coverage = compute_scale_coverage(i, model.begin_ends)

        config = {
            'layer_idx': i,
            'sparsity_attn': 0.2,  # 默认
            'sparsity_ffn': 0.2,
            'headsize': 64,  # 默认
        }

        # 早期尺度层（处理1x1 - 4x4）
        if scale_coverage['early'] > 0.5:
            config['sparsity_attn'] = 0.1  # 注意力保守剪枝
            config['sparsity_ffn'] = 0.15

        # 后期尺度层（处理10x10 - 16x16）
        elif scale_coverage['late'] > 0.5:
            config['sparsity_attn'] = 0.3  # 注意力激进剪枝
            config['sparsity_ffn'] = 0.35
            # 考虑channel-wise剪枝而非head-wise
            config['headsize'] = 1

        pruning_config.append(config)

    return pruning_config
```

---

## 5. 创新点总结

### 5.1 核心创新方向

#### 🔥 创新1: 推理对齐的Token生成策略

**动机**: VAR推理时存在自回归误差累积，校准数据应模拟此过程

**方案**:
1. ✅ **渐进式AR剪枝**: 逐尺度使用AR生成的tokens
2. ✅ **混合Token策略**: 真实tokens + AR生成tokens
3. ✅ **误差注入训练**: 剪枝后微调时注入推理噪声

**实验**:
```bash
# Baseline: 全真实tokens
python model_slimming_basic.py --use_images --sparsity 0.3

# 创新: 逐尺度AR tokens
python model_slimming_progressive_ar.py --ar_mode progressive --sparsity 0.3

# 创新: 混合tokens
python model_slimming_mixed.py --mixing_ratio 0.5 --sparsity 0.3
```

**预期**: AR策略在高剪枝率(>30%)时FID提升10-20%

---

#### 🔥 创新2: Scale-Mul引导的智能剪枝

**动机**: scale_mul_1H11是训练学到的head重要性先验

**方案**:
1. ✅ **Scale-Aware Head Pruning**: 融合Hessian + scale_mul
2. ✅ **Dynamic Sparsity by Scale**: 根据scale分布调整剪枝率
3. ✅ **Scale-Mul Distillation**: 剪枝后蒸馏scale分布

**实验**:
```bash
# Baseline: 纯SlimGPT (Hessian)
python model_slimming_basic.py --sparsity 0.3

# 创新: Scale引导
python model_slimming_scale_guided.py --sparsity 0.3 --scale_weight 0.3

# 创新: Scale动态稀疏度
python model_slimming_dynamic.py --base_sparsity 0.3 --scale_adaptive
```

**关键实验**: 验证scale_mul与Hessian重要性的相关性

```python
# 分析脚本
python analyze_scale_importance.py \
    --var_ckpt path/to/var_d16.pth \
    --num_samples 256 \
    --output scale_correlation.json
```

---

#### 🔥 创新3: 尺度感知的校准与剪枝

**动机**: VAR的10尺度金字塔有不同的语义和敏感度

**方案**:
1. ✅ **Scale-Balanced Sampling**: 每尺度独立采样校准集
2. ✅ **Scale-Progressive Pruning**: 不同尺度不同剪枝策略
3. ✅ **Scale-Specific Sparsity**: 早期保守、后期激进

**实验设计**:
```python
# 对比实验
configs = {
    'uniform': {  # 基线：均匀剪枝
        'sparsity': [0.3] * 16,
        'calibration': 'random_256'
    },
    'scale_aware': {  # 创新：尺度感知
        'sparsity': [0.1]*4 + [0.25]*6 + [0.4]*6,  # 早→晚递增
        'calibration': 'scale_balanced_250'
    },
    'scale_progressive': {  # 创新：逐尺度剪枝
        'sparsity': 'adaptive',
        'calibration': 'progressive_ar'
    }
}
```

---

### 5.2 实验验证计划

#### 阶段1: 基线对比 (1-2天)

```bash
# 1. 当前方案（全真实tokens）
bash run_baseline.sh

# 2. 收集指标
# - FID-50K
# - Inference time (tokens/s)
# - Model size (MB)
# - Per-scale FID (分尺度评估)
```

#### 阶段2: Token策略消融 (2-3天)

```bash
# 对比3种token策略
for strategy in real_tokens ar_tokens mixed_tokens; do
    python model_slimming_ablation.py \
        --token_strategy $strategy \
        --sparsity 0.3 \
        --num_samples 256
done
```

#### 阶段3: Scale-Mul集成 (2-3天)

```bash
# 对比scale_weight参数
for weight in 0.0 0.1 0.2 0.3 0.5; do
    python model_slimming_scale_guided.py \
        --sparsity 0.3 \
        --scale_weight $weight
done
```

#### 阶段4: 尺度感知优化 (3-4天)

```bash
# 对比校准策略
python model_slimming_scale_balanced.py \
    --calibration_mode scale_balanced \
    --samples_per_scale 25

# 对比剪枝策略
python model_slimming_scale_progressive.py \
    --sparsity_schedule early_conservative
```

---

### 5.3 理论假设与验证

#### 假设1: AR Token优于真实Token（后层剪枝）

**理论**: 后层需要适应前层剪枝后的激活分布

**验证**:
```python
# 对比实验
layer_ranges = [(0, 4), (4, 8), (8, 12), (12, 16)]
for start, end in layer_ranges:
    fid_real = prune_with_real_tokens(start, end)
    fid_ar = prune_with_ar_tokens(start, end)
    print(f"Layers {start}-{end}: Real={fid_real:.2f}, AR={fid_ar:.2f}")
```

**预期**:
- 前4层: Real ≈ AR
- 后12层: AR < Real (FID降低5-15%)

---

#### 假设2: scale_mul反映Head重要性

**理论**: 训练时高scale_mul的head学到了更关键的特征

**验证**:
```python
# 1. 计算相关性
correlation = compute_scale_hessian_correlation(model)

# 2. 对比剪枝效果
# - 按scale_mul剪枝
# - 按Hessian剪枝
# - 按random剪枝
```

**预期**:
- Correlation > 0.5 (中等正相关)
- scale_mul剪枝效果接近Hessian (FID差距<5%)

---

#### 假设3: 尺度感知剪枝优于均匀剪枝

**理论**: 早期尺度更重要，应更保守剪枝

**验证**:
```python
# 分尺度FID
fid_per_scale = evaluate_per_scale_fid(pruned_model)
# fid_per_scale = {
#     'scale_0-2': 25.3,  # 早期尺度（整体语义）
#     'scale_3-6': 32.1,  # 中期尺度（结构）
#     'scale_7-9': 45.6,  # 后期尺度（细节）
# }
```

**预期**: 尺度感知剪枝的早期FID显著优于均匀剪枝

---

### 5.4 潜在发现与贡献

#### 贡献1: 首个VAR剪枝基线
- **意义**: 建立VAR模型压缩的baseline
- **影响**: 为VAR部署提供实用方案

#### 贡献2: 推理对齐的剪枝范式
- **意义**: 证明AR token策略优于teacher forcing
- **影响**: 启发自回归模型的剪枝方法论

#### 贡献3: Attention Scale作为剪枝先验
- **意义**: 发现scale_mul与head重要性的关系
- **影响**: 提供新的结构化剪枝指导信号

#### 贡献4: 多尺度模型剪枝策略
- **意义**: 针对金字塔结构的专门剪枝设计
- **影响**: 可迁移到其他层次生成模型（如LDM）

---

### 5.5 快速验证实验（1天内）

#### 实验A: Token策略对比（简化版）

```bash
# 只对比2层（layer 8-9）
python quick_ablation.py \
    --layers 8 9 \
    --strategies real_tokens,ar_tokens \
    --num_samples 64 \
    --eval_samples 500  # 少量样本快速评估
```

#### 实验B: Scale-Mul相关性

```bash
# 分析现有checkpoint
python analyze_scale_mul.py \
    --var_ckpt /path/to/var_d16.pth \
    --output scale_analysis.png
```

**输出**: 可视化每层的scale_mul分布 + Hessian重要性对比

---

## 6. 其他VAR特有差异与创新空间

### 6.1 条件输入强度

**差异**: VAR有强类别条件，LLM无/弱条件

**创新**: 条件感知剪枝
```python
# 不同类别对head的依赖可能不同
def class_aware_pruning(model, calibration_labels):
    # 按类别聚类，分析head激活模式
    class_head_importance = cluster_head_importance_by_class(model, calibration_labels)
    # 保留对所有类别都重要的heads
    universal_heads = find_universal_important_heads(class_head_importance)
```

### 6.2 位置编码差异

**VAR**: Absolute + Level embedding
**LLM**: RoPE/ALiBi

**创新**: 位置感知剪枝（不同位置的tokens可能需要不同的头）

### 6.3 VQVAE耦合

**VAR**: 依赖VQVAE编码器
**LLM**: 独立tokenizer

**创新**: 联合VAE-VAR剪枝（同时压缩VQVAE和VAR）

---

## 7. 实施路线图

### Week 1: 基础设施
- [ ] 实现AR token生成模块
- [ ] 添加scale_mul分析工具
- [ ] 搭建评估pipeline（per-scale FID）

### Week 2: Token策略实验
- [ ] 实现3种token策略
- [ ] 对比实验 (sparsity=0.2, 0.3)
- [ ] 分析结果

### Week 3: Scale-Mul集成
- [ ] 实现scale-guided pruning
- [ ] 超参数搜索 (scale_weight)
- [ ] 消融实验

### Week 4: 尺度感知优化
- [ ] 实现scale-balanced sampling
- [ ] 实现scale-progressive pruning
- [ ] 综合对比

### Week 5: 论文撰写
- [ ] 整理实验结果
- [ ] 撰写方法部分
- [ ] 准备可视化

---

## 总结

针对VAR使用SlimGPT，我们识别了**3大核心创新方向**：

1. **推理对齐的Token策略** - 解决AR误差累积问题
2. **Scale-Mul引导剪枝** - 利用VAR特有的注意力先验
3. **尺度感知剪枝** - 针对10层金字塔的专门设计

这些创新不仅能提升VAR剪枝效果，还可能为自回归图像生成模型的压缩提供新的方法论。
