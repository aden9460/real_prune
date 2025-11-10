# 多尺度SlimGPT重要性评估实现方案

## 理论基础

VAR模型的独特之处是10个尺度的autoregressive生成：
```python
patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # 680 total tokens
```

每个尺度的生成模式本质不同（1×1的global structure vs 16×16的fine details），因此需要针对性的重要性评估策略。

## 核心实现方案

### 方法2：逐尺度评估重要性和加权 (推荐)

**理论优势**:
- **信息分离性**: 每个尺度的生成模式本质不同
- **尺度特异性**: 某些神经元可能只在特定尺度下重要
- **可解释性**: 可以分析每个尺度对最终重要性的贡献

### 基于现有Catcher机制的多尺度实现

```python
class MultiScaleVARCatcher:
    def __init__(self, module, scales=10):
        self.scales = scales
        self.catchers_per_scale = [
            VARCatcher(module) for _ in range(scales)
        ]

    def add_batch_for_scale(self, scale_idx, inp, out):
        """为特定尺度添加数据"""
        self.catchers_per_scale[scale_idx].add_batch(inp, out)

    def compute_multi_scale_importance(self, scale_weights=None):
        """计算多尺度加权重要性"""
        if scale_weights is None:
            scale_weights = [1.0] * self.scales  # 均匀权重

        importance_per_scale = []
        for i, catcher in enumerate(self.catchers_per_scale):
            # 使用现有的SlimGPT公式
            H = catcher.H
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

            # OBS重要性公式：权重²/Hessian逆对角
            if catcher.headsize > 1:  # 多头注意力的结构化重要性
                Hinv_diag = torch.stack([
                    Hinv[j:j+catcher.headsize, j:j+catcher.headsize]
                    for j in range(0, catcher.columns, catcher.headsize)
                ])
                Hinv_diag = torch.diagonal(
                    torch.linalg.cholesky(Hinv_diag), dim1=-2, dim2=-1
                ).reshape(-1)
                Hinv_diag = Hinv_diag ** 2
            else:
                Hinv_diag = Hinv.diag()

            # 计算重要性
            W = catcher.layer.weight.data
            error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
            importance_per_scale.append(error)

        # 尺度加权组合
        final_importance = sum(
            w * imp for w, imp in zip(scale_weights, importance_per_scale)
        )
        return final_importance, importance_per_scale
```

### 与现有代码的集成方案

```python
# 修改 model_slimming_basic.py 中的实现
def enhanced_var_pruning():
    """增强的VAR剪枝实现"""

    # Phase 1: 收集多尺度激活
    multi_scale_catchers = {}

    for layer_name in ["attn.proj", "ffn.fc2"]:
        layer_module = get_layer_by_name(model, layer_name)
        multi_scale_catchers[layer_name] = MultiScaleVARCatcher(
            layer_module, scales=10
        )

    # Phase 2: 使用不同尺度的校准数据
    for scale in range(10):
        print(f"Collecting data for scale {scale}")

        # 准备该尺度的校准数据
        calibration_data_scale = prepare_scale_specific_calibration_data(
            scale, num_samples=args.nsamples
        )

        # 为每个目标层收集该尺度的激活
        for batch in calibration_data_scale:
            with torch.no_grad():
                # 前向传播到目标层
                _ = model(batch)  # 触发forward hooks

                # Catchers会自动收集激活数据

    # Phase 3: 计算多尺度重要性
    scale_weights = compute_scale_weights()

    layer_importances = {}
    for layer_name, catcher in multi_scale_catchers.items():
        importance, importance_per_scale = catcher.compute_multi_scale_importance(
            scale_weights
        )
        layer_importances[layer_name] = {
            'final': importance,
            'per_scale': importance_per_scale
        }

        # 可选：保存每个尺度的重要性分析
        save_scale_analysis(layer_name, importance_per_scale, scale_weights)

    # Phase 4: 基于多尺度重要性进行剪枝
    for layer_name, importance_data in layer_importances.items():
        importance = importance_data['final']

        # 选择要剪枝的索引（重要性最低的）
        num_pruned = int(len(importance) * sparsity_ratio)
        pruned_indices = torch.argsort(importance)[:num_pruned]

        # 执行结构化剪枝
        apply_structured_pruning(layer_name, pruned_indices)

    return model

def prepare_scale_specific_calibration_data(scale_idx, num_samples=128):
    """准备特定尺度的校准数据"""
    # 根据尺度索引，生成或筛选相应的校准数据
    # 可以是：
    # 1. 只包含特定分辨率的图像patch
    # 2. 强调特定尺度特征的数据增强
    # 3. 基于VAR生成过程中特定阶段的中间状态

    if args.use_images:
        # 使用真实图像数据
        return get_imagenet_calibration_data(num_samples)
    else:
        # 使用label-only数据，但考虑尺度特异性
        return generate_scale_aware_labels(scale_idx, num_samples)
```

### 尺度权重的设计原则

```python
def compute_scale_weights():
    """计算尺度权重"""
    # 基于人类视觉感知和VAR生成特点的权重设计

    # 方案1：基于感知重要性
    # 粗尺度(1-4): 结构重要性高 → 权重大
    # 细尺度(8-16): 细节重要性高 → 权重中等
    # 中等尺度(5-7): 过渡性质 → 权重相对小
    perceptual_weights = [
        0.15, 0.15, 0.12, 0.12,  # 1,2,3,4 - 粗尺度
        0.08, 0.08, 0.08,        # 5,6,8 - 中等尺度
        0.1, 0.1, 0.12           # 10,13,16 - 细尺度
    ]

    # 方案2：自适应权重（基于数据驱动）
    adaptive_weights = compute_adaptive_scale_weights()

    # 方案3：均匀权重（作为baseline）
    uniform_weights = [1.0/10] * 10

    # 可以通过实验选择最佳权重方案
    return perceptual_weights

def compute_adaptive_scale_weights():
    """基于数据驱动的自适应尺度权重"""
    # 分析每个尺度对最终生成质量的影响
    scale_sensitivities = []

    for scale in range(10):
        # 测量该尺度扰动对输出的影响
        sensitivity = measure_scale_sensitivity(scale)
        scale_sensitivities.append(sensitivity)

    # 归一化为权重
    total_sensitivity = sum(scale_sensitivities)
    weights = [s / total_sensitivity for s in scale_sensitivities]

    return weights
```

### 数据收集和校准策略

```python
def generate_scale_aware_labels(scale_idx, num_samples):
    """生成尺度感知的标签数据"""
    # VAR的10个尺度对应不同的语义层次
    # 可以设计不同的类别采样策略

    if scale_idx < 4:  # 粗尺度：关注大类别差异
        # 选择视觉差异大的类别
        preferred_classes = select_visually_distinct_classes()
    elif scale_idx < 7:  # 中等尺度：关注中等粒度特征
        preferred_classes = select_medium_grain_classes()
    else:  # 细尺度：关注纹理和细节
        preferred_classes = select_texture_rich_classes()

    # 生成校准标签
    labels = torch.randint(0, len(preferred_classes), (num_samples,))
    return map_to_imagenet_classes(labels, preferred_classes)

def save_scale_analysis(layer_name, importance_per_scale, scale_weights):
    """保存尺度分析结果"""
    analysis = {
        'layer_name': layer_name,
        'importance_per_scale': [imp.cpu().numpy() for imp in importance_per_scale],
        'scale_weights': scale_weights,
        'weighted_importance': [
            (w * imp).cpu().numpy()
            for w, imp in zip(scale_weights, importance_per_scale)
        ]
    }

    save_path = f"scale_analysis_{layer_name}.json"
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=lambda x: x.tolist())
```

### 实验验证方案

```python
def validate_multi_scale_pruning():
    """验证多尺度剪枝的有效性"""

    # 对比实验：
    experiments = [
        {'name': 'uniform', 'weights': [1.0/10] * 10},
        {'name': 'perceptual', 'weights': compute_perceptual_weights()},
        {'name': 'adaptive', 'weights': compute_adaptive_scale_weights()},
        {'name': 'coarse_bias', 'weights': [0.2, 0.2, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]},
        {'name': 'fine_bias', 'weights': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.15, 0.2, 0.25]}
    ]

    results = {}
    for exp in experiments:
        print(f"Testing {exp['name']} weighting strategy...")

        # 使用该权重策略进行剪枝
        pruned_model = enhanced_var_pruning(scale_weights=exp['weights'])

        # 评估剪枝后的性能
        fid_score = evaluate_fid(pruned_model)
        is_score = evaluate_inception_score(pruned_model)

        results[exp['name']] = {
            'fid': fid_score,
            'is': is_score,
            'weights': exp['weights']
        }

    # 保存结果
    with open('multi_scale_pruning_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results
```

## 总结

这个多尺度重要性评估方案的核心创新点：

1. **分尺度Fisher信息计算**: 为每个尺度独立计算Hessian矩阵
2. **加权重要性组合**: 基于感知重要性或自适应方法设计尺度权重
3. **无缝集成**: 基于现有VARCatcher机制，最小化代码修改
4. **实验验证**: 提供完整的对比实验框架

该方案充分利用了VAR模型的多尺度特性，有望在保持生成质量的同时实现更有效的模型压缩。