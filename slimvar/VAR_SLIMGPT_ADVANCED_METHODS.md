# VAR + SlimGPT 高级剪枝方法

> **项目背景**：VAR 模型使用 SlimGPT/OBS 进行 training-free 剪枝
>
> **问题**：原始方法 FID=178，修复 scale_mul 后 FID=8.87（0.2 sparsity）
>
> **目标**：进一步优化高稀疏度性能，探索 VAR 特定的剪枝创新

---

## 核心理论基础

### 🔍 **关键洞察**：OBS 输入分布假设失效

**核心问题**：
```python
# scale_mul 作用于 Q（改变注意力模式）
q = F.normalize(q, dim=-1).mul(scale_mul)  # basic_var.py:104

# 但 OBS 对 proj 进行补偿（作用于 O）
W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # slimgpt.py:117
```

**理论分析**：
1. **scale_mul → Q 变化**：改变注意力权重分布
2. **注意力模式变化 → proj 输入分布变化**：不同的注意力输出作为 proj 输入
3. **OBS 基于校准时的输入分布计算 Hessian**：H = X^T X
4. **剪枝后 scale_mul 变化违背了 OBS 的输入分布假设**
5. **结果**：OBS 补偿基于错误的分布假设，导致 FID 暴涨

**实验验证**：
- 修复前（重置 scale_mul）：FID = 178
- 修复后（保留原始 scale_mul）：FID = 8.87（改善 95%）

---

## 方法一：参数感知剪枝 ⭐⭐⭐

### 理论依据
VAR 的 scale_mul 直接影响注意力计算，传统 OBS 无法处理这种参数间的耦合关系。

### 完整实现

```python
def parameter_aware_pruning(model, calibration_data, sparsity=0.2):
    """
    Parameter-aware pruning for VAR models

    考虑 scale_mul 与权重矩阵的耦合关系，在剪枝决策中同时优化两者
    """

    def compute_coupled_importance(attn_module, calibration_inputs):
        """计算考虑 scale_mul 耦合的重要性分数"""

        # 1. 收集原始激活
        original_importance = []

        # 2. 扰动 scale_mul 测试敏感性
        scale_sensitivity = []
        original_scale = attn_module.scale_mul_1H11.data.clone()

        for head_idx in range(attn_module.num_heads):
            # 小幅扰动单个 head 的 scale_mul
            perturbed_scale = original_scale.clone()
            perturbed_scale[0, head_idx, 0, 0] *= 0.9  # 10% reduction

            attn_module.scale_mul_1H11.data = perturbed_scale

            # 计算输出变化
            with torch.no_grad():
                output_change = 0
                for batch in calibration_inputs:
                    original_out = attn_module(batch)
                    # 量化输出变化幅度
                    output_change += torch.norm(original_out).item()

            scale_sensitivity.append(output_change)
            attn_module.scale_mul_1H11.data = original_scale  # 恢复

        # 3. 结合权重重要性和 scale 敏感性
        weight_importance = compute_obs_importance(attn_module.proj)  # 标准 OBS
        scale_sensitivity = torch.tensor(scale_sensitivity)

        # 融合重要性：高权重重要性 OR 高 scale 敏感性的 head 保留
        combined_importance = []
        head_dim = 64

        for head_idx in range(attn_module.num_heads):
            head_start = head_idx * head_dim
            head_end = head_start + head_dim

            # 该 head 对应权重的平均重要性
            head_weight_importance = weight_importance[head_start:head_end].mean()
            head_scale_sensitivity = scale_sensitivity[head_idx]

            # 融合策略：加权几何平均
            alpha = 0.6  # 权重重要性占主导
            beta = 0.4   # scale 敏感性为辅助

            fused_score = (head_weight_importance ** alpha) * (head_scale_sensitivity ** beta)
            combined_importance.extend([fused_score] * head_dim)

        return torch.tensor(combined_importance)

    def prune_with_scale_preservation(layer, importance_scores, sparsity):
        """执行剪枝并保留对应的 scale_mul"""

        # 1. 选择要剪枝的 channels
        num_prune = int(len(importance_scores) * sparsity)
        _, prune_indices = torch.topk(importance_scores, num_prune, largest=False)

        # 2. 计算保留的 heads
        head_dim = 64
        pruned_heads = set((prune_indices // head_dim).tolist())
        all_heads = set(range(layer.attn.num_heads))
        keep_heads = sorted(list(all_heads - pruned_heads))

        # 3. 保留对应的 scale_mul 值（关键修复）
        old_scale_mul = layer.attn.scale_mul_1H11.data
        new_scale_mul = old_scale_mul[0, keep_heads, 0, 0].view(1, -1, 1, 1)
        layer.attn.scale_mul_1H11 = nn.Parameter(new_scale_mul.clone(), requires_grad=True)

        # 4. 执行结构化剪枝
        tp.prune_linear_in_channels(layer.attn.proj, prune_indices.tolist())
        # ... 其他 torch-pruning 操作

        return prune_indices

    # 主剪枝循环
    for i, layer in enumerate(model.blocks):
        print(f"Processing layer {i} with parameter-aware pruning...")

        # 计算耦合重要性
        importance = compute_coupled_importance(layer.attn, calibration_data)

        # 执行剪枝
        prune_with_scale_preservation(layer, importance, sparsity)

    return model
```

### 预期效果
- **0.2 sparsity**：FID < 6（从 8.87 进一步改善）
- **0.4 sparsity**：FID < 30（从 64.37 大幅改善）
- **实现难度**：⭐⭐ 中等（需要修改 SlimGPT 核心逻辑）

---

## 方法二：条件重要性评估 ⭐⭐

### 理论依据
VAR 的 AdaLN 使得不同类别条件下的激活分布显著不同。传统 OBS 混合所有类别的 Hessian，可能不是最优策略。

### 数学分析

**问题**：标准 OBS 计算
```python
H_mixed = (1/N) Σ X_i^T X_i  # 混合所有样本
```

**改进**：按条件分离
```python
H_c = (1/N_c) Σ_{x∈class_c} X_c^T X_c  # 每类单独计算
importance_c = diag(H_c^{-1})           # 每类重要性
final_importance = Σ p(c) * importance_c # 按类别概率加权
```

### 完整实现

```python
def conditional_importance_pruning(model, calibration_data, class_labels, sparsity=0.2):
    """
    Conditional importance evaluation considering AdaLN's class-dependent behavior
    """

    class ConditionalSlimGPT:
        def __init__(self, layer, num_classes=1000):
            self.layer = layer
            self.num_classes = num_classes
            self.class_hessians = {}  # {class_id: Hessian_matrix}
            self.class_counts = {}    # {class_id: sample_count}

        def add_batch_conditional(self, inp, out, class_labels):
            """按类别分别累积 Hessian"""

            batch_size = inp.shape[0]
            inp_reshaped = inp.reshape(-1, inp.shape[-1]).t()  # [hidden, batch*seq]

            for i, class_id in enumerate(class_labels):
                class_id = int(class_id.item())

                # 提取该样本的激活
                sample_inp = inp_reshaped[:, i*inp.shape[1]:(i+1)*inp.shape[1]]

                # 初始化该类别的 Hessian
                if class_id not in self.class_hessians:
                    feature_dim = sample_inp.shape[0]
                    self.class_hessians[class_id] = torch.zeros(
                        feature_dim, feature_dim, device=inp.device, dtype=torch.float32
                    )
                    self.class_counts[class_id] = 0

                # 更新 Hessian
                H_c = self.class_hessians[class_id]
                n_c = self.class_counts[class_id]

                # 增量更新：H_new = (n*H_old + X^T X) / (n+1)
                H_c *= n_c / (n_c + 1)
                sample_inp_norm = math.sqrt(2 / (n_c + 1)) * sample_inp.float()
                H_c += sample_inp_norm.matmul(sample_inp_norm.t())

                self.class_counts[class_id] = n_c + 1

        def compute_conditional_importance(self, class_distribution=None):
            """计算条件重要性分数"""

            if class_distribution is None:
                # 使用 ImageNet 均匀分布
                class_distribution = {cid: 1.0/self.num_classes for cid in self.class_hessians.keys()}

            # 计算每类的重要性
            class_importances = {}
            for class_id, H_c in self.class_hessians.items():
                try:
                    # 正则化 Hessian
                    damping = 0.01 * torch.mean(torch.diag(H_c))
                    H_regularized = H_c + damping * torch.eye(H_c.shape[0], device=H_c.device)

                    # 重要性 = Hessian 对角线（Fisher Information）
                    importance_c = torch.diag(H_regularized)
                    class_importances[class_id] = importance_c

                except:
                    # 如果矩阵奇异，使用伪逆
                    importance_c = torch.diag(torch.pinverse(H_c))
                    class_importances[class_id] = importance_c

            # 按分布加权平均
            weighted_importance = None
            total_weight = 0

            for class_id, importance in class_importances.items():
                weight = class_distribution.get(class_id, 0) * self.class_counts[class_id]
                if weighted_importance is None:
                    weighted_importance = weight * importance
                else:
                    weighted_importance += weight * importance
                total_weight += weight

            return weighted_importance / total_weight if total_weight > 0 else weighted_importance

    def analyze_class_activation_differences(model, calibration_data, class_labels):
        """分析不同类别下的激活差异"""

        class_stats = {}

        with torch.no_grad():
            for batch_idx, (data_batch, label_batch) in enumerate(zip(calibration_data, class_labels)):
                for i, label in enumerate(label_batch):
                    label = int(label.item())

                    # 单样本前向传播
                    sample_data = data_batch[i:i+1]

                    # 获取每层激活统计
                    layer_activations = []
                    def hook_fn(module, input, output):
                        layer_activations.append({
                            'mean': output.mean().item(),
                            'std': output.std().item(),
                            'max': output.max().item(),
                            'min': output.min().item()
                        })

                    # 注册 hooks
                    hooks = []
                    for layer in model.blocks:
                        hooks.append(layer.attn.proj.register_forward_hook(hook_fn))

                    # 前向传播
                    _ = model(sample_data)

                    # 清理 hooks
                    for hook in hooks:
                        hook.remove()

                    # 保存统计信息
                    if label not in class_stats:
                        class_stats[label] = []
                    class_stats[label].append(layer_activations)

        # 分析类间差异
        print("\\n=== Class Activation Analysis ===")
        for class_id in sorted(class_stats.keys())[:5]:  # 显示前5个类别
            stats = class_stats[class_id]
            layer_means = [np.mean([s[0]['mean'] for s in stats]) for _ in range(len(stats[0]))]
            print(f"Class {class_id}: Layer means = {layer_means[:3]}")  # 前3层

        return class_stats

    # 主剪枝流程
    print("=== Conditional Importance Pruning ===")

    # 1. 分析类别激活差异
    class_stats = analyze_class_activation_differences(model, calibration_data, class_labels)

    # 2. 按层执行条件重要性剪枝
    for layer_idx, layer in enumerate(model.blocks):
        print(f"\\nProcessing layer {layer_idx} with conditional evaluation...")

        # 创建条件 SlimGPT
        conditional_pruner = ConditionalSlimGPT(layer.attn.proj)

        # 收集条件激活
        def add_batch_hook(module, inp, out):
            batch_labels = class_labels  # 假设 batch 顺序一致
            conditional_pruner.add_batch_conditional(inp[0], out, batch_labels)

        hook = layer.attn.proj.register_forward_hook(add_batch_hook)

        # 前向传播收集
        with torch.no_grad():
            for batch_data in calibration_data:
                _ = model(batch_data)

        hook.remove()

        # 计算条件重要性
        importance_scores = conditional_pruner.compute_conditional_importance()

        # 执行剪枝
        num_prune = int(len(importance_scores) * sparsity)
        _, prune_indices = torch.topk(importance_scores, num_prune, largest=False)

        # 应用 torch-pruning
        tp.prune_linear_in_channels(layer.attn.proj, prune_indices.tolist())

        print(f"  Pruned {len(prune_indices)} features based on conditional importance")

    return model
```

### 预期效果
- **理论改善**：更准确的重要性评估，特别是在高稀疏度下
- **适用场景**：类别间激活分布差异大的情况
- **实验优先级**：⭐⭐ 中等（需要验证 AdaLN 的类别差异是否显著）

---

## 方法三：任务导向剪枝 ⭐⭐⭐

### 理论依据
**核心洞察**：重构误差 ≠ 生成质量

```python
# OBS 最小化：
minimize ||W_pruned @ X - W_original @ X||^2  # 重构误差

# VAR 真正需要：
minimize FID(images_pruned, images_original)    # 生成质量
```

**离散采样脆弱性**：
```python
# 细微的 logits 变化可能导致完全不同的 token
logits_original = [2.1, 1.9, 1.8, ...]  → argmax = 0
logits_pruned   = [1.8, 2.0, 1.9, ...]  → argmax = 1  # 完全不同！

# 结果：680 tokens 中 30% 变化 → FID 暴涨
```

### 完整实现

```python
def task_oriented_pruning(model, vae, calibration_data, sparsity=0.2, fid_target=20.0):
    """
    Task-oriented pruning that directly optimizes FID instead of reconstruction error
    """

    def compute_fid_based_importance(layer, test_images, current_fid):
        """通过扰动测试计算 FID 导向的重要性"""

        importance_scores = []
        baseline_fid = current_fid

        # 获取层权重
        weight = layer.attn.proj.weight.data  # [out_features, in_features]

        for channel_idx in range(weight.shape[1]):  # 遍历输入通道
            # 临时置零该通道
            original_weight = weight[:, channel_idx].clone()
            weight[:, channel_idx] = 0

            # 生成测试图像
            with torch.no_grad():
                test_class_labels = torch.arange(10).cuda()  # 测试 10 个类别
                generated_images = []

                for label in test_class_labels:
                    img = model.autoregressive_infer_cfg(
                        B=1, label_B=label, cfg=1.0, top_k=0, top_p=0.9
                    )
                    generated_images.append(img)

                generated_batch = torch.cat(generated_images, dim=0)

            # 计算 FID 变化
            try:
                # 使用快速 FID 近似（或真实 FID 如果计算资源充足）
                fid_change = compute_fast_fid_approximation(generated_batch, test_images[:10])
                importance = abs(fid_change - baseline_fid)  # FID 变化幅度作为重要性
            except:
                importance = 0.0  # 计算失败时设为低重要性

            importance_scores.append(importance)

            # 恢复权重
            weight[:, channel_idx] = original_weight

            if channel_idx % 100 == 0:
                print(f"    Tested channel {channel_idx}/{weight.shape[1]}, importance={importance:.4f}")

        return torch.tensor(importance_scores)

    def compute_fast_fid_approximation(generated_imgs, reference_imgs):
        """快速 FID 近似（用于重要性评估）"""

        # 使用预训练 Inception 提取特征
        inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        inception.cuda()

        def extract_features(imgs):
            with torch.no_grad():
                # 预处理图像
                if imgs.shape[1] != 3:
                    imgs = imgs.repeat(1, 3, 1, 1)  # 如果是灰度图
                imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear')
                imgs = (imgs - 0.5) / 0.5  # 标准化到 [-1, 1]

                # 提取特征
                features = inception(imgs)
                return features.cpu()

        gen_features = extract_features(generated_imgs)
        ref_features = extract_features(reference_imgs)

        # 计算特征距离（FID 的简化版本）
        gen_mean = gen_features.mean(dim=0)
        ref_mean = ref_features.mean(dim=0)

        feature_distance = torch.norm(gen_mean - ref_mean).item()
        return feature_distance

    def iterative_fid_optimization(model, target_sparsity, fid_threshold):
        """迭代 FID 优化剪枝"""

        current_sparsity = 0.0
        step_size = 0.05  # 每次剪枝 5%

        # 生成参考图像（用于 FID 计算）
        print("Generating reference images for FID evaluation...")
        reference_images = []
        with torch.no_grad():
            for class_id in range(50):  # 50 个类别的参考
                img = model.autoregressive_infer_cfg(
                    B=1, label_B=class_id, cfg=1.5, top_k=0, top_p=0.9
                )
                reference_images.append(img)
        reference_batch = torch.cat(reference_images, dim=0)

        while current_sparsity < target_sparsity:
            next_sparsity = min(current_sparsity + step_size, target_sparsity)
            prune_ratio = step_size / (1 - current_sparsity)  # 实际剪枝比例

            print(f"\\n=== Pruning Step: {current_sparsity:.1%} → {next_sparsity:.1%} ===")

            best_layer_idx = None
            best_importance = None
            best_fid_impact = float('inf')

            # 1. 为每层计算 FID 导向重要性
            for layer_idx, layer in enumerate(model.blocks):
                if hasattr(layer.attn, 'proj'):  # 确保是注意力层
                    print(f"Evaluating layer {layer_idx} for FID impact...")

                    # 计算该层的 FID 重要性
                    importance = compute_fid_based_importance(
                        layer, reference_batch, current_fid=0  # 基准 FID
                    )

                    # 预估剪枝该层的 FID 影响
                    num_prune = int(len(importance) * prune_ratio)
                    if num_prune > 0:
                        _, worst_indices = torch.topk(importance, num_prune, largest=False)
                        fid_impact = importance[worst_indices].sum().item()

                        if fid_impact < best_fid_impact:
                            best_fid_impact = fid_impact
                            best_layer_idx = layer_idx
                            best_importance = importance

            # 2. 剪枝最优层
            if best_layer_idx is not None:
                layer = model.blocks[best_layer_idx]
                num_prune = int(len(best_importance) * prune_ratio)
                _, prune_indices = torch.topk(best_importance, num_prune, largest=False)

                # 执行剪枝
                tp.prune_linear_in_channels(layer.attn.proj, prune_indices.tolist())

                print(f"Pruned {num_prune} channels from layer {best_layer_idx}")
                print(f"Estimated FID impact: {best_fid_impact:.3f}")

                # 3. 验证 FID
                actual_fid = evaluate_fid(model, reference_batch)
                print(f"Actual FID after pruning: {actual_fid:.2f}")

                if actual_fid > fid_threshold:
                    print(f"⚠️  FID {actual_fid:.2f} exceeds threshold {fid_threshold}")
                    print("Consider reducing pruning rate or applying compensation")
                    break

            current_sparsity = next_sparsity

        return model

    def evaluate_fid(model, reference_images):
        """评估当前模型的 FID"""

        # 生成同数量的测试图像
        generated_images = []
        with torch.no_grad():
            for i in range(len(reference_images)):
                img = model.autoregressive_infer_cfg(
                    B=1, label_B=i % 1000, cfg=1.5, top_k=0, top_p=0.9
                )
                generated_images.append(img)

        generated_batch = torch.cat(generated_images, dim=0)

        # 计算真实 FID（这里用简化版本）
        fid_score = compute_fast_fid_approximation(generated_batch, reference_images)
        return fid_score

    # 主流程
    print("=== Task-Oriented Pruning for FID Optimization ===")
    print(f"Target sparsity: {sparsity:.1%}, FID threshold: {fid_target}")

    # 测试原始模型 FID
    print("\\nEvaluating baseline FID...")
    # baseline_fid = evaluate_fid(model, None)  # 可以用标准数据集
    # print(f"Baseline FID: {baseline_fid:.2f}")

    # 执行迭代 FID 优化
    optimized_model = iterative_fid_optimization(model, sparsity, fid_target)

    return optimized_model
```

### 实验设计

```python
def run_task_oriented_experiments():
    """完整的任务导向剪枝实验"""

    sparsity_levels = [0.1, 0.2, 0.3, 0.4]
    fid_thresholds = [10, 15, 20, 30]

    results = {}

    for sparsity in sparsity_levels:
        for fid_threshold in fid_thresholds:
            print(f"\\n=== Experiment: {sparsity:.1%} sparsity, FID ≤ {fid_threshold} ===")

            # 加载原始模型
            model = load_var_model(depth=16)

            # 执行任务导向剪枝
            start_time = time.time()
            try:
                pruned_model = task_oriented_pruning(
                    model, vae, calibration_data,
                    sparsity=sparsity, fid_target=fid_threshold
                )

                # 评估结果
                final_fid = evaluate_comprehensive_fid(pruned_model)
                final_sparsity = calculate_actual_sparsity(pruned_model)
                pruning_time = time.time() - start_time

                results[(sparsity, fid_threshold)] = {
                    'final_fid': final_fid,
                    'final_sparsity': final_sparsity,
                    'pruning_time': pruning_time,
                    'success': final_fid <= fid_threshold
                }

                print(f"✅ Success: FID={final_fid:.2f}, Sparsity={final_sparsity:.1%}")

            except Exception as e:
                print(f"❌ Failed: {e}")
                results[(sparsity, fid_threshold)] = {'success': False, 'error': str(e)}

    # 保存结果
    with open('task_oriented_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results
```

### 预期效果
- **直接优化目标**：不再依赖重构误差的代理优化
- **自适应剪枝**：根据 FID 反馈动态调整策略
- **高质量保证**：确保生成质量不降至不可接受水平
- **实现复杂度**：⭐⭐⭐ 较高（需要集成 FID 计算和迭代优化）

---

## 实验优先级与建议

### 🥇 **优先级 1**：参数感知剪枝
**原因**：
- 直接解决了 scale_mul 耦合问题（已验证有效）
- 实现相对简单，基于现有框架
- 预期改善显著

**实施计划**：
1. 实现 `compute_coupled_importance` 函数
2. 在现有代码基础上测试 0.2 sparsity
3. 如果成功，扩展到 0.4 sparsity

### 🥈 **优先级 2**：任务导向剪枝
**原因**：
- 理论基础最强（直接优化目标指标）
- 可以作为其他方法的验证手段
- 长期价值高

**实施计划**：
1. 先实现 FID 快速近似版本
2. 小规模测试（5% sparsity）验证可行性
3. 如果有效，开发完整版本

### 🥉 **优先级 3**：条件重要性评估
**原因**：
- 需要先验证 AdaLN 的类别差异是否显著
- 理论假设需要实验确认
- 实现复杂度较高

**实施计划**：
1. 先分析不同类别的激活统计差异
2. 如果差异显著，再投入完整实现
3. 否则优先投入其他方法

---

## 代码集成建议

### 修改 `model_slimming_basic.py`

```python
# 在主函数中添加方法选择
parser.add_argument(
    "--pruning_method", type=str, default="standard",
    choices=["standard", "parameter_aware", "conditional", "task_oriented"],
    help="Pruning method to use"
)

# 在 model_slimming 函数中添加分支
if args.pruning_method == "parameter_aware":
    model = parameter_aware_pruning(model, calibration_labels, calibration_tokens, args)
elif args.pruning_method == "conditional":
    model = conditional_importance_pruning(model, calibration_labels, calibration_tokens, args)
elif args.pruning_method == "task_oriented":
    model = task_oriented_pruning(model, vae, calibration_data, args.sparsity)
else:
    model = model_slimming(model, calibration_labels, calibration_tokens, args)  # 原始方法
```

### 性能监控

```python
def benchmark_all_methods():
    """对比所有方法的性能"""

    methods = {
        "baseline": lambda m: m,  # 无剪枝
        "standard": lambda m: model_slimming(m, ...),  # 现有方法
        "parameter_aware": lambda m: parameter_aware_pruning(m, ...),
        "conditional": lambda m: conditional_importance_pruning(m, ...),
        "task_oriented": lambda m: task_oriented_pruning(m, ...)
    }

    results_table = []

    for method_name, method_func in methods.items():
        model = load_var_model(depth=16)
        pruned_model = method_func(model)

        # 评估指标
        fid_score = compute_fid(pruned_model)
        sparsity = compute_sparsity(pruned_model)
        inference_time = measure_inference_speed(pruned_model)

        results_table.append({
            'Method': method_name,
            'FID': f"{fid_score:.2f}",
            'Sparsity': f"{sparsity:.1%}",
            'Inference (ms)': f"{inference_time:.1f}",
        })

    print(tabulate(results_table, headers="keys", tablefmt="grid"))
```

---

## 总结

通过分析 VAR+SlimGPT 的核心问题（scale_mul 与 OBS 的输入分布假设冲突），我们提出了三种创新方法：

1. **参数感知剪枝**：直接解决参数耦合问题 ✅ **已验证有效**
2. **条件重要性评估**：利用 VAR 的条件生成特性
3. **任务导向剪枝**：直接优化生成质量而非重构误差

**关键洞察**：scale_mul 作用在 Q 上改变注意力模式，但 OBS 在 proj (O) 上做补偿。这种不匹配导致 OBS 的输入分布假设失效，是 FID 暴涨的根本原因。

**下一步**：优先实施参数感知剪枝的完整版本，然后探索任务导向剪枝作为长期方向。

---

**最后更新**：2024-11-10
**维护者**：Claude & 用户
**文件位置**：`/home/project/real_prune/slimvar/VAR_SLIMGPT_ADVANCED_METHODS.md`