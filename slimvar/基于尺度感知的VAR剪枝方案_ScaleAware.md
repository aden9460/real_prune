# 基于尺度感知的VAR剪枝方案（Scale-Aware Pruning）

**创建日期**: 2025-11-12
**方案版本**: v1.0
**目标模型**: VAR-d16/d30
**状态**: ⭐ 推荐优先实施

---

## ⭐ 核心创新

**利用VAR的多尺度生成特性，将heads分为"整体关注型"和"细节关注型"，然后进行尺度感知的剪枝与组内补偿。**

### 与之前concentration方案的关键区别

| 方面 | Attention Concentration | **Scale-Aware（本方案）** |
|------|------------------------|--------------------------|
| 分组依据 | 集中度（熵、Gini、span） | **尺度偏好（关注哪些尺度）** |
| 理论基础 | 弱（集中≠功能） | **强（尺度=功能角色）** |
| 与VAR结合 | 无 | **高度契合** |
| 块对角假设 | 难满足 | **更可能满足** |
| 成功概率 | 20-30% | **40-50%** |

---

## 1. 理论基础

### 1.1 VAR的多尺度生成机制

VAR在每个尺度的生成过程：

```
尺度 k 的生成：
1. 接收：插值上采样的前 k-1 个尺度（继承粗粒度信息）
2. 生成：自回归生成当前尺度 k 的 k² 个token
3. Attention：可以看到所有历史token（causal mask）

具体尺度序列：
- 尺度 0: 1个token (first_l, class embedding)
- 尺度 1: 2×2 = 4 tokens
- 尺度 2: 3×3 = 9 tokens
- ...
- 尺度 10: 16×16 = 256 tokens
总计：1 + 4 + 9 + ... + 256 = 680 tokens
```

### 1.2 核心假设

**假设1：尺度偏好反映功能角色**

```
关注早期尺度(1-4) → 整合全局结构 → "整体关注型"头
关注当前尺度(10+) → 生成局部细节 → "细节关注型"头
```

**假设2：功能相似的heads输出相关性更高**

```
H[global_heads, global_heads] 内部相关性高
H[detail_heads, detail_heads] 内部相关性高
H[global_heads, detail_heads] 跨组相关性低 → 块对角
```

**数学依据**：

即使共享V矩阵，如果attention patterns在尺度维度分离：
```
global_head_output = Σ attn[early_tokens] * V[early_tokens]
detail_head_output = Σ attn[current_tokens] * V[current_tokens]

如果 early_tokens 和 current_tokens 的V特征分离
→ 输出低相关 → H 趋于块对角
```

### 1.3 为什么优于concentration方案？

**Concentration方案的问题**：

- ❌ 一个head可以"集中"关注早期1×1 token（集中但关注整体）
- ❌ 一个head可以"平缓"关注当前16×16所有token（平缓但关注细节）
- ❌ **集中度和功能是正交的**

**Scale-Aware方案的优势**：

- ✅ 尺度偏好直接反映在VAR的生成流程中
- ✅ 与模型架构天然对齐
- ✅ 可以自适应剪枝率（整体重要，细节冗余）

---

## 2. 量化指标

### 2.1 指标1：Scale-Weighted Attention Distance

**定义**：Attention关注的平均尺度距离

```python
def compute_scale_weighted_attention_distance(attention_weights, token_scales):
    """
    Args:
        attention_weights: [B, H, L, L] - attention maps
        token_scales: [L] - 每个token的尺度编号 [0,1,1,1,1,2,2,2,...]

    Returns:
        scale_distance: [H] - 每个head的平均尺度距离

    物理意义:
        - 小值(0-2): 关注相近尺度 → 细节关注
        - 中值(3-5): 混合关注
        - 大值(6+): 跨尺度关注 → 整体关注
    """
    B, H, L, _ = attention_weights.shape
    scale_distances = []

    for h in range(H):
        head_attn = attention_weights[:, h, :, :]  # [B, L, L]
        distances = []

        for b in range(B):
            for q_pos in range(L):
                query_scale = token_scales[q_pos]
                attn_dist = head_attn[b, q_pos, :]  # [L]

                # 加权平均尺度差
                weighted_dist = 0
                for k_pos in range(q_pos + 1):  # causal mask
                    key_scale = token_scales[k_pos]
                    scale_diff = abs(query_scale - key_scale)
                    weighted_dist += attn_dist[k_pos].item() * scale_diff

                distances.append(weighted_dist)

        scale_distances.append(np.mean(distances))

    return torch.tensor(scale_distances)
```

### 2.2 指标2：Coarse-Token Attention Ratio

**定义**：分配给早期尺度token的注意力比例

```python
def compute_coarse_attention_ratio(attention_weights, token_scales,
                                   coarse_threshold=4):
    """
    Args:
        coarse_threshold: 尺度 <= threshold 被认为是"粗粒度"

    Returns:
        coarse_ratios: [H] - 每个head关注粗粒度token的比例

    物理意义:
        - 高值(>0.5): 主要关注早期尺度 → 整体关注
        - 中值(0.3-0.5): 混合
        - 低值(<0.3): 主要关注后期尺度 → 细节关注
    """
    B, H, L, _ = attention_weights.shape
    coarse_ratios = []

    for h in range(H):
        head_attn = attention_weights[:, h, :, :]
        ratios = []

        for b in range(B):
            for q_pos in range(L):
                attn_dist = head_attn[b, q_pos, :]

                # 累加分配给粗粒度token的权重
                coarse_weight = sum(
                    attn_dist[k].item()
                    for k in range(q_pos + 1)
                    if token_scales[k] <= coarse_threshold
                )
                ratios.append(coarse_weight)

        coarse_ratios.append(np.mean(ratios))

    return torch.tensor(coarse_ratios)
```

### 2.3 指标3：Spatial Receptive Field

**定义**：Attention的空间感受野大小

```python
def compute_spatial_receptive_field(attention_weights, token_positions):
    """
    Args:
        token_positions: [L, 2] - 每个token的归一化空间坐标(x,y) ∈ [0,1]²

    Returns:
        receptive_fields: [H] - 每个head的平均空间感受野半径

    物理意义:
        - 大值(>0.5): 空间跨度大 → 整体关注
        - 小值(<0.3): 空间跨度小 → 细节关注
    """
    B, H, L, _ = attention_weights.shape
    receptive_fields = []

    for h in range(H):
        head_attn = attention_weights[:, h, :, :]
        fields = []

        for b in range(B):
            for q_pos in range(L):
                q_coord = token_positions[q_pos]  # [2]
                attn_dist = head_attn[b, q_pos, :]

                # 加权空间距离
                weighted_distance = sum(
                    attn_dist[k].item() * torch.norm(q_coord - token_positions[k]).item()
                    for k in range(q_pos + 1)
                )
                fields.append(weighted_distance)

        receptive_fields.append(np.mean(fields))

    return torch.tensor(receptive_fields)
```

### 2.4 辅助函数：Token映射

```python
def build_token_scale_mapping():
    """
    构建每个token所属的尺度

    Returns:
        token_scales: [680] - 每个position对应的尺度编号
    """
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    token_scales = []
    token_scales.append(0)  # first_l token，特殊标记为尺度0

    for scale_idx, patch_num in enumerate(patch_nums, start=1):
        num_tokens = patch_num ** 2
        token_scales.extend([scale_idx] * num_tokens)

    return torch.tensor(token_scales)  # [680]

def build_token_position_mapping():
    """
    构建每个token的空间坐标（归一化到[0,1]²）

    Returns:
        token_positions: [680, 2] - (x, y)坐标
    """
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    token_positions = []
    token_positions.append([0.5, 0.5])  # first_l token放在中心

    for patch_num in patch_nums:
        for i in range(patch_num):
            for j in range(patch_num):
                # 归一化坐标
                x = (j + 0.5) / patch_num
                y = (i + 0.5) / patch_num
                token_positions.append([x, y])

    return torch.tensor(token_positions)  # [680, 2]
```

---

## 3. 聚类与分组

### 3.1 K-means聚类（k=3）

```python
def cluster_heads_by_scale_preference(scale_distances, coarse_ratios,
                                     spatial_fields, k=3):
    """
    用K-means将heads分为3组：整体、混合、细节

    Args:
        scale_distances: [num_total_heads]
        coarse_ratios: [num_total_heads]
        spatial_fields: [num_total_heads]
        k: 聚类数（默认3）

    Returns:
        head_groups: dict, {"layer_0_head_3": "global", ...}
        cluster_stats: 聚类统计信息
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    # 归一化特征
    features = np.stack([
        scale_distances.numpy(),
        coarse_ratios.numpy(),
        spatial_fields.numpy()
    ], axis=1)  # [N, 3]

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # K-means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features_norm)

    # Silhouette score（聚类质量）
    silhouette = silhouette_score(features_norm, labels)

    # 反标准化中心，确定每个cluster的类型
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_types = assign_cluster_types(centers_original)

    # 构建head_groups
    head_groups = {}
    num_heads_per_layer = len(scale_distances) // 16  # 假设16层

    for idx, label in enumerate(labels):
        layer_idx = idx // num_heads_per_layer
        head_idx = idx % num_heads_per_layer
        key = f"layer_{layer_idx}_head_{head_idx}"
        head_groups[key] = cluster_types[label]  # "global", "mixed", or "detail"

    # 统计
    cluster_stats = {
        'silhouette_score': silhouette,
        'num_global': sum(1 for v in head_groups.values() if v == 'global'),
        'num_mixed': sum(1 for v in head_groups.values() if v == 'mixed'),
        'num_detail': sum(1 for v in head_groups.values() if v == 'detail'),
        'cluster_centers': centers_original.tolist()
    }

    return head_groups, cluster_stats

def assign_cluster_types(centers):
    """
    根据聚类中心的特征值判断类型

    Args:
        centers: [k, 3] - k个聚类中心，特征为[scale_dist, coarse_ratio, spatial_field]

    Returns:
        types: list of str - ["global", "mixed", "detail"]
    """
    k = len(centers)
    types = []

    for i in range(k):
        scale_dist = centers[i, 0]
        coarse_ratio = centers[i, 1]
        spatial_field = centers[i, 2]

        # 判断逻辑（可调整阈值）
        if coarse_ratio > 0.5 or spatial_field > 0.5:
            # 高coarse_ratio或大空间感受野 → 整体关注
            types.append('global')
        elif coarse_ratio < 0.25 and spatial_field < 0.3:
            # 低coarse_ratio且小空间感受野 → 细节关注
            types.append('detail')
        else:
            # 中间值 → 混合
            types.append('mixed')

    # 确保至少有一个global和一个detail
    if 'global' not in types:
        # 找coarse_ratio最大的
        max_idx = np.argmax(centers[:, 1])
        types[max_idx] = 'global'
    if 'detail' not in types:
        # 找coarse_ratio最小的
        min_idx = np.argmin(centers[:, 1])
        types[min_idx] = 'detail'

    return types
```

---

## 4. 尺度感知剪枝与补偿

### 4.1 自适应剪枝率分配

```python
def allocate_sparsity_by_scale_type(head_groups, global_sparsity=0.2):
    """
    根据尺度类型分配剪枝率

    核心思想：
        - Global heads重要（提供全局约束）→ 剪枝率低
        - Detail heads冗余（局部可互补）→ 剪枝率高

    Args:
        head_groups: dict, {"layer_X_head_Y": "global"/"mixed"/"detail"}
        global_sparsity: 全局平均剪枝率

    Returns:
        sparsity_dict: dict, {group_type: sparsity}
    """
    # 基础剪枝率配置
    base_sparsity = {
        'global': global_sparsity * 0.5,   # 整体: 10%（如果global=0.2）
        'mixed': global_sparsity,          # 混合: 20%
        'detail': global_sparsity * 1.5    # 细节: 30%
    }

    # 计算每种类型的head数量
    type_counts = {
        'global': sum(1 for v in head_groups.values() if v == 'global'),
        'mixed': sum(1 for v in head_groups.values() if v == 'mixed'),
        'detail': sum(1 for v in head_groups.values() if v == 'detail')
    }

    total_heads = sum(type_counts.values())

    # 验证加权平均是否等于global_sparsity
    weighted_avg = sum(
        base_sparsity[t] * type_counts[t] / total_heads
        for t in ['global', 'mixed', 'detail']
    )

    # 调整以匹配global_sparsity
    adjustment = global_sparsity / weighted_avg
    sparsity_dict = {
        t: base_sparsity[t] * adjustment
        for t in ['global', 'mixed', 'detail']
    }

    return sparsity_dict
```

### 4.2 分组剪枝与组内补偿

```python
def scale_aware_pruning_with_compensation(
    model, calibration_data, head_groups, sparsity_dict
):
    """
    尺度感知的分组剪枝与组内补偿

    流程：
    1. 对每一层，按尺度类型分组
    2. 每组使用对应的剪枝率
    3. 组内进行SlimGPT补偿
    """
    from slim_utils.slimgpt import SlimGPT

    num_layers = len(model.blocks)
    all_pruned_info = {}

    for layer_idx in range(num_layers):
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx} - Scale-Aware Pruning")
        print(f"{'='*60}")

        # 1. 收集该层的head分组
        groups = {'global': [], 'mixed': [], 'detail': []}
        num_heads = model.blocks[layer_idx].attn.num_heads

        for h in range(num_heads):
            key = f"layer_{layer_idx}_head_{h}"
            group_type = head_groups.get(key, 'mixed')  # 默认混合
            groups[group_type].append(h)

        print(f"  Global heads: {len(groups['global'])}")
        print(f"  Mixed heads: {len(groups['mixed'])}")
        print(f"  Detail heads: {len(groups['detail'])}")

        # 2. 准备SlimGPT
        O_weight = model.blocks[layer_idx].attn.proj.weight.data.clone()
        slimgpt = SlimGPT(O_weight)

        # 添加激活（需要提前收集）
        attn_outputs = collect_attention_outputs(
            model, layer_idx, calibration_data
        )
        slimgpt.add_batch(attn_outputs)

        # 3. 对每组分别剪枝和补偿
        layer_pruned_heads = []

        for group_type in ['global', 'mixed', 'detail']:
            heads_in_group = groups[group_type]
            if len(heads_in_group) == 0:
                continue

            group_sparsity = sparsity_dict[group_type]
            n_prune = int(len(heads_in_group) * group_sparsity)

            if n_prune == 0:
                print(f"  {group_type}: Skip (too few heads)")
                continue

            print(f"  {group_type}: Pruning {n_prune}/{len(heads_in_group)} heads")

            # 组内剪枝与补偿
            pruned_heads = slimgpt.struct_prune_grouped(
                sparsity=group_sparsity,
                head_groups={f"layer_{layer_idx}_head_{h}": 0
                           for h in heads_in_group},
                layer_idx=layer_idx,
                headsize=64
            )

            layer_pruned_heads.extend(pruned_heads)

        # 4. 应用新权重到模型
        model.blocks[layer_idx].attn.proj.weight.data = slimgpt.W

        all_pruned_info[layer_idx] = {
            'pruned_heads': layer_pruned_heads,
            'group_sizes': {k: len(v) for k, v in groups.items()},
            'sparsity_used': {k: sparsity_dict[k] for k in groups.keys()}
        }

    return all_pruned_info

def collect_attention_outputs(model, layer_idx, calibration_data):
    """
    收集指定层的attention输出（O projection的输入）
    """
    # 实现细节参考verification D
    pass
```

---

## 5. 验证实验

### 5.1 验证实验F：尺度依赖的块对角性

**目的**：验证按尺度偏好分组是否比按concentration分组更块对角

```python
def verify_scale_dependent_block_diagonality(
    model, calibration_data, scale_groups, concentration_groups
):
    """
    对比两种分组方法的块对角程度

    Returns:
        comparison: dict with BDR scores
    """
    # 1. 计算head output correlation matrix
    correlation_matrix = compute_head_output_correlation(
        model, calibration_data, layer_idx=7
    )

    # 2. 测量scale-based分组的BDR
    BDR_scale = measure_block_diagonality(
        correlation_matrix, scale_groups, layer_idx=7
    )

    # 3. 测量concentration-based分组的BDR
    BDR_conc = measure_block_diagonality(
        correlation_matrix, concentration_groups, layer_idx=7
    )

    # 4. 对比
    improvement = (BDR_conc - BDR_scale) / BDR_conc * 100

    results = {
        'BDR_scale_aware': BDR_scale,
        'BDR_concentration': BDR_conc,
        'improvement_percent': improvement,
        'verdict': 'pass' if BDR_scale < BDR_conc * 0.8 else 'fail'
    }

    print(f"\n{'='*60}")
    print("Verification F: Scale-Dependent Block Diagonality")
    print(f"{'='*60}")
    print(f"BDR (Scale-Aware):     {BDR_scale:.3f}")
    print(f"BDR (Concentration):   {BDR_conc:.3f}")
    print(f"Improvement:           {improvement:+.1f}%")

    if results['verdict'] == 'pass':
        print("✅ Scale-aware grouping is significantly more block-diagonal!")
        print("→ Proceed with full experiment")
    else:
        print("❌ Scale-aware grouping is not better")
        print("→ Consider hybrid approach")

    return results
```

### 5.2 验证实验G：重建误差对比

```python
def verify_scale_aware_reconstruction(
    model, calibration_data, scale_groups, sparsity=0.2
):
    """
    对比尺度感知剪枝 vs 全局剪枝 的重建误差
    """
    # 实现与verification D类似，但使用scale-aware分组
    # ...

    if mse_scale_aware < mse_global * 0.95:
        print("✅ Scale-aware pruning is significantly better!")
        return 'proceed'
    else:
        print("⚠️ Scale-aware pruning is not better than global")
        return 'marginal'
```

---

## 6. 完整实施流程

### Phase 1: 尺度偏好分析（2天）

```bash
# 步骤1: 创建分析脚本 analyze_scale_preference.py
python analyze_scale_preference.py \
    --model_depth 16 \
    --num_samples 100 \
    --layers 0,5,7,11,15 \
    --output scale_preference_analysis/

# 输出:
#   - head_scale_groups.json: {"layer_X_head_Y": "global"/"mixed"/"detail"}
#   - scale_preference_stats.json: 聚类统计
#   - visualizations: t-SNE图
```

### Phase 2: 验证实验（1天）

```bash
# 验证F: 块对角性对比
python verify_scale_block_diagonality.py \
    --head_groups scale_preference_analysis/head_scale_groups.json \
    --layers 7,11

# 验证G: 重建误差对比
python verify_scale_reconstruction.py \
    --head_groups scale_preference_analysis/head_scale_groups.json \
    --sparsity 0.2
```

**决策标准**：
- 如果验证F通过（BDR降低>20%）且验证G通过（MSE降低>5%）→ 继续Phase 3
- 否则 → 转向混合策略或其他方向

### Phase 3: 完整剪枝实验（5-7天）

```bash
# 尺度感知剪枝
python model_slimming_scale_aware.py \
    --config pruning_config_scale_aware.json \
    --head_groups scale_preference_analysis/head_scale_groups.json \
    --output checkpoints/pruned_scale_aware_d16.pth

# Baseline（对比）
python model_slimming_basic.py \
    --config pruning_config_20percent.json \
    --output checkpoints/pruned_baseline_d16.pth

# Finetune两个模型
# 评估FID
```

---

## 7. 预期结果

### 7.1 定量预测

| 指标 | Baseline | Scale-Aware | 改进 |
|-----|---------|------------|------|
| FID | 5.20 | 4.85 | **6.7%** ⬆️ |
| Silhouette | N/A | 0.40 | N/A |
| BDR | 0.55 | 0.35 | 36% ⬇️ |

**保守估计**：FID改进 3-5%
**乐观估计**：FID改进 7-10%

### 7.2 定性发现（预期）

1. **层级差异**：
   - 浅层（0-5）：大部分为global heads
   - 深层（11-15）：大部分为detail heads

2. **尺度分布**：
   - Global heads主要关注尺度1-4
   - Detail heads主要关注尺度8-10

3. **块对角性**：
   - `H[global, global]`和`H[detail, detail]`内部相关强
   - `H[global, detail]`跨组相关弱

---

## 8. 风险与应对

### 风险1：聚类质量差

**表现**：Silhouette < 0.25

**应对**：
1. 调整coarse_threshold（默认4，可尝试3或5）
2. 只用2组（global vs detail，去掉mixed）
3. 手动标注几个典型层

### 风险2：块对角假设不成立

**表现**：BDR > 0.5

**应对**：
1. 软分组（加权补偿）
2. 混合策略（50% scale-aware + 50% global）

### 风险3：Detail heads不够冗余

**表现**：Detail组剪枝后FID下降明显

**应对**：
1. 降低detail组的剪枝率（30% → 20%）
2. 只剪枝global和mixed组

---

## 9. 为什么推荐优先尝试此方案？

### 对比之前所有方案

| 方案 | 理论 | VAR结合 | 成功率 | 时间 |
|-----|------|---------|--------|------|
| Scale_mul聚类 | ❌ | ❌ | 0% | 已失败 |
| Concentration | ⚠️ | ❌ | 20-30% | 10天 |
| **Scale-Aware** | ✅ | ✅ | **40-50%** | **7-10天** |
| Attention-Guided Criteria | ✅ | ⚠️ | 30-40% | 5天 |

**Scale-Aware的独特优势**：

1. ✅ **唯一利用VAR多尺度特性的方案**
2. ✅ 理论基础强且直观
3. ✅ 即使失败也有研究价值（head的尺度偏好分析）
4. ✅ 可以与其他方案结合（如与Attention-Guided Criteria）

---

## 10. 总结

### 核心创新点

1. **首次利用VAR的多尺度生成特性指导剪枝**
2. **用尺度偏好替代集中度进行分组**
3. **自适应剪枝率（整体少剪，细节多剪）**

### 预期贡献

- **实用**：提升剪枝后FID（预期3-7%）
- **理论**：首次分析VAR heads的尺度偏好
- **通用**：可推广到其他多尺度生成模型

### 下一步

✅ **立即开始验证实验**（2-3天）
✅ **如果验证通过，投入完整实验**（5-7天）
✅ **如果验证失败，转向Attention-Guided Criteria**

---

**文档版本**: v1.0
**最后更新**: 2025-11-12
**推荐度**: ⭐⭐⭐⭐⭐
