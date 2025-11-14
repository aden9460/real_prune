# Scale_mul聚类假设验证指南

**创建日期**: 2025-11-11
**目标**: 验证核心假设 - "scale_mul接近的heads关注的注意力模式高度相似"
**状态**: 待验证

---

## 执行摘要

本文档提供5种方法来验证VAR模型剪枝策略1（聚类剪枝）的核心假设。这个假设是整个策略的理论基础，必须通过严格的实证验证。

**核心假设**：
```
如果两个heads的scale_mul非常接近（差异 < 阈值）
→ 它们的注意力模式高度相似
→ 功能冗余
→ 可以剪枝其中一个而影响较小
```

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [方法1: 直接注意力相似度](#2-方法1-直接注意力相似度)
3. [方法2: CKA输出表示相似度](#3-方法2-cka输出表示相似度)
4. [方法3: 聚类内部vs间相似度](#4-方法3-聚类内部vs间相似度)
5. [方法4: 消融实验验证](#5-方法4-消融实验验证)
6. [方法5: 可视化验证](#6-方法5-可视化验证)
7. [综合验证协议](#7-综合验证协议)
8. [实施指南](#8-实施指南)
9. [预期结果](#9-预期结果)

---

## 1. 背景与动机

### 1.1 为什么需要验证？

策略1（聚类剪枝）提出的核心思想是：
```python
# 识别scale_mul聚类
clusters = identify_clusters(scale_mul, threshold=std*0.3)
# 例如：[[0,1,2], [5,6,7,8], [12,13]]

# 每个聚类保留1个代表，剪除其余
for cluster in clusters:
    keep = cluster[len(cluster)//2]  # 中位数
    prune = cluster - {keep}
```

这个策略的有效性完全依赖于假设：**聚类内的heads功能冗余**。

如果假设不成立，那么：
- ❌ 聚类剪枝可能破坏重要的功能多样性
- ❌ 性能损失可能比预期严重
- ❌ 整个策略1需要重新设计

因此，**必须先验证假设，再大规模应用策略**。

### 1.2 Scale_mul的物理意义

在VAR的attention机制中：
```python
attention_scores = scale_mul * (Q @ K.T) / sqrt(d)
attention_weights = softmax(attention_scores)
```

**scale_mul的作用**：
- 高scale_mul → 注意力分布尖锐 → 关注特定局部模式
- 低scale_mul → 注意力分布平滑 → 关注全局整合

**假设的逻辑**：
```
scale_mul接近 → 注意力尖锐度相似 → 关注模式相似
```

但这个逻辑链条需要实证验证！因为：
- Q, K矩阵可能不同
- 即使尖锐度相同，关注的位置可能不同
- 需要实际数据验证

---

## 2. 方法1: 直接注意力相似度

### 2.1 核心思路

**最直接的验证方法**：直接比较attention weights的相似度。

```python
# 对于同一层的两个heads i和j，在相同输入上
attn_i = get_attention_map(model, image, layer_idx, head_i)  # [L, L]
attn_j = get_attention_map(model, image, layer_idx, head_j)  # [L, L]

# 计算相似度
similarity = cosine_similarity(attn_i.flatten(), attn_j.flatten())

# 如果scale_mul接近 → similarity应该高
# 如果scale_mul差异大 → similarity应该低
```

### 2.2 实施步骤

#### Step 1: Hook获取Attention Maps

```python
def extract_attention_maps(model, images, layer_idx):
    """
    从指定层提取所有heads的attention maps

    Args:
        model: VAR model
        images: [B, 3, H, W] input images
        layer_idx: target layer index

    Returns:
        attention_maps: [B, num_heads, L, L] attention weights (after softmax)
    """
    attention_maps = []

    def attention_hook(module, input, output):
        # VAR的attention实现：需要找到softmax之后的位置
        # 根据VAR代码结构，attention weights在哪里？
        # 需要hook正确的位置！
        attn = extract_attn_weights_from_var_block(output)
        attention_maps.append(attn)

    # 注册hook
    hook_handle = model.blocks[layer_idx].attn.register_forward_hook(attention_hook)

    # Forward pass
    with torch.no_grad():
        # 准备VAR需要的输入
        if hasattr(model, 'vae'):
            tokens = model.vae.img_to_idxBl(images)
            x_BLCv = model.vae.quantize.idxBl_to_var_input(tokens)
        else:
            # 简化：使用class labels
            labels = torch.arange(len(images)).cuda()
            _ = model(labels)

    # 移除hook
    hook_handle.remove()

    return torch.stack(attention_maps)  # [B, num_heads, L, L]
```

**关键问题**：VAR的attention实现细节
- 需要查看`VAR/models/basic_var.py`中的attention实现
- 确定softmax的位置
- 确保hook获取的是最终的attention weights

#### Step 2: 计算成对相似度

```python
def compute_pairwise_attention_similarity(attention_maps, metric='cosine'):
    """
    计算所有head pairs的平均注意力相似度

    Args:
        attention_maps: [B, num_heads, L, L]
        metric: 'cosine' or 'js_divergence'

    Returns:
        similarity_matrix: [num_heads, num_heads]
    """
    B, num_heads, L, _ = attention_maps.shape
    similarity_matrix = torch.zeros(num_heads, num_heads)

    for i in range(num_heads):
        for j in range(i, num_heads):
            # 在B个样本上计算平均相似度
            sims = []
            for b in range(B):
                attn_i = attention_maps[b, i].flatten()  # [L*L]
                attn_j = attention_maps[b, j].flatten()

                if metric == 'cosine':
                    sim = F.cosine_similarity(attn_i, attn_j, dim=0)
                elif metric == 'js_divergence':
                    # Jensen-Shannon divergence (更适合概率分布)
                    from scipy.spatial.distance import jensenshannon
                    sim = 1 - jensenshannon(
                        attn_i.cpu().numpy(),
                        attn_j.cpu().numpy()
                    )

                sims.append(sim)

            # 平均相似度
            avg_sim = np.mean(sims)
            similarity_matrix[i, j] = avg_sim
            similarity_matrix[j, i] = avg_sim

    return similarity_matrix
```

#### Step 3: 分析Scale_mul差异 vs 相似度

```python
def analyze_scale_vs_similarity(scale_mul, similarity_matrix):
    """
    分析scale_mul差异与注意力相似度的关系

    Returns:
        correlation: Pearson相关系数
        p_value: 显著性
        scatter_data: (scale_diffs, similarities) 用于绘图
    """
    num_heads = len(scale_mul)
    scale_diffs = []
    similarities = []

    for i in range(num_heads):
        for j in range(i+1, num_heads):
            # Scale_mul差异
            diff = abs(scale_mul[i] - scale_mul[j])
            scale_diffs.append(diff)

            # 注意力相似度
            sim = similarity_matrix[i, j]
            similarities.append(sim)

    # Pearson相关性检验
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(scale_diffs, similarities)

    return correlation, p_value, (scale_diffs, similarities)
```

#### Step 4: 可视化

```python
def plot_similarity_vs_scale_diff(scale_diffs, similarities, layer_idx, save_path):
    """
    绘制散点图：X=scale_mul差异, Y=注意力相似度
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(scale_diffs, similarities, alpha=0.5, s=20)

    # 拟合线
    z = np.polyfit(scale_diffs, similarities, 1)
    p = np.poly1d(z)
    plt.plot(scale_diffs, p(scale_diffs), "r--", alpha=0.8, linewidth=2)

    plt.xlabel('Scale_mul Difference |s_i - s_j|', fontsize=12)
    plt.ylabel('Attention Similarity', fontsize=12)
    plt.title(f'Layer {layer_idx}: Scale_mul Diff vs Attention Similarity', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    from scipy.stats import pearsonr
    r, p = pearsonr(scale_diffs, similarities)
    plt.text(0.05, 0.95, f'Pearson r = {r:.3f}\np-value = {p:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

### 2.3 成功标准

#### 定量标准
```
✅ Pearson相关系数 r < -0.3 (中等负相关)
✅ p-value < 0.05 (统计显著性)
✅ 散点图呈现明显的负相关趋势
```

#### 预期结果
```
如果假设成立：
- scale_mul差异越小 → 相似度越高
- 散点图左上角密集（小差异、高相似度）
- 散点图右下角分散（大差异、低相似度）
```

---

## 3. 方法2: CKA输出表示相似度

### 3.1 核心思路

方法1只看attention weights，但最终功能还取决于V矩阵。方法2更全面：直接比较heads的输出表示。

```python
# 计算每个head的输出
out_i = attn_i @ V_i  # [B, L, head_dim]
out_j = attn_j @ V_j

# 使用CKA (Centered Kernel Alignment) 度量相似度
similarity = CKA(out_i, out_j)
```

**CKA的优势**：
- 对线性变换不变（更robust）
- 广泛用于神经网络表示比较
- 能捕捉功能相似性而非简单的数值相似性

### 3.2 CKA计算

```python
def compute_CKA(X, Y):
    """
    Centered Kernel Alignment between two representations

    Args:
        X, Y: [batch*seq, dim] representations

    Returns:
        cka_score: float in [0, 1], 1表示完全相似
    """
    # Gram matrices (using linear kernel)
    K = X @ X.T  # [n, n]
    L = Y @ Y.T  # [n, n]

    # Center the gram matrices
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    # CKA score
    numerator = np.trace(K_c @ L_c)
    denominator = np.sqrt(np.trace(K_c @ K_c) * np.trace(L_c @ L_c))

    cka = numerator / (denominator + 1e-10)

    return cka

def extract_head_outputs(model, images, layer_idx):
    """
    提取每个head的输出表示 (attention @ V)

    Returns:
        head_outputs: [num_heads, B*L, head_dim]
    """
    # 实现细节：需要hook V矩阵和attention weights
    # 然后计算 attn @ V 对每个head
    pass
```

### 3.3 分析流程

与方法1类似，但用CKA相似度代替cosine相似度：

```python
# 1. 提取所有heads的输出表示
head_outputs = extract_head_outputs(model, calibration_images, layer_idx)

# 2. 计算16×16 CKA矩阵
cka_matrix = compute_pairwise_cka(head_outputs)

# 3. 分析与scale_mul的关系
correlation, p_value = analyze_scale_vs_similarity(scale_mul, cka_matrix)
```

### 3.4 成功标准

```
✅ Pearson相关系数 r < -0.25 (CKA通常比cosine宽松一些)
✅ p-value < 0.05
✅ 趋势与方法1一致
```

---

## 4. 方法3: 聚类内部vs间相似度

### 4.1 核心思路

**最直接验证聚类有效性的方法**：

```
如果聚类有效：
- 聚类内部相似度 >> 聚类间相似度
- Silhouette score > 0.3
```

这个方法不需要散点图，而是直接评估聚类质量。

### 4.2 实施步骤

#### Step 1: 识别聚类

```python
def identify_clusters(scale_mul, threshold):
    """
    基于scale_mul识别聚类

    Args:
        scale_mul: [num_heads] scale_mul值
        threshold: 聚类阈值（例如 std * 0.3）

    Returns:
        clusters: List[List[int]] 聚类列表
    """
    num_heads = len(scale_mul)

    # 排序
    sorted_indices = torch.argsort(scale_mul)
    sorted_scales = scale_mul[sorted_indices]

    # 计算相邻差异
    diffs = sorted_scales[1:] - sorted_scales[:-1]

    # 识别聚类边界
    clusters = []
    current_cluster = [sorted_indices[0].item()]

    for i in range(len(diffs)):
        if diffs[i] < threshold:
            current_cluster.append(sorted_indices[i+1].item())
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [sorted_indices[i+1].item()]

    if len(current_cluster) > 1:
        clusters.append(current_cluster)

    return clusters
```

#### Step 2: 计算聚类内部相似度

```python
def compute_intra_cluster_similarity(clusters, similarity_matrix):
    """
    计算聚类内部的平均相似度
    """
    intra_sims = []

    for cluster in clusters:
        if len(cluster) > 1:
            # 所有cluster内的pairs
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    head_i = cluster[i]
                    head_j = cluster[j]
                    sim = similarity_matrix[head_i, head_j]
                    intra_sims.append(sim)

    return np.mean(intra_sims) if intra_sims else 0.0
```

#### Step 3: 计算聚类间相似度

```python
def compute_inter_cluster_similarity(clusters, similarity_matrix):
    """
    计算聚类之间的平均相似度
    """
    inter_sims = []

    # 所有cluster pairs
    from itertools import combinations
    for cluster_a, cluster_b in combinations(clusters, 2):
        for head_i in cluster_a:
            for head_j in cluster_b:
                sim = similarity_matrix[head_i, head_j]
                inter_sims.append(sim)

    return np.mean(inter_sims) if inter_sims else 0.0
```

#### Step 4: Silhouette Score

```python
def compute_silhouette_score(clusters, similarity_matrix):
    """
    计算聚类的Silhouette score

    Returns:
        score: float in [-1, 1], >0.3表示聚类有效
    """
    # 转换为sklearn格式
    from sklearn.metrics import silhouette_score

    # 创建cluster labels
    num_heads = similarity_matrix.shape[0]
    labels = np.zeros(num_heads, dtype=int)

    for cluster_id, cluster in enumerate(clusters):
        for head_idx in cluster:
            labels[head_idx] = cluster_id

    # 孤立的heads标记为独立cluster
    singleton_id = len(clusters)
    for i in range(num_heads):
        if labels[i] == 0 and i not in clusters[0]:
            labels[i] = singleton_id
            singleton_id += 1

    # 转换similarity为distance
    distance_matrix = 1 - similarity_matrix

    # 计算silhouette score
    score = silhouette_score(distance_matrix, labels, metric='precomputed')

    return score
```

### 4.3 可视化：相似度热力图

```python
def plot_similarity_heatmap(similarity_matrix, scale_mul, clusters, save_path):
    """
    绘制16×16相似度热力图，标注聚类边界

    期望：如果聚类有效，应该看到块对角结构
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 按scale_mul排序（聚类会自然分组）
    sorted_indices = np.argsort(scale_mul)
    sorted_sim_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_scale_mul = scale_mul[sorted_indices]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(sorted_sim_matrix,
                cmap='RdYlGn',
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Attention Similarity'},
                ax=ax)

    # 标注聚类边界
    cluster_boundaries = []
    current_pos = 0
    for cluster in clusters:
        cluster_boundaries.append(current_pos + len(cluster))
        current_pos += len(cluster)

    for boundary in cluster_boundaries:
        ax.axhline(y=boundary, color='blue', linewidth=2, alpha=0.7)
        ax.axvline(x=boundary, color='blue', linewidth=2, alpha=0.7)

    # 添加scale_mul值标注
    ax.set_xlabel('Head Index (sorted by scale_mul)', fontsize=12)
    ax.set_ylabel('Head Index (sorted by scale_mul)', fontsize=12)
    ax.set_title(f'Attention Similarity Matrix with Clusters', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

### 4.4 成功标准

```
✅ Silhouette score > 0.3 (good clustering)
✅ Intra-cluster similarity > Inter-cluster similarity × 1.5
✅ 热力图呈现明显的块对角结构
```

---

## 5. 方法4: 消融实验验证

### 5.1 核心思路

**最终的功能验证**：如果聚类内真的冗余，剪除聚类内heads应该比剪除不同聚类heads的性能损失小。

```
实验A: 剪除3个聚类内heads → FID_A
实验B: 剪除3个不同聚类heads → FID_B

如果假设成立: FID_A < FID_B (性能损失更小)
```

### 5.2 实验设计

#### 实验配置

| 实验组 | 剪枝策略 | heads数量 | 预期FID |
|--------|---------|----------|---------|
| **Baseline** | 不剪枝 | 16 | 3.57 |
| **A1** | 聚类内剪枝(小聚类) | 13 | 3.60-3.65 |
| **A2** | 聚类内剪枝(大聚类) | 13 | 3.65-3.70 |
| **B** | 跨聚类剪枝 | 13 | 3.75-3.85 |

#### 具体实施

```python
def ablation_study_clustering(model, vae, clusters, layer_idx):
    """
    消融实验：对比聚类内vs跨聚类剪枝

    Args:
        clusters: 识别的聚类，例如 [[0,1,2], [5,6,7,8], [12,13]]

    Returns:
        results: {
            'baseline_fid': float,
            'intra_cluster_fid': float,
            'inter_cluster_fid': float,
            'fid_ratio': float  # inter / intra，期望>1.2
        }
    """
    # 实验A: 聚类内剪枝
    # 选择最大的聚类
    largest_cluster = max(clusters, key=len)
    prune_heads_intra = largest_cluster[:3]  # 剪3个

    model_A = copy.deepcopy(model)
    prune_specific_heads(model_A, layer_idx, prune_heads_intra)
    fid_A = evaluate_fid(model_A, vae)

    # 实验B: 跨聚类剪枝
    # 从不同聚类各选1个
    prune_heads_inter = [clusters[0][0], clusters[1][0], clusters[2][0]]

    model_B = copy.deepcopy(model)
    prune_specific_heads(model_B, layer_idx, prune_heads_inter)
    fid_B = evaluate_fid(model_B, vae)

    # 计算比率
    fid_ratio = fid_B / fid_A

    return {
        'intra_cluster_fid': fid_A,
        'inter_cluster_fid': fid_B,
        'fid_ratio': fid_ratio,
        'success': fid_ratio > 1.1  # B比A差至少10%
    }
```

### 5.3 成功标准

```
✅ FID_B / FID_A > 1.2 (跨聚类剪枝性能差20%)
✅ FID_A - baseline < 0.1 (聚类内剪枝影响小)
✅ 重复3次实验，结果一致
```

---

## 6. 方法5: 可视化验证

### 6.1 核心思路

定性观察：直接可视化聚类内heads的attention patterns，看是否视觉上相似。

```python
def visualize_cluster_attention_patterns(model, image, layer_idx, cluster):
    """
    可视化一个聚类内所有heads的attention patterns

    Args:
        cluster: 例如 [5, 6, 7, 8]
    """
    fig, axes = plt.subplots(1, len(cluster), figsize=(15, 3))

    for idx, head_idx in enumerate(cluster):
        # 获取attention map
        attn_map = get_attention_map(model, image, layer_idx, head_idx)

        # 可视化（选择一个token的attention分布）
        token_idx = 340  # 中间token
        attn_to_visualize = attn_map[token_idx, :].reshape(26, 26)  # 假设680 tokens -> 26×26 grid

        axes[idx].imshow(attn_to_visualize, cmap='viridis')
        axes[idx].set_title(f'Head {head_idx}\nScale={scale_mul[head_idx]:.2f}')
        axes[idx].axis('off')

    plt.suptitle(f'Cluster {cluster}: scale_mul差异 < {threshold}')
    plt.savefig(f'cluster_visualization_layer{layer_idx}.png')
```

### 6.2 预期观察

```
✅ 聚类内的attention maps视觉上相似
✅ 都关注相似的图像区域
✅ attention分布形状相似
```

**注意**：这是辅助验证，主观性强，不能作为主要证据。

---

## 7. 综合验证协议

### 7.1 验证流程

#### Phase 1: 快速验证（1-2天）

**目标**：初步评估假设有效性

```bash
# 步骤1: 单层验证（Layer 7，最高方差层）
python verify_scale_mul_clustering.py \
    --model_depth 16 \
    --layer_idx 7 \
    --num_samples 50 \
    --methods "attention_similarity,clustering_quality" \
    --output_dir results/quick_verify

# 步骤2: 查看结果
# - results/quick_verify/layer_7_similarity_scatter.png
# - results/quick_verify/layer_7_similarity_heatmap.png
# - results/quick_verify/statistics.json
```

**决策点**：
- 如果Pearson r < -0.3且p < 0.05 → 继续Phase 2
- 如果不满足 → 重新审视假设或调整阈值

#### Phase 2: 全面验证（3-5天）

**目标**：多层、多方法验证

```bash
# 步骤1: 所有层验证（0-15）
python verify_scale_mul_clustering.py \
    --model_depth 16 \
    --layer_idx all \
    --num_samples 100 \
    --methods "attention_similarity,cka_similarity,clustering_quality" \
    --output_dir results/full_verify

# 步骤2: 生成综合报告
python generate_verification_report.py \
    --input_dir results/full_verify \
    --output_file VERIFICATION_RESULTS.md
```

#### Phase 3: 消融验证（1周）

**目标**：功能性验证

```bash
# 步骤1: 消融实验
python ablation_clustering.py \
    --model_depth 16 \
    --layer_idx 7,11,12 \  # 高方差层
    --output_dir results/ablation

# 步骤2: 评估FID
# 对比聚类内vs跨聚类剪枝的性能
```

### 7.2 判断标准

#### 强验证通过 (95%置信度)

```
✅ 方法1: Pearson r < -0.4, p < 0.01
✅ 方法2: CKA r < -0.3, p < 0.05
✅ 方法3: Silhouette > 0.4
✅ 方法4: FID_ratio > 1.3
✅ 方法5: 可视化一致
```

#### 中等验证通过 (75%置信度)

```
✅ 方法1: Pearson r < -0.3, p < 0.05
✅ 方法3: Silhouette > 0.3
✅ 其他3个方法中至少1个通过
```

#### 弱验证通过 (60%置信度)

```
✅ 方法1或方法3通过
✅ 趋势正确但不显著
```

#### 验证失败

```
❌ 方法1和方法3都不通过
❌ 或相关性为正（与假设相反）
→ 需要重新设计策略1
```

---

## 8. 实施指南

### 8.1 环境准备

```bash
# 安装依赖
pip install scipy scikit-learn seaborn

# 准备数据
cd /home/project/real_prune/slimvar
# 确保有校准图像或使用类别标签
```

### 8.2 代码集成

```python
# 在model_slimming_basic.py中添加验证选项
parser.add_argument("--verify_clustering", action="store_true",
                   help="Verify scale_mul clustering hypothesis")
parser.add_argument("--verify_layers", type=str, default="7",
                   help="Layers to verify (comma-separated or 'all')")

if args.verify_clustering:
    from verify_scale_mul_clustering import verify_all_methods

    verify_results = verify_all_methods(
        model=var,
        calibration_labels=calibration_labels,
        layer_indices=parse_layer_indices(args.verify_layers),
        num_samples=args.num_samples,
        output_dir=args.save_dir
    )

    # 保存结果
    with open(os.path.join(args.save_dir, 'verification_results.json'), 'w') as f:
        json.dump(verify_results, f, indent=2)

    print("\n" + "="*60)
    print("Verification Results Summary:")
    print("="*60)
    for layer_idx, result in verify_results.items():
        print(f"\nLayer {layer_idx}:")
        print(f"  Pearson r: {result['pearson_r']:.3f} (p={result['p_value']:.4f})")
        print(f"  Silhouette: {result['silhouette_score']:.3f}")
        print(f"  Status: {'✅ PASS' if result['pass'] else '❌ FAIL'}")
```

### 8.3 快速开始

```bash
# 最小验证（推荐先运行）
python verify_scale_mul_clustering.py \
    --model_depth 16 \
    --layer_idx 7 \
    --num_samples 20 \
    --quick_mode

# 预期运行时间：5-10分钟
```

---

## 9. 预期结果

### 9.1 理论预测

基于scale_mul的物理意义和VAR的架构，我们预测：

#### 高方差层（Layer 7, 11, 12）
```
预测：假设更可能成立
理由：
- 高方差 = heads高度分化
- 分化过程中自然形成功能聚类
- scale_mul接近的heads = 相似尖锐度 = 相似功能

预期指标：
- Pearson r: -0.35 到 -0.45
- Silhouette: 0.35 到 0.50
```

#### 低方差层（Layer 0, 1, 2）
```
预测：假设可能较弱
理由：
- 低方差 = heads统一，差异小
- scale_mul本身差异就小，聚类不明显
- 可能没有明显的功能聚类

预期指标：
- Pearson r: -0.15 到 -0.25
- Silhouette: 0.15 到 0.30
```

### 9.2 失败情况分析

如果验证失败，可能的原因：

#### 原因1: 阈值设置不当
```
症状：聚类过多或过少
解决：测试不同阈值 (0.2σ, 0.3σ, 0.5σ)
```

#### 原因2: Q, K矩阵主导
```
症状：scale_mul相似但Q@K.T差异大
推论：需要同时考虑scale_mul和Q@K.T的分布
解决：修改策略1，加入Q@K.T的相似性判断
```

#### 原因3: 假设根本不成立
```
症状：所有方法都失败，相关性为正或接近0
推论：scale_mul与attention pattern无强关联
解决：放弃策略1，采用策略2或策略3
```

---

## 10. 总结

### 10.1 核心要点

1. **假设必须验证**：策略1的有效性完全依赖于这个假设
2. **多方法验证**：5种方法互补，增加置信度
3. **快速迭代**：先快速验证，再深入分析
4. **准备Plan B**：如果验证失败，策略2和策略3仍然可用

### 10.2 时间预算

| 阶段 | 内容 | 时间 |
|------|------|------|
| 实现 | 编写验证代码 | 2-3小时 |
| Phase 1 | 快速验证（1层，50样本） | 10分钟 |
| Phase 2 | 全面验证（16层，100样本） | 1小时 |
| Phase 3 | 消融实验 | 1-2天 |
| 分析 | 结果分析和报告 | 2-3小时 |

**总计**：约3-4小时代码 + 2-3小时运行 + 1-2天消融（可选）

### 10.3 风险提示

⚠️ **关键风险**：如果假设不成立，策略1需要重新设计

但这不是坏事！因为：
1. 我们及早发现了问题（比大规模实验后发现好）
2. 策略2（方差条件）和策略3（组合）仍然可用
3. 验证过程本身会提供新的洞察

---

**文档版本**: v1.0
**最后更新**: 2025-11-11
**状态**: 待验证 → 实施中
**下一步**: 实现`verify_scale_mul_clustering.py`
