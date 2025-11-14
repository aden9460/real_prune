#!/usr/bin/env python3
"""
Scale_mul聚类假设验证脚本
验证核心假设: scale_mul接近的heads注意力模式相似

实现方法:
- 方法1: 直接注意力相似度分析
- 方法3: 聚类质量评估
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# 添加VAR路径
sys.path.append("VAR/")
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


# =====================================================================
# 模型加载 (复用analyze_scale_mul.py的函数)
# =====================================================================

def load_var_model(model_depth, vae_ckpt_path, var_ckpt_path, device='cuda'):
    """加载VAR模型"""
    from VAR.models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=False
    )

    vae.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt_path, map_location='cpu'), strict=False)

    vae.eval()
    var.eval()

    print(f'✓ VAR-d{model_depth} model loaded successfully.')
    return vae, var


def extract_scale_mul(model):
    """
    提取所有层的scale_mul参数

    Returns:
        scale_mul_matrix: (num_layers, num_heads) 实际scale值
    """
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads

    scale_mul_matrix = np.zeros((num_layers, num_heads))

    for i, block in enumerate(model.blocks):
        # scale_mul_1H11: (1, num_heads, 1, 1)
        raw_values = block.attn.scale_mul_1H11.data.cpu().squeeze().numpy()  # (num_heads,)
        scale_values = np.exp(raw_values)  # 转换为实际scale
        scale_mul_matrix[i] = scale_values

    return scale_mul_matrix


# =====================================================================
# Attention Map提取器
# =====================================================================

class AttentionMapExtractor:
    """
    Hook VAR模型的attention层，提取attention weights

    处理3种attention实现:
    - slow_attn: 可以直接hook
    - flash_attn: 无法直接获取，需要关闭flash并使用slow路径
    - xformers: 同上
    """

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.attention_maps = []
        self.hook_handle = None

        # 获取目标层
        self.target_block = model.blocks[layer_idx]
        self.target_attn = self.target_block.attn

        # 保存原始设置
        self.original_using_flash = self.target_attn.using_flash
        self.original_using_xform = self.target_attn.using_xform

        # 强制使用slow attention以便hook
        self.target_attn.using_flash = False
        self.target_attn.using_xform = False

    def _compute_slow_attn_with_capture(self, query, key, value, scale, attn_mask=None):
        """
        复制slow_attn的逻辑，但捕获attention weights

        Args:
            query, key, value: [B, num_heads, L, head_dim]
            scale: float
            attn_mask: [1, 1, L, L] or None

        Returns:
            output: [B, num_heads, L, head_dim]
        """
        # 计算attention scores: [B, H, L, L]
        attn_scores = query.mul(scale) @ key.transpose(-2, -1)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Softmax得到attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 收集最后一个尺度的attention (256个tokens，对应16x16最高分辨率)
        # 在autoregressive生成中，每个尺度的tokens数 = patch_size^2
        # 最后一个尺度是256 (16x16)
        L = attn_weights.shape[2]

        # Debug: 记录遇到的序列长度
        if not hasattr(self, '_seen_lengths'):
            self._seen_lengths = set()
        self._seen_lengths.add(L)

        if L == 256:  # 最后一个尺度 (16x16)
            # 保存attention weights [B, H, 256, 256]
            self.attention_maps.append(attn_weights.detach().cpu())

        # 计算输出
        output = attn_weights @ value

        return output

    def _attention_forward_hook(self, module, input_args, output):
        """
        Hook attention模块的forward

        注意: 由于我们已经强制使用slow_attn，需要在forward中捕获
        """
        # 这个hook实际上不会被使用，因为我们直接monkey patch了forward
        pass

    def extract(self, inputs):
        """
        提取attention maps

        Args:
            inputs: class labels [B] 或 tokens

        Returns:
            attention_maps: [B, num_heads, L, L]
        """
        self.attention_maps = []

        # Monkey patch slow_attn函数
        original_forward = self.target_attn.forward

        def patched_forward(x, attn_bias):
            """
            Patched forward函数，复制原始逻辑但捕获attention
            """
            B, L, C = x.shape

            # QKV projection (复制原始代码)
            qkv = F.linear(
                input=x,
                weight=self.target_attn.mat_qkv.weight,
                bias=torch.cat((
                    self.target_attn.q_bias,
                    self.target_attn.zero_k_bias,
                    self.target_attn.v_bias
                ))
            ).view(B, L, 3, self.target_attn.num_heads, 64)

            # 分离q, k, v: BHLc格式 (for slow_attn)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # q/k/v: [B, H, L, 64]

            # L2 normalize if needed
            if self.target_attn.attn_l2_norm:
                scale_mul = self.target_attn.scale_mul_1H11.clamp_max(
                    self.target_attn.max_scale_mul
                ).exp()
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)

            # 使用我们的捕获版本slow_attn
            oup = self._compute_slow_attn_with_capture(
                query=q, key=k, value=v,
                scale=self.target_attn.scale,
                attn_mask=attn_bias
            ).transpose(1, 2).reshape(B, L, C)

            # Projection
            return self.target_attn.proj_drop(self.target_attn.proj(oup))

        # 临时替换forward
        self.target_attn.forward = patched_forward

        try:
            # 执行forward pass
            with torch.no_grad():
                if isinstance(inputs, torch.Tensor) and inputs.ndim == 1:
                    # Class labels
                    # Use autoregressive_infer_cfg which supports conditional input
                    B = inputs.shape[0]

                    # Process each sample (can't batch due to autoregressive nature)
                    for b in range(B):
                        label = inputs[b:b+1]  # [1]
                        # Run autoregressive inference
                        # This will trigger full forward pass through all layers including our target
                        _ = self.model.autoregressive_infer_cfg(
                            B=1,
                            label_B=label,
                            cfg=1.5,
                            top_p=0.96,
                            g_seed=None,
                        )
                else:
                    raise NotImplementedError("Only class labels input supported")
        finally:
            # 恢复原始forward
            self.target_attn.forward = original_forward

        # 合并所有batch的attention maps
        if len(self.attention_maps) == 0:
            raise ValueError(
                f"No attention maps were captured! "
                f"Expected sequence length 680, but only saw lengths: {getattr(self, '_seen_lengths', 'unknown')}. "
                f"This likely means the patched forward was not called or the model uses a different attention path."
            )

        attention_maps = torch.cat(self.attention_maps, dim=0)  # [total_B, H, L, L]

        return attention_maps

    def restore(self):
        """恢复原始设置"""
        self.target_attn.using_flash = self.original_using_flash
        self.target_attn.using_xform = self.original_using_xform


# =====================================================================
# 方法1: 直接注意力相似度分析
# =====================================================================

def compute_pairwise_attention_similarity(attention_maps, metric='cosine'):
    """
    计算所有head pairs的平均注意力相似度

    Args:
        attention_maps: [B, num_heads, L, L]
        metric: 'cosine' or 'pearson'

    Returns:
        similarity_matrix: [num_heads, num_heads]
    """
    B, num_heads, L, _ = attention_maps.shape
    similarity_matrix = np.zeros((num_heads, num_heads))

    print(f"  Computing pairwise similarities for {num_heads} heads...")

    for i in range(num_heads):
        for j in range(i, num_heads):
            # 在B个样本上计算平均相似度
            sims = []
            for b in range(B):
                attn_i = attention_maps[b, i].flatten()  # [L*L]
                attn_j = attention_maps[b, j].flatten()

                if metric == 'cosine':
                    sim = F.cosine_similarity(attn_i, attn_j, dim=0).item()
                elif metric == 'pearson':
                    # Pearson correlation
                    sim, _ = pearsonr(attn_i.numpy(), attn_j.numpy())
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                sims.append(sim)

            # 平均相似度
            avg_sim = np.mean(sims)
            similarity_matrix[i, j] = avg_sim
            similarity_matrix[j, i] = avg_sim

    return similarity_matrix


def analyze_scale_vs_similarity(scale_mul, similarity_matrix):
    """
    分析scale_mul差异与注意力相似度的关系

    Returns:
        correlation: Pearson相关系数
        p_value: 显著性
        scatter_data: (scale_diffs, similarities)
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
    correlation, p_value = pearsonr(scale_diffs, similarities)

    return correlation, p_value, (scale_diffs, similarities)


def plot_similarity_vs_scale_diff(scale_diffs, similarities, layer_idx,
                                   correlation, p_value, save_path):
    """
    绘制散点图: X=scale_mul差异, Y=注意力相似度
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(scale_diffs, similarities, alpha=0.5, s=30, c='blue', edgecolors='black', linewidth=0.5)

    # 拟合线
    z = np.polyfit(scale_diffs, similarities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(scale_diffs), max(scale_diffs), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear Fit')

    plt.xlabel('Scale_mul Difference |s_i - s_j|', fontsize=13)
    plt.ylabel('Attention Similarity', fontsize=13)
    plt.title(f'Layer {layer_idx}: Scale_mul Diff vs Attention Similarity', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    textstr = f'Pearson r = {correlation:.3f}\np-value = {p_value:.4f}\n'
    if p_value < 0.001:
        textstr += '*** p < 0.001'
    elif p_value < 0.01:
        textstr += '** p < 0.01'
    elif p_value < 0.05:
        textstr += '* p < 0.05'
    else:
        textstr += 'Not significant'

    # 判断结果
    if correlation < -0.3 and p_value < 0.05:
        result = '✓ PASS'
        color = 'green'
    elif correlation < -0.2 and p_value < 0.05:
        result = '~ WEAK PASS'
        color = 'orange'
    else:
        result = '✗ FAIL'
        color = 'red'

    textstr += f'\n\nResult: {result}'

    plt.text(0.05, 0.95, textstr,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
             fontsize=11)

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved scatter plot: {save_path}")


# =====================================================================
# 方法3: 聚类质量分析
# =====================================================================

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
    sorted_indices = np.argsort(scale_mul)
    sorted_scales = scale_mul[sorted_indices]

    # 计算相邻差异
    diffs = sorted_scales[1:] - sorted_scales[:-1]

    # 识别聚类边界
    clusters = []
    current_cluster = [int(sorted_indices[0])]

    for i in range(len(diffs)):
        if diffs[i] < threshold:
            current_cluster.append(int(sorted_indices[i+1]))
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [int(sorted_indices[i+1])]

    # 添加最后一个聚类
    if len(current_cluster) > 1:
        clusters.append(current_cluster)

    return clusters


def compute_intra_cluster_similarity(clusters, similarity_matrix):
    """
    计算聚类内部的平均相似度
    """
    intra_sims = []

    for cluster in clusters:
        if len(cluster) > 1:
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    head_i = cluster[i]
                    head_j = cluster[j]
                    sim = similarity_matrix[head_i, head_j]
                    intra_sims.append(sim)

    return np.mean(intra_sims) if intra_sims else 0.0


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


def compute_silhouette_score_custom(clusters, similarity_matrix):
    """
    计算聚类的Silhouette score

    Returns:
        score: float in [-1, 1], >0.3表示聚类有效
    """
    num_heads = similarity_matrix.shape[0]

    # 如果聚类太少，无法计算
    if len(clusters) < 2:
        return 0.0

    # 创建cluster labels
    labels = np.full(num_heads, -1, dtype=int)

    for cluster_id, cluster in enumerate(clusters):
        for head_idx in cluster:
            labels[head_idx] = cluster_id

    # 孤立的heads标记为独立cluster
    singleton_id = len(clusters)
    for i in range(num_heads):
        if labels[i] == -1:
            labels[i] = singleton_id
            singleton_id += 1

    # 转换similarity为distance
    distance_matrix = 1 - similarity_matrix

    # 计算silhouette score
    try:
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
    except:
        score = 0.0

    return score


def plot_similarity_heatmap(similarity_matrix, scale_mul, clusters, layer_idx, save_path):
    """
    绘制16x16相似度热力图，标注聚类边界

    期望: 如果聚类有效，应该看到块对角结构
    """
    # 按scale_mul排序 (聚类会自然分组)
    sorted_indices = np.argsort(scale_mul)
    sorted_sim_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
    sorted_scale_mul = scale_mul[sorted_indices]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(11, 10))

    im = ax.imshow(sorted_sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Similarity', fontsize=12)

    # 标注聚类边界
    # 需要找到sorted_indices中聚类的位置
    cluster_boundaries = []
    for cluster in clusters:
        # 找到cluster中的head在sorted_indices中的位置
        positions = [np.where(sorted_indices == head)[0][0] for head in cluster]
        positions.sort()
        if len(positions) > 1:
            # 标记边界
            start = positions[0]
            end = positions[-1] + 1
            cluster_boundaries.append((start, end))

    # 绘制边界线
    for start, end in cluster_boundaries:
        ax.plot([start-0.5, end-0.5], [start-0.5, start-0.5], 'b-', linewidth=2.5)
        ax.plot([start-0.5, start-0.5], [start-0.5, end-0.5], 'b-', linewidth=2.5)
        ax.plot([end-0.5, end-0.5], [start-0.5, end-0.5], 'b-', linewidth=2.5)
        ax.plot([start-0.5, end-0.5], [end-0.5, end-0.5], 'b-', linewidth=2.5)

    ax.set_xlabel('Head Index (sorted by scale_mul)', fontsize=13)
    ax.set_ylabel('Head Index (sorted by scale_mul)', fontsize=13)
    ax.set_title(f'Layer {layer_idx}: Attention Similarity Matrix with Clusters',
                 fontsize=15, fontweight='bold')

    # 设置ticks
    ax.set_xticks(range(len(scale_mul)))
    ax.set_yticks(range(len(scale_mul)))
    ax.set_xticklabels(sorted_indices, fontsize=9)
    ax.set_yticklabels(sorted_indices, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved heatmap: {save_path}")


# =====================================================================
# 主验证流程
# =====================================================================

def verify_clustering_hypothesis(
    model, layer_idx, num_samples=50,
    threshold_factor=0.3, output_dir='./verification_results', device='cuda'
):
    """
    验证scale_mul聚类假设

    Args:
        model: VAR模型
        layer_idx: 要验证的层
        num_samples: 校准样本数
        threshold_factor: 聚类阈值系数 (threshold = std * factor)
        output_dir: 输出目录

    Returns:
        results: dict 验证结果
    """
    print(f"\n{'='*70}")
    print(f" Verifying Layer {layer_idx} ".center(70, '='))
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 提取scale_mul ====================
    print("Step 1: Extracting scale_mul values...")
    scale_mul_matrix = extract_scale_mul(model)
    scale_mul = scale_mul_matrix[layer_idx]

    print(f"  Layer {layer_idx} scale_mul:")
    print(f"    Mean: {scale_mul.mean():.2f}")
    print(f"    Std:  {scale_mul.std():.2f}")
    print(f"    Range: [{scale_mul.min():.2f}, {scale_mul.max():.2f}]")

    # 保存scale_mul
    np.save(f'{output_dir}/scale_mul_layer{layer_idx}.npy', scale_mul)

    # ==================== 2. 提取Attention Maps ====================
    print(f"\nStep 2: Extracting attention maps from {num_samples} samples...")

    extractor = AttentionMapExtractor(model, layer_idx)

    # 生成class labels (batch处理)
    batch_size = 10
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_attention_maps = []

    for i in tqdm(range(num_batches), desc="  Extracting"):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        labels = torch.randint(0, 1000, (current_batch_size,)).to(device)

        attn_maps = extractor.extract(labels)  # [B, H, L, L]
        all_attention_maps.append(attn_maps)

    # 合并
    attention_maps = torch.cat(all_attention_maps, dim=0)[:num_samples]  # [num_samples, H, L, L]

    print(f"  ✓ Extracted attention maps: {attention_maps.shape}")

    # 恢复设置
    extractor.restore()

    # ==================== 3. 方法1: 直接注意力相似度 ====================
    print("\nStep 3: Method 1 - Computing pairwise attention similarity...")

    similarity_matrix = compute_pairwise_attention_similarity(attention_maps, metric='cosine')

    # 保存相似度矩阵
    np.save(f'{output_dir}/attention_similarity_matrix.npy', similarity_matrix)

    # 分析相关性
    correlation, p_value, (scale_diffs, similarities) = analyze_scale_vs_similarity(
        scale_mul, similarity_matrix
    )

    print(f"  Pearson correlation: r = {correlation:.4f}, p = {p_value:.4f}")

    # 判断方法1
    method1_pass = correlation < -0.3 and p_value < 0.05
    method1_weak = correlation < -0.2 and p_value < 0.05

    if method1_pass:
        print(f"  ✓ Method 1: PASS (strong negative correlation)")
    elif method1_weak:
        print(f"  ~ Method 1: WEAK PASS (moderate negative correlation)")
    else:
        print(f"  ✗ Method 1: FAIL (no significant negative correlation)")

    # 可视化
    plot_similarity_vs_scale_diff(
        scale_diffs, similarities, layer_idx, correlation, p_value,
        f'{output_dir}/method1_scatter_layer{layer_idx}.png'
    )

    # ==================== 4. 方法3: 聚类质量分析 ====================
    print("\nStep 4: Method 3 - Clustering quality analysis...")

    # 识别聚类
    threshold = scale_mul.std() * threshold_factor
    clusters = identify_clusters(scale_mul, threshold)

    print(f"  Threshold: {threshold:.2f} (std={scale_mul.std():.2f} × {threshold_factor})")
    print(f"  Identified {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        cluster_scales = [scale_mul[h] for h in cluster]
        print(f"    Cluster {i}: {cluster} (scales: {np.mean(cluster_scales):.1f} ± {np.std(cluster_scales):.1f})")

    # 保存聚类
    with open(f'{output_dir}/clusters.json', 'w') as f:
        json.dump({
            'threshold': float(threshold),
            'threshold_factor': threshold_factor,
            'num_clusters': len(clusters),
            'clusters': clusters,
        }, f, indent=2)

    # 计算聚类内/间相似度
    intra_sim = compute_intra_cluster_similarity(clusters, similarity_matrix)
    inter_sim = compute_inter_cluster_similarity(clusters, similarity_matrix)

    print(f"\n  Intra-cluster similarity: {intra_sim:.4f}")
    print(f"  Inter-cluster similarity: {inter_sim:.4f}")
    print(f"  Ratio (intra/inter): {intra_sim/inter_sim:.2f}")

    # Silhouette score
    silhouette = compute_silhouette_score_custom(clusters, similarity_matrix)
    print(f"  Silhouette score: {silhouette:.4f}")

    # 判断方法3
    method3_pass = silhouette > 0.3 and intra_sim > inter_sim * 1.5
    method3_weak = silhouette > 0.2 and intra_sim > inter_sim * 1.2

    if method3_pass:
        print(f"  ✓ Method 3: PASS (good clustering)")
    elif method3_weak:
        print(f"  ~ Method 3: WEAK PASS (moderate clustering)")
    else:
        print(f"  ✗ Method 3: FAIL (poor clustering)")

    # 可视化
    plot_similarity_heatmap(
        similarity_matrix, scale_mul, clusters, layer_idx,
        f'{output_dir}/method3_heatmap_layer{layer_idx}.png'
    )

    # ==================== 5. 综合判断 ====================
    print(f"\n{'='*70}")
    print(" Verification Summary ".center(70, '='))
    print(f"{'='*70}\n")

    # 综合判断
    if method1_pass and method3_pass:
        overall = 'PASS'
        confidence = 'HIGH'
        recommendation = 'Clustering pruning is highly recommended for this layer.'
    elif (method1_pass or method3_pass) and (method1_weak or method3_weak):
        overall = 'PASS'
        confidence = 'MEDIUM'
        recommendation = 'Clustering pruning is recommended but with caution.'
    elif method1_weak and method3_weak:
        overall = 'WEAK_PASS'
        confidence = 'LOW'
        recommendation = 'Clustering pruning may work but needs careful validation.'
    else:
        overall = 'FAIL'
        confidence = 'N/A'
        recommendation = 'Clustering pruning is NOT recommended. Use alternative strategies.'

    print(f"Overall Result: {overall}")
    print(f"Confidence: {confidence}")
    print(f"Recommendation: {recommendation}\n")

    # ==================== 6. 保存结果 ====================
    results = {
        'layer_idx': layer_idx,
        'num_samples': num_samples,
        'scale_mul': {
            'mean': float(scale_mul.mean()),
            'std': float(scale_mul.std()),
            'min': float(scale_mul.min()),
            'max': float(scale_mul.max()),
        },
        'method1': {
            'pearson_r': float(correlation),
            'p_value': float(p_value),
            'pass': bool(method1_pass),  # Convert numpy bool to Python bool
            'status': 'PASS' if method1_pass else ('WEAK' if method1_weak else 'FAIL'),
        },
        'method3': {
            'num_clusters': len(clusters),
            'intra_similarity': float(intra_sim),
            'inter_similarity': float(inter_sim),
            'ratio': float(intra_sim / inter_sim) if inter_sim > 0 else 0,
            'silhouette_score': float(silhouette),
            'pass': bool(method3_pass),  # Convert numpy bool to Python bool
            'status': 'PASS' if method3_pass else ('WEAK' if method3_weak else 'FAIL'),
        },
        'overall': {
            'result': overall,
            'confidence': confidence,
            'recommendation': recommendation,
        }
    }

    # 保存JSON
    with open(f'{output_dir}/verification_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_dir}/\n")

    return results


# =====================================================================
# 主函数
# =====================================================================

def main(args):
    print("\n" + "="*80)
    print(" Scale_mul Clustering Hypothesis Verification ".center(80, "="))
    print("="*80 + "\n")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 加载模型
    print("Loading VAR model...")
    vae_ckpt = args.vae_ckpt if args.vae_ckpt else '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt if args.var_ckpt else f'/home/project/daily/AR/model_zoo/var_d{args.model_depth}.pth'

    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device=device)

    # 验证
    results = verify_clustering_hypothesis(
        model=var,
        layer_idx=args.layer_idx,
        num_samples=args.num_samples,
        threshold_factor=args.threshold_factor,
        output_dir=args.output_dir,
        device=device
    )

    print("="*80)
    print(" Verification Complete! ".center(80, "="))
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify scale_mul clustering hypothesis for VAR model"
    )

    parser.add_argument(
        "--model_depth", type=int, default=16,
        help="VAR model depth"
    )
    parser.add_argument(
        "--layer_idx", type=int, default=7,
        help="Layer index to verify (recommend high-variance layer like 7)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--threshold_factor", type=float, default=0.3,
        help="Clustering threshold factor (threshold = std * factor)"
    )
    parser.add_argument(
        "--vae_ckpt", type=str, default="",
        help="Path to VQVAE checkpoint"
    )
    parser.add_argument(
        "--var_ckpt", type=str, default="",
        help="Path to VAR checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./verification_results_d16",
        help="Directory to save verification results"
    )

    args = parser.parse_args()
    main(args)
