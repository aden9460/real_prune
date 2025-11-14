#!/usr/bin/env python3
"""
Scale_Mul分析脚本
分析VAR模型中scale_mul_1H11参数的分布特征，为剪枝策略提供数据支持
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

# 添加VAR路径
sys.path.append("VAR/")
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


def load_var_model(model_depth, vae_ckpt_path, var_ckpt_path, device='cuda'):
    """加载VAR模型"""
    from models import build_vae_var

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
        scale_mul_raw: (num_layers, num_heads) log空间的原始值
    """
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads

    scale_mul_matrix = np.zeros((num_layers, num_heads))
    scale_mul_raw = np.zeros((num_layers, num_heads))

    for i, block in enumerate(model.blocks):
        # scale_mul_1H11: (1, num_heads, 1, 1)
        raw_values = block.attn.scale_mul_1H11.data.cpu().squeeze().numpy()  # (num_heads,)
        scale_values = np.exp(raw_values)  # 转换为实际scale

        scale_mul_raw[i] = raw_values
        scale_mul_matrix[i] = scale_values

    return scale_mul_matrix, scale_mul_raw


def analyze_scale_mul_statistics(scale_mul_matrix):
    """
    统计分析scale_mul分布

    Returns:
        dict: 包含各种统计信息
    """
    num_layers, num_heads = scale_mul_matrix.shape

    stats = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'per_layer': [],
        'global': {},
        'classification': {},
    }

    # 每层统计
    for i in range(num_layers):
        layer_scales = scale_mul_matrix[i]

        layer_stat = {
            'layer_idx': int(i),
            'mean': float(layer_scales.mean()),
            'std': float(layer_scales.std()),
            'min': float(layer_scales.min()),
            'max': float(layer_scales.max()),
            'median': float(np.median(layer_scales)),
            'q25': float(np.percentile(layer_scales, 25)),
            'q75': float(np.percentile(layer_scales, 75)),
            'variance': float(layer_scales.var()),
            # 分类统计
            'high_scale_ratio': float((layer_scales > 50).sum() / num_heads),
            'mid_scale_ratio': float(((layer_scales >= 5) & (layer_scales <= 50)).sum() / num_heads),
            'low_scale_ratio': float((layer_scales < 5).sum() / num_heads),
            'high_scale_heads': int((layer_scales > 50).sum()),
            'low_scale_heads': int((layer_scales < 5).sum()),
        }
        stats['per_layer'].append(layer_stat)

    # 全局统计
    all_scales = scale_mul_matrix.flatten()
    stats['global'] = {
        'mean': float(all_scales.mean()),
        'std': float(all_scales.std()),
        'min': float(all_scales.min()),
        'max': float(all_scales.max()),
        'median': float(np.median(all_scales)),
        'total_heads': int(all_scales.size),
    }

    # 全局分类
    stats['classification'] = {
        'high_scale_heads': int((all_scales > 50).sum()),
        'mid_scale_heads': int(((all_scales >= 5) & (all_scales <= 50)).sum()),
        'low_scale_heads': int((all_scales < 5).sum()),
        'high_scale_ratio': float((all_scales > 50).sum() / all_scales.size),
        'mid_scale_ratio': float(((all_scales >= 5) & (all_scales <= 50)).sum() / all_scales.size),
        'low_scale_ratio': float((all_scales < 5).sum() / all_scales.size),
    }

    return stats


def visualize_scale_mul(scale_mul_matrix, stats, output_dir):
    """可视化scale_mul分布"""
    os.makedirs(output_dir, exist_ok=True)

    num_layers, num_heads = scale_mul_matrix.shape

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ==================== 1. 热力图：所有层所有head的scale_mul ====================
    plt.figure(figsize=(16, 10))

    # 使用对数scale显示（因为scale_mul范围很大）
    scale_mul_log = np.log10(scale_mul_matrix + 1e-8)

    sns.heatmap(
        scale_mul_log.T,  # 转置：横轴=层，纵轴=head
        cmap='viridis',
        cbar_kws={'label': 'log10(scale_mul)'},
        xticklabels=range(num_layers),
        yticklabels=range(num_heads),
        linewidths=0.5,
    )
    plt.title(f'Scale_Mul Heatmap (log10 scale)\nModel: VAR-d{num_layers}', fontsize=16)
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('Head Index', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scale_mul_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/scale_mul_heatmap.png")

    # ==================== 2. 箱线图：每层的scale_mul分布 ====================
    plt.figure(figsize=(20, 6))

    # 使用log scale
    scale_mul_log_list = [np.log10(scale_mul_matrix[i] + 1e-8) for i in range(num_layers)]

    plt.boxplot(scale_mul_log_list, labels=range(num_layers))
    plt.title(f'Scale_Mul Distribution per Layer (log10 scale)', fontsize=16)
    plt.xlabel('Layer Index', fontsize=14)
    plt.ylabel('log10(scale_mul)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scale_mul_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/scale_mul_boxplot.png")

    # ==================== 3. 柱状图：高/中/低scale head比例 ====================
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 3a. 堆叠柱状图
    layers = list(range(num_layers))
    high_ratios = [s['high_scale_ratio'] for s in stats['per_layer']]
    mid_ratios = [s['mid_scale_ratio'] for s in stats['per_layer']]
    low_ratios = [s['low_scale_ratio'] for s in stats['per_layer']]

    axes[0].bar(layers, high_ratios, label='High (>50)', color='red', alpha=0.7)
    axes[0].bar(layers, mid_ratios, bottom=high_ratios, label='Mid (5-50)', color='orange', alpha=0.7)
    axes[0].bar(layers, low_ratios, bottom=np.array(high_ratios) + np.array(mid_ratios),
                label='Low (<5)', color='blue', alpha=0.7)
    axes[0].set_xlabel('Layer Index', fontsize=12)
    axes[0].set_ylabel('Head Ratio', fontsize=12)
    axes[0].set_title('Head Classification by Scale_Mul (Stacked)', fontsize=14)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 3b. 分组柱状图
    x = np.arange(num_layers)
    width = 0.25

    axes[1].bar(x - width, high_ratios, width, label='High (>50)', color='red', alpha=0.7)
    axes[1].bar(x, mid_ratios, width, label='Mid (5-50)', color='orange', alpha=0.7)
    axes[1].bar(x + width, low_ratios, width, label='Low (<5)', color='blue', alpha=0.7)
    axes[1].set_xlabel('Layer Index', fontsize=12)
    axes[1].set_ylabel('Head Ratio', fontsize=12)
    axes[1].set_title('Head Classification by Scale_Mul (Grouped)', fontsize=14)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scale_mul_classification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/scale_mul_classification.png")

    # ==================== 4. 折线图：层级统计趋势 ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    layers = list(range(num_layers))
    means = [s['mean'] for s in stats['per_layer']]
    stds = [s['std'] for s in stats['per_layer']]
    variances = [s['variance'] for s in stats['per_layer']]

    # 4a. 均值趋势
    axes[0, 0].plot(layers, means, marker='o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Layer Index', fontsize=12)
    axes[0, 0].set_ylabel('Mean Scale_Mul', fontsize=12)
    axes[0, 0].set_title('Mean Scale_Mul per Layer', fontsize=14)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axhline(y=stats['global']['mean'], color='r', linestyle='--', label='Global Mean')
    axes[0, 0].legend()

    # 4b. 标准差趋势
    axes[0, 1].plot(layers, stds, marker='s', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Layer Index', fontsize=12)
    axes[0, 1].set_ylabel('Std Dev', fontsize=12)
    axes[0, 1].set_title('Standard Deviation per Layer', fontsize=14)
    axes[0, 1].grid(alpha=0.3)

    # 4c. 方差趋势
    axes[1, 0].plot(layers, variances, marker='^', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Layer Index', fontsize=12)
    axes[1, 0].set_ylabel('Variance', fontsize=12)
    axes[1, 0].set_title('Variance per Layer (Head Differentiation)', fontsize=14)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axhline(y=20, color='r', linestyle='--', label='High Variance Threshold')
    axes[1, 0].legend()

    # 4d. Min-Max范围
    mins = [s['min'] for s in stats['per_layer']]
    maxs = [s['max'] for s in stats['per_layer']]
    axes[1, 1].fill_between(layers, mins, maxs, alpha=0.3, color='purple')
    axes[1, 1].plot(layers, means, marker='o', linewidth=2, markersize=6, color='purple', label='Mean')
    axes[1, 1].set_xlabel('Layer Index', fontsize=12)
    axes[1, 1].set_ylabel('Scale_Mul', fontsize=12)
    axes[1, 1].set_title('Min-Max Range per Layer', fontsize=14)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scale_mul_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/scale_mul_trends.png")

    # ==================== 5. 直方图：全局scale_mul分布 ====================
    plt.figure(figsize=(12, 6))

    all_scales = scale_mul_matrix.flatten()
    plt.hist(np.log10(all_scales + 1e-8), bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.log10(5), color='b', linestyle='--', label='Low/Mid boundary (5)', linewidth=2)
    plt.axvline(x=np.log10(50), color='r', linestyle='--', label='Mid/High boundary (50)', linewidth=2)
    plt.axvline(x=np.log10(stats['global']['mean']), color='g', linestyle='--',
                label=f"Global Mean ({stats['global']['mean']:.1f})", linewidth=2)

    plt.xlabel('log10(scale_mul)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Global Scale_Mul Distribution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scale_mul_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/scale_mul_histogram.png")


def recommend_pruning_strategy(stats, num_layers):
    """
    基于scale_mul分布推荐剪枝策略

    Returns:
        dict: 包含每层推荐剪枝率和理由
    """
    strategy = {
        'per_layer_sparsity': [],
        'scale_groups': [],
        'rationale': {},
    }

    base_sparsity = 0.25  # 基准剪枝率

    for i, layer_stat in enumerate(stats['per_layer']):
        # 决策因子
        variance = layer_stat['variance']
        high_ratio = layer_stat['high_scale_ratio']
        low_ratio = layer_stat['low_scale_ratio']
        mean = layer_stat['mean']

        # 决策逻辑
        if high_ratio > 0.4:  # 超过40%是专家型heads
            sparsity = base_sparsity * 0.6  # 保守剪枝
            reason = f"High expert heads ratio ({high_ratio:.2%}) - conservative pruning"
        elif low_ratio > 0.3:  # 超过30%是通才型heads
            sparsity = base_sparsity * 1.4  # 激进剪枝
            reason = f"High generalist heads ratio ({low_ratio:.2%}) - aggressive pruning"
        elif variance > 20:  # 高方差，head分化明显
            sparsity = base_sparsity * 1.2
            reason = f"High variance ({variance:.1f}) - can prune low-scale heads"
        elif variance < 10:  # 低方差，功能相似
            sparsity = base_sparsity * 0.8
            reason = f"Low variance ({variance:.1f}) - difficult to distinguish importance"
        else:
            sparsity = base_sparsity
            reason = "Balanced distribution - default sparsity"

        # 限制剪枝率范围
        sparsity = max(0.1, min(0.5, sparsity))

        strategy['per_layer_sparsity'].append({
            'layer': i,
            'sparsity': float(sparsity),
            'reason': reason,
        })

    # 分3个尺度组的策略
    # 早期层 (0-5): 处理早期尺度 (0-3)
    # 中期层 (6-11): 处理中期尺度 (3-7)
    # 后期层 (12-15): 处理后期尺度 (7-10)

    early_layers = stats['per_layer'][:num_layers//3]
    mid_layers = stats['per_layer'][num_layers//3:2*num_layers//3]
    late_layers = stats['per_layer'][2*num_layers//3:]

    strategy['scale_groups'] = [
        {
            'group': 'early',
            'layers': list(range(0, num_layers//3)),
            'scales': '0-3 (1²-4²)',
            'avg_variance': float(np.mean([s['variance'] for s in early_layers])),
            'avg_high_ratio': float(np.mean([s['high_scale_ratio'] for s in early_layers])),
            'recommended_sparsity': float(np.mean([s['sparsity'] for s in strategy['per_layer_sparsity'][:num_layers//3]])),
        },
        {
            'group': 'mid',
            'layers': list(range(num_layers//3, 2*num_layers//3)),
            'scales': '3-7 (5²-8²)',
            'avg_variance': float(np.mean([s['variance'] for s in mid_layers])),
            'avg_high_ratio': float(np.mean([s['high_scale_ratio'] for s in mid_layers])),
            'recommended_sparsity': float(np.mean([s['sparsity'] for s in strategy['per_layer_sparsity'][num_layers//3:2*num_layers//3]])),
        },
        {
            'group': 'late',
            'layers': list(range(2*num_layers//3, num_layers)),
            'scales': '7-10 (10²-16²)',
            'avg_variance': float(np.mean([s['variance'] for s in late_layers])),
            'avg_high_ratio': float(np.mean([s['high_scale_ratio'] for s in late_layers])),
            'recommended_sparsity': float(np.mean([s['sparsity'] for s in strategy['per_layer_sparsity'][2*num_layers//3:]])),
        },
    ]

    return strategy


def print_analysis_report(stats, strategy):
    """打印分析报告"""
    print("\n" + "="*80)
    print(" Scale_Mul Analysis Report ".center(80, "="))
    print("="*80)

    # 全局统计
    print("\n📊 Global Statistics:")
    print(f"  Total Heads: {stats['global']['total_heads']}")
    print(f"  Mean Scale_Mul: {stats['global']['mean']:.2f}")
    print(f"  Std Dev: {stats['global']['std']:.2f}")
    print(f"  Range: [{stats['global']['min']:.2f}, {stats['global']['max']:.2f}]")
    print(f"  Median: {stats['global']['median']:.2f}")

    # 分类统计
    print("\n🎯 Head Classification:")
    cls = stats['classification']
    print(f"  High Scale (>50):  {cls['high_scale_heads']:3d} heads ({cls['high_scale_ratio']:.1%})")
    print(f"  Mid Scale (5-50):  {cls['mid_scale_heads']:3d} heads ({cls['mid_scale_ratio']:.1%})")
    print(f"  Low Scale (<5):    {cls['low_scale_heads']:3d} heads ({cls['low_scale_ratio']:.1%})")

    # 层级趋势
    print("\n📈 Layer-wise Trends:")
    variances = [s['variance'] for s in stats['per_layer']]
    high_var_layers = [i for i, v in enumerate(variances) if v > 20]
    low_var_layers = [i for i, v in enumerate(variances) if v < 10]

    print(f"  High Variance Layers (>20): {high_var_layers}")
    print(f"  Low Variance Layers (<10):  {low_var_layers}")
    print(f"  Avg Variance: {np.mean(variances):.2f}")

    # 剪枝策略建议
    print("\n✂️  Recommended Pruning Strategy:")
    print("\n  Per-Layer Sparsity (Top 5 / Bottom 5):")

    sparsities = [(s['layer'], s['sparsity']) for s in strategy['per_layer_sparsity']]
    sparsities_sorted = sorted(sparsities, key=lambda x: x[1])

    print("    Most Conservative (Low Sparsity):")
    for layer, sp in sparsities_sorted[:5]:
        print(f"      Layer {layer:2d}: {sp:.2%}")

    print("    Most Aggressive (High Sparsity):")
    for layer, sp in sparsities_sorted[-5:]:
        print(f"      Layer {layer:2d}: {sp:.2%}")

    # 三尺度组策略
    print("\n  Three-Scale-Group Strategy:")
    for group in strategy['scale_groups']:
        print(f"\n    {group['group'].upper()} Group (Layers {group['layers'][0]}-{group['layers'][-1]}):")
        print(f"      Handles Scales: {group['scales']}")
        print(f"      Avg Variance: {group['avg_variance']:.2f}")
        print(f"      High-Scale Heads: {group['avg_high_ratio']:.1%}")
        print(f"      → Recommended Sparsity: {group['recommended_sparsity']:.2%}")

    print("\n" + "="*80 + "\n")


def main(args):
    print("\n" + "="*80)
    print(" VAR Scale_Mul Analysis Tool ".center(80, "="))
    print("="*80 + "\n")

    # 1. 加载模型
    print("Step 1: Loading VAR model...")
    vae_ckpt = args.vae_ckpt if args.vae_ckpt else '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt if args.var_ckpt else f'/home/project/daily/AR/model_zoo/var_d{args.model_depth}.pth'

    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device='cpu')

    # 2. 提取scale_mul
    print("\nStep 2: Extracting scale_mul parameters...")
    scale_mul_matrix, scale_mul_raw = extract_scale_mul(var)
    print(f"  Shape: {scale_mul_matrix.shape} (layers x heads)")
    print(f"  Range: [{scale_mul_matrix.min():.2f}, {scale_mul_matrix.max():.2f}]")

    # 3. 统计分析
    print("\nStep 3: Analyzing scale_mul distribution...")
    stats = analyze_scale_mul_statistics(scale_mul_matrix)

    # 4. 可视化
    print("\nStep 4: Generating visualizations...")
    visualize_scale_mul(scale_mul_matrix, stats, args.output_dir)

    # 5. 推荐剪枝策略
    print("\nStep 5: Recommending pruning strategy...")
    strategy = recommend_pruning_strategy(stats, args.model_depth)

    # 6. 保存结果
    print("\nStep 6: Saving analysis results...")

    # 保存统计数据
    with open(f'{args.output_dir}/scale_mul_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {args.output_dir}/scale_mul_analysis.json")

    # 保存剪枝策略
    with open(f'{args.output_dir}/pruning_strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)
    print(f"✓ Saved: {args.output_dir}/pruning_strategy.json")

    # 保存原始数据
    np.save(f'{args.output_dir}/scale_mul_matrix.npy', scale_mul_matrix)
    print(f"✓ Saved: {args.output_dir}/scale_mul_matrix.npy")

    # 7. 打印报告
    print_analysis_report(stats, strategy)

    print("✅ Analysis complete!")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Scale_Mul distribution in VAR model")

    parser.add_argument(
        "--model_depth", type=int, default=16,
        choices=[16, 20, 24, 30],
        help="VAR model depth"
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
        "--output_dir", type=str, default="./scale_mul_analysis",
        help="Directory to save analysis results"
    )

    args = parser.parse_args()
    main(args)
