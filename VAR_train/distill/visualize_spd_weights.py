"""
文件名: visualize_spd_weights.py
位置: /home/project/real_prune/VAR_train/distill/
功能: 生成SPD权重调度的可视化图表

使用方法:
    python visualize_spd_weights.py --output_dir ./figures
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from pathlib import Path

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_spd_weight_schedule(save_path='spd_weight_schedule.pdf'):
    """绘制SPD权重调度图 - 主图"""

    # 你们的权重配置
    scales = np.arange(10)
    weights = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    scale_names = ['1×1', '2×2', '3×3', '4×4', '5×5',
                   '8×8', '10×10', '13×13', '16×16', 'Full']
    scale_tokens = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]

    fig, ax = plt.subplots(figsize=(12, 5))

    # 渐进色彩：从深红（重要）到浅蓝（次要）
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 10))

    # 绘制柱状图
    bars = ax.bar(scales, weights, color=colors, edgecolor='black',
                   linewidth=1.5, alpha=0.9)

    # 添加渐进箭头
    ax.annotate('', xy=(9, 0.15), xytext=(0, 2.05),
                arrowprops=dict(arrowstyle='->', lw=3, color='darkred', alpha=0.7))
    ax.text(4.5, 1.75, 'Progressive Decrease\n(Coarse → Fine)',
            fontsize=13, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # 标注三个阶段（背景色）
    ax.axvspan(-0.5, 3.5, alpha=0.12, color='red', label='Coarse Scales (High Weight)')
    ax.axvspan(3.5, 6.5, alpha=0.12, color='yellow', label='Medium Scales')
    ax.axvspan(6.5, 9.5, alpha=0.12, color='blue', label='Fine Scales (Low Weight)')

    # 在柱子上标注权重值
    for i, (scale, weight) in enumerate(zip(scales, weights)):
        ax.text(scale, weight + 0.08, f'{weight:.1f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
        # 在底部标注token数量
        ax.text(scale, -0.15, f'{scale_tokens[i]}t',
                ha='center', va='top', fontsize=8, color='gray', style='italic')

    # 轴标签和标题
    ax.set_xlabel('Scale Index (Resolution)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Distillation Weight', fontsize=14, fontweight='bold')
    ax.set_title('Scale-Progressive Distillation: Weight Schedule',
                 fontsize=16, fontweight='bold', pad=20)

    # X轴刻度
    ax.set_xticks(scales)
    ax.set_xticklabels(scale_names, rotation=0, ha='center')

    # Y轴范围
    ax.set_ylim(-0.3, 2.3)

    # 图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # 网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 权重调度图已保存至: {save_path}")

    # 同时保存PNG格式
    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def plot_weight_comparison(save_path='spd_weight_comparison.pdf'):
    """对比不同权重策略"""

    scales = np.arange(10)
    scale_names = ['1×1', '2×2', '3×3', '4×4', '5×5',
                   '8×8', '10×10', '13×13', '16×16', 'Full']

    # 不同策略
    strategies = {
        'SPD (Ours)': np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]),
        'Uniform': np.array([1.0] * 10),
        'Inverse': np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]),
        'Sqrt': np.array([np.sqrt(2.0 - 0.2*i) for i in range(10)])
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, weights in strategies.items():
        linestyle = '-' if strategy == 'SPD (Ours)' else '--'
        linewidth = 3 if strategy == 'SPD (Ours)' else 2
        marker = 'o' if strategy == 'SPD (Ours)' else 's'
        markersize = 10 if strategy == 'SPD (Ours)' else 7
        alpha = 1.0 if strategy == 'SPD (Ours)' else 0.7

        ax.plot(scales, weights, label=strategy,
                linestyle=linestyle, linewidth=linewidth,
                marker=marker, markersize=markersize, alpha=alpha)

    ax.set_xlabel('Scale Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=13, fontweight='bold')
    ax.set_title('Comparison of Different Weighting Strategies',
                 fontsize=15, fontweight='bold')

    ax.set_xticks(scales)
    ax.set_xticklabels(scale_names, rotation=45, ha='right')

    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 策略对比图已保存至: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def plot_weight_vs_tokens(save_path='spd_weight_vs_tokens.pdf'):
    """权重与token数量的关系图"""

    scales = np.arange(10)
    weights = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    scale_tokens = np.array([1, 4, 9, 16, 25, 36, 64, 100, 169, 256])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：权重和token数量的双轴图
    color1 = 'tab:red'
    ax1.set_xlabel('Scale Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distillation Weight', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(scales, weights, color=color1, marker='o', linewidth=3,
                     markersize=8, label='Weight')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color2 = 'tab:blue'
    ax1_twin.set_ylabel('Number of Tokens', color=color2, fontsize=12, fontweight='bold')
    line2 = ax1_twin.plot(scales, scale_tokens, color=color2, marker='s', linewidth=3,
                          markersize=8, linestyle='--', label='Tokens')
    ax1_twin.tick_params(axis='y', labelcolor=color2)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)
    ax1.set_title('Weight vs. Token Count per Scale', fontsize=14, fontweight='bold')

    # 右图：加权后的有效强度（weight * tokens）
    effective_strength = weights * scale_tokens
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, 10))

    ax2.bar(scales, effective_strength, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Scale Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Effective Supervision (Weight × Tokens)', fontsize=12, fontweight='bold')
    ax2.set_title('Effective Supervision per Scale', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # 标注数值
    for i, val in enumerate(effective_strength):
        ax2.text(i, val + 5, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 权重-Token关系图已保存至: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def plot_information_density(save_path='spd_information_density.pdf'):
    """信息密度假设可视化"""

    scales = np.arange(10)
    scale_names = ['1×1', '2×2', '3×3', '4×4', '5×5',
                   '8×8', '10×10', '13×13', '16×16', 'Full']
    scale_tokens = np.array([1, 4, 9, 16, 25, 36, 64, 100, 169, 256])

    # 信息密度 ≈ 1 / tokens (归一化)
    info_density = 1.0 / scale_tokens
    info_density_norm = info_density / info_density.max()

    # 我们的权重（归一化到0-1）
    weights = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制信息密度
    ax.plot(scales, info_density_norm, marker='s', linewidth=3, markersize=10,
            label='Information Density (∝ 1/tokens)', color='steelblue', alpha=0.7)

    # 绘制我们的权重
    ax.plot(scales, weights_norm, marker='o', linewidth=3, markersize=10,
            label='SPD Weights (normalized)', color='crimson', alpha=0.9)

    # 填充区域表示相关性
    ax.fill_between(scales, info_density_norm, weights_norm, alpha=0.15, color='purple')

    ax.set_xlabel('Scale Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
    ax.set_title('SPD Weight Design Follows Information Density Principle',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(scales)
    ax.set_xticklabels(scale_names, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')

    # 添加注释
    ax.annotate('High density\n→ High weight', xy=(1, 0.9), xytext=(2, 0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkred'),
                fontsize=11, weight='bold', color='darkred')

    ax.annotate('Low density\n→ Low weight', xy=(8, 0.1), xytext=(6, 0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='navy'),
                fontsize=11, weight='bold', color='navy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 信息密度图已保存至: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate SPD weight visualization figures')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成 Scale-Progressive Distillation 权重可视化图表")
    print("=" * 60)
    print()

    # 生成所有图表
    print("[1/5] 生成主权重调度图...")
    plot_spd_weight_schedule(str(output_dir / 'spd_weight_schedule.pdf'))
    print()

    print("[2/5] 生成策略对比图...")
    plot_weight_comparison(str(output_dir / 'spd_weight_comparison.pdf'))
    print()

    print("[3/5] 生成权重-Token关系图...")
    plot_weight_vs_tokens(str(output_dir / 'spd_weight_vs_tokens.pdf'))
    print()

    print("[4/5] 生成信息密度理论图...")
    plot_information_density(str(output_dir / 'spd_information_density.pdf'))
    print()

    print("=" * 60)
    print(f"✓ 所有图表生成完成！")
    print(f"✓ 输出目录: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == '__main__':
    main()
