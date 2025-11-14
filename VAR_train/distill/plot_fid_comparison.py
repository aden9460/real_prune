"""
文件名: plot_fid_comparison.py
位置: /home/project/real_prune/VAR_train/distill/
功能: 绘制不同方法的FID对比曲线

使用方法:
    python plot_fid_comparison.py --results_file results.json --output_dir ./figures
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_fid_vs_pruning_rate(results_dict, save_path='fid_comparison.pdf'):
    """
    绘制FID vs. Pruning Rate曲线

    Args:
        results_dict: {
            'pruning_rates': [0.1, 0.2, 0.3, 0.4],
            'methods': {
                'Teacher': [10.2, 10.2, 10.2, 10.2],
                'Pruned Only': [32.5, 45.8, 68.3, 95.7],
                'Normal KD': [18.4, 28.3, 42.1, 61.5],
                'SPD (Ours)': [12.8, 15.7, 22.4, 35.8]
            }
        }
    """

    pruning_rates = np.array(results_dict['pruning_rates']) * 100  # 转换为百分比
    methods = results_dict['methods']

    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色和样式配置
    styles = {
        'Teacher': {'color': 'black', 'linestyle': '--', 'marker': '*', 'linewidth': 2.5, 'markersize': 15},
        'Pruned Only': {'color': 'red', 'linestyle': '-.', 'marker': 'x', 'linewidth': 2, 'markersize': 10},
        'Normal KD': {'color': 'orange', 'linestyle': '--', 'marker': 's', 'linewidth': 2, 'markersize': 8},
        'SPD (Ours)': {'color': 'green', 'linestyle': '-', 'marker': 'o', 'linewidth': 3, 'markersize': 10}
    }

    # 绘制每条曲线
    for method_name, fid_values in methods.items():
        style = styles.get(method_name, {'color': 'gray', 'linestyle': '-', 'marker': 'o'})
        ax.plot(pruning_rates, fid_values, label=method_name, **style, alpha=0.9)

        # 在最后一个点标注数值
        if method_name != 'Teacher':
            last_fid = fid_values[-1]
            ax.text(pruning_rates[-1] + 1, last_fid, f'{last_fid:.1f}',
                   fontsize=9, verticalalignment='center')

    ax.set_xlabel('Pruning Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('FID Score (lower is better)', fontsize=13, fontweight='bold')
    ax.set_title('FID vs. Pruning Rate: Method Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')

    # 设置y轴从0开始
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ FID对比曲线已保存至: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def plot_fid_improvement_bar(results_dict, save_path='fid_improvement.pdf'):
    """绘制FID改进的柱状图"""

    pruning_rates = np.array(results_dict['pruning_rates']) * 100
    methods = results_dict['methods']

    # 计算相对于Pruned Only的改进
    baseline = np.array(methods['Pruned Only'])
    improvements = {}

    for method_name, fid_values in methods.items():
        if method_name not in ['Teacher', 'Pruned Only']:
            improvement = (baseline - np.array(fid_values)) / baseline * 100
            improvements[method_name] = improvement

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pruning_rates))
    width = 0.25

    colors = {'Normal KD': 'orange', 'SPD (Ours)': 'green'}

    for i, (method_name, impr) in enumerate(improvements.items()):
        offset = (i - len(improvements)/2 + 0.5) * width
        bars = ax.bar(x + offset, impr, width, label=method_name,
                     color=colors.get(method_name, 'gray'), alpha=0.8)

        # 标注数值
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Pruning Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('FID Improvement vs. Pruned Only (%)', fontsize=13, fontweight='bold')
    ax.set_title('Relative FID Improvement by Method', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(pr)}%' for pr in pruning_rates])
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ FID改进柱状图已保存至: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def plot_weight_strategy_ablation(ablation_dict, save_path='weight_ablation.pdf'):
    """
    绘制权重策略消融实验结果

    Args:
        ablation_dict: {
            'strategies': ['Uniform', 'Inverse', 'Sqrt', 'Exp', 'Linear (Ours)'],
            'fid_20': [28.3, 32.1, 22.4, 19.8, 15.7],
            'fid_30': [42.1, 48.3, 35.6, 30.2, 22.4]
        }
    """

    strategies = ablation_dict['strategies']
    fid_20 = ablation_dict['fid_20']
    fid_30 = ablation_dict['fid_30']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, fid_20, width, label='20% Pruning',
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, fid_30, width, label='30% Pruning',
                   color='lightcoral', alpha=0.8, edgecolor='black')

    # 标注数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Weight Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('FID Score', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: Weight Strategy Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha='right')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # 高亮最优结果
    best_idx = strategies.index('Linear (Ours)')
    ax.axvspan(best_idx - 0.5, best_idx + 0.5, alpha=0.1, color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 权重策略消融图已保存至: {save_path}")

    png_path = save_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG版本已保存至: {png_path}")

    plt.close()

def create_sample_results():
    """创建示例结果数据（用于测试）"""

    results = {
        'pruning_rates': [0.1, 0.2, 0.3, 0.4],
        'methods': {
            'Teacher': [10.2, 10.2, 10.2, 10.2],
            'Pruned Only': [32.5, 45.8, 68.3, 95.7],
            'Normal KD': [18.4, 28.3, 42.1, 61.5],
            'SPD (Ours)': [12.8, 15.7, 22.4, 35.8]
        }
    }

    ablation = {
        'strategies': ['Uniform', 'Inverse', 'Sqrt', 'Exp', 'Linear (Ours)'],
        'fid_20': [28.3, 32.1, 22.4, 19.8, 15.7],
        'fid_30': [42.1, 48.3, 35.6, 30.2, 22.4]
    }

    return results, ablation

def main():
    parser = argparse.ArgumentParser(description='Plot FID comparison figures')
    parser.add_argument('--results_file', type=str, default=None,
                       help='JSON file containing FID results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use sample data for testing')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成 FID 对比可视化图表")
    print("=" * 60)
    print()

    # 加载或创建数据
    if args.results_file and Path(args.results_file).exists():
        print(f"从文件加载结果: {args.results_file}")
        with open(args.results_file, 'r') as f:
            data = json.load(f)
        results_dict = data['results']
        ablation_dict = data['ablation']
    else:
        if not args.use_sample_data:
            print("⚠️  未找到结果文件，使用示例数据")
            print("提示：使用 --use_sample_data 标志来明确使用示例数据")
        else:
            print("使用示例数据进行测试")
        results_dict, ablation_dict = create_sample_results()

    print()

    # 生成图表
    print("[1/3] 生成FID vs. Pruning Rate曲线...")
    plot_fid_vs_pruning_rate(results_dict, str(output_dir / 'fid_comparison.pdf'))
    print()

    print("[2/3] 生成FID改进柱状图...")
    plot_fid_improvement_bar(results_dict, str(output_dir / 'fid_improvement.pdf'))
    print()

    print("[3/3] 生成权重策略消融图...")
    plot_weight_strategy_ablation(ablation_dict, str(output_dir / 'weight_ablation.pdf'))
    print()

    print("=" * 60)
    print(f"✓ 所有FID对比图表生成完成！")
    print(f"✓ 输出目录: {output_dir.absolute()}")
    print("=" * 60)

    # 保存示例JSON文件
    if args.use_sample_data or not args.results_file:
        sample_file = output_dir / 'sample_results.json'
        with open(sample_file, 'w') as f:
            json.dump({
                'results': results_dict,
                'ablation': ablation_dict
            }, f, indent=2)
        print(f"\n提示：示例数据已保存至 {sample_file}")
        print("     你可以基于此格式准备自己的实验结果文件")

if __name__ == '__main__':
    main()
