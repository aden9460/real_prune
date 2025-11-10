"""
分析 Attention 的 O (output projection) 矩阵
判断 F2 方法是否适合用于剪枝 O 矩阵
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns

def analyze_attention_structure():
    """分析 Multi-Head Attention 的结构"""

    print("="*80)
    print("Multi-Head Attention 结构分析")
    print("="*80)

    # 典型的 Transformer 配置
    d_model = 768  # VAR/ViT 常用
    num_heads = 12
    head_dim = d_model // num_heads  # 64

    print(f"\n【配置】")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - head_dim: {head_dim}")

    print(f"\n【Attention 流程】")
    print(f"""
    1. 输入 X: [batch, seq_len, {d_model}]

    2. Linear 投影:
       Q = X @ W_q  → [{d_model}, {d_model}]
       K = X @ W_k  → [{d_model}, {d_model}]
       V = X @ W_v  → [{d_model}, {d_model}]

    3. 分头重塑:
       Q → [batch, {num_heads}, seq_len, {head_dim}]
       K → [batch, {num_heads}, seq_len, {head_dim}]
       V → [batch, {num_heads}, seq_len, {head_dim}]

    4. 计算 Attention:
       Attention = softmax(QK^T / √{head_dim}) @ V
       → [batch, {num_heads}, seq_len, {head_dim}]

    5. 拼接所有 head:
       concat(head_1, ..., head_{num_heads})
       → [batch, seq_len, {num_heads * head_dim}]
       = [batch, seq_len, {d_model}]

    6. ⭐ O 矩阵投影（我们要分析的）:
       output = concat_heads @ W_o
       W_o: [{d_model}, {d_model}]
            [{num_heads * head_dim}, {d_model}]

       输入维度: {d_model} = {num_heads} heads × {head_dim} per head
       输出维度: {d_model}
    """)

    print("\n" + "="*80)
    print("O 矩阵的输入和输出特性分析")
    print("="*80)

    print(f"\n【O 矩阵输入：来自 {num_heads} 个 head 的拼接】")
    print(f"  输入维度划分:")
    for i in range(num_heads):
        start_idx = i * head_dim
        end_idx = (i + 1) * head_dim
        print(f"    Head {i:2d}: [{start_idx:3d}:{end_idx:3d}]  ({head_dim} dims)")

    print(f"\n  ⚠️ 关键问题1: 不同 head 的输出是否相关？")
    print(f"  答案: 有一定相关性！")
    print(f"  原因:")
    print(f"    - 所有 head 都来自同一个输入 X")
    print(f"    - 虽然每个 head 关注不同的子空间")
    print(f"    - 但它们共享相同的位置编码和底层语义")
    print(f"    - Multi-head 的设计就是为了捕捉不同的关系模式")
    print(f"    → 输入 A 矩阵 [{d_model}, {d_model}] 可能有显著的非对角元素")

    print(f"\n【O 矩阵输出：d_model 维度的特征】")
    print(f"  输出维度: [{d_model}]")

    print(f"\n  ⚠️ 关键问题2: 输出神经元之间是否相关？")
    print(f"  答案: 很可能有强相关性！")
    print(f"  原因:")
    print(f"    - O 矩阵的输出会经过 LayerNorm")
    print(f"    - LayerNorm 会在特征维度上归一化")
    print(f"    - 这会引入输出神经元之间的依赖关系")
    print(f"    - 后续的 FFN 也会利用这些特征的联合信息")
    print(f"    → 输出 G 矩阵 [{d_model}, {d_model}] 可能有显著的非对角元素")

    print("\n" + "="*80)
    print("F2 方法的适用性分析")
    print("="*80)

    print(f"\n【F2 方法假设】")
    print(f"  F2 = A ⊗ B，其中 B (对应 G) 是对角矩阵")
    print(f"  ↓")
    print(f"  假设：输出神经元之间无相关性")

    print(f"\n【对于 Attention O 矩阵】")
    print(f"  ❌ 输出神经元之间很可能有相关性")
    print(f"  ❌ G 矩阵可能不是对角占优的")
    print(f"  ❌ F2 的对角假设可能不成立")

    print(f"\n【建议】")
    print(f"  ✅ 方案1: 使用完整 KFAC (kfac_full_pruner.py)")
    print(f"     - 考虑 A 和 G 的对角元素的外积")
    print(f"     - 虽然也是近似，但比 F2 更保守")

    print(f"\n  ✅ 方案2: 直接测量 G 矩阵的对角性")
    print(f"     - 在实际数据上计算 G 矩阵")
    print(f"     - 分析 ||diag(G)|| / ||G|| 的比例")
    print(f"     - 如果对角占优 > 80%，F2 可以接受")
    print(f"     - 如果对角占优 < 60%，应该用完整方法")

    print(f"\n  ✅ 方案3: Head 级别剪枝")
    print(f"     - 不剪输出神经元，而是剪整个 head")
    print(f"     - 每个 head 是 {head_dim} 维的单元")
    print(f"     - Head 之间相对独立，更适合剪枝")

    # 创建可视化
    create_attention_visualization(d_model, num_heads, head_dim)

def simulate_attention_statistics():
    """模拟 Attention 的统计特性"""

    print("\n" + "="*80)
    print("模拟实验：测量 G 矩阵的对角性")
    print("="*80)

    torch.manual_seed(42)
    batch_size = 32
    seq_len = 256
    d_model = 768
    num_heads = 12
    head_dim = d_model // num_heads

    print(f"\n【实验设置】")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")

    # 模拟 attention 输出（拼接后的）
    # 我们假设不同 head 之间有一定的相关性
    print(f"\n【模拟场景】")

    scenarios = {
        "独立的 heads (理想情况)": {
            "correlation": 0.0,
            "description": "每个 head 完全独立，无相关性"
        },
        "弱相关 heads": {
            "correlation": 0.3,
            "description": "Head 之间有轻微相关性"
        },
        "中等相关 heads (现实情况)": {
            "correlation": 0.6,
            "description": "Head 之间有中等相关性（更接近真实）"
        },
        "强相关 heads": {
            "correlation": 0.9,
            "description": "Head 之间高度相关"
        }
    }

    results = {}

    for scenario_name, config in scenarios.items():
        print(f"\n场景: {scenario_name}")
        print(f"  {config['description']}")

        # 生成带相关性的数据
        correlation = config['correlation']

        # 基础随机向量
        base = torch.randn(batch_size, seq_len, d_model)
        noise = torch.randn(batch_size, seq_len, d_model)

        # 混合基础和噪声来控制相关性
        attention_output = correlation * base + (1 - correlation) * noise

        # 模拟梯度（通过 loss 反向传播得到）
        # 假设梯度也有类似的结构
        grad_output = torch.randn(batch_size, seq_len, d_model)

        # 计算 A 矩阵（输入协方差）
        # A = E[x ⊗ x]
        attention_output_flat = attention_output.view(-1, d_model)
        A = (attention_output_flat.t() @ attention_output_flat) / (batch_size * seq_len)

        # 计算 G 矩阵（输出梯度协方差）
        # G = E[g ⊗ g]
        grad_output_flat = grad_output.view(-1, d_model)
        G = (grad_output_flat.t() @ grad_output_flat) / (batch_size * seq_len)

        # 分析对角性
        diag_G = torch.diag(torch.diag(G))
        off_diag_G = G - diag_G

        diag_norm = torch.norm(diag_G, 'fro').item()
        off_diag_norm = torch.norm(off_diag_G, 'fro').item()
        total_norm = torch.norm(G, 'fro').item()

        diag_ratio = diag_norm / total_norm
        off_diag_ratio = off_diag_norm / total_norm

        print(f"  A 矩阵分析:")
        diag_A = torch.diag(torch.diag(A))
        off_diag_A = A - diag_A
        a_diag_ratio = torch.norm(diag_A, 'fro').item() / torch.norm(A, 'fro').item()
        print(f"    对角占比: {a_diag_ratio*100:.2f}%")

        print(f"  G 矩阵分析:")
        print(f"    ||diag(G)||_F = {diag_norm:.2f}")
        print(f"    ||off-diag(G)||_F = {off_diag_norm:.2f}")
        print(f"    对角占比: {diag_ratio*100:.2f}%")
        print(f"    非对角占比: {off_diag_ratio*100:.2f}%")

        # 判断
        if diag_ratio > 0.8:
            recommendation = "✅ F2 方法合理"
        elif diag_ratio > 0.6:
            recommendation = "⚠️ F2 可以用，但建议谨慎"
        else:
            recommendation = "❌ 不建议用 F2，用完整 KFAC"

        print(f"    → {recommendation}")

        results[scenario_name] = {
            'A': A,
            'G': G,
            'diag_ratio': diag_ratio,
            'off_diag_ratio': off_diag_ratio,
            'recommendation': recommendation
        }

    # 可视化结果
    visualize_scenarios(results, d_model)

    print("\n" + "="*80)
    print("实际建议")
    print("="*80)
    print(f"""
基于模拟实验，对于 Attention 的 O 矩阵：

1. 📊 先测量，再决定:
   ```python
   # 在实际数据上收集统计
   G = compute_gradient_covariance(model, dataloader)
   diag_ratio = torch.norm(torch.diag(torch.diag(G))) / torch.norm(G)

   if diag_ratio > 0.7:
       use_f2 = True
   else:
       use_f2 = False  # 用完整 KFAC
   ```

2. 🎯 更好的选择 - Head 级别剪枝:
   - 不直接剪 O 矩阵的输出神经元
   - 而是剪整个 attention head
   - 剪掉 Head i 意味着:
     * W_o 的输入维度 [{head_dim}*i : {head_dim}*(i+1)] 被剪掉
     * 对应 {head_dim} 个输入神经元
   - Head 之间更加独立，更适合剪枝

3. 🔬 分层策略:
   - 早期层（靠近输入）: Head 相关性较弱 → F2 可能可以
   - 中间层: Head 相关性中等 → 建议用完整 KFAC
   - 后期层（靠近输出）: Head 相关性较强 → 必须用完整 KFAC

4. 📝 实践中的做法:
   - 大多数工作在 Attention 上用 structured pruning (head-level)
   - 而不是 unstructured pruning (neuron-level)
   - 原因就是 O 矩阵的输入输出都有相关性
""")

def create_attention_visualization(d_model, num_heads, head_dim):
    """可视化 Attention 结构"""
    fig = plt.figure(figsize=(18, 10))

    # 图1: Attention 结构图
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Multi-Head Attention Structure', fontsize=14, fontweight='bold')

    # 输入
    input_box = FancyBboxPatch((4, 10.5), 2, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(5, 10.9, 'Input X', ha='center', va='center', fontsize=11, fontweight='bold')

    # Q, K, V 投影
    for i, name in enumerate(['Q', 'K', 'V']):
        x_pos = 1.5 + i * 3
        box = FancyBboxPatch((x_pos, 8.5), 1.5, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
        ax1.add_patch(box)
        ax1.text(x_pos + 0.75, 8.9, f'W_{name.lower()}', ha='center', va='center',
                fontsize=10, fontweight='bold')
        # 箭头
        ax1.arrow(5, 10.5, x_pos + 0.75 - 5, -1.5, head_width=0.2, head_length=0.2,
                 fc='black', ec='black', alpha=0.5)

    # Multi-head attention
    mha_box = FancyBboxPatch((2, 6.5), 6, 1.2, boxstyle="round,pad=0.05",
                            edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax1.add_patch(mha_box)
    ax1.text(5, 7.4, f'{num_heads} Heads', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax1.text(5, 6.9, f'Each: {head_dim} dims', ha='center', va='center',
            fontsize=9, style='italic')

    # Concat
    concat_box = FancyBboxPatch((3, 4.5), 4, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='purple', facecolor='plum', linewidth=2)
    ax1.add_patch(concat_box)
    ax1.text(5, 4.9, f'Concat: {d_model} dims', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # O 矩阵 (重点标注)
    o_box = FancyBboxPatch((3, 2.5), 4, 1.2, boxstyle="round,pad=0.05",
                          edgecolor='orange', facecolor='lightyellow', linewidth=3)
    ax1.add_patch(o_box)
    ax1.text(5, 3.4, '⭐ W_o Matrix ⭐', ha='center', va='center',
            fontsize=12, fontweight='bold', color='red')
    ax1.text(5, 2.9, f'[{d_model}, {d_model}]', ha='center', va='center',
            fontsize=9)

    # 输出
    output_box = FancyBboxPatch((4, 0.5), 2, 0.8, boxstyle="round,pad=0.05",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(5, 0.9, 'Output', ha='center', va='center', fontsize=11, fontweight='bold')

    # 图2: W_o 输入结构（head 拼接）
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_xlim(0, num_heads)
    ax2.set_ylim(0, 10)
    ax2.set_title('W_o Input: Concatenated Heads', fontsize=13, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, num_heads))
    for i in range(num_heads):
        rect = Rectangle((i, 0), 1, 8, facecolor=colors[i], edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(i + 0.5, 4, f'H{i}', ha='center', va='center',
                fontsize=9, fontweight='bold', rotation=90)

    ax2.text(num_heads/2, 9, f'Input dim: {num_heads} × {head_dim} = {d_model}',
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax2.set_xlim(0, num_heads)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('Head Index', fontsize=10)
    ax2.set_yticks([])

    # 图3: 相关性示意
    ax3 = plt.subplot(2, 3, 3)
    # 创建一个示意性的相关矩阵
    corr_matrix = np.zeros((num_heads, num_heads))
    for i in range(num_heads):
        for j in range(num_heads):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # 模拟中等相关性
                corr_matrix[i, j] = 0.3 + 0.4 * np.random.rand()

    im = ax3.imshow(corr_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax3.set_title('Head Correlation (Simulated)\nInput Correlation', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Head Index', fontsize=10)
    ax3.set_ylabel('Head Index', fontsize=10)
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # 图4: F2 假设 vs 实际
    ax4 = plt.subplot(2, 3, 4)

    # 创建示意图
    x = np.arange(d_model)

    # F2 假设：对角占优
    f2_values = np.zeros(d_model)
    f2_values[::64] = 10  # 只有对角有值
    f2_values += np.random.randn(d_model) * 0.5  # 小噪声

    ax4.plot(x, f2_values, label='F2 Assumption (diagonal)', linewidth=2, alpha=0.7)

    # 实际情况：更多非对角
    actual_values = np.ones(d_model) * 5 + np.random.randn(d_model) * 2
    actual_values[::64] = 10  # 对角稍大

    ax4.plot(x, actual_values, label='Actual (with correlations)', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Neuron Index', fontsize=10)
    ax4.set_ylabel('Importance (arbitrary)', fontsize=10)
    ax4.set_title('F2 Assumption vs Reality', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 图5: 建议方案对比
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('Pruning Strategies Comparison', fontsize=13, fontweight='bold')

    strategies = [
        ("F2", "Fast, assumes diagonal G", "⚠️ May be inaccurate", "yellow"),
        ("Full KFAC", "More accurate, uses diag(A)⊗diag(G)", "✅ Better for Attn", "lightgreen"),
        ("Head-level", "Prune entire heads", "✅ Best for Attn", "lightblue"),
    ]

    y_pos = 0.9
    for name, desc, status, color in strategies:
        box = FancyBboxPatch((0.05, y_pos - 0.15), 0.9, 0.13, boxstyle="round,pad=0.01",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax5.add_patch(box)
        ax5.text(0.1, y_pos - 0.05, name, fontsize=11, fontweight='bold', va='top')
        ax5.text(0.1, y_pos - 0.09, desc, fontsize=8, va='top', style='italic')
        ax5.text(0.1, y_pos - 0.13, status, fontsize=9, va='top', fontweight='bold')
        y_pos -= 0.2

    # 图6: 决策流程图
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('Decision Flow', fontsize=13, fontweight='bold')

    flow_text = """
    1. Measure G matrix on real data
       ↓
    2. Compute: ratio = ||diag(G)|| / ||G||
       ↓
    3. Decision:
       • ratio > 0.8  → Use F2
       • 0.6 < ratio < 0.8 → Use Full KFAC
       • ratio < 0.6 → Use Head-level pruning
       ↓
    4. For Attention O matrix:
       → Prefer Head-level pruning
    """

    ax6.text(0.1, 0.9, flow_text, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('attention_o_matrix_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: attention_o_matrix_analysis.png")
    plt.close()

def visualize_scenarios(results, d_model):
    """可视化不同场景的 G 矩阵"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (scenario_name, data) in enumerate(results.items()):
        row = idx // 2
        col = (idx % 2) * 2

        # G 矩阵热图
        ax1 = axes[row, col]
        G_show = data['G'][:64, :64].numpy()  # 只显示前64x64
        im1 = ax1.imshow(G_show, cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'{scenario_name}\nG Matrix (64×64 subset)', fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # 对角占比饼图
        ax2 = axes[row, col + 1]
        sizes = [data['diag_ratio'], data['off_diag_ratio']]
        labels = [f"Diagonal\n{data['diag_ratio']*100:.1f}%",
                 f"Off-diagonal\n{data['off_diag_ratio']*100:.1f}%"]
        colors = ['lightgreen', 'lightcoral']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
               textprops={'fontsize': 9, 'fontweight': 'bold'})
        ax2.set_title(f'{data["recommendation"]}', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('g_matrix_scenarios.png', dpi=150, bbox_inches='tight')
    print("✓ Scenario visualization saved to: g_matrix_scenarios.png")
    plt.close()

if __name__ == "__main__":
    analyze_attention_structure()
    simulate_attention_statistics()
