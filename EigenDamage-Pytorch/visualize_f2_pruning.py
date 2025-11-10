"""
可视化解释 KFAC OBS F2 剪枝方法
用简单的例子展示整个计算流程
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_f2_pruning():
    # 使用小尺寸矩阵便于可视化（8x8 而不是 1024x1024）
    in_dim = 8
    out_dim = 8
    prune_num = 3  # 剪掉3个神经元

    # 创建示例数据
    torch.manual_seed(42)
    w = torch.randn(out_dim, in_dim) * 0.5  # 权重矩阵
    A = torch.randn(in_dim, in_dim)
    A = (A @ A.t()) + torch.eye(in_dim) * 0.1  # 输入协方差矩阵（对称正定）
    G = torch.randn(out_dim, out_dim)
    G = (G @ G.t()) + torch.eye(out_dim) * 0.1  # 输出梯度协方差矩阵

    # 计算 G_inv
    d_g, Q_g = torch.linalg.eigh(G)  # 注意：eigh返回(特征值, 特征向量)
    eps = 1e-10
    G_inv = Q_g @ torch.diag(1.0 / (d_g + eps)) @ Q_g.t()

    # 计算重要性
    w_imps = torch.sum(w**2 @ A, dim=1)
    importances = w_imps / torch.diag(G_inv)

    # 选择要剪枝的神经元（重要性最低的）
    _, sorted_indices = torch.sort(importances, descending=True)
    keep_indices = sorted_indices[:out_dim - prune_num]
    prune_indices = sorted_indices[out_dim - prune_num:]

    # 创建大图
    fig = plt.figure(figsize=(20, 14))

    # ========== 第一部分：整体流程图 ==========
    ax0 = plt.subplot(2, 3, 1)
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 12)
    ax0.axis('off')
    ax0.set_title('F2 Pruning Overall Pipeline', fontsize=14, fontweight='bold', pad=20)

    # 绘制流程框
    boxes = [
        (1, 10, "Input Data\n& Gradients", '#E8F4F8'),
        (1, 8.5, "Collect Statistics\nA (input cov)\nG (gradient cov)", '#B3E5FC'),
        (1, 6.5, "Eigen Decompose\nG = Q @ diag(d) @ Q^T", '#81D4FA'),
        (1, 4.5, "Compute G_inv\nG_inv = Q @ diag(1/d) @ Q^T", '#4FC3F7'),
        (1, 2.5, "Importance Score\nimp = (w²@A).sum(1) / diag(G_inv)", '#29B6F6'),
        (1, 0.5, "Prune & Compensate\nUpdate remaining weights", '#0288D1'),
    ]

    for i, (y_pos, y_val, text, color) in enumerate(boxes):
        box = FancyBboxPatch((y_pos, y_val), 8, 1.3, boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=color, linewidth=2)
        ax0.add_patch(box)
        ax0.text(5, y_val + 0.65, text, ha='center', va='center',
                fontsize=10, fontweight='bold')

        if i < len(boxes) - 1:
            arrow = FancyArrowPatch((5, y_val), (5, boxes[i+1][1] + 1.3),
                                   arrowstyle='->', mutation_scale=30, linewidth=2,
                                   color='black')
            ax0.add_patch(arrow)

    # ========== 第二部分：矩阵维度图 ==========
    ax1 = plt.subplot(2, 3, 2)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Matrix Dimensions (8x8 example)', fontsize=14, fontweight='bold', pad=20)

    # 权重矩阵 W
    w_rect = FancyBboxPatch((1, 6), 3, 3, linewidth=2, edgecolor='blue', facecolor='lightblue')
    ax1.add_patch(w_rect)
    ax1.text(2.5, 7.5, 'W\n[8, 8]', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(2.5, 5.5, 'out × in', ha='center', va='center', fontsize=9)

    # 输入协方差矩阵 A
    a_rect = FancyBboxPatch((5.5, 6), 3, 3, linewidth=2, edgecolor='green', facecolor='lightgreen')
    ax1.add_patch(a_rect)
    ax1.text(7, 7.5, 'A\n[8, 8]', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(7, 5.5, 'input cov', ha='center', va='center', fontsize=9)

    # 梯度协方差矩阵 G
    g_rect = FancyBboxPatch((1, 1.5), 3, 3, linewidth=2, edgecolor='red', facecolor='lightcoral')
    ax1.add_patch(g_rect)
    ax1.text(2.5, 3, 'G\n[8, 8]', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(2.5, 1, 'gradient cov', ha='center', va='center', fontsize=9)

    # G_inv
    ginv_rect = FancyBboxPatch((5.5, 1.5), 3, 3, linewidth=2, edgecolor='purple', facecolor='plum')
    ax1.add_patch(ginv_rect)
    ax1.text(7, 3, 'G_inv\n[8, 8]', ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(7, 1, 'inverse Fisher', ha='center', va='center', fontsize=9)

    # ========== 第三部分：重要性计算可视化 ==========
    ax2 = plt.subplot(2, 3, 3)
    ax2.bar(range(out_dim), importances.numpy(), color=['red' if i in prune_indices else 'green'
                                                          for i in range(out_dim)])
    ax2.set_xlabel('Output Neuron Index', fontsize=11)
    ax2.set_ylabel('Importance Score', fontsize=11)
    ax2.set_title('Neuron Importance Scores\n(Red = Pruned, Green = Kept)',
                  fontsize=13, fontweight='bold')
    ax2.axhline(y=importances[prune_indices].max(), color='orange', linestyle='--',
                linewidth=2, label='Pruning Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== 第四部分：权重矩阵热图（剪枝前） ==========
    ax3 = plt.subplot(2, 3, 4)
    im1 = ax3.imshow(w.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax3.set_title('Weight Matrix (Before Pruning)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Input Dim (8)', fontsize=11)
    ax3.set_ylabel('Output Dim (8)', fontsize=11)

    # 标记要剪枝的行
    for idx in prune_indices:
        ax3.axhspan(idx.item()-0.5, idx.item()+0.5, color='red', alpha=0.3)
    plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)

    # ========== 第五部分：剪枝补偿计算 ==========
    ax4 = plt.subplot(2, 3, 5)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Surgery (Weight Compensation)', fontsize=13, fontweight='bold', pad=20)

    # 显示公式
    formulas = [
        "Step 1: G_inv_diag = diag(G_inv)",
        "Step 2: G_inv[:, pruned_cols] = 0",
        "Step 3: coeff = G_inv @ diag(1/G_inv_diag)",
        "Step 4: delta_w = -coeff @ w",
        "Step 5: w_new = w + delta_w",
    ]

    for i, formula in enumerate(formulas):
        y_pos = 8.5 - i * 1.5
        box = FancyBboxPatch((0.5, y_pos-0.3), 9, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='lightyellow', linewidth=1.5)
        ax4.add_patch(box)
        ax4.text(5, y_pos + 0.1, formula, ha='center', va='center',
                fontsize=10, family='monospace', fontweight='bold')

    # ========== 第六部分：剪枝后的权重矩阵 ==========
    ax5 = plt.subplot(2, 3, 6)

    # 执行实际的surgery
    G_inv_surgery = G_inv.clone()
    G_inv_diag = torch.diag(G_inv)
    G_inv_surgery[:, keep_indices] = 0  # 保留的列清零（surgery只补偿被剪枝的）

    # 注意：这里的逻辑是将要保留的列清零，让被剪枝的神经元影响保留的神经元
    # 实际代码中是 G_inv[:, m.out_indices] = 0，out_indices是保留的索引
    coeff = G_inv_surgery @ torch.diag(1.0 / (G_inv_diag + eps))
    delta_w = -coeff @ w
    w_new = w + delta_w

    # 只显示保留的行
    w_pruned = w_new[keep_indices, :]
    im2 = ax5.imshow(w_pruned.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax5.set_title(f'Weight Matrix (After Pruning)\n{out_dim-prune_num}x{in_dim}',
                  fontsize=13, fontweight='bold')
    ax5.set_xlabel('Input Dim (8)', fontsize=11)
    ax5.set_ylabel(f'Output Dim ({out_dim-prune_num})', fontsize=11)
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('f2_pruning_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: f2_pruning_visualization.png")
    plt.show()

    # 打印详细信息
    print("\n" + "="*60)
    print("DETAILED CALCULATION BREAKDOWN")
    print("="*60)
    print(f"\n1. Matrix Dimensions:")
    print(f"   - Weight W: [{out_dim}, {in_dim}] (output × input)")
    print(f"   - Input cov A: [{in_dim}, {in_dim}]")
    print(f"   - Gradient cov G: [{out_dim}, {out_dim}]")
    print(f"   - G_inv: [{out_dim}, {out_dim}]")

    print(f"\n2. Importance Calculation:")
    print(f"   w² @ A → [{out_dim}, {in_dim}] @ [{in_dim}, {in_dim}] = [{out_dim}, {in_dim}]")
    print(f"   sum(dim=1) → [{out_dim}]  (one score per output neuron)")
    print(f"   divide by diag(G_inv) → [{out_dim}] / [{out_dim}] = [{out_dim}]")

    print(f"\n3. Pruning Decision:")
    print(f"   - Keep neurons (high importance): {keep_indices.tolist()}")
    print(f"   - Prune neurons (low importance): {prune_indices.tolist()}")
    print(f"   - Importance scores:")
    for i in range(out_dim):
        status = "❌ PRUNE" if i in prune_indices else "✓ KEEP"
        print(f"     Neuron {i}: {importances[i]:.4f}  {status}")

    print(f"\n4. Surgery (Compensation):")
    print(f"   - Original weight norm: {torch.norm(w).item():.4f}")
    print(f"   - Delta weight norm: {torch.norm(delta_w).item():.4f}")
    print(f"   - New weight norm: {torch.norm(w_new).item():.4f}")
    print(f"   - Pruned weight norm: {torch.norm(w_pruned).item():.4f}")

    print(f"\n5. Final Model:")
    print(f"   - Original: {out_dim} output neurons")
    print(f"   - After pruning: {out_dim - prune_num} output neurons")
    print(f"   - Compression ratio: {(1 - prune_num/out_dim)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    visualize_f2_pruning()
