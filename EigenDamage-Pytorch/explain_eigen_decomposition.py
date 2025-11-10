"""
详细解释特征值分解和逆矩阵计算
包含可视化和数值示例
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import Ellipse

def explain_eigen_decomposition():
    """用简单例子解释特征值分解"""

    # 创建一个4x4的对称正定矩阵（类似于 Fisher 信息矩阵）
    torch.manual_seed(42)
    n = 4
    A = torch.randn(n, n)
    G = A @ A.t() + torch.eye(n) * 2  # 确保正定

    # 特征值分解
    d_g, Q_g = torch.linalg.eigh(G)  # eigenvalues, eigenvectors

    print("="*70)
    print("第一问：什么是特征值和特征向量？")
    print("="*70)

    print("\n【定义】")
    print("对于方阵 G，如果存在标量 λ 和非零向量 v 满足：")
    print("    G @ v = λ * v")
    print("则：")
    print("  - λ 是特征值 (eigenvalue)")
    print("  - v 是对应的特征向量 (eigenvector)")

    print("\n【物理意义】")
    print("特征向量是矩阵 G 作用下'不改变方向'的向量")
    print("特征值是该向量被'拉伸或压缩'的倍数")

    print(f"\n【示例：G 是 {n}×{n} 矩阵】")
    print(f"\n原始矩阵 G =")
    print(G.numpy().round(2))

    print(f"\n特征值 d_g = [{', '.join([f'{x:.2f}' for x in d_g])}]")
    print(f"  - 维度: [{len(d_g)}]  (一维向量，有{n}个特征值)")
    print(f"  - 物理意义: 每个特征值代表在对应特征方向上的'重要性'")

    print(f"\n特征向量矩阵 Q_g = ")
    print(Q_g.numpy().round(3))
    print(f"  - 维度: [{n}, {n}]")
    print(f"  - 每一列是一个特征向量")
    print(f"  - 正交矩阵: Q^T @ Q = I (单位矩阵)")

    # 验证正交性
    I_check = Q_g.t() @ Q_g
    print(f"\n验证正交性: Q_g^T @ Q_g =")
    print(I_check.numpy().round(3))
    print("  ↑ 应该是单位矩阵（对角线为1，其他为0）")

    # 验证特征值分解
    print("\n验证特征值方程：")
    for i in range(n):
        v = Q_g[:, i]  # 第i个特征向量
        Gv = G @ v
        lambda_v = d_g[i] * v
        error = torch.norm(Gv - lambda_v)
        print(f"  第{i}个特征向量: ||G@v - λ*v|| = {error:.6f} ✓")

    print("\n" + "="*70)
    print("第二问：为什么 G_inv = Q @ diag(1/λ) @ Q^T ？")
    print("="*70)

    print("\n【特征值分解公式】")
    print("对于对称矩阵 G，可以分解为：")
    print("    G = Q @ diag(λ) @ Q^T")
    print("其中：")
    print("  - Q 是正交矩阵（列向量是特征向量）")
    print("  - diag(λ) 是对角矩阵（对角线是特征值）")

    print("\n【求逆过程】")
    print("两边同时取逆：")
    print("    G^(-1) = (Q @ diag(λ) @ Q^T)^(-1)")
    print("           = Q^(-T) @ diag(λ)^(-1) @ Q^(-1)")
    print("           = Q @ diag(1/λ) @ Q^T")
    print("                   ↑")
    print("因为 Q 是正交矩阵，所以 Q^(-1) = Q^T")

    print("\n【数值验证】")
    # 方法1：直接求逆
    G_inv_direct = torch.inverse(G)

    # 方法2：用特征值分解
    eps = 1e-10
    G_inv_eigen = Q_g @ torch.diag(1.0 / (d_g + eps)) @ Q_g.t()

    print(f"直接求逆 G^(-1) =")
    print(G_inv_direct.numpy().round(3))

    print(f"\n用特征值分解 Q @ diag(1/λ) @ Q^T =")
    print(G_inv_eigen.numpy().round(3))

    error = torch.norm(G_inv_direct - G_inv_eigen)
    print(f"\n两种方法的误差: {error:.8f} ✓")

    # 验证 G @ G_inv = I
    identity_check = G @ G_inv_eigen
    print(f"\n验证 G @ G^(-1) = I:")
    print(identity_check.numpy().round(3))

    print("\n【为什么用特征值分解求逆？】")
    print("优点：")
    print("  1. 数值稳定：可以过滤掉接近0的特征值（加 eps）")
    print("  2. 可以只计算部分特征值（对于大矩阵）")
    print("  3. 可以分析矩阵的'病态程度'（条件数 = max(λ)/min(λ)）")
    print(f"  4. 条件数: {(d_g.max() / d_g.min()).item():.2f}")

    print("\n" + "="*70)
    print("第三问：代码中的 _update_inv 和 _get_unit_importance")
    print("="*70)

    print("\n【_update_inv 函数】")
    print("作用：对收集的统计信息进行特征值分解")
    print("""
步骤1: 对输入协方差矩阵分解
    m_aa = m_aa / steps  # 平均化
    d_a, Q_a = torch.symeig(m_aa, eigenvectors=True)
    # d_a: [input_dim]  特征值
    # Q_a: [input_dim, input_dim]  特征向量

步骤2: 对输出梯度协方差矩阵分解
    m_gg = m_gg / steps  # 平均化
    d_g, Q_g = torch.symeig(m_gg, eigenvectors=True)
    # d_g: [output_dim]  特征值
    # Q_g: [output_dim, output_dim]  特征向量

步骤3: 过滤小特征值（数值稳定性）
    d_a.mul_((d_a > eps).float())  # 小于eps的设为0
    d_g.mul_((d_g > eps).float())
    """)

    print("\n【_get_unit_importance 函数】")
    print("作用：计算每个输出神经元的重要性得分")

    # 模拟代码中的计算
    print("\n以 kfac_full_pruner.py 中的完整方法为例：")
    print("""
步骤1: 获取权重矩阵
    w = fetch_mat_weights(m, False)  # [output_dim, input_dim]

步骤2: 计算 A_inv 和 G_inv
    A_inv = Q_a @ diag(1/(d_a + eps)) @ Q_a^T  # [in_dim, in_dim]
    G_inv = Q_g @ diag(1/(d_g + eps)) @ Q_g^T  # [out_dim, out_dim]

步骤3: 提取对角元素
    A_inv_diag = diag(A_inv)  # [in_dim]
    G_inv_diag = diag(G_inv)  # [out_dim]

步骤4: 计算重要性（逐元素版本）
    w_imp = w² / (G_inv_diag[:, None] @ A_inv_diag[None, :])
    # w²: [out_dim, in_dim]
    # G_inv_diag[:, None]: [out_dim, 1]
    # A_inv_diag[None, :]: [1, in_dim]
    # 外积: [out_dim, 1] @ [1, in_dim] = [out_dim, in_dim]
    # 结果: [out_dim, in_dim] - 每个权重的重要性

步骤5: 聚合到神经元级别
    out_neuron_imp = w_imp.sum(dim=1)  # [out_dim]
    # 对每个输出神经元的所有输入连接求和
    """)

    # 数值示例
    print("\n【数值示例：计算单个神经元重要性】")
    w_example = torch.randn(4, 4) * 0.5
    A_inv_example = Q_g @ torch.diag(1.0/(d_g + 1e-10)) @ Q_g.t()
    G_inv_example = Q_g @ torch.diag(1.0/(d_g + 1e-10)) @ Q_g.t()

    A_inv_diag = torch.diag(A_inv_example)
    G_inv_diag = torch.diag(G_inv_example)

    print(f"\n权重矩阵 w =")
    print(w_example.numpy().round(2))

    print(f"\nG_inv_diag = {G_inv_diag.numpy().round(3)}")
    print(f"A_inv_diag = {A_inv_diag.numpy().round(3)}")

    # 计算重要性
    w_imp = w_example**2 / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
    neuron_imp = w_imp.sum(1)

    print(f"\n每个权重的重要性 w_imp =")
    print(w_imp.numpy().round(2))

    print(f"\n每个神经元的重要性（按行求和）:")
    for i, imp in enumerate(neuron_imp):
        print(f"  神经元 {i}: {imp:.3f}")

    print("\n【物理意义】")
    print("重要性 = w² / (G_inv_diag ⊗ A_inv_diag)")
    print("  - 分子 w²: 权重的平方（能量）")
    print("  - 分母 G_inv_diag: Fisher矩阵逆的对角元素")
    print("    → 删除该神经元的'代价'（越大代价越小）")
    print("  - 分母 A_inv_diag: 输入协方差逆的对角元素")
    print("    → 该输入维度的'冗余度'")
    print("  - 重要性越高 → 越不应该剪枝")

    # 创建可视化
    create_visualization(G, Q_g, d_g, G_inv_eigen)

def create_visualization(G, Q_g, d_g, G_inv):
    """创建可视化图"""
    fig = plt.figure(figsize=(18, 10))

    # ========== 图1: 特征值分解可视化 ==========
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-1, 5)
    ax1.set_aspect('equal')
    ax1.set_title('Eigenvalue Decomposition Concept', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # G矩阵
    g_box = FancyBboxPatch((0, 3), 1, 1, boxstyle="round,pad=0.05",
                           edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax1.add_patch(g_box)
    ax1.text(0.5, 3.5, 'G', ha='center', va='center', fontsize=16, fontweight='bold')

    # 等号
    ax1.text(1.5, 3.5, '=', ha='center', va='center', fontsize=20)

    # Q矩阵
    q_box = FancyBboxPatch((2, 3), 0.7, 1, boxstyle="round,pad=0.05",
                           edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(q_box)
    ax1.text(2.35, 3.5, 'Q', ha='center', va='center', fontsize=14, fontweight='bold')

    # 乘号
    ax1.text(3, 3.5, '@', ha='center', va='center', fontsize=16)

    # diag(λ)
    diag_box = FancyBboxPatch((3.2, 3), 1, 1, boxstyle="round,pad=0.05",
                              edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax1.add_patch(diag_box)
    ax1.text(3.7, 3.5, 'diag(λ)', ha='center', va='center', fontsize=11, fontweight='bold')

    # 画对角线示意
    for i in np.linspace(3.3, 4.1, 4):
        ax1.plot([i, i+0.15], [3.8-0.6*(i-3.3), 3.2-0.6*(i-3.3)], 'r-', linewidth=2)

    # 乘号
    ax1.text(4.5, 3.5, '@', ha='center', va='center', fontsize=16)

    # Q^T
    qt_box = FancyBboxPatch((0.5, 1.5), 0.7, 1, boxstyle="round,pad=0.05",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(qt_box)
    ax1.text(0.85, 2, 'Q^T', ha='center', va='center', fontsize=14, fontweight='bold')

    # 标注
    ax1.text(2.5, 2.2, 'Eigenvectors\n(orthogonal)', ha='center', fontsize=9, style='italic')
    ax1.text(3.7, 2.2, 'Eigenvalues\n(diagonal)', ha='center', fontsize=9, style='italic')

    # ========== 图2: 逆矩阵计算 ==========
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(-1, 5)
    ax2.set_aspect('equal')
    ax2.set_title('Computing Inverse via Eigendecomposition', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # G^(-1)
    ginv_box = FancyBboxPatch((0, 3), 1.2, 1, boxstyle="round,pad=0.05",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax2.add_patch(ginv_box)
    ax2.text(0.6, 3.5, 'G^(-1)', ha='center', va='center', fontsize=14, fontweight='bold')

    ax2.text(1.6, 3.5, '=', ha='center', va='center', fontsize=20)

    # Q
    q_box2 = FancyBboxPatch((2, 3), 0.7, 1, boxstyle="round,pad=0.05",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax2.add_patch(q_box2)
    ax2.text(2.35, 3.5, 'Q', ha='center', va='center', fontsize=14, fontweight='bold')

    ax2.text(3, 3.5, '@', ha='center', va='center', fontsize=16)

    # diag(1/λ)
    diag_box2 = FancyBboxPatch((3.2, 3), 1.2, 1, boxstyle="round,pad=0.05",
                               edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax2.add_patch(diag_box2)
    ax2.text(3.8, 3.5, 'diag(1/λ)', ha='center', va='center', fontsize=11, fontweight='bold')

    ax2.text(4.7, 3.5, '@', ha='center', va='center', fontsize=16)

    # Q^T
    qt_box2 = FancyBboxPatch((0.5, 1.5), 0.7, 1, boxstyle="round,pad=0.05",
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax2.add_patch(qt_box2)
    ax2.text(0.85, 2, 'Q^T', ha='center', va='center', fontsize=14, fontweight='bold')

    # 标注关键点
    ax2.text(3.8, 2.2, '← Invert eigenvalues', ha='center', fontsize=9,
             style='italic', color='red')
    ax2.text(2.5, 1, 'Key: Q^(-1) = Q^T for orthogonal matrices',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # ========== 图3: 特征值可视化 ==========
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(range(len(d_g)), d_g.numpy(), color='steelblue', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Eigenvalue Index', fontsize=11)
    ax3.set_ylabel('Eigenvalue Magnitude', fontsize=11)
    ax3.set_title('Eigenvalues of G Matrix', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 标注最大和最小
    max_idx = d_g.argmax()
    min_idx = d_g.argmin()
    ax3.text(max_idx, d_g[max_idx], f' Max: {d_g[max_idx]:.2f}',
             va='bottom', fontsize=9, color='red', fontweight='bold')
    ax3.text(min_idx, d_g[min_idx], f' Min: {d_g[min_idx]:.2f}',
             va='top', fontsize=9, color='blue', fontweight='bold')

    # ========== 图4: G矩阵热图 ==========
    ax4 = plt.subplot(2, 3, 4)
    im1 = ax4.imshow(G.numpy(), cmap='RdBu_r', aspect='auto')
    ax4.set_title('Original Matrix G', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Column', fontsize=10)
    ax4.set_ylabel('Row', fontsize=10)
    plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    # ========== 图5: G_inv矩阵热图 ==========
    ax5 = plt.subplot(2, 3, 5)
    im2 = ax5.imshow(G_inv.numpy(), cmap='RdBu_r', aspect='auto')
    ax5.set_title('Inverse Matrix G^(-1)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Column', fontsize=10)
    ax5.set_ylabel('Row', fontsize=10)
    plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    # ========== 图6: 特征向量可视化 ==========
    ax6 = plt.subplot(2, 3, 6)
    im3 = ax6.imshow(Q_g.numpy(), cmap='viridis', aspect='auto')
    ax6.set_title('Eigenvector Matrix Q_g', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Eigenvector Index', fontsize=10)
    ax6.set_ylabel('Component', fontsize=10)
    plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)
    ax6.text(0.5, -0.15, 'Each column is an eigenvector',
             transform=ax6.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('eigen_decomposition_explained.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: eigen_decomposition_explained.png")
    plt.close()

if __name__ == "__main__":
    explain_eigen_decomposition()
