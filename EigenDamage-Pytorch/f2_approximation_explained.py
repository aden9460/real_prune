"""
详细解释F2近似在哪一步发生
对比完整KFAC vs F2方法
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def explain_f2_approximation():
    """用数值示例对比完整KFAC和F2近似"""

    print("="*80)
    print("F2 近似发生在哪一步？")
    print("="*80)

    # 创建示例数据
    torch.manual_seed(42)
    out_dim, in_dim = 8, 8

    # 权重矩阵
    w = torch.randn(out_dim, in_dim) * 0.5

    # 输入协方差矩阵 A
    A = torch.randn(in_dim, in_dim)
    A = (A @ A.t()) + torch.eye(in_dim) * 0.5

    # 输出梯度协方差矩阵 G (完整的，非对角)
    G = torch.randn(out_dim, out_dim)
    G = (G @ G.t()) + torch.eye(out_dim) * 0.5

    # 计算逆矩阵
    eps = 1e-10
    d_g, Q_g = torch.linalg.eigh(G)
    A_inv = torch.inverse(A)
    G_inv = Q_g @ torch.diag(1.0 / (d_g + eps)) @ Q_g.t()

    print("\n【Fisher 信息矩阵的 KFAC 近似】")
    print("完整的 Fisher 矩阵: F [out*in, out*in]")
    print("KFAC 近似: F ≈ A ⊗ G")
    print(f"  - A: [{in_dim}, {in_dim}] 输入协方差")
    print(f"  - G: [{out_dim}, {out_dim}] 输出梯度协方差")
    print(f"  - Kronecker 积后: [{out_dim*in_dim}, {out_dim*in_dim}]")

    print("\n" + "="*80)
    print("方法对比")
    print("="*80)

    # ========== 完整 KFAC 方法 ==========
    print("\n【方法1: 完整 KFAC (Full)】")
    print("来自 kfac_full_pruner.py 的 _get_unit_importance")
    print("-"*80)

    print("\n步骤1: 计算 A_inv 和 G_inv")
    print(f"  A_inv = Q_a @ diag(1/λ_a) @ Q_a^T  [{in_dim}, {in_dim}]")
    print(f"  G_inv = Q_g @ diag(1/λ_g) @ Q_g^T  [{out_dim}, {out_dim}]")

    print("\n步骤2: 提取对角元素")
    A_inv_diag = torch.diag(A_inv)
    G_inv_diag = torch.diag(G_inv)
    print(f"  A_inv_diag = diag(A_inv)  [{in_dim}]")
    print(f"  G_inv_diag = diag(G_inv)  [{out_dim}]")
    print(f"\n  G_inv_diag = {G_inv_diag.numpy().round(3)}")

    print("\n步骤3: 计算每个权重的重要性 (关键步骤！)")
    print("  代码: w_imp = w² / (G_inv_diag[:, None] @ A_inv_diag[None, :])")

    # 这里使用了G_inv和A_inv的对角元素的外积
    w_imp_full = w**2 / (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
    print(f"  结果维度: [{out_dim}, {in_dim}] - 每个权重都有独立的重要性")

    print("\n步骤4: 聚合到神经元")
    print("  代码: out_neuron_imp = w_imp.sum(dim=1)")
    out_neuron_imp_full = w_imp_full.sum(1)
    print(f"  结果维度: [{out_dim}]")

    print("\n✓ 完整KFAC使用了：")
    print("  - A_inv 的对角元素（近似：忽略输入之间的耦合）")
    print("  - G_inv 的对角元素（近似：忽略输出之间的耦合）")
    print("  - 对每个权重单独计算重要性")

    # ========== F2 方法 ==========
    print("\n" + "="*80)
    print("\n【方法2: F2 近似】")
    print("来自 kfac_OBS_F2.py 的 _get_unit_importance")
    print("-"*80)

    print("\n步骤1: 计算 G_inv (完整矩阵)")
    print(f"  G_inv = Q_g @ diag(1/λ_g) @ Q_g^T  [{out_dim}, {out_dim}]")
    print("  ⚠️ 注意：这里先计算完整的 G_inv")

    print("\n步骤2: 权重加权求和 (考虑输入相关性)")
    print("  代码: w_imps = sum(w² @ A, dim=1)")
    w_imps_f2 = torch.sum(w**2 @ A, dim=1)  # 注意这里用的是A，不是A_inv！
    print(f"  - w²: [{out_dim}, {in_dim}]")
    print(f"  - A: [{in_dim}, {in_dim}]  (完整的输入协方差矩阵)")
    print(f"  - w² @ A: [{out_dim}, {in_dim}]")
    print(f"  - sum(dim=1): [{out_dim}]")

    print("\n步骤3: 除以 G_inv 的对角元素 (⭐ F2近似发生在这里！)")
    print("  代码: out_neuron_imp = w_imps / diag(G_inv)")
    print("                                    ^^^^^^^^^^^^")
    print("                                    只用对角元素！")
    out_neuron_imp_f2 = w_imps_f2 / torch.diag(G_inv)
    print(f"  - w_imps: [{out_dim}]")
    print(f"  - diag(G_inv): [{out_dim}]")
    print(f"  - 结果: [{out_dim}]")

    print("\n✓ F2方法的关键区别：")
    print("  - 使用完整的 A 矩阵（w² @ A 考虑了输入之间的相关性）")
    print("  - 但只使用 G_inv 的对角元素（忽略输出之间的相关性）❗")

    # ========== 近似对比 ==========
    print("\n" + "="*80)
    print("关键对比：近似在哪里？")
    print("="*80)

    print("\n【完整KFAC】")
    print("```python")
    print("# 步骤1: 先分离对角")
    print("A_inv_diag = diag(A_inv)  # [in_dim]")
    print("G_inv_diag = diag(G_inv)  # [out_dim]")
    print("")
    print("# 步骤2: 外积 → [out_dim, in_dim]")
    print("denominator = G_inv_diag[:, None] @ A_inv_diag[None, :]")
    print("")
    print("# 步骤3: 逐元素除法")
    print("w_imp = w² / denominator  # [out_dim, in_dim]")
    print("")
    print("# 步骤4: 求和")
    print("importance = w_imp.sum(1)  # [out_dim]")
    print("```")
    print("📊 近似: diag(A_inv) ⊗ diag(G_inv) 而不是完整的 A_inv ⊗ G_inv")

    print("\n【F2方法】")
    print("```python")
    print("# 步骤1: 使用完整的 A 矩阵")
    print("w_imps = sum(w² @ A, dim=1)  # [out_dim]  ← A是完整矩阵！")
    print("")
    print("# 步骤2: 只用 G_inv 的对角")
    print("importance = w_imps / diag(G_inv)  # [out_dim] ← 只用对角！")
    print("```")
    print("📊 近似: 假设 G (输出协方差) 是对角矩阵")

    # ========== 可视化差异 ==========
    print("\n" + "="*80)
    print("G 矩阵的结构分析")
    print("="*80)

    G_diag = torch.diag(torch.diag(G))  # 只保留对角元素
    G_off_diag = G - G_diag  # 非对角元素

    diag_ratio = torch.norm(G_diag) / torch.norm(G)
    off_diag_ratio = torch.norm(G_off_diag) / torch.norm(G)

    print(f"\nG 矩阵的 Frobenius 范数分析:")
    print(f"  - 完整矩阵 ||G||_F = {torch.norm(G):.4f}")
    print(f"  - 对角部分 ||diag(G)||_F = {torch.norm(G_diag):.4f} ({diag_ratio*100:.1f}%)")
    print(f"  - 非对角部分 ||off-diag(G)||_F = {torch.norm(G_off_diag):.4f} ({off_diag_ratio*100:.1f}%)")

    print("\n⚠️ F2近似假设: 非对角部分可以忽略")
    print(f"   如果 {off_diag_ratio*100:.1f}% 很小，近似就很好")
    print(f"   如果 {off_diag_ratio*100:.1f}% 很大，近似误差就大")

    # ========== 数值结果对比 ==========
    print("\n" + "="*80)
    print("数值结果对比")
    print("="*80)

    print("\n神经元重要性得分:")
    print(f"{'神经元':>6} | {'完整KFAC':>12} | {'F2方法':>12} | {'相对差异':>12}")
    print("-"*60)
    for i in range(out_dim):
        diff_pct = abs(out_neuron_imp_full[i] - out_neuron_imp_f2[i]) / out_neuron_imp_full[i] * 100
        print(f"  {i:>4} | {out_neuron_imp_full[i]:>12.4f} | {out_neuron_imp_f2[i]:>12.4f} | {diff_pct:>11.2f}%")

    # 相关系数
    corr = torch.corrcoef(torch.stack([out_neuron_imp_full, out_neuron_imp_f2]))[0, 1]
    print(f"\n两种方法的相关系数: {corr:.4f}")
    print("(越接近1说明两种方法排序越一致)")

    # ========== 可视化 ==========
    create_comparison_visualization(G, G_inv, w, A,
                                   out_neuron_imp_full, out_neuron_imp_f2,
                                   w_imp_full, w_imps_f2)

    # ========== 总结 ==========
    print("\n" + "="*80)
    print("总结：F2近似在哪一步？")
    print("="*80)

    print("""
🎯 F2近似发生在这一行代码：

    out_neuron_imp = w_imps / torch.diag(G_inv)
                               ^^^^^^^^^^^^^^^
                               只用对角元素！

📌 完整公式推导：

完整的OBS重要性（理论）:
    importance_i = w_i^T @ (A⊗G)^(-1) @ w_i

KFAC近似:
    (A⊗G)^(-1) ≈ A^(-1) ⊗ G^(-1)

完整KFAC（代码实现）:
    用 diag(A^(-1)) ⊗ diag(G^(-1)) 进一步近似
    → 忽略了A和G的非对角元素

F2近似（kfac_OBS_F2.py）:
    分子: 用完整的 A 计算 w² @ A
    分母: 只用 diag(G^(-1))
    → 假设 G 是对角矩阵（输出神经元之间无相关性）

💡 为什么叫 F2？
    F = A ⊗ G
    F2 特指 G 是对角矩阵的特殊情况
    → 计算复杂度大大降低
    → 适合大规模模型（如1024×1024的层）

✅ F2适用场景：
    - 输出神经元之间相关性弱
    - 需要快速剪枝
    - 内存受限

❌ F2可能不适合：
    - 输出神经元高度相关（如注意力机制）
    - 需要最精确的重要性估计
""")

def create_comparison_visualization(G, G_inv, w, A, imp_full, imp_f2, w_imp_full, w_imps_f2):
    """创建对比可视化"""
    fig = plt.figure(figsize=(18, 10))

    # 图1: G矩阵热图
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(G.numpy(), cmap='RdBu_r', aspect='auto')
    ax1.set_title('G Matrix (Full)\nHas off-diagonal elements', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 图2: G的对角矩阵
    ax2 = plt.subplot(2, 4, 2)
    G_diag_only = torch.diag(torch.diag(G))
    im2 = ax2.imshow(G_diag_only.numpy(), cmap='RdBu_r', aspect='auto')
    ax2.set_title('diag(G) Only\nF2 approximation', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 图3: G_inv矩阵
    ax3 = plt.subplot(2, 4, 3)
    im3 = ax3.imshow(G_inv.numpy(), cmap='RdBu_r', aspect='auto')
    ax3.set_title('G_inv (Full)', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 图4: G_inv对角元素
    ax4 = plt.subplot(2, 4, 4)
    ax4.bar(range(len(G_inv)), torch.diag(G_inv).numpy(), color='steelblue', edgecolor='black')
    ax4.set_title('diag(G_inv)\nUsed in F2', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Neuron Index')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)

    # 图5: 完整KFAC的w_imp
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.imshow(w_imp_full.numpy(), cmap='viridis', aspect='auto')
    ax5.set_title('Full KFAC: w_imp [out, in]\nPer-weight importance', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Input Dim')
    ax5.set_ylabel('Output Dim')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # 图6: F2的w_imps (一维)
    ax6 = plt.subplot(2, 4, 6)
    ax6.bar(range(len(w_imps_f2)), w_imps_f2.numpy(), color='coral', edgecolor='black')
    ax6.set_title('F2: w_imps [out]\nPer-neuron (before /diag(G_inv))', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Output Neuron')
    ax6.set_ylabel('Value')
    ax6.grid(True, alpha=0.3)

    # 图7: 重要性对比
    ax7 = plt.subplot(2, 4, 7)
    x = np.arange(len(imp_full))
    width = 0.35
    ax7.bar(x - width/2, imp_full.numpy(), width, label='Full KFAC', color='steelblue', edgecolor='black')
    ax7.bar(x + width/2, imp_f2.numpy(), width, label='F2', color='coral', edgecolor='black')
    ax7.set_title('Importance Comparison', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Output Neuron')
    ax7.set_ylabel('Importance Score')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 图8: 散点图
    ax8 = plt.subplot(2, 4, 8)
    ax8.scatter(imp_full.numpy(), imp_f2.numpy(), s=100, alpha=0.7, edgecolors='black')
    ax8.plot([imp_full.min(), imp_full.max()], [imp_full.min(), imp_full.max()],
             'r--', linewidth=2, label='y=x (perfect match)')
    ax8.set_xlabel('Full KFAC Importance', fontsize=10)
    ax8.set_ylabel('F2 Importance', fontsize=10)
    ax8.set_title('Correlation Plot', fontsize=11, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 计算相关系数
    corr = np.corrcoef(imp_full.numpy(), imp_f2.numpy())[0, 1]
    ax8.text(0.05, 0.95, f'Correlation: {corr:.4f}',
             transform=ax8.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('f2_approximation_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: f2_approximation_comparison.png")
    plt.close()

if __name__ == "__main__":
    explain_f2_approximation()
