"""
Head维度剪枝使用示例

这个文件展示如何使用 head_dim_prune 方法对attention层进行维度剪枝
"""

import torch
import torch.nn as nn
from slimgpt import SlimGPT

# ========== 示例1：基本使用 ==========

def example_basic():
    """
    基本使用示例：对一个线性层进行head_dim剪枝
    """
    print("=== 示例1：基本使用 ===\n")

    # 创建一个模拟的attention投影层
    # 假设：12个head，每个64维，共768维
    in_features = 768
    out_features = 768
    layer = nn.Linear(in_features, out_features)

    # 创建模拟的输入数据用于统计Hessian
    batch_size = 32
    seq_len = 128
    hidden_size = 768

    # 创建SlimGPT pruner（需要args对象）
    class Args:
        no_compensate = False  # 启用误差补偿

    args = Args()
    pruner = SlimGPT(layer, layer_idx=0, args=args)

    # 添加数据batch来统计Hessian矩阵
    print("统计Hessian矩阵...")
    for i in range(10):  # 使用10个batch
        inp = torch.randn(batch_size, seq_len, hidden_size)
        out = layer(inp)
        pruner.add_batch(inp, out)

    # 执行head_dim剪枝
    # sparsity=0.25 表示每个head删除25%的维度
    # headsize=64 表示每个head有64维
    print("\n执行head_dim剪枝（每个head删除25%维度）...")
    pruned_indices = pruner.head_dim_prune(
        sparsity=0.25,
        headsize=64,
        percdamp=0.01,
        layer_idx=0
    )

    print(f"\n剪枝完成！")
    print(f"删除的维度索引: {pruned_indices.tolist()[:10]}... (共{len(pruned_indices)}个)")
    print(f"原始维度: 12 heads × 64 dim = 768")
    print(f"剪枝后: 12 heads × 48 dim = 576")
    print(f"压缩率: {len(pruned_indices) / 768 * 100:.1f}%")

    # 验证权重形状不变（只是部分列被清零）
    print(f"\n权重形状: {layer.weight.shape}")
    print(f"非零元素比例: {(layer.weight != 0).float().mean():.2%}")


# ========== 示例2：与struct_prune对比 ==========

def example_comparison():
    """
    对比示例：head_dim_prune vs struct_prune
    """
    print("\n\n=== 示例2：head_dim_prune vs struct_prune ===\n")

    # 创建两个相同的层
    layer_head_prune = nn.Linear(768, 768)
    layer_dim_prune = nn.Linear(768, 768)

    # 复制权重使其相同
    layer_dim_prune.load_state_dict(layer_head_prune.state_dict())

    class Args:
        no_compensate = False
    args = Args()

    # 创建两个pruner
    pruner_head = SlimGPT(layer_head_prune, layer_idx=0, args=args)
    pruner_dim = SlimGPT(layer_dim_prune, layer_idx=1, args=args)

    # 使用相同的数据
    torch.manual_seed(42)
    for i in range(10):
        inp = torch.randn(32, 128, 768)
        out_head = layer_head_prune(inp)
        out_dim = layer_dim_prune(inp)
        pruner_head.add_batch(inp, out_head)
        pruner_dim.add_batch(inp, out_dim)

    # 方法1：struct_prune（删除完整head）
    print("方法1: struct_prune（删除完整head）")
    print("  - 删除25%的head（12 → 9）")
    print("  - 每个head保持64维")
    pruned_head = pruner_head.struct_prune(
        sparsity=0.25,
        headsize=64,
        percdamp=0.01,
        layer_idx=0
    )
    print(f"  - 删除了 {len(pruned_head)} 个维度")
    print(f"  - 最终: 9 heads × 64 dim = 576 维")

    # 方法2：head_dim_prune（减少head维度）
    print("\n方法2: head_dim_prune（减少每个head的维度）")
    print("  - 保持12个head")
    print("  - 每个head删除25%维度（64 → 48）")
    pruned_dim = pruner_dim.head_dim_prune(
        sparsity=0.25,
        headsize=64,
        percdamp=0.01,
        layer_idx=1
    )
    print(f"  - 删除了 {len(pruned_dim)} 个维度")
    print(f"  - 最终: 12 heads × 48 dim = 576 维")

    # 对比
    print("\n=== 对比总结 ===")
    print(f"{'特性':<20} {'struct_prune':<20} {'head_dim_prune':<20}")
    print("-" * 60)
    print(f"{'删除维度数':<20} {len(pruned_head):<20} {len(pruned_dim):<20}")
    print(f"{'head数量':<20} {'9':<20} {'12':<20}")
    print(f"{'每个head维度':<20} {'64':<20} {'48':<20}")
    print(f"{'最终总维度':<20} {'576':<20} {'576':<20}")
    print(f"{'个性化剪枝':<20} {'否':<20} {'是':<20}")
    print(f"{'Global Update':<20} {'需要':<20} {'不需要':<20}")


# ========== 示例3：不同稀疏度 ==========

def example_different_sparsity():
    """
    不同稀疏度的效果
    """
    print("\n\n=== 示例3：不同稀疏度的效果 ===\n")

    sparsity_levels = [0.125, 0.25, 0.375, 0.5]

    print(f"{'稀疏度':<15} {'每head删除':<15} {'每head剩余':<15} {'总维度':<15}")
    print("-" * 60)

    for sparsity in sparsity_levels:
        headsize = 64
        dims_to_remove = round(headsize * sparsity)
        dims_remaining = headsize - dims_to_remove
        total_dims = 12 * dims_remaining

        print(f"{sparsity:<15.1%} {dims_to_remove:<15} {dims_remaining:<15} {total_dims:<15}")


# ========== 示例4：实际应用场景 ==========

def example_real_world():
    """
    实际应用场景：处理Transformer模型的QKV投影
    """
    print("\n\n=== 示例4：实际应用场景 ===\n")
    print("场景：对Transformer的QKV投影层进行head_dim剪枝")
    print()

    # 模拟一个Transformer层的attention
    class MockAttention(nn.Module):
        def __init__(self, hidden_size=768, num_heads=12):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads

            # QKV投影
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

    attention = MockAttention()

    print(f"原始配置:")
    print(f"  - hidden_size: 768")
    print(f"  - num_heads: 12")
    print(f"  - head_dim: 64")
    print()

    # 对每个投影层进行剪枝
    projections = ['q_proj', 'k_proj', 'v_proj']
    sparsity = 0.25  # 每个head删除25%维度

    class Args:
        no_compensate = False
    args = Args()

    print(f"执行head_dim剪枝（sparsity={sparsity}）:")

    for proj_name in projections:
        layer = getattr(attention, proj_name)
        pruner = SlimGPT(layer, layer_idx=0, args=args)

        # 模拟数据统计
        for i in range(5):
            inp = torch.randn(16, 64, 768)
            out = layer(inp)
            pruner.add_batch(inp, out)

        # 剪枝
        pruned = pruner.head_dim_prune(
            sparsity=sparsity,
            headsize=64,
            percdamp=0.01,
            layer_idx=0
        )

        print(f"  - {proj_name}: 删除 {len(pruned)} 维")

    print()
    print("剪枝后配置:")
    print(f"  - hidden_size: 768（权重形状不变）")
    print(f"  - num_heads: 12（head数量不变）")
    print(f"  - head_dim: 48（每个head有效维度）")
    print(f"  - 实际计算维度: 576")
    print()
    print("优势:")
    print("  ✓ 保持多头结构")
    print("  ✓ 每个head保留最重要的维度")
    print("  ✓ reshape兼容（可直接使用原有代码）")


# ========== 主函数 ==========

if __name__ == "__main__":
    print("=" * 70)
    print("Head维度剪枝 (head_dim_prune) 使用示例")
    print("=" * 70)

    try:
        # 运行示例1
        example_basic()

        # 运行示例2
        example_comparison()

        # 运行示例3
        example_different_sparsity()

        # 运行示例4
        example_real_world()

        print("\n" + "=" * 70)
        print("所有示例运行完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
