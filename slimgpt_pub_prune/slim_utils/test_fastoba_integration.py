"""
快速测试FastOBA集成

这个脚本使用较小的参数快速验证FastOBA集成是否正常工作
"""

import torch
from compare_pruning_methods import ComparisonExperiment

def main():
    print("=" * 80)
    print("FastOBA集成快速测试")
    print("=" * 80)
    print()

    # 创建较小规模的实验以加快速度
    experiment = ComparisonExperiment(
        hidden_size=768,
        num_heads=12,
        batch_size=4,       # 减小batch size
        seq_len=32,         # 减小序列长度
        num_batches=5,      # 减少batch数量
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 只测试一个稀疏度
    sparsity_levels = [0.25]

    print("测试配置:")
    print(f"  Hidden size: {experiment.hidden_size}")
    print(f"  Num heads: {experiment.num_heads}")
    print(f"  Batch size: {experiment.batch_size}")
    print(f"  Seq len: {experiment.seq_len}")
    print(f"  Num batches: {experiment.num_batches}")
    print(f"  Sparsity: {sparsity_levels[0]:.1%}")
    print(f"  Device: {experiment.device}")
    print()

    print("=" * 80)
    print("测试1: SlimGPT Attention对比 (原有功能)")
    print("=" * 80)
    print()

    try:
        results_slimgpt = experiment.run_comparison(
            sparsity=0.25,
            test_type='attention',
            use_fastoba=False
        )
        print("\n✓ SlimGPT对比测试通过")
        print(f"  Head剪枝误差: {results_slimgpt['head_prune']['relative_error']:.6e}")
        print(f"  Head-dim剪枝误差: {results_slimgpt['head_dim_prune']['relative_error']:.6e}")
    except Exception as e:
        print(f"\n✗ SlimGPT对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("测试2: FastOBA Attention对比 (新功能)")
    print("=" * 80)
    print()

    try:
        results_fastoba = experiment.run_comparison(
            sparsity=0.25,
            test_type='attention',
            use_fastoba=True
        )
        print("\n✓ FastOBA对比测试通过")
        print(f"  SlimGPT Head误差: {results_fastoba['slimgpt_head']['relative_error']:.6e}")
        print(f"  SlimGPT Head-dim误差: {results_fastoba['slimgpt_headdim']['relative_error']:.6e}")
        print(f"  FastOBA Head误差: {results_fastoba['fastoba_head']['relative_error']:.6e}")
        print(f"  FastOBA Head-dim误差: {results_fastoba['fastoba_headdim']['relative_error']:.6e}")

        # 对比分析
        print("\n对比分析:")
        best_slimgpt = min(
            results_fastoba['slimgpt_head']['relative_error'],
            results_fastoba['slimgpt_headdim']['relative_error']
        )
        best_fastoba = min(
            results_fastoba['fastoba_head']['relative_error'],
            results_fastoba['fastoba_headdim']['relative_error']
        )
        improvement = (best_slimgpt - best_fastoba) / best_slimgpt * 100

        print(f"  最佳SlimGPT误差: {best_slimgpt:.6e}")
        print(f"  最佳FastOBA误差: {best_fastoba:.6e}")
        print(f"  FastOBA相对改进: {improvement:+.2f}%")

    except Exception as e:
        print(f"\n✗ FastOBA对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！FastOBA集成成功！")
    print("=" * 80)
    print()
    print("下一步:")
    print("  运行完整实验: python compare_pruning_methods.py")
    print("  选择选项2 (FastOBA) 或选项3 (两者对比)")
    print()


if __name__ == "__main__":
    main()
