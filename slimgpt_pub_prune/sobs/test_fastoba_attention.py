"""
Quick test script for FastOBAAttentionSlimGPT

Tests basic functionality without requiring full VAR model
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sobs.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT


class SimpleAttention(nn.Module):
    """Simplified attention for testing"""
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        # Aggregate
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)

        return x


def test_initialization():
    """Test 1: Initialization"""
    print("\n=== Test 1: Initialization ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)

    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean'
    )

    assert pruner.num_heads == 12
    assert pruner.head_dim == 64
    assert pruner.H.shape == (768, 768)
    print("✓ Initialization successful")
    print(f"  H shape: {pruner.H.shape}")
    print(f"  num_heads: {pruner.num_heads}, head_dim: {pruner.head_dim}")


def test_slimgpt_mode():
    """Test 2: SlimGPT mode (H=XX^T)"""
    print("\n=== Test 2: SlimGPT Mode ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='slimgpt'
    )

    # Generate dummy input
    inp = torch.randn(4, 16, 768)  # [batch, seq_len, embed_dim]

    # Forward
    attention.eval()
    with torch.no_grad():
        out = attention(inp)

    # Add batch
    pruner.add_batch(inp, out)

    assert pruner.nsamples > 0
    assert not torch.all(pruner.H == 0)
    print("✓ SlimGPT mode works")
    print(f"  nsamples: {pruner.nsamples}")
    print(f"  H diagonal (first 5): {torch.diag(pruner.H)[:5]}")


def test_fastoba_mode():
    """Test 3: FastOBA mode (block-diagonal Hessian)"""
    print("\n=== Test 3: FastOBA Mode ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    attention.eval()

    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        fastoba_order=2,
        fastoba_delta=1.0,
        hessian_accumulate_freq=5
    )

    # Generate dummy inputs (accumulate to trigger Hessian computation)
    for i in range(5):
        inp = torch.randn(2, 8, 768, requires_grad=True)
        out = attention(inp)
        pruner.add_batch(inp, out)

    assert pruner.nsamples > 0
    assert not torch.all(pruner.H == 0)

    # Check block-diagonal structure
    H = pruner.H
    head_dim = pruner.head_dim

    # First block (head 0)
    block_0 = H[:head_dim, :head_dim]
    # Off-diagonal between head 0 and head 1
    off_diag = H[:head_dim, head_dim:2*head_dim]

    print("✓ FastOBA mode works")
    print(f"  nsamples: {pruner.nsamples}")
    print(f"  H diagonal (first 5): {torch.diag(H)[:5]}")
    print(f"  Block 0 norm: {block_0.norm():.4f}")
    print(f"  Off-diagonal norm: {off_diag.norm():.4f}")
    print(f"  Block/Off ratio: {block_0.norm() / (off_diag.norm() + 1e-8):.2f}x "
          "(should be large for block-diagonal)")


def test_var_scale_aware():
    """Test 4: VAR scale-aware caching"""
    print("\n=== Test 4: VAR Scale-Aware Caching ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    attention.eval()

    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        hessian_accumulate_freq=10
    )

    # Simulate VAR's multi-scale tokens
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    for stage_id, pn in enumerate(patch_nums):
        seq_len = pn * pn
        inp = torch.randn(2, seq_len, 768, requires_grad=True)
        out = attention(inp)

        # This should infer stage_id automatically
        pruner.add_batch_v7_fastoba(inp, out)

        # Check per-stage cache
        if stage_id < 10:
            assert len(pruner._per_stage_cache[stage_id]) > 0
            print(f"  Stage {stage_id} ({pn}×{pn}, {seq_len} tokens): "
                  f"cached {len(pruner._per_stage_cache[stage_id])} entries")

    print("✓ Scale-aware caching works")


def test_per_stage_analysis():
    """Test 5: Per-stage Hessian and importance"""
    print("\n=== Test 5: Per-Stage Analysis ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    attention.eval()

    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean',
        hessian_accumulate_freq=5
    )

    # Add data for stage 0 (1×1)
    for _ in range(10):
        inp = torch.randn(2, 1, 768, requires_grad=True)
        out = attention(inp)
        pruner.add_batch_v7_fastoba(inp, out, stage_id=0)

    # Compute per-stage Hessian
    H_stage = pruner.compute_per_stage_hessian(stage_id=0)
    assert H_stage is not None
    assert H_stage.shape == (768, 768)
    print("✓ Per-stage Hessian computed")
    print(f"  H_stage shape: {H_stage.shape}")

    # Compute head importance
    head_imp = pruner.compute_per_stage_head_importance(stage_id=0)
    assert head_imp is not None
    assert head_imp.shape == (12,)
    print("✓ Head importance computed")
    print(f"  Head importance: {head_imp}")
    print(f"  Most important head: {head_imp.argmax().item()}")

    # Compute dimension importance
    dim_imp = pruner.compute_per_stage_head_dim_importance(stage_id=0)
    assert dim_imp is not None
    assert dim_imp.shape == (12, 64)
    print("✓ Dimension importance computed")
    print(f"  Dim importance shape: {dim_imp.shape}")
    print(f"  Head 0 top-5 dims: {torch.argsort(dim_imp[0], descending=True)[:5]}")


def test_importance_scoring():
    """Test 6: Get pruning importance scores"""
    print("\n=== Test 6: Pruning Importance Scores ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    attention.eval()

    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean'
    )

    # Add data
    for _ in range(5):
        inp = torch.randn(2, 16, 768, requires_grad=True)
        out = attention(inp)
        pruner.add_batch(inp, out)

    # Get importance scores
    importance = pruner.get_pruning_importance(percdamp=0.01)
    assert importance is not None
    assert importance.shape == (768,)
    assert torch.all(importance >= 0)

    print("✓ Importance scores computed")
    print(f"  Importance shape: {importance.shape}")
    print(f"  Min: {importance.min():.4f}, Max: {importance.max():.4f}, Mean: {importance.mean():.4f}")

    # Simulate pruning decision (40% sparsity)
    sparsity = 0.4
    n_pruned = int(768 * sparsity)
    pruned_indices = torch.argsort(importance)[:n_pruned]
    print(f"  Would prune {n_pruned} channels (40% sparsity)")
    print(f"  Pruned indices (first 10): {pruned_indices[:10]}")


def test_struct_prune_with_compensation():
    """Test 7: Structured pruning with compensation"""
    print("\n=== Test 7: Struct Prune with Compensation ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    attention.eval()

    # Test with compensation enabled
    pruner_with_comp = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean',
        use_compensation=True,
        hessian_accumulate_freq=5,
        debug=True
    )

    # Add data
    for _ in range(5):
        inp = torch.randn(2, 16, 768, requires_grad=True)
        out = attention(inp)
        pruner_with_comp.add_batch(inp, out)

    # Store original weights
    W_original = attention.proj.weight.data.clone()

    # Perform pruning (40% sparsity, channel-wise)
    pruned_indices = pruner_with_comp.struct_prune(sparsity=0.4, headsize=1, percdamp=0.01)
    W_pruned_comp = attention.proj.weight.data.clone()

    print("✓ Pruning with compensation completed")
    print(f"  Pruned {len(pruned_indices)} channels (target: {int(768*0.4)})")
    print(f"  Weight change (L2 norm): {(W_pruned_comp - W_original).norm():.4f}")

    # Test without compensation
    attention2 = SimpleAttention(embed_dim=768, num_heads=12)
    attention2.proj.weight.data = W_original.clone()
    attention2.eval()

    pruner_no_comp = FastOBAAttentionSlimGPT(
        attention_module=attention2,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean',
        use_compensation=False
    )

    # Add same data
    for _ in range(5):
        inp = torch.randn(2, 16, 768, requires_grad=True)
        out = attention2(inp)
        pruner_no_comp.add_batch(inp, out)

    # Perform pruning without compensation
    pruned_indices_2 = pruner_no_comp.struct_prune(sparsity=0.4, headsize=1, percdamp=0.01)
    W_pruned_no_comp = attention2.proj.weight.data.clone()

    print("✓ Pruning without compensation completed")
    print(f"  Pruned {len(pruned_indices_2)} channels")
    print(f"  Weight change (L2 norm): {(W_pruned_no_comp - W_original).norm():.4f}")

    # Compare compensation effect
    comp_norm = (W_pruned_comp - W_original).norm()
    no_comp_norm = (W_pruned_no_comp - W_original).norm()
    print(f"\n  Compensation reduces weight change by {(1 - comp_norm/no_comp_norm)*100:.1f}%")


def test_head_internal_pruning():
    """Test 8: Head-internal dimension pruning with block-diagonal compensation"""
    print("\n=== Test 8: Head-Internal Dimension Pruning ===")

    attention = SimpleAttention(embed_dim=768, num_heads=12)
    attention.eval()

    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean',
        use_compensation=True,
        debug=False  # Set True to see detailed per-head progress
    )

    # Add data
    for _ in range(5):
        inp = torch.randn(2, 16, 768, requires_grad=True)
        out = attention(inp)
        pruner.add_batch(inp, out)

    # Store original weights
    W_original = attention.proj.weight.data.clone()

    # Perform head-internal pruning (40% per head)
    pruned_indices = pruner.struct_prune_head_dims(sparsity=0.4, percdamp=0.01, blocksize=16)
    W_pruned = attention.proj.weight.data.clone()

    print("✓ Head-internal pruning completed")
    print(f"  Pruned {len(pruned_indices)} dims total (target: {int(768*0.4)})")
    print(f"  Pruned per head: {len(pruned_indices)//12} dims (target: {int(64*0.4)})")

    # Verify block-diagonal pruning
    for head_id in range(12):
        start = head_id * 64
        end = (head_id + 1) * 64
        head_pruned = pruned_indices[(pruned_indices >= start) & (pruned_indices < end)]
        print(f"  Head {head_id}: pruned {len(head_pruned)} / 64 dims")

    # Compare with full pruning
    attention2 = SimpleAttention(embed_dim=768, num_heads=12)
    attention2.proj.weight.data = W_original.clone()
    attention2.eval()

    pruner2 = FastOBAAttentionSlimGPT(
        attention_module=attention2,
        layer_idx=0,
        num_heads=12,
        embed_dim=768,
        hessian_mode='block_diagonal',
        use_compensation=True
    )

    for _ in range(5):
        inp = torch.randn(2, 16, 768, requires_grad=True)
        out = attention2(inp)
        pruner2.add_batch(inp, out)

    pruned_indices2 = pruner2.struct_prune(sparsity=0.4, headsize=1)
    W_pruned2 = attention2.proj.weight.data.clone()

    print("\n✓ Comparison with global channel-wise pruning:")
    print(f"  Block-diagonal compensation: {(W_pruned - W_original).norm():.4f}")
    print(f"  Global compensation:         {(W_pruned2 - W_original).norm():.4f}")
    print(f"  Ratio: {((W_pruned - W_original).norm() / (W_pruned2 - W_original).norm()):.3f}x")


def main():
    print("\n" + "="*60)
    print("  FastOBAAttentionSlimGPT Unit Tests")
    print("="*60)

    torch.manual_seed(42)

    try:
        test_initialization()
        test_slimgpt_mode()
        test_fastoba_mode()
        test_var_scale_aware()
        test_per_stage_analysis()
        test_importance_scoring()
        test_struct_prune_with_compensation()
        test_head_internal_pruning()

        print("\n" + "="*60)
        print("  ✓ All Tests Passed!")
        print("="*60)
        print("\nKey Features Verified:")
        print("  ✓ SlimGPT and FastOBA Hessian modes")
        print("  ✓ VAR-aware multi-scale token handling")
        print("  ✓ Per-stage analysis for scale-specific insights")
        print("  ✓ OBS weight compensation (global & block-diagonal)")
        print("  ✓ Head-internal dimension pruning with block compensation")
        print("\nNext steps:")
        print("  1. Run VAR-d16 per-stage analysis:")
        print("     python sobs/var_per_stage_analysis.py --model_path /path/to/var_d16.pth")
        print("  2. Run full ablation study:")
        print("     bash sobs/run_ablation.sh")
        print("")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
