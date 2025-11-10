"""
VAR Per-Stage Head Analysis Experiment

Analyzes how different VAR scales (1×1 to 16×16) affect:
1. Head importance rankings
2. Head dimension selection for pruning
3. Scale-specific head/dimension identification

Model: VAR-d16
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sobs.fastoba_attention_slimgpt import FastOBAAttentionSlimGPT
from models import VAR, VQVAE


def load_var_model(model_path: str, device: str = 'cuda'):
    """Load VAR-d16 model"""
    print(f"Loading model from {model_path}")

    # Load VAR model
    vae_ckpt = torch.load(model_path, map_location='cpu')
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16))
    vae.load_state_dict(vae_ckpt['state_dict'], strict=True)
    vae.eval()

    # Load transformer
    var = VAR(vae=vae, num_classes=1000, depth=16, embed_dim=1024,
             num_heads=16, drop_rate=0.0, attn_drop_rate=0.0,
             drop_path_rate=0.0)

    if os.path.exists(model_path.replace('vae', 'var')):
        var_ckpt = torch.load(model_path.replace('vae', 'var'), map_location='cpu')
        var.load_state_dict(var_ckpt['state_dict'], strict=True)

    var.eval()
    var.to(device)

    return var, vae


def collect_stage_data(model, dataloader, layer_idx: int, num_samples: int = 128):
    """
    Collect Attention input/output for each stage

    Returns:
        stage_data: Dict[stage_id, List[Dict]]
    """
    print(f"Collecting data for layer {layer_idx}...")

    stage_data = {s: [] for s in range(10)}
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    model.eval()
    device = next(model.parameters()).device

    # Hook to capture attention I/O
    layer = model.blocks[layer_idx]
    attention = layer.attn

    captured_data = {}

    def forward_hook(module, inp, out):
        captured_data['inp'] = inp[0].detach()
        captured_data['out'] = out.detach()

    handle = attention.register_forward_hook(forward_hook)

    count = 0
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Collecting")):
        if count >= num_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            # VAR forward (autoregressive generation)
            # For each stage, capture the attention I/O
            for stage_id, pn in enumerate(patch_nums):
                # Forward through this stage
                # This requires modifying VAR's forward to return intermediate states
                # For now, we'll use a simplified approach

                # Generate tokens for this stage
                if stage_id == 0:
                    inp_seq = model.class_emb(labels).unsqueeze(1)  # [B, 1, C]
                else:
                    # Use tokens from previous stages
                    pass

                # Forward through transformer layer
                _ = layer(inp_seq)

                if 'inp' in captured_data and 'out' in captured_data:
                    stage_data[stage_id].append({
                        'inp': captured_data['inp'].cpu(),
                        'out': captured_data['out'].cpu(),
                        'stage_id': stage_id,
                        'seqlen': captured_data['inp'].shape[1]
                    })

        count += 1

    handle.remove()

    return stage_data


def experiment_per_stage_head_analysis(model, dataloader, layer_idx: int,
                                       num_heads: int = 16,
                                       embed_dim: int = 1024,
                                       num_samples: int = 128,
                                       output_dir: str = 'results'):
    """
    Main experiment: analyze head importance across scales

    Returns:
        results: Dict[stage_id, Dict]
    """
    print(f"\n=== Per-Stage Head Analysis for Layer {layer_idx} ===\n")

    # Initialize pruner
    attention_module = model.blocks[layer_idx].attn
    pruner = FastOBAAttentionSlimGPT(
        attention_module=attention_module,
        layer_idx=layer_idx,
        num_heads=num_heads,
        embed_dim=embed_dim,
        hessian_mode='block_diagonal',
        head_importance_mode='block_mean',
        fastoba_order=2,
        fastoba_delta=1.0,
        hessian_accumulate_freq=10
    )

    # Collect stage data
    stage_data = collect_stage_data(model, dataloader, layer_idx, num_samples)

    # Feed data to pruner (with stage-aware caching)
    print("\nFeeding data to pruner...")
    for stage_id in range(10):
        for entry in tqdm(stage_data[stage_id], desc=f"Stage {stage_id}"):
            pruner.add_batch_v7_fastoba(
                inp=entry['inp'],
                out=entry['out'],
                stage_id=stage_id
            )

    # Compute per-stage Hessian and importance
    print("\nComputing per-stage Hessian and importance...")
    results = {}
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    for stage_id in range(10):
        print(f"  Stage {stage_id} ({patch_nums[stage_id]}×{patch_nums[stage_id]})...")

        # Compute Hessian
        H_stage = pruner.compute_per_stage_hessian(stage_id)
        if H_stage is None:
            print(f"    Warning: No data for stage {stage_id}")
            continue

        # Compute head importance
        head_imp = pruner.compute_per_stage_head_importance(stage_id)

        # Compute head dimension importance
        dim_imp = pruner.compute_per_stage_head_dim_importance(stage_id)

        # Ranking
        head_ranking = torch.argsort(head_imp, descending=True)

        results[f'stage_{stage_id}'] = {
            'patch_size': patch_nums[stage_id],
            'token_count': patch_nums[stage_id] ** 2,
            'head_importance': head_imp.cpu().numpy(),
            'head_ranking': head_ranking.cpu().numpy(),
            'dim_importance': dim_imp.cpu().numpy(),  # [num_heads, head_dim]
        }

    return results


def analyze_head_ranking_consistency(results):
    """
    Analyze head ranking consistency across scales using Kendall's tau

    Returns:
        correlation_matrix: [num_stages, num_stages]
    """
    print("\n=== Head Ranking Consistency Analysis ===\n")

    num_stages = len(results)
    correlation_matrix = np.zeros((num_stages, num_stages))

    for si in range(num_stages):
        for sj in range(num_stages):
            if f'stage_{si}' not in results or f'stage_{sj}' not in results:
                continue

            ranking_i = results[f'stage_{si}']['head_ranking']
            ranking_j = results[f'stage_{sj}']['head_ranking']

            tau, p_value = kendalltau(ranking_i, ranking_j)
            correlation_matrix[si, sj] = tau

            if si < sj:
                print(f"Stage {si} ({results[f'stage_{si}']['patch_size']}×{results[f'stage_{si}']['patch_size']}) "
                      f"vs Stage {sj} ({results[f'stage_{sj}']['patch_size']}×{results[f'stage_{sj}']['patch_size']}): "
                      f"τ={tau:.3f}, p={p_value:.4f}")

    return correlation_matrix


def analyze_head_dim_selection_overlap(results, sparsity: float = 0.3):
    """
    Analyze head dimension selection overlap using Jaccard similarity

    Prints overlap statistics for each head across scales
    """
    print(f"\n=== Head Dimension Selection Overlap (Sparsity={sparsity}) ===\n")

    num_stages = len(results)
    if num_stages == 0:
        return

    num_heads = results['stage_0']['dim_importance'].shape[0]
    head_dim = results['stage_0']['dim_importance'].shape[1]

    # For each stage, determine which dimensions to prune per head
    stage_pruned_dims = {}
    for stage_id in range(num_stages):
        if f'stage_{stage_id}' not in results:
            continue

        dim_imp = results[f'stage_{stage_id}']['dim_importance']
        n_pruned_per_head = int(head_dim * sparsity)

        pruned_dims_per_head = []
        for h in range(num_heads):
            head_imp = dim_imp[h]
            pruned_idx = np.argsort(head_imp)[:n_pruned_per_head]
            pruned_dims_per_head.append(set(pruned_idx))

        stage_pruned_dims[stage_id] = pruned_dims_per_head

    # Compute Jaccard similarity
    for head_id in range(num_heads):
        print(f"\nHead {head_id}:")
        for si in range(num_stages):
            if si not in stage_pruned_dims:
                continue
            for sj in range(si + 1, num_stages):
                if sj not in stage_pruned_dims:
                    continue

                set_i = stage_pruned_dims[si][head_id]
                set_j = stage_pruned_dims[sj][head_id]

                jaccard = len(set_i & set_j) / len(set_i | set_j) if len(set_i | set_j) > 0 else 0

                print(f"  Stage {si} vs Stage {sj}: Jaccard={jaccard:.3f} "
                      f"(共同删除 {len(set_i & set_j)}/{len(set_i)} 维)")


def find_scale_specific_heads(results, top_k: int = 3):
    """
    Find heads that are important in one scale but not others

    Returns:
        scale_specific: Dict[key, Dict]
    """
    print(f"\n=== Scale-Specific Heads (top-{top_k}) ===\n")

    num_stages = len(results)
    scale_specific = {}

    for si in range(num_stages):
        if f'stage_{si}' not in results:
            continue

        ranking_si = results[f'stage_{si}']['head_ranking']
        top_heads = set(ranking_si[:top_k])

        for head_id in top_heads:
            # Compute average rank in other stages
            ranks_in_other = []
            for sj in range(num_stages):
                if sj != si and f'stage_{sj}' in results:
                    ranking_sj = results[f'stage_{sj}']['head_ranking']
                    rank = np.where(ranking_sj == head_id)[0][0]
                    ranks_in_other.append(rank)

            if len(ranks_in_other) == 0:
                continue

            avg_rank_other = np.mean(ranks_in_other)
            rank_in_si = np.where(ranking_si == head_id)[0][0]

            # Scale-specific if ranked high in current scale but low in others
            num_heads = len(ranking_si)
            if avg_rank_other > num_heads // 2:  # Average rank in bottom half
                key = f'stage_{si}_head_{head_id}'
                scale_specific[key] = {
                    'stage_id': si,
                    'patch_size': results[f'stage_{si}']['patch_size'],
                    'head_id': int(head_id),
                    'rank_in_stage': int(rank_in_si),
                    'avg_rank_other': float(avg_rank_other),
                    'importance': float(results[f'stage_{si}']['head_importance'][head_id])
                }

                print(f"Stage {si} Head {head_id}: rank={rank_in_si}, "
                      f"avg_other_rank={avg_rank_other:.1f}, "
                      f"importance={scale_specific[key]['importance']:.4f}")

    return scale_specific


def visualize_results(results, correlation_matrix, output_dir: str = 'results'):
    """Generate visualizations"""
    print(f"\n=== Generating Visualizations ===\n")

    os.makedirs(output_dir, exist_ok=True)

    num_stages = len(results)
    if num_stages == 0:
        print("No results to visualize")
        return

    num_heads = results['stage_0']['head_importance'].shape[0]

    # 1. Head importance heatmap (10 stages × num_heads)
    print("  Generating head importance heatmap...")
    fig, ax = plt.subplots(figsize=(14, 8))
    importance_matrix = np.zeros((num_stages, num_heads))
    for si in range(num_stages):
        if f'stage_{si}' in results:
            importance_matrix[si, :] = results[f'stage_{si}']['head_importance']

    sns.heatmap(importance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'H{i}' for i in range(num_heads)],
                yticklabels=[f'S{si}({results[f"stage_{si}"]["patch_size"]}²)'
                            if f'stage_{si}' in results else f'S{si}'
                            for si in range(num_stages)],
                ax=ax)
    ax.set_title('Head Importance by Stage (FastOBA Block-Diagonal Hessian)')
    ax.set_xlabel('Head ID')
    ax.set_ylabel('Stage (Resolution)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'head_importance_by_stage.png'), dpi=300)
    plt.close()

    # 2. Head ranking correlation matrix
    print("  Generating correlation matrix...")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, center=0,
                xticklabels=[f'S{i}' for i in range(num_stages)],
                yticklabels=[f'S{i}' for i in range(num_stages)],
                ax=ax)
    ax.set_title("Head Ranking Correlation Between Stages (Kendall's τ)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage_head_ranking_correlation.png'), dpi=300)
    plt.close()

    # 3. Per-head dimension importance (for first 4 heads)
    print("  Generating per-head dimension importance plots...")
    head_dim = results['stage_0']['dim_importance'].shape[1]
    for head_id in range(min(4, num_heads)):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for si in range(num_stages):
            if f'stage_{si}' not in results:
                continue

            ax = axes[si // 5, si % 5]
            dim_imp = results[f'stage_{si}']['dim_importance'][head_id]

            ax.bar(range(head_dim), dim_imp)
            ax.set_title(f'Stage {si} ({results[f"stage_{si}"]["patch_size"]}²)')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Importance')

        plt.suptitle(f'Head {head_id} Dimension Importance Across Stages')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'head_{head_id}_dim_importance_per_stage.png'), dpi=300)
        plt.close()

    print(f"\nVisualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='VAR Per-Stage Head Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to VAR model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--layer_idx', type=int, default=0,
                       help='Layer index to analyze')
    parser.add_argument('--num_samples', type=int, default=128,
                       help='Number of samples for analysis')
    parser.add_argument('--output_dir', type=str, default='results/per_stage_analysis',
                       help='Output directory for results')
    parser.add_argument('--sparsity', type=float, default=0.4,
                       help='Target sparsity for overlap analysis (default: 0.4 = 40%)')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, vae = load_var_model(args.model_path, device)

    # Load data (placeholder - adjust based on actual data loading)
    from torch.utils.data import DataLoader, TensorDataset
    # For testing, create dummy data
    dummy_images = torch.randn(args.num_samples, 3, 256, 256)
    dummy_labels = torch.randint(0, 1000, (args.num_samples,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Run experiment
    results = experiment_per_stage_head_analysis(
        model=model,
        dataloader=dataloader,
        layer_idx=args.layer_idx,
        num_heads=16,  # VAR-d16 default
        embed_dim=1024,  # VAR-d16 default
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # Save raw results
    results_json = {}
    for key, val in results.items():
        results_json[key] = {
            'patch_size': val['patch_size'],
            'token_count': val['token_count'],
            'head_importance': val['head_importance'].tolist(),
            'head_ranking': val['head_ranking'].tolist(),
        }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Analysis
    correlation_matrix = analyze_head_ranking_consistency(results)
    analyze_head_dim_selection_overlap(results, sparsity=args.sparsity)
    scale_specific = find_scale_specific_heads(results, top_k=3)

    # Save scale-specific heads
    with open(os.path.join(args.output_dir, 'scale_specific_heads.json'), 'w') as f:
        json.dump(scale_specific, f, indent=2)

    # Visualizations
    visualize_results(results, correlation_matrix, args.output_dir)

    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
