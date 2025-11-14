#!/usr/bin/env python3
"""
Per-head Attention Sharpness Analysis (Full Sequence, Real Images)

Requirements from user:
- Use attention weights (post-softmax)
- Use the full sequence (VAR teacher forcing → L ≈ 680), not only last scale
- Data source: real ImageNet images (refer to model_slimming_basic.py loader)
- Per-layer KMeans clustering (k=2) on per-head metrics to label heads as
  'sharp'(尖锐) vs 'smooth'(平缓)
- Produce heatmaps + per-layer mean/variance of metrics

Implementation notes:
- Reuse utilities from verify_scale_concentration.py:
  * load_var_model, prepare_calibration_data
  * AttentionMapExtractor (captures attention weights via slow path)
  * compute_attention_entropy, compute_effective_span, compute_gini_coefficient
- Outputs under head_sharpness_images/ by default:
  * metrics_per_layer.json: per-layer per-head metrics + labels
  * labels_heatmap.png, entropy_heatmap.png, span_heatmap.png, gini_heatmap.png
  * layer_stats.csv: per-layer mean/var for entropy/span/gini

CLI example:
  python analyze_head_sharpness_from_images.py \
      --model_depth 16 \
      --num_samples 32 \
      --imagenet_dir /home/project/ImageNet-1K \
      --layers all \
      --plots

"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def kmeans_per_layer(ent, span, gini):
    """Layerwise KMeans(k=2) on standardized features.

    Returns:
      labels: np.ndarray [H], 1=sharp (low entropy/span, high gini), 0=smooth
    """
    feats = np.stack([ent, span, gini], axis=1)

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        km = KMeans(n_clusters=2, n_init=20, random_state=42)
        km.fit(X)
        labels = km.labels_.astype(int)
        centers = scaler.inverse_transform(km.cluster_centers_)
    except Exception:
        # Fallback: simple numpy k-means (k=2)
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-8
        X = (feats - mu) / sd
        idx_sorted = np.argsort(X[:, 0])
        c0 = X[idx_sorted[: max(1, len(X)//4)]].mean(axis=0)
        c1 = X[idx_sorted[-max(1, len(X)//4):]].mean(axis=0)
        C = np.stack([c0, c1], axis=0)
        for _ in range(50):
            d = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
            lab = d.argmin(axis=1)
            newC = np.stack([
                X[lab == 0].mean(axis=0) if np.any(lab == 0) else C[0],
                X[lab == 1].mean(axis=0) if np.any(lab == 1) else C[1],
            ], axis=0)
            if np.allclose(newC, C):
                break
            C = newC
        centers = C * sd + mu
        labels = lab.astype(int)

    # Map cluster to sharp via score S = -entropy - span + gini
    (c0_e, c0_s, c0_g), (c1_e, c1_s, c1_g) = centers
    c0_score = -c0_e - c0_s + c0_g
    c1_score = -c1_e - c1_s + c1_g
    sharp_cluster = 0 if c0_score > c1_score else 1
    labels = (labels == sharp_cluster).astype(int)
    return labels


def main():
    ap = argparse.ArgumentParser(description="Per-head attention sharpness (full sequence, real images)")
    ap.add_argument('--model_depth', type=int, default=16)
    ap.add_argument('--num_samples', type=int, default=32)
    ap.add_argument('--layers', type=str, default='all', help="'all' or CSV of layer indices")
    ap.add_argument('--imagenet_dir', type=str, default='/home/project/ImageNet-1K')
    ap.add_argument('--vae_ckpt', type=str, default='')
    ap.add_argument('--var_ckpt', type=str, default='')
    ap.add_argument('--out_dir', type=str, default='head_sharpness_images')
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()

    # Heavy imports after arg parsing
    from verify_scale_concentration import (
        load_var_model,
        prepare_calibration_data,
        AttentionMapExtractor,
        compute_attention_entropy,
        compute_effective_span,
        compute_gini_coefficient,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    vae_ckpt = args.vae_ckpt or f'/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt or f'/home/project/daily/AR/model_zoo/var_d{args.model_depth}.pth'
    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device=device)

    # Prepare real ImageNet tokenized inputs
    labels, tokens = prepare_calibration_data(
        vae, num_samples=args.num_samples, imagenet_dir=args.imagenet_dir, device=device
    )

    # Select layers
    if args.layers == 'all':
        test_layers = list(range(len(var.blocks)))
    else:
        test_layers = [int(x.strip()) for x in args.layers.split(',')]

    # Storage for heatmaps (LxH)
    # Heads per layer assumed constant; we infer from first layer after extraction
    ENT = []
    SPN = []
    GIN = []
    LAB = []

    per_layer_records = []

    for layer_idx in test_layers:
        print(f"\n[Layer {layer_idx}] Extracting attention weights (full sequence, teacher forcing)...")
        extractor = AttentionMapExtractor(var, layer_idx)
        # We want teacher forcing on pre-encoded tokens to ensure full sequence.
        attn = extractor.extract(inputs=labels, use_teacher_forcing=True, input_tokens=tokens)
        extractor.restore()

        # attn: [B, H, L, L] on CPU
        print(f"  captured: {tuple(attn.shape)} (B,H,L,L)")

        # Compute metrics per head
        ent = compute_attention_entropy(attn).cpu().numpy()
        spn = compute_effective_span(attn).cpu().numpy()
        gin = compute_gini_coefficient(attn).cpu().numpy()

        # KMeans per layer
        labels_h = kmeans_per_layer(ent, spn, gin)

        # Persist
        ENT.append(ent)
        SPN.append(spn)
        GIN.append(gin)
        LAB.append(labels_h)

        # Per-layer stats
        rec = {
            'layer': layer_idx,
            'num_heads': int(len(ent)),
            'metrics': {
                'entropy': {'mean': float(np.mean(ent)), 'var': float(np.var(ent))},
                'span': {'mean': float(np.mean(spn)), 'var': float(np.var(spn))},
                'gini': {'mean': float(np.mean(gin)), 'var': float(np.var(gin))},
            },
            'labels': labels_h.tolist(),
            'raw': {
                'entropy': ent.tolist(),
                'span': spn.tolist(),
                'gini': gin.tolist(),
            }
        }
        per_layer_records.append(rec)

    ENT = np.stack(ENT, axis=0)
    SPN = np.stack(SPN, axis=0)
    GIN = np.stack(GIN, axis=0)
    LAB = np.stack(LAB, axis=0)

    # Save JSON
    out_json = {
        'layers': per_layer_records,
        'shapes': {
            'entropy': ENT.shape,
            'span': SPN.shape,
            'gini': GIN.shape,
            'labels': LAB.shape,
        }
    }
    (out_dir / 'metrics_per_layer.json').write_text(json.dumps(out_json, ensure_ascii=False, indent=2))

    # Save per-layer mean/var CSV
    with (out_dir / 'layer_stats.csv').open('w') as f:
        f.write('layer,ent_mean,ent_var,span_mean,span_var,gini_mean,gini_var,sharp_count\n')
        for rec in per_layer_records:
            l = rec['layer']
            em = rec['metrics']['entropy']['mean']
            ev = rec['metrics']['entropy']['var']
            sm = rec['metrics']['span']['mean']
            sv = rec['metrics']['span']['var']
            gm = rec['metrics']['gini']['mean']
            gv = rec['metrics']['gini']['var']
            sharp = int(sum(rec['labels']))
            f.write(f"{l},{em:.6f},{ev:.6f},{sm:.6f},{sv:.6f},{gm:.6f},{gv:.6f},{sharp}\n")

    # Optional plots
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')  # headless backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        def heatmap(mat, title, fname, cmap='viridis', vmin=None, vmax=None):
            plt.figure(figsize=(12, 6))
            sns.heatmap(mat, cmap=cmap, vmin=vmin, vmax=vmax, cbar=True,
                        linewidths=0.2, linecolor='gray')
            plt.xlabel('Head Index')
            plt.ylabel('Layer Index')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=180)
            plt.close()

        # Heatmaps for metrics
        heatmap(ENT, 'Entropy per Head (lower = sharper)', 'entropy_heatmap.png', cmap='magma')
        heatmap(SPN, 'Effective Span per Head (lower = sharper)', 'span_heatmap.png', cmap='magma')
        heatmap(GIN, 'Gini per Head (higher = sharper)', 'gini_heatmap.png', cmap='magma')

        # Labels heatmap (binary)
        plt.figure(figsize=(12, 6))
        sns.heatmap(LAB, cmap=sns.color_palette(['#3572A5', '#D62728']), vmin=0, vmax=1,
                    cbar=False, linewidths=0.2, linecolor='gray')
        plt.xlabel('Head Index')
        plt.ylabel('Layer Index')
        plt.title('Head Sharpness Labels (1=sharp/red, 0=smooth/blue)')
        plt.tight_layout()
        plt.savefig(out_dir / 'labels_heatmap.png', dpi=180)
        plt.close()

    print(f"\n✓ Saved metrics JSON to {out_dir}/metrics_per_layer.json")
    print(f"✓ Saved layer stats CSV to {out_dir}/layer_stats.csv")
    if args.plots:
        print(f"✓ Saved heatmaps to {out_dir}/")


if __name__ == '__main__':
    main()
