#!/usr/bin/env python3
"""
Compute and visualize equal-weight composite sharpness score S_w:
  S_w = mean( -z(entropy), -z(span), +z(gini) )

Standardization z(.) is computed globally across all layers and heads
to enable cross-layer comparison.

Outputs (in the same dir as metrics JSON by default):
  - sw_heatmap.png           (L x H heatmap of S_w)
  - sw_means_vars.png        (per-layer mean/var curves)
  - sw_hist_grid.png         (4x4 layer histograms)
  - sw_means_vars.csv        (per-layer mean/var table)
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_metrics(metrics_json: Path):
    obj = json.loads(metrics_json.read_text())
    layers = obj["layers"]
    L = len(layers)
    ent = []
    spn = []
    gin = []
    for rec in layers:
        ent.append(np.array(rec["raw"]["entropy"], dtype=float))
        spn.append(np.array(rec["raw"]["span"], dtype=float))
        gin.append(np.array(rec["raw"]["gini"], dtype=float))
    ent = np.stack(ent, axis=0)  # [L, H]
    spn = np.stack(spn, axis=0)
    gin = np.stack(gin, axis=0)
    return ent, spn, gin


def zscore_global(M):
    v = M.reshape(-1)
    mu = v.mean()
    sd = v.std() + 1e-12
    Z = (M - mu) / sd
    return Z, mu, sd


def main():
    ap = argparse.ArgumentParser(description="Visualize equal-weight S_w from entropy/span/gini")
    ap.add_argument('--metrics_json', type=str, default='head_sharpness_images/metrics_per_layer.json')
    ap.add_argument('--out_dir', type=str, default='')
    ap.add_argument('--bins', type=int, default=16)
    args = ap.parse_args()

    mj = Path(args.metrics_json)
    out_dir = Path(args.out_dir) if args.out_dir else mj.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ent, spn, gin = load_metrics(mj)
    L, H = ent.shape

    # Global z across all layers/heads
    z_ent, mu_e, sd_e = zscore_global(ent)
    z_spn, mu_s, sd_s = zscore_global(spn)
    z_gin, mu_g, sd_g = zscore_global(gin)

    # Equal-weight composite
    SW = (-z_ent - z_spn + z_gin) / 3.0

    # Per-layer stats
    sw_mean = SW.mean(axis=1)
    sw_var  = SW.var(axis=1)

    # Save CSV
    with (out_dir / 'sw_means_vars.csv').open('w') as f:
        f.write('layer,sw_mean,sw_var\n')
        for i in range(L):
            f.write(f"{i},{sw_mean[i]:.6f},{sw_var[i]:.6f}\n")

    # Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(SW, cmap='RdBu_r', center=0.0, linewidths=0.2, linecolor='gray')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title('Composite Sharpness S_w (equal weights, global z)')
    plt.tight_layout()
    plt.savefig(out_dir / 'sw_heatmap.png', dpi=180)
    plt.close()

    # Means/Vars curves
    layers = np.arange(L)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(layers, sw_mean, '-o', label='S_w mean')
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.4)
    ax.set_ylabel('Mean')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Per-Layer S_w Mean')

    ax = axes[1]
    ax.plot(layers, sw_var, '-o', label='S_w var')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Variance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Per-Layer S_w Variance')
    plt.tight_layout()
    plt.savefig(out_dir / 'sw_means_vars.png', dpi=180)
    plt.close()

    # Histograms per layer
    R, C = 4, 4
    fig, axes = plt.subplots(R, C, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    sw_min = float(SW.min()); sw_max = float(SW.max())
    for i in range(L):
        ax = axes[i]
        ax.hist(SW[i], bins=args.bins, range=(sw_min, sw_max), color='tab:red', edgecolor='black', linewidth=0.5)
        ax.set_title(f'L{i}', fontsize=9)
    for i in range(L, R*C):
        axes[i].axis('off')
    fig.suptitle('S_w Histograms per Layer (equal weights)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / 'sw_hist_grid.png', dpi=180)
    plt.close()

    print(f"Saved: {out_dir/'sw_heatmap.png'}")
    print(f"Saved: {out_dir/'sw_means_vars.png'}")
    print(f"Saved: {out_dir/'sw_hist_grid.png'}")


if __name__ == '__main__':
    main()

