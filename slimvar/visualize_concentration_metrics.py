#!/usr/bin/env python3
"""
Visualize per-layer concentration metrics (entropy/span/gini):
 - Means across layers
 - Variances across layers
 - Layer-wise histograms (4x4 grid per metric)

Input: head_sharpness_images/metrics_per_layer.json (default)
Output images written next to the JSON by default.
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


def main():
    ap = argparse.ArgumentParser(description="Visualize entropy/span/gini stats and histograms per layer")
    ap.add_argument('--metrics_json', type=str, default='head_sharpness_images/metrics_per_layer.json')
    ap.add_argument('--out_dir', type=str, default='')
    ap.add_argument('--bins', type=int, default=16)
    args = ap.parse_args()

    mj = Path(args.metrics_json)
    out_dir = Path(args.out_dir) if args.out_dir else mj.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ent, spn, gin = load_metrics(mj)
    L, H = ent.shape

    # Means and vars per layer
    ent_mean = ent.mean(axis=1)
    ent_var = ent.var(axis=1)
    spn_mean = spn.mean(axis=1)
    spn_var = spn.var(axis=1)
    gin_mean = gin.mean(axis=1)
    gin_var = gin.var(axis=1)

    # Save CSV
    with (out_dir / 'metrics_means_vars.csv').open('w') as f:
        f.write('layer,ent_mean,ent_var,span_mean,span_var,gini_mean,gini_var\n')
        for i in range(L):
            f.write(f"{i},{ent_mean[i]:.6f},{ent_var[i]:.6f},{spn_mean[i]:.6f},{spn_var[i]:.6f},{gin_mean[i]:.6f},{gin_var[i]:.6f}\n")

    # Plots (non-interactive)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    layers = np.arange(L)

    # Means/Vars figure: two rows
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.plot(layers, ent_mean, '-o', label='entropy mean')
    ax.plot(layers, spn_mean, '-o', label='span mean')
    ax.plot(layers, gin_mean, '-o', label='gini mean')
    ax.set_ylabel('Mean')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Per-Layer Means (entropy/span/gini)')

    ax = axes[1]
    ax.plot(layers, ent_var, '-o', label='entropy var')
    ax.plot(layers, spn_var, '-o', label='span var')
    ax.plot(layers, gin_var, '-o', label='gini var')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Variance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Per-Layer Variances (entropy/span/gini)')

    plt.tight_layout()
    plt.savefig(out_dir / 'metrics_means_vars.png', dpi=180)
    plt.close()

    # Histogram grids per metric (4x4 for L=16)
    def hist_grid(data, title, fname, bins, rng=None):
        R, C = 4, 4
        fig, axes = plt.subplots(R, C, figsize=(12, 9), sharex=True, sharey=True)
        axes = axes.flatten()
        for i in range(L):
            ax = axes[i]
            ax.hist(data[i], bins=bins, range=rng, color='tab:blue', edgecolor='black', linewidth=0.5)
            ax.set_title(f'L{i}', fontsize=9)
        # Hide unused axes if L<16
        for i in range(L, R*C):
            axes[i].axis('off')
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()

    # Ranges
    ent_rng = (float(ent.min()), float(ent.max()))
    spn_rng = (float(spn.min()), float(spn.max()))
    gin_rng = (max(0.0, float(gin.min()-0.02)), min(1.0, float(gin.max()+0.02)))

    hist_grid(ent, 'Entropy Histograms per Layer', 'entropy_hist_grid.png', args.bins, ent_rng)
    hist_grid(spn, 'Span@0.9 Histograms per Layer', 'span_hist_grid.png', args.bins, spn_rng)
    hist_grid(gin, 'Gini Histograms per Layer', 'gini_hist_grid.png', args.bins, gin_rng)

    print(f"Saved: {out_dir/'metrics_means_vars.png'}")
    print(f"Saved: {out_dir/'entropy_hist_grid.png'}")
    print(f"Saved: {out_dir/'span_hist_grid.png'}")
    print(f"Saved: {out_dir/'gini_hist_grid.png'}")


if __name__ == '__main__':
    main()

