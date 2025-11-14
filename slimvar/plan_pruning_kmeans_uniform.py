#!/usr/bin/env python3
"""
Plan uniform per-layer pruning at a fixed rate (default 40%),
allocating pruned heads proportionally to KMeans clusters within each layer.

Inputs:
  - head_sharpness_images/metrics_per_layer.json (contains per-layer KMeans labels and raw metrics)

Policy:
  - For each layer with H heads, prune_count = round(H * rate)
  - Count cluster sizes (label 0 = smooth, 1 = sharp from earlier script)
  - Allocate pruned quota proportionally: to_prune_k = round(prune_count * (n_k / H)),
    and adjust by largest-remainder to hit exact prune_count
  - Within each cluster, rank heads by composite score S_w (equal weights, global z)
    ascending (lower S_w = 更平缓/更不尖锐) and prune the lowest S_w heads

Outputs in the same directory by default:
  - pruning_plan_40pct.json: list per layer of heads to prune, plus per cluster counts
  - pruning_mask.csv: L x H matrix: 0=keep, 1=prune(smooth cluster), 2=prune(sharp cluster)
  - pruning_stacked_bars.png: per-layer stacked bars of pruned counts by cluster
  - pruning_mask_heatmap.png: tri-color heatmap of pruning mask
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
    labels = []
    for rec in layers:
        ent.append(np.array(rec["raw"]["entropy"], dtype=float))
        spn.append(np.array(rec["raw"]["span"], dtype=float))
        gin.append(np.array(rec["raw"]["gini"], dtype=float))
        labels.append(np.array(rec["labels"], dtype=int))  # 1=sharp, 0=smooth
    ent = np.stack(ent, axis=0)
    spn = np.stack(spn, axis=0)
    gin = np.stack(gin, axis=0)
    labels = np.stack(labels, axis=0)
    return ent, spn, gin, labels


def zscore_global(M):
    v = M.reshape(-1)
    mu = v.mean()
    sd = v.std() + 1e-12
    Z = (M - mu) / sd
    return Z


def largest_remainder_allocation(total, weights):
    """Allocate integer counts summing to total proportionally to weights.

    Returns integer array of same length as weights.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        # evenly spread
        base = np.full_like(weights, fill_value=total // len(weights), dtype=int)
        rem = total - base.sum()
        base[:rem] += 1
        return base
    prop = total * weights / weights.sum()
    floor = np.floor(prop).astype(int)
    rem = total - floor.sum()
    if rem > 0:
        remainders = prop - floor
        idx = np.argsort(-remainders)[:rem]
        floor[idx] += 1
    return floor


def main():
    ap = argparse.ArgumentParser(description="Uniform per-layer pruning proportional to KMeans clusters")
    ap.add_argument('--metrics_json', type=str, default='head_sharpness_images/metrics_per_layer.json')
    ap.add_argument('--rate', type=float, default=0.4, help='Per-layer pruning rate (e.g., 0.4 for 40%)')
    ap.add_argument('--out_dir', type=str, default='')
    args = ap.parse_args()

    mj = Path(args.metrics_json)
    out_dir = Path(args.out_dir) if args.out_dir else mj.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ent, spn, gin, labs = load_metrics(mj)
    L, H = ent.shape

    # Global z for composite score
    z_ent = zscore_global(ent)
    z_spn = zscore_global(spn)
    z_gin = zscore_global(gin)
    SW = (-z_ent - z_spn + z_gin) / 3.0

    plan = {"rate": args.rate, "layers": []}
    mask = np.zeros((L, H), dtype=int)  # 0 keep, 1 prune(smooth), 2 prune(sharp)

    for li in range(L):
        prune_count = int(round(H * args.rate))
        labels_l = labs[li]
        # counts per cluster
        n_smooth = int((labels_l == 0).sum())
        n_sharp = int((labels_l == 1).sum())
        alloc = largest_remainder_allocation(prune_count, np.array([n_smooth, n_sharp]))
        to_prune_smooth, to_prune_sharp = int(alloc[0]), int(alloc[1])

        # rank within cluster by SW ascending (lower = less sharp)
        sw_l = SW[li]
        idx_smooth = np.where(labels_l == 0)[0]
        idx_sharp  = np.where(labels_l == 1)[0]
        order_smooth = idx_smooth[np.argsort(sw_l[idx_smooth])]
        order_sharp  = idx_sharp[np.argsort(sw_l[idx_sharp])]

        prune_smooth = order_smooth[:to_prune_smooth].tolist()
        prune_sharp  = order_sharp[:to_prune_sharp].tolist()
        pruned = sorted(prune_smooth + prune_sharp)

        # set mask
        for h in prune_smooth:
            mask[li, h] = 1
        for h in prune_sharp:
            mask[li, h] = 2

        plan["layers"].append({
            "layer": li,
            "num_heads": H,
            "n_smooth": n_smooth,
            "n_sharp": n_sharp,
            "prune_count": prune_count,
            "prune_smooth": prune_smooth,
            "prune_sharp": prune_sharp,
            "pruned": pruned,
        })

    # Save plan
    (out_dir / f'pruning_plan_{int(args.rate*100)}pct.json').write_text(
        json.dumps(plan, ensure_ascii=False, indent=2)
    )
    np.savetxt(out_dir / 'pruning_mask.csv', mask, fmt='%d', delimiter=',')

    # Visualizations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Stacked bars: pruned per cluster
    pruned_s = [len(rec['prune_smooth']) for rec in plan['layers']]
    pruned_h = [len(rec['prune_sharp']) for rec in plan['layers']]
    layers = np.arange(L)
    plt.figure(figsize=(12, 4))
    plt.bar(layers, pruned_s, color='#3572A5', edgecolor='black', label='pruned(smooth)')
    plt.bar(layers, pruned_h, bottom=pruned_s, color='#D62728', edgecolor='black', label='pruned(sharp)')
    plt.axhline(int(round(H*args.rate)), color='k', linestyle='--', alpha=0.5, label='per-layer quota')
    plt.xlabel('Layer Index')
    plt.ylabel('Pruned Heads')
    plt.title(f'Per-Layer Pruning (rate={args.rate:.0%}) proportional to KMeans clusters')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'pruning_stacked_bars.png', dpi=180)
    plt.close()

    # Tri-color mask heatmap: 0 keep, 1 smooth pruned, 2 sharp pruned
    cmap = sns.color_palette(['#EEEEEE', '#6BAED6', '#E6550D'], as_cmap=False)
    # create ListedColormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cm = ListedColormap(cmap)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cm.N)

    plt.figure(figsize=(12, 6))
    im = plt.imshow(mask, cmap=cm, norm=norm, aspect='auto')
    plt.xlabel('Head Index')
    plt.ylabel('Layer Index')
    plt.title('Pruning Mask (0=keep, 1=pruned smooth, 2=pruned sharp)')
    plt.colorbar(im, ticks=[0,1,2])
    plt.tight_layout()
    plt.savefig(out_dir / 'pruning_mask_heatmap.png', dpi=180)
    plt.close()

    print(f"Saved plan to: {out_dir / f'pruning_plan_{int(args.rate*100)}pct.json'}")
    print(f"Saved mask to: {out_dir / 'pruning_mask.csv'}")
    print(f"Saved stacked bars: {out_dir / 'pruning_stacked_bars.png'}")
    print(f"Saved mask heatmap: {out_dir / 'pruning_mask_heatmap.png'}")


if __name__ == '__main__':
    main()

