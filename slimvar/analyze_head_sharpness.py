#!/usr/bin/env python3
"""
Analyze per-head attention sharpness across all layers (d16)

Reads existing concentration metrics saved by verify_scale_concentration.py
under scale_concentration_verification/layer_*_results.json and classifies
each head in each layer as 'sharp' (尖锐) or 'smooth' (平缓).

Metrics used per head:
  - entropy (lower ⇒ sharper)
  - effective span@0.9 (lower ⇒ sharper)
  - gini coefficient (higher ⇒ sharper)

Classification methods:
  - kmeans (default): per-layer KMeans(k=2) on standardized [entropy, span, gini],
    mapping the lower-entropy/higher-gini cluster to 'sharp'. Robust, no manual thresholds.
  - median: per-layer median-based rule on a combined score S = -z(entropy) - z(span) + z(gini).

Outputs (under head_sharpness_analysis/ by default):
  - head_sharpness_summary.json: all metrics per layer/head + labels
  - head_sharpness_matrix.csv: 16xH matrix (1=sharp, 0=smooth) for quick glance
  - per-layer bar plots and an overall heatmap (optional; enabled by --plots)

Usage:
  python analyze_head_sharpness.py \
      --results_dir scale_concentration_verification \
      --method kmeans \
      --plots

Notes:
  - This script only aggregates/analyzes existing results. To recompute attention
    metrics, run verify_scale_concentration.py first with real images (preferred)
    or random generation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_layer_results(results_dir: Path, max_layers: int = 64):
    """Load per-layer JSONs produced by verify_scale_concentration.py.

    Returns:
      layers: sorted list of available layer indices
      data: dict[layer_idx] -> dict with 'entropies', 'spans', 'ginis', 'scale_mul'
    """
    data = {}
    layers = []
    for i in range(max_layers):
        p = results_dir / f"layer_{i}_results.json"
        if p.exists():
            try:
                obj = json.loads(p.read_text())
                cms = obj.get("concentration_metrics", {})
                ent = cms.get("entropies", [])
                spn = cms.get("spans", [])
                gin = cms.get("ginis", [])
                scl = obj.get("scale_mul", {}).get("values", [])
                if len(ent) and len(spn) and len(gin):
                    data[i] = {
                        "entropies": np.array(ent, dtype=np.float64),
                        "spans": np.array(spn, dtype=np.float64),
                        "ginis": np.array(gin, dtype=np.float64),
                        "scale_mul": np.array(scl, dtype=np.float64) if len(scl) else None,
                    }
                    layers.append(i)
            except Exception:
                # skip malformed
                pass
    layers.sort()
    return layers, data


def classify_per_layer_kmeans(ent, spn, gin):
    """KMeans(k=2) on standardized [entropy, span, gini].

    Returns:
      labels: np.ndarray of 0/1 with '1' denoting 'sharp' (lower entropy, higher gini)
      centers: cluster centers in original feature space
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
    except ModuleNotFoundError:
        # Fallback: simple numpy k-means (k=2) on standardized features
        feats = np.stack([ent, spn, gin], axis=1)
        # standardize
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-8
        X = (feats - mu) / sd

        # init centers by percentiles along entropy axis
        idx_sorted = np.argsort(X[:, 0])  # entropy
        c0 = X[idx_sorted[: max(1, len(X)//4)]].mean(axis=0)
        c1 = X[idx_sorted[-max(1, len(X)//4):]].mean(axis=0)
        C = np.stack([c0, c1], axis=0)

        for _ in range(50):
            # assign
            dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            # update
            new_C = np.stack([
                X[labels == 0].mean(axis=0),
                X[labels == 1].mean(axis=0)
            ], axis=0)
            # handle empty cluster
            for k in (0, 1):
                if np.any(np.isnan(new_C[k])):
                    new_C[k] = C[k]
            if np.allclose(new_C, C):
                break
            C = new_C

        # back to original scale
        centers_orig = C * sd + mu
        labels = labels.astype(int)
    else:
        feats = np.stack([ent, spn, gin], axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)

        # robust KMeans
        km = KMeans(n_clusters=2, n_init=20, random_state=42)
        km.fit(X)
        labels = km.labels_.astype(int)

        # Map cluster with lower mean entropy and higher mean gini to 'sharp' (1)
        centers_orig = scaler.inverse_transform(km.cluster_centers_)
    c0_e, c0_s, c0_g = centers_orig[0]
    c1_e, c1_s, c1_g = centers_orig[1]

    # Define sharp cluster by a simple score: S = -entropy - span + gini
    c0_score = -c0_e - c0_s + c0_g
    c1_score = -c1_e - c1_s + c1_g
    sharp_cluster = 0 if c0_score > c1_score else 1

    sharp_labels = (labels == sharp_cluster).astype(int)
    return sharp_labels, centers_orig


def classify_per_layer_median(ent, spn, gin):
    """Median rule on combined z-score S = -z(entropy) - z(span) + z(gini).

    Returns:
      labels: np.ndarray of 0/1 with '1' denoting 'sharp'.
    """
    def z(x):
        mu, sd = np.mean(x), np.std(x) + 1e-8
        return (x - mu) / sd

    S = -z(ent) - z(spn) + z(gin)
    med = np.median(S)
    labels = (S >= med).astype(int)
    return labels


def main():
    ap = argparse.ArgumentParser(description="Per-head attention sharpness classification (d16)")
    ap.add_argument("--results_dir", type=str, default="scale_concentration_verification",
                    help="Directory containing layer_*_results.json from verify_scale_concentration.py")
    ap.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "median"],
                    help="Classification method: kmeans or median rule")
    ap.add_argument("--plots", action="store_true", help="Generate per-layer plots and overall heatmap")
    ap.add_argument("--out_dir", type=str, default="head_sharpness_analysis", help="Output directory")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers, data = load_layer_results(results_dir)
    if not layers:
        raise SystemExit(f"No layer_*_results.json found under {results_dir}")

    # Aggregate labels and build outputs
    summary = {"method": args.method, "layers": [], "matrix": []}

    for layer in layers:
        ent = data[layer]["entropies"]
        spn = data[layer]["spans"]
        gin = data[layer]["ginis"]

        if args.method == "kmeans":
            labels, centers = classify_per_layer_kmeans(ent, spn, gin)
            centers = centers.tolist()
        else:
            labels = classify_per_layer_median(ent, spn, gin)
            centers = None

        layer_record = {
            "layer": layer,
            "num_heads": int(len(ent)),
            "labels": labels.astype(int).tolist(),  # 1=sharp, 0=smooth
            "metrics": {
                "entropies": ent.tolist(),
                "spans": spn.tolist(),
                "ginis": gin.tolist(),
            },
        }
        if centers is not None:
            layer_record["cluster_centers"] = centers

        summary["layers"].append(layer_record)
        summary["matrix"].append(labels.astype(int).tolist())

    # Save JSON summary
    (out_dir / "head_sharpness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # Save CSV matrix (layers x heads)
    # Infer max heads across layers for consistent width
    max_heads = max(len(data[l]["entropies"]) for l in layers)
    M = np.full((len(layers), max_heads), fill_value=-1, dtype=int)
    for i, layer in enumerate(layers):
        row = np.array(summary["matrix"][i], dtype=int)
        M[i, : len(row)] = row
    np.savetxt(out_dir / "head_sharpness_matrix.csv", M, fmt="%d", delimiter=",")

    # Optional plots
    if args.plots:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Per-layer bar plots (entropy as proxy, colored by label)
        for i, layer in enumerate(layers):
            ent = np.array(summary["layers"][i]["metrics"]["entropies"])  # shape [H]
            labels = np.array(summary["layers"][i]["labels"])  # 1=sharp, 0=smooth
            H = len(ent)
            order = np.argsort(ent)  # low→high entropy

            plt.figure(figsize=(10, 3.2))
            colors = ["tab:red" if labels[h] == 1 else "tab:blue" for h in order]
            plt.bar(range(H), ent[order], color=colors, edgecolor="black", linewidth=0.5)
            plt.xticks(range(H), order, rotation=0)
            plt.xlabel("Head (sorted by entropy, low→high)")
            plt.ylabel("Entropy")
            plt.title(f"Layer {layer}: Head Sharpness by Entropy (red=sharp, blue=smooth)")
            plt.tight_layout()
            plt.savefig(out_dir / f"layer_{layer}_entropy_bars.png", dpi=160)
            plt.close()

        # Overall heatmap of labels (layers x heads)
        plt.figure(figsize=(12, 6))
        sns.heatmap(M, cmap=sns.color_palette(["#3572A5", "#D62728"]) , vmin=0, vmax=1,
                    cbar=False, linewidths=0.2, linecolor="gray")
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.title("Head Sharpness Map (1=sharp/red, 0=smooth/blue)")
        plt.tight_layout()
        plt.savefig(out_dir / "head_sharpness_heatmap.png", dpi=180)
        plt.close()

    print(f"✓ Saved summary to {out_dir}/head_sharpness_summary.json")
    print(f"✓ Saved matrix to  {out_dir}/head_sharpness_matrix.csv")
    if args.plots:
        print(f"✓ Saved plots to   {out_dir}/")


if __name__ == "__main__":
    main()
