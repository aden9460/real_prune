# ===== pruning_viz.py =====
import os, json, math
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True); return d

def build_plan_from_importance(
    importance_json_path: str,
    target_global_sparsity: float,
    alpha: float = 1.0,                   # 层重要性幂次（保护重要层）
    s_min_max: Tuple[float,float] = (0.0, 0.6),
    depth_discount_gamma: float = 1.0     # 深度折扣（<1.0 则越深越保守）
):
    """
    读 layer_importance.json，返回：
      - sparsity_per_layer: {layer_id: s_i}
      - prune_indices: {layer_id: [indices_to_prune]}
      - layer_meta: {layer_id: {"curvature_density": float, "num_units": int}}
    说明：
      - num_units = len(per_head_or_col)，即该层可剪单元数（attn.proj 为 head 数；fc2 为列数）
      - s_i 的归一化按“保留量 ∝ curvature_density^alpha”，总量满足 target_global_sparsity
    """
    with open(importance_json_path, "r") as f:
        imp = json.load(f)

    layers = sorted(int(k) for k in imp.keys())
    # 收集层指标
    curv = {i: float(imp[str(i)]["curvature_density"]) for i in layers}
    scores = {i: np.array(imp[str(i)]["obs_scores"]["per_head_or_col"], dtype=float) for i in layers}
    num_units = {i: scores[i].shape[0] for i in layers}

    # 深度折扣 + 幂次
    L = len(layers)
    weights = []
    for rank, lid in enumerate(layers):
        depth_w = (depth_discount_gamma ** (rank/(L-1))) if L>1 else 1.0
        w = max(1e-12, curv[lid]) ** alpha
        w *= depth_w
        weights.append(w)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    # 按保留量分摊，推回每层 sparsity
    total_units = sum(num_units.values())
    keep_units_target = (1.0 - target_global_sparsity) * total_units
    keep_units = {lid: keep_units_target * weights[idx] for idx, lid in enumerate(layers)}

    sparsity_per_layer = {}
    for lid in layers:
        s_i = 1.0 - keep_units[lid] / max(1, num_units[lid])
        s_i = float(np.clip(s_i, s_min_max[0], s_min_max[1]))
        # 若这一层可剪单位很少，避免出现“负剪枝”或“全剪光”
        if num_units[lid] <= 0:
            s_i = 0.0
        sparsity_per_layer[lid] = s_i

    # 选具体要剪的 indices：分数越小越先剪
    prune_indices = {}
    for lid in layers:
        k = int(np.floor(sparsity_per_layer[lid] * num_units[lid]))
        order = np.argsort(scores[lid])  # 升序，越小越先剪
        prune_indices[lid] = order[:k].tolist()

    # 元信息汇总
    layer_meta = {lid: {"curvature_density": curv[lid], "num_units": int(num_units[lid])} for lid in layers}
    return sparsity_per_layer, prune_indices, layer_meta

def plot_layer_plan(layer_meta: Dict[int,dict], sparsity_per_layer: Dict[int,float], outdir: str):
    _ensure_dir(outdir)
    lids = sorted(layer_meta.keys())
    curv = [layer_meta[i]["curvature_density"] for i in lids]
    sp   = [sparsity_per_layer.get(i, 0.0) for i in lids]

    # 双轴：柱(剪枝率) + 折线(曲率密度)
    fig, ax1 = plt.subplots(figsize=(max(6, len(lids)*0.5), 4))
    x = np.arange(len(lids))
    b = ax1.bar(x, sp, width=0.6)
    ax1.set_ylabel("planned sparsity per layer")
    ax1.set_ylim(0, max(0.65, max(sp)+0.05))
    ax1.set_xticks(x); ax1.set_xticklabels([f"L{i}" for i in lids], rotation=0)
    ax2 = ax1.twinx()
    ax2.plot(x, curv, marker='o')
    ax2.set_ylabel("curvature density (trace/token)")
    ax2.set_title("Layer plan: sparsity vs curvature")
    fig.tight_layout()
    p = os.path.join(outdir, "layer_plan.png")
    plt.savefig(p, dpi=150); plt.close(fig)
    # Pareto 散点
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(curv, sp)
    for i,c,s in zip(lids, curv, sp):
        ax.annotate(f"L{i}", (c, s), textcoords="offset points", xytext=(4,2), fontsize=8)
    ax.set_xlabel("curvature density (trace/token)")
    ax.set_ylabel("planned sparsity")
    ax.set_title("Pareto of curvature vs planned sparsity")
    fig.tight_layout()
    p2 = os.path.join(outdir, "pareto_scatter.png")
    plt.savefig(p2, dpi=150); plt.close(fig)
    return p, p2

import math

def plot_headmap_with_plan(importance_json_path: str,
                           prune_indices: Dict[int, list],
                           outdir: str,
                           max_cols: int = 2048,
                           dpi: int = 150):
    """
    生成层×单元重要性热图，并在被剪单元位置打 ✕ 标记。
    - 自动将列做分组聚合，控制可视化列数不超过 max_cols（默认 2048）。
    - 自动根据 dpi 限制图像像素，避免超过 2^16。
    """
    _ensure_dir(outdir)
    with open(importance_json_path, "r") as f:
        imp = json.load(f)

    lids = sorted(int(k) for k in imp.keys())
    rows = len(lids)
    widths = [len(imp[str(i)]["obs_scores"]["per_head_or_col"]) for i in lids]
    if not widths:
        return None
    Wmax = max(widths)

    # ---- 列分组：聚合相邻 'group' 个单元为一列，减少可视化宽度 ----
    group = max(1, math.ceil(Wmax / max_cols))
    Wvis = math.ceil(Wmax / group)

    M = np.full((rows, Wvis), np.nan, dtype=float)   # 重要性（层内 min-max 归一）
    mask = np.zeros_like(M, dtype=bool)              # 是否有被剪单元落在该分组

    for r, lid in enumerate(lids):
        s = np.array(imp[str(lid)]["obs_scores"]["per_head_or_col"], dtype=float)
        # 层内 min-max 归一化，便于跨层比较
        if np.isfinite(s).any():
            s_norm = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
        else:
            s_norm = s

        # 填充到 group 的整数倍再 reshape -> 分组取均值作为展示值
        pad = (-len(s_norm)) % group
        if pad > 0:
            s_norm = np.pad(s_norm, (0, pad), constant_values=np.nan)
        s_grp = s_norm.reshape(-1, group)            # [Wvis, group]
        M[r, :s_grp.shape[0]] = np.nanmean(s_grp, axis=1)

        # 被剪单元打到分组：组索引 = idx // group
        for idx in prune_indices.get(lid, []):
            gcol = idx // group
            if gcol < Wvis:
                mask[r, gcol] = True

    # ---- 计算安全画布尺寸（像素限制）----
    PX_LIMIT = 2**16 - 1024  # 留一点余量
    # 基于“每列/每行”的基准尺寸（英寸），再根据 DPI 兜底限制像素
    unit_w_in = 0.08         # 每列的英寸宽度，更小以保证安全
    unit_h_in = 0.35         # 每行的英寸高度
    width_in  = max(8.0,  Wvis * unit_w_in)
    height_in = max(4.0,   rows * unit_h_in)

    # 若超像素上限，按比例收缩
    width_px  = width_in  * dpi
    height_px = height_in * dpi
    if width_px > PX_LIMIT:
        scale = PX_LIMIT / width_px * 0.98
        width_in *= scale
    if height_px > PX_LIMIT:
        scale = PX_LIMIT / height_px * 0.98
        height_in *= scale

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    im = ax.imshow(M, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="normalized importance (higher = keep)")
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([f"L{i}" for i in lids])

    # x 轴刻度太密集时稀疏显示
    if Wvis <= 40:
        ax.set_xticks(np.arange(Wvis))
    else:
        step = max(1, Wvis // 40)
        ax.set_xticks(np.arange(0, Wvis, step))
    ax.set_xlabel(f"unit index (grouped by {group})")
    ax.set_title("Per-layer unit importance with prune marks")

    # 叠加“✕”标记（按分组）
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        ax.text(x, y, "✕", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    p = os.path.join(outdir, "headmap_plan.png")
    plt.savefig(p, dpi=dpi)
    plt.close(fig)
    return p

def plot_schedule(sparsity_per_layer: Dict[int,float], layer_meta: Dict[int,dict],
                  prune_indices: Dict[int,list], steps: int, outdir: str):
    _ensure_dir(outdir)
    lids = sorted(layer_meta.keys())
    # 将总配额均匀切分到 steps（四舍五入保总和）
    schedule = np.zeros((steps, len(lids)), dtype=int)
    for c, lid in enumerate(lids):
        total = len(prune_indices.get(lid, []))
        if total <= 0: continue
        base = total // steps
        extra = total % steps
        schedule[:, c] = base
        if extra > 0:
            schedule[:extra, c] += 1  # 前面几步多剪一个
    # 画步×层热图（数值越大颜色越深）
    fig, ax = plt.subplots(figsize=(max(6, len(lids)*0.5), max(4, steps*0.3)))
    im = ax.imshow(schedule, aspect='auto')
    plt.colorbar(im, ax=ax, label="#units pruned at step")
    ax.set_yticks(np.arange(steps))
    ax.set_yticklabels([f"s{t+1}" for t in range(steps)])
    ax.set_xticks(np.arange(len(lids)))
    ax.set_xticklabels([f"L{i}" for i in lids], rotation=0)
    ax.set_title("Gentle iterative pruning schedule")
    # 在格子里标注数字
    for i in range(steps):
        for j in range(len(lids)):
            v = schedule[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "schedule_plan.png")
    plt.savefig(p, dpi=150); plt.close(fig)
    return p
# ===== end pruning_viz.py =====

# from pruning_viz import build_plan_from_importance, plot_layer_plan, plot_headmap_with_plan, plot_schedule

s_min_max = (0, 0.6)
sparsity_per_layer, prune_indices, layer_meta = build_plan_from_importance(
    importance_json_path="/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens/layer_importance.json",
    target_global_sparsity=0.2,
    alpha=1,
    s_min_max=s_min_max,
    depth_discount_gamma=1
)
plot_layer_plan(layer_meta, sparsity_per_layer, outdir="/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens")
plot_headmap_with_plan("/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens/layer_importance.json", prune_indices, outdir="/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens")
plot_schedule(sparsity_per_layer, layer_meta, prune_indices, steps=10, outdir="/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens")

# 也可以把计划另存 JSON，后面直接喂剪枝器
with open(os.path.join("/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens", "planned_sparsity.json"), "w") as f:
    json.dump({str(k): float(v) for k,v in sparsity_per_layer.items()}, f, indent=2)
with open(os.path.join("/home/suanba/EdgeVAR/real_prune/hcurv_fc2_v9_full_equal_tokens", "planned_indices.json"), "w") as f:
    json.dump({str(k): v for k,v in prune_indices.items()}, f, indent=2)

