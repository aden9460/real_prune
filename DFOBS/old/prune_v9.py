###全尺度评估改为单个相乘 之后平均
##这里选择 进行1.曲率一致性可视化 2.数值稳定性分析 3.收敛/停采集
##CurvatureMonitor
# 太好了，我们直接给你一版**面向“逐层 Transformer 重要性分析 + OBS 决策”**的全新 CurvatureMonitor。它有这些能力：
# 逐层×尺度统计（支持自定义顺序：1,4,9,16,25,36,64,100,169,256）
# 流式累计 
# X
# ⊤
# X
# X 
# ⊤
#  X（分块、GPU算/CPU存，省显存）
# 对大维度层（如 ffn.fc2 的 4096 维）默认不存整 H，只存 trace/diag，避免 OOM
# 提供公平合成（先 per-token，再尺度等权；可选 unit_trace 或 corr 相关矩阵规范化）
# 输出“逐层重要性”表 & 可选 OBS 重要性 proxy（有整 H 用真 OBS 公式；没整 H 自动退化到 对角近似）
# 可画 layer×scale 的 trace/token 热图，以及 逐层重要性柱状图
# 保留收敛/停采与数值稳定性门槛
# 说明：剪谁（排名/重要性）推荐用“公平合成”的 
# H
# H（去掉 token 长度偏置、可选相关化）；补偿（更新权重）建议用合成的原始 
# H
# H+阻尼（或直接在你的 SlimGPT 内继续使用现有 H 统计与阻尼）。
import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
from eval import eval_ppl
from slim_utils.slimgpt import SlimGPT
from slim_utils.slim_dataset import get_loaders
from slim_utils.params_remove import LLaMAParamsPruner
from ppl_eval.ppl_eval import ppl_metric
from torchvision.utils import save_image
import sys
sys.path.append("/home/suanba/EdgeVAR/Torch-Pruning")
from importlib.metadata import version
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import os
import torch_pruning as tp
from torchviz import make_dot
import torch_pruning.pruner.function as tfun
from models import VQVAE, build_vae_var
import gc
from contextlib import contextmanager
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
import os, json, re


# ===== curvature_monitor_v2.py =====
import os, re, json
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

@torch.no_grad()
def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    # [B,T,C] or [N,C] -> [N,C]
    if x.dim() == 3:
        return x.flatten(0, 1)
    return x

def parse_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("fp32","float32","f32"): return torch.float32
    if s in ("bf16","bfloat16"):      return torch.bfloat16
    if s in ("fp64","float64","f64"): return torch.float64
    return torch.float32

class CurvatureMonitor:
    """
    逐层×尺度统计 + 公平合成 + 逐层重要性分析（含 OBS 代理）
    - Hook: forward_pre_hook 抓 Linear 的输入 X
    - 统计：
        trace_sum[(layer, scale)], tokens[(layer, scale)],
        diag_sum[(layer, scale)]   （始终存，CPU/float32）
        H_sum[(layer, scale)]      （仅当 C<=max_full_dim 时存整块 H，CPU/float32）
    - 合成：
        H̄_s = H_s / N_s (per-token)；how='equal_scales'/'equal_tokens'/'unit_trace'
        norm='none' or 'corr'（相关矩阵：D^{-1/2} H̄ D^{-1/2}）
    - 逐层重要性：
        * full: 有整 H → OBS 评分（damped inv diag），列→head 聚合
        * diag: 无整 H → 对角近似（H_jj * ||W[:,j]||^2）
        * weights_only: 兜底（仅权重能量）
    - 设备策略：
        * 统计与重要性分析统一在 CPU / stat_dtype（默认 float32）
        * X^T X 计算用 work_device（cuda/cpu）分块 GEMM，结果落 CPU
    """

    def __init__(self,
                 model: nn.Module,
                 tap_regex: str = r'.*attn\.proj$',
                 regex_layer: str = r'blocks\.(\d+)\.',
                 include_types: Tuple[type, ...] = (nn.Linear,),
                 max_full_dim: int = 1024,
                 scale_tokens_order: Optional[List[int]] = None,  # [1,4,9,16,25,36,64,100,169,256]
                 key_prefix: str = "T",
                 store_device: str = "cpu",
                 work_device: str = "cuda",
                 stat_dtype: torch.dtype = torch.float32,
                 stream_chunk: int = 8192,
                 ):
        self.model = model
        self.tap_re = re.compile(tap_regex)
        self.layer_re = re.compile(regex_layer)
        self.include_types = include_types
        self.max_full_dim = max_full_dim
        self.store_device = store_device
        self.work_device = work_device
        self.stat_dtype = stat_dtype
        self.stream_chunk = stream_chunk

        # 逐层×尺度字典
        self.trace_sum: Dict[Tuple[int,str], float] = {}
        self.tokens:    Dict[Tuple[int,str], int]   = {}
        self.diag_sum:  Dict[Tuple[int,str], torch.Tensor] = {}
        self.H_sum:     Dict[Tuple[int,str], torch.Tensor] = {}

        # 收敛曲线历史（逐尺度聚合，不按层）
        self.history: Dict[str, List[Dict[str, float]]] = {}

        # 尺度排序
        self.key_prefix = key_prefix
        self._order_map = {f"{key_prefix}{t}": i for i,t in enumerate(scale_tokens_order)} if scale_tokens_order else None

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def close(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()

    # ---------- Hooks ----------
    def _register_hooks(self):
        for name, mod in self.model.named_modules():
            if not isinstance(mod, self.include_types):
                continue
            if self.tap_re.match(name):
                h = mod.register_forward_pre_hook(lambda m, inp, n=name: self._on_inp(n, m, inp))
                self.hooks.append(h)

    def _layer_id_from_name(self, name: str) -> int:
        m = self.layer_re.search(name or "")
        return int(m.group(1)) if m else -1

    def _scale_key_from_x(self, x: torch.Tensor) -> str:
        T = int(x.shape[1]) if x.dim() >= 2 else 1
        key = f"{self.key_prefix}{T}"
        if self._order_map is not None and key not in self._order_map:
            self._order_map[key] = max(self._order_map.values(), default=-1) + 1
        return key

    def _sorted_scales(self) -> List[str]:
        keys = sorted(set(k[1] for k in self.trace_sum.keys()))
        if self._order_map:
            keys.sort(key=lambda s: self._order_map.get(s, 10**9))
        else:
            def _tok(s):
                m = re.search(r'(\d+)', s)
                return int(m.group(1)) if m else 10**9
            keys.sort(key=_tok)
        return keys

    def _ensure_cell(self, layer_id: int, scale_key: str, C: int, want_full: bool):
        k = (layer_id, scale_key)
        if k not in self.trace_sum: self.trace_sum[k] = 0.0
        if k not in self.tokens:    self.tokens[k]    = 0
        if k not in self.diag_sum:  self.diag_sum[k]  = torch.zeros(C, device=self.store_device, dtype=self.stat_dtype)
        if want_full and k not in self.H_sum:
            self.H_sum[k] = torch.zeros(C, C, device=self.store_device, dtype=self.stat_dtype)

    @torch.no_grad()
    def _accum_xxt_streaming(self, X: torch.Tensor, key):
        N, C = X.shape
        H_acc = self.H_sum[key]
        for i in range(0, N, self.stream_chunk):
            Xi = X[i:i+self.stream_chunk]
            if self.work_device == "cuda":
                Xi = Xi.to("cuda", non_blocking=True).to(torch.float32)
                G = Xi.t().matmul(Xi).to(self.store_device, dtype=self.stat_dtype, non_blocking=True)
            else:
                Xi = Xi.to(self.store_device, dtype=self.stat_dtype)
                G  = Xi.t().matmul(Xi)
            H_acc.add_(G)

    @torch.no_grad()
    def _on_inp(self, name: str, mod: nn.Module, inputs):
        x = inputs[0]                      # [B,T,C]
        X = _flatten_bt(x)                 # [N,C]
        if X.dim()!=2 or X.numel()==0: return

        N, C = X.shape
        layer_id  = self._layer_id_from_name(name)
        scale_key = self._scale_key_from_x(x)
        want_full = (C <= self.max_full_dim)

        self._ensure_cell(layer_id, scale_key, C, want_full=False)
        k = (layer_id, scale_key)

        # 统计：CPU/float32
        Xcpu = X.detach().to(self.store_device, self.stat_dtype)
        self.tokens[k]    += int(N)
        self.trace_sum[k] += float((Xcpu*Xcpu).sum().item())
        self.diag_sum[k]  += (Xcpu*Xcpu).sum(dim=0)

        if want_full:
            if k not in self.H_sum:
                self.H_sum[k] = torch.zeros(C, C, device=self.store_device, dtype=self.stat_dtype)
            self._accum_xxt_streaming(X, k)

    # ---------- 读数 ----------
    @torch.no_grad()
    def layer_scale_metrics(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        out: Dict[int, Dict[str, Dict[str, float]]] = {}
        for (l, s), tr in self.trace_sum.items():
            N = max(1, self.tokens[(l,s)])
            md = float(self.diag_sum[(l,s)].mean().item())
            out.setdefault(l, {})[s] = {
                "tokens": float(N),
                "trace_per_token": float(tr / N),
                "mean_diag": md,
            }
        return out

    # ---------- 公平合成（逐层） ----------
    @torch.no_grad()
    def fair_H_for_layer(self, layer_id: int, how="equal_scales", norm="none", eps=1e-6):
        scales = [s for (l,s) in self.trace_sum.keys() if l==layer_id]
        if not scales: return None
        # 需要每个尺度都存了整 H
        if any((layer_id, s) not in self.H_sum for s in scales):
            return None

        Hbars = {}
        Ns = {}
        for s in scales:
            k = (layer_id, s)
            Ns[s] = max(1, self.tokens[k])
            Hbar = self.H_sum[k] / Ns[s]
            if norm == "corr":
                D = torch.diag(Hbar).clamp_min(eps)
                D_inv_sqrt = torch.diag(1.0/torch.sqrt(D))
                Hbar = D_inv_sqrt @ Hbar @ D_inv_sqrt
            Hbars[s] = Hbar

        if how == "equal_scales":
            w = {s: 1.0/len(scales) for s in scales}
        elif how == "equal_tokens":
            tot = sum(Ns.values()); w = {s: Ns[s]/tot for s in scales}
        elif how == "unit_trace":
            w = {s: 1.0/len(scales) for s in scales}
            for s in scales:
                tr = torch.trace(Hbars[s]).clamp_min(eps)
                Hbars[s] = Hbars[s] * (Hbars[s].shape[0] / tr)
        else:
            raise ValueError(how)

        H = None
        for s in scales:
            H = Hbars[s]*w[s] if H is None else H + Hbars[s]*w[s]
        # 重要：返回 CPU / stat_dtype，避免设备不一致
        return H.to(self.store_device, self.stat_dtype)

    # ---------- 逐层重要性（含 OBS 代理） ----------
    @torch.no_grad()
    def layer_importance(self,
                         get_weight_fn,         # (layer_id)-> (W: [out,in]) 要剪的 Linear 权重
                         layer_ids: Optional[List[int]] = None,
                         headsize: int = 64,
                         damping_perc: float = 0.02,
                         fair_how: str = "equal_scales",
                         fair_norm: str = "none",
                         diag_fallback: bool = True) -> Dict[int, Dict]:
        scales = self._sorted_scales()
        if layer_ids is None:
            layer_ids = sorted(set(l for (l,_) in self.trace_sum.keys()))

        results = {}
        for l in layer_ids:
            # 1) 曲率密度（跨尺度等权的 trace/token，用于层间可比）
            curv = 0.0; cnt = 0
            for s in scales:
                k = (l, s)
                if k in self.trace_sum and self.tokens.get(k,0) > 0:
                    curv += self.trace_sum[k] / max(1, self.tokens[k])
                    cnt += 1
            curvature_density = float(curv / max(1, cnt))

            # 2) OBS 代理
            W = get_weight_fn(l)  # [out, in]
            if W is None:
                results[l] = {"curvature_density": curvature_density}
                continue

            # 统一到 CPU / float32（避免 CPU×CUDA 混算）
            Wc = W.detach().to(self.store_device, dtype=self.stat_dtype)
            col_energy = (Wc * Wc).sum(dim=0)  # [in_features] on CPU

            in_features = Wc.shape[1]
            have_full = all(((l,s) in self.H_sum) for s in scales) and in_features <= self.max_full_dim

            col_scores = None
            mode = "weights_only"

            if have_full:
                H = self.fair_H_for_layer(l, how=fair_how, norm=fair_norm)
                if H is None:
                    have_full = False
            if have_full:
                # H 在 CPU / float32；做阻尼 + Cholesky + inv diag
                Hc = H  # already on CPU
                diag_mean = float(torch.diag(Hc).mean().item())
                lam = float(damping_perc) * max(1e-12, diag_mean)
                Hd = Hc.clone()
                Hd.diagonal().add_(lam)
                try:
                    L = torch.linalg.cholesky(Hd)
                    Hinv = torch.cholesky_inverse(L)
                    Hinv_diag = Hinv.diag().clamp_min(1e-12)
                    col_scores = (col_energy / Hinv_diag).cpu().numpy()
                    mode = "full"
                except Exception:
                    have_full = False

            if not have_full and diag_fallback:
                # 等权的 per-token diag(H)（CPU/float32）
                Hjj = None; S = 0
                for s in scales:
                    k = (l, s)
                    if k in self.diag_sum and self.tokens.get(k,0)>0:
                        hbar_j = self.diag_sum[k] / max(1, self.tokens[k])  # [C] on CPU
                        Hjj = hbar_j if Hjj is None else Hjj + hbar_j
                        S += 1
                if Hjj is None or S==0 or Hjj.numel()!=Wc.shape[1]:
                    col_scores = col_energy.cpu().numpy()
                    mode = "weights_only"
                else:
                    Hjj = (Hjj / S).to(self.stat_dtype).clamp_min(1e-12)
                    col_scores = (col_energy * Hjj).cpu().numpy()
                    mode = "diag"

            # 头或列聚合
            if headsize > 1 and (Wc.shape[1] % headsize == 0):
                Hn = Wc.shape[1] // headsize
                head_scores = np.array([col_scores[h*headsize:(h+1)*headsize].sum() for h in range(Hn)])
            else:
                head_scores = col_scores

            summary = {
                "mean": float(np.mean(head_scores)),
                "median": float(np.median(head_scores)),
                "p25": float(np.percentile(head_scores, 25)),
                "p75": float(np.percentile(head_scores, 75)),
                "min": float(np.min(head_scores)),
                "max": float(np.max(head_scores)),
                "num_heads_or_cols": int(len(head_scores)),
            }
            results[l] = {
                "curvature_density": curvature_density,
                "obs_scores": {
                    "mode": mode,
                    "per_head_or_col": head_scores.tolist(),
                    "summary": summary
                }
            }
        return results

    # ---------- 收敛 ----------
    @torch.no_grad()
    def checkpoint(self):
        scales = self._sorted_scales()
        for s in scales:
            vals = []
            for (l, ss), tr in self.trace_sum.items():
                if ss != s: continue
                N = max(1, self.tokens[(l, ss)])
                vals.append(tr / N)
            if not vals: continue
            v = float(np.mean(vals))
            self.history.setdefault(s, []).append({"trace_per_token": v})

    @torch.no_grad()
    def is_converged(self, eps_trace=0.01, patience=3, min_tokens_per_scale=5000):
        ok_map = {}
        for s, hist in self.history.items():
            tot_tokens = sum(self.tokens.get((l, s), 0) for (l, ss) in self.tokens.keys() if ss==s)
            if tot_tokens < min_tokens_per_scale or len(hist) < patience:
                ok_map[s] = False; continue
            tail = [h["trace_per_token"] for h in hist[-patience:]]
            base = max(1e-12, abs(np.mean(tail)))
            ok_map[s] = (max(tail)-min(tail))/base <= eps_trace
        return (len(ok_map)>0 and all(ok_map.values())), ok_map

    # ---------- 可视化 ----------
    def save_layer_scale_heatmap(self, out_dir="hcurv"):
        os.makedirs(out_dir, exist_ok=True)
        layers = sorted(set(l for (l,_) in self.trace_sum.keys()))
        scales = self._sorted_scales()
        if not layers or not scales: return None
        M = np.zeros((len(layers), len(scales)), dtype=np.float64)
        for i,l in enumerate(layers):
            for j,s in enumerate(scales):
                k = (l,s)
                if k in self.trace_sum and self.tokens.get(k,0)>0:
                    M[i,j] = self.trace_sum[k] / max(1, self.tokens[k])
        plt.figure(figsize=(max(6, len(scales)*0.6), max(4, len(layers)*0.4)))
        plt.imshow(M, aspect='auto')
        plt.colorbar(label="trace/token")
        plt.yticks(range(len(layers)), [f"L{l}" for l in layers])
        plt.xticks(range(len(scales)), scales, rotation=45, ha='right')
        plt.title("Per-layer × Per-scale curvature (trace/token)")
        plt.tight_layout()
        p = os.path.join(out_dir, "layer_scale_trace_heatmap.png")
        plt.savefig(p, dpi=150); plt.close()
        return p

    def save_layer_importance_bar(self, results: Dict[int, Dict], out_dir="hcurv", topk=None, key="curvature_density"):
        os.makedirs(out_dir, exist_ok=True)
        items = sorted([(l, v.get(key, 0.0)) for l, v in results.items()], key=lambda x: x[1], reverse=True)
        if topk is not None: items = items[:topk]
        xs = [f"L{l}" for l,_ in items]; ys = [v for _,v in items]
        if not xs: return None
        plt.figure(figsize=(max(6, len(xs)*0.5), 4))
        plt.bar(xs, ys)
        plt.ylabel(key)
        plt.title(f"Layer importance by {key}")
        plt.tight_layout()
        p = os.path.join(out_dir, f"layer_importance_{key}.png")
        plt.savefig(p, dpi=150); plt.close()
        return p

    def dump_metrics_json(self, out_path="hcurv/metrics_layer_scale.json"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.layer_scale_metrics(), f, indent=2)
        return out_path
        # ---------- 额外可视化：收敛曲线 ----------
    def save_convergence_trace(self, out_dir="hcurv", fname="convergence_trace.png"):
        os.makedirs(out_dir, exist_ok=True)
        if not self.history:
            return None
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        for s, hist in self.history.items():
            ys = [h["trace_per_token"] for h in hist]
            xs = list(range(1, len(ys)+1))
            plt.plot(xs, ys, label=str(s))
        plt.xlabel("checkpoint index")
        plt.ylabel("trace/token")
        plt.title("Convergence trace per scale (avg over layers)")
        plt.legend(ncol=2, fontsize=8)
        p = os.path.join(out_dir, fname)
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        return p

    def save_diag_heatmap(self, out_dir="hcurv", fname="diag_heatmap.png", mode="cv"):
        """
        mode:
          - 'mean'   : mean(diag)/token（与你的 trace/token 只差常数，图样会很像）
          - 'cv'     : diag 各维的变异系数 std(diag_per_feature)/mean(diag_per_feature)，反映“能量是否集中到少数维度”
          - 'srdiag' : 对角稳定秩 proxy: (sum d)^2 / sum(d^2)，d = diag(H_bar)（只用对角，和完整稳定秩不同，但能反映集中度）
        """
        import numpy as np, matplotlib.pyplot as plt, os
        os.makedirs(out_dir, exist_ok=True)
        layers = sorted(set(l for (l,_) in self.trace_sum.keys()))
        scales = self._sorted_scales()
        if not layers or not scales: return None

        M = np.zeros((len(layers), len(scales)), dtype=np.float64)
        eps = 1e-12
        for i,l in enumerate(layers):
            for j,s in enumerate(scales):
                k = (l,s)
                N = self.tokens.get(k, 0)
                if (k not in self.diag_sum) or (N <= 0):
                    M[i,j] = np.nan; continue
                d = (self.diag_sum[k] / float(N)).to(torch.float64).cpu().numpy()  # per-feature diag(H_bar)
                if mode == "mean":
                    M[i,j] = float(np.mean(d))
                elif mode == "cv":
                    mu = float(np.mean(d) + eps)
                    sd = float(np.std(d))
                    M[i,j] = sd / mu
                elif mode == "srdiag":
                    num = float(np.sum(d))**2
                    den = float(np.sum(d*d) + eps)
                    M[i,j] = num / den
                else:
                    raise ValueError(mode)

        plt.figure(figsize=(max(6, len(scales)*0.6), max(4, len(layers)*0.4)))
        im = plt.imshow(M, aspect='auto')
        plt.colorbar(im, label=f"diag metric ({mode})")
        plt.yticks(range(len(layers)), [f"L{l}" for l in layers])
        plt.xticks(range(len(scales)), scales, rotation=45, ha='right')
        plt.title(f"Per-layer × Per-scale diag metric: {mode}")
        plt.tight_layout()
        p = os.path.join(out_dir, fname)
        plt.savefig(p, dpi=150); plt.close()
        return p
    # ---------- 额外可视化：阻尼 L-curve（指定层） ----------
    @torch.no_grad()
    def save_damping_lcurve(self, layer_id: int, get_weight_fn,
                            fair_how="equal_scales", fair_norm="none",
                            percs=(1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1),
                            out_dir="hcurv", fname="damping_lcurve.png"):
        import numpy as np, matplotlib.pyplot as plt, os, math
        os.makedirs(out_dir, exist_ok=True)

        # 拿 H（若没有整块 H，直接退化为对角近似曲线）
        H = self.fair_H_for_layer(layer_id, how=fair_how, norm=fair_norm)
        W = get_weight_fn(layer_id)
        if W is None:
            return None

        # CPU / 升精度
        Wc = W.detach().to(self.store_device, dtype=torch.float64)
        col_energy = (Wc * Wc).sum(dim=0)  # [in]
        finite_points = []

        if H is not None:
            Hc = H.to(self.store_device, torch.float64)
            # 数值对称化
            Hc = 0.5 * (Hc + Hc.transpose(0,1))
            d = torch.diag(Hc).clamp_min(1e-18)
            dmean = float(d.mean().item())
            lam_base = max(1e-12, 1e-6 * dmean)   # 底座阻尼

            lam_list, loss_proxy, kappa_est = [], [], []
            for p in percs:
                lam = max(float(p) * dmean, lam_base)
                Hd = Hc.clone()
                Hd.diagonal().add_(lam)
                try:
                    L = torch.linalg.cholesky(Hd)
                    Hinv = torch.cholesky_inverse(L)
                    Hinv_diag = Hinv.diag().clamp_min(1e-18)
                    loss = float((col_energy / Hinv_diag).sum().item())
                    # 条件数粗估（对角上界近似）
                    dmax = float(d.max().item()); dmin = float(d.min().item())
                    kappa = (dmax + lam) / max(1e-18, dmin + lam)
                    lam_list.append(lam); loss_proxy.append(loss); kappa_est.append(kappa)
                except RuntimeError:
                    lam_list.append(lam); loss_proxy.append(float("nan")); kappa_est.append(float("nan"))

            lam_np = np.array(lam_list); loss_np = np.array(loss_proxy); kap_np = np.array(kappa_est)
            m1 = np.isfinite(loss_np); m2 = np.isfinite(kap_np)
            finite_points.append(np.count_nonzero(m1 & m2))

            # 如果没有一个点是有效的，就退化到对角近似
            if np.count_nonzero(m1) == 0:
                H = None
            else:
                # 画图（只画有效点）
                plt.figure(figsize=(8,4))
                ax1 = plt.subplot(1,2,1)
                ax1.set_xscale("log"); ax1.set_yscale("log")
                ax1.plot(lam_np[m1], loss_np[m1], marker="o")
                ax1.set_xlabel("lambda"); ax1.set_ylabel("OBS total proxy")
                ax1.set_title(f"L{layer_id} loss vs damping")
                ax2 = plt.subplot(1,2,2)
                ax2.set_xscale("log")
                ax2.plot(lam_np[m2], kap_np[m2], marker="o")
                ax2.set_xlabel("lambda"); ax2.set_ylabel("kappa_est ~ (max_diag+λ)/(min_diag+λ)")
                ax2.set_title("conditioning vs damping")
                plt.tight_layout()
                p = os.path.join(out_dir, fname)
                plt.savefig(p, dpi=150); plt.close()
                return p

        # —— 退化：对角近似 L-curve（保证有图）——
        # 等权的 per-token diag(H)（CPU/float32→64）
        Hjj = None; S = 0
        scales = self._sorted_scales()
        for s in scales:
            k = (layer_id, s)
            if k in self.diag_sum and self.tokens.get(k,0)>0:
                hbar_j = (self.diag_sum[k] / max(1, self.tokens[k])).to(torch.float64)
                Hjj = hbar_j if Hjj is None else Hjj + hbar_j
                S += 1
        if Hjj is None or S==0:
            return None

        Hjj = (Hjj / S).clamp_min(1e-18)
        dmean = float(Hjj.mean().item())
        lam_base = max(1e-12, 1e-6 * dmean)

        lam_list, loss_proxy, kappa_est = [], [], []
        for p in percs:
            lam = max(float(p) * dmean, lam_base)
            loss = float((col_energy * (Hjj + lam)).sum().item())  # 近似：1/(Hinv_jj)≈(H_jj+λ)
            dmax = float(Hjj.max().item()); dmin = float(Hjj.min().item())
            kappa = (dmax + lam) / max(1e-18, dmin + lam)
            lam_list.append(lam); loss_proxy.append(loss); kappa_est.append(kappa)

        lam_np = np.array(lam_list); loss_np = np.array(loss_proxy); kap_np = np.array(kappa_est)
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(1,2,1)
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.plot(lam_np, loss_np, marker="o")
        ax1.set_xlabel("lambda"); ax1.set_ylabel("OBS total proxy (diag)")
        ax1.set_title(f"L{layer_id} loss vs damping (diag)")
        ax2 = plt.subplot(1,2,2)
        ax2.set_xscale("log")
        ax2.plot(lam_np, kap_np, marker="o")
        ax2.set_xlabel("lambda"); ax2.set_ylabel("kappa_est (diag)")
        ax2.set_title("conditioning vs damping (diag)")
        plt.tight_layout()
        p = os.path.join(out_dir, fname)
        plt.savefig(p, dpi=150); plt.close()
        return p

# ===== end curvature_monitor_v2.py =====

 

# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    # use_cache = model.config.use_cache
    # model.config.use_cache = False

    # layers = model.model.layers
    layers = model.blocks
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    # model.config.use_cache = use_cache
    return float(count) / total_params


def get_model(model_dir):
    # def skip(*args, **kwargs):
    #     pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    model =  LlamaForCausalLM.from_pretrained(
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        pretrained_model_name_or_path = model_dir
    )
    model.seqlen = 2048

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_dir, 
                                              use_fast=False)
    tokenizer.bos_token = '<s>'  # token_id 1
    tokenizer.eos_token = tokenizer.pad_token = tokenizer.unk_token = '</s>'  # token_id 2 
    return model, tokenizer


class Catcher(nn.Module):
    def __init__(self, seqlen, hidden_size, num_samples, batch_samples, cache_dev='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.layer_inputs = torch.zeros(
            (num_samples, seqlen, hidden_size), 
            dtype=dtype, device=cache_dev
        )
        if cache_dev == 'cpu':
            self.batch_inputs = torch.zeros(
                (batch_samples, seqlen, hidden_size), 
                dtype=dtype, device='cuda'
            )
        self.batch_samples = batch_samples
        self.row_idx = 0
        self.batch_idx = 0
        self.attention_mask = None
        self.cache_dev = cache_dev

    def forward(self, inputs, **kwargs):
        if self.cache_dev == 'cpu':
            self.batch_inputs[self.row_idx] = inputs
            self.row_idx += 1
            if self.row_idx == self.batch_samples:
                self.layer_inputs[self.batch_idx: self.batch_idx + self.batch_samples] = self.batch_inputs.to(self.cache_dev)
                self.row_idx = 0
                self.batch_idx += 1
        else:
            self.layer_inputs[self.row_idx] = inputs
            self.row_idx += 1            

        if self.attention_mask is None and kwargs["attention_mask"] is not None:
            self.attention_mask = kwargs["attention_mask"]

        raise ValueError


def prepare_calibration_input(args, model, dataloader, device="cuda"):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.num_samples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def last_input(args, model, dataloader, layer): #单次获得每个layer的 逐layer
    layers = model.blocks

    # 为每一层准备 cache 和 outs
    # num_layers = len(layers)
    num_batches = len(dataloader)
    caches = [None for _ in range(num_batches*10)]
    outs = [None for _ in range(num_batches*10)] 
    class Catcher(nn.Module):
        def __init__(self, module, layer_idx,batch_idx):
            super().__init__()
            self.module = module
            self.layer_idx = layer_idx
            self.batch_idx = batch_idx
        def forward(self, x, **kwargs):
            
            caches[self.layer_idx][self.batch_idx] = {
                "i": self.batch_idx,
                "x": x,
                "cond_BD": kwargs.get("cond_BD", None),
            }
            # outs[self.layer_idx][self.batch_idx] = {
            #     "i": self.batch_idx,
            #     "x": None,
            #     "cond_BD": kwargs.get("cond_BD", None),
            # }
            self.batch_idx += 1

            return self.module(x, **kwargs)

    # 将 Catcher 应用到每一层
    # 所有层都会被捕获
    
    layers[layer] = Catcher(layers[layer], layer,batch_idx=0)
    try:
        for batch_idx, batch in enumerate(dataloader):
            model(batch)
    except ValueError:
        pass
    layers[layer] = layers[layer].module


    cond_BD_or_gss = None
    return cond_BD_or_gss, outs, cond_BD_or_gss, caches



def get_module_by_name(layer, name):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module


# @torch.no_grad()
def model_slimming(model, dataloader, args):

    batch = len(dataloader)

    

    dev = "cuda" if torch.cuda.is_available() else 'cpu'

    layers = model.blocks

    var = model
    with torch.no_grad():

        t1 = time.time()

        num_batches = len(dataloader)
        print("pruning...")

        caches = [None for _ in range(num_batches*10)]
        outs = [None for _ in range(num_batches*10)] 
        for i in range(len(layers)):
            
            layer = layers[i].to(dev)
            if args.minlayer <= i < args.maxlayer:
                all_module_dict = find_layers(layer)

            sequential = [####################################
                ["attn.proj"],
                ["ffn.fc2"],
            ]
            for names in sequential:
                module_dict = {name: all_module_dict[name] for name in names}
                pruner_dict = {}
                for name in module_dict:
                    pruner_dict[name] = SlimGPT(module_dict[name], i, args)  # 对每一个层都初始化一个剪枝器

                def add_batch(name):
                    def func(_, inp, out):
                        pruner_dict[name].add_batch_v7(inp[0].data, out.data)  # calculate H
                    return func


                handles = []
                for name in module_dict:
                    handles.append(module_dict[name].register_forward_hook(add_batch(name))) #注册钩子
                for b in model.blocks: b.attn.kv_caching(True)

                for batch_idx, batch in enumerate(dataloader):
                    model(batch)

                for b in model.blocks: b.attn.kv_caching(False)
                for h in handles:
                    h.remove()
                
                for name in module_dict:
                    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                    print(f"layer {i}: {name} sparsity {sparsity}")
                    if args.prune_method == "slimgpt":
                        idx = pruner_dict[name].struct_prune(    #执行剪枝操作
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "magnitude":
                        idx = pruner_dict[name].magnitude_prune(    #执行剪枝操作
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "taylor":
                        idx = pruner_dict[name].taylor_prune(    #执行剪枝操作
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    pruner_dict[name].free()
                    target_layer = get_module_by_name(model.blocks[i], name)
                    if name == "ffn.fc2":
                        target_layer_b = get_module_by_name(model.blocks[i], "ffn.fc1")
                        idx = idx.tolist()
                        tp.prune_linear_in_channels(target_layer,idx)
                        tp.prune_linear_out_channels(target_layer_b,idx)
                    elif name == "attn.proj" :
                        # model.blocks[i].attn.pruned_indices.copy_(idx.int())
                        sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                        model.blocks[i].attn.num_heads = torch.round(torch.tensor(model.num_heads*(1-sparsity))).int()
                        idx_m = idx.to(dtype=torch.long)    
                        idx = idx.tolist()
                        keep_idxs = list(set(range(target_layer.in_features)) - set(idx))
                        model.blocks[i].attn.q_bias = nn.Parameter(model.blocks[i].attn.q_bias.data[keep_idxs])
                        zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
                        model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
                        model.blocks[i].attn.v_bias = nn.Parameter(model.blocks[i].attn.v_bias.data[keep_idxs])
                        model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
                        torch.full(size=(1, model.blocks[i].attn.num_heads, 1, 1),fill_value=4.0,device='cuda').log(),requires_grad=True)
                        target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
                        tp.prune_linear_in_channels(target_layer, idx) #proj的inchannel
                        
                        hidden = 16 * 64

                        rm_feat_q = idx_m                                # 直接就是单段里的通道
                        # 映射到合并 QKV 的 out 索引（mat_qkv 的第0维）
                        rm_qkv = torch.cat([rm_feat_q,
                                            rm_feat_q + hidden,
                                            rm_feat_q + 2*hidden], dim=0)   # [3m]

                        rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
                        tp.prune_linear_out_channels(target_layer_b,rm_qkv_list )

            del pruner_dict   
            print(model.blocks[i].ffn.fc1.weight.shape)
            print(model.blocks[i].ffn.fc2.weight.shape)
            print(model.blocks[i].attn.mat_qkv.weight.shape)
            print(model.blocks[i].attn.proj.weight.shape)

            del layer
            torch.cuda.empty_cache()

    print("本次评估用用时为{}".format(time.time()-t1))
    
    return model




def main(args):
    print('load model...')
    # model, tokenizer = get_model(args.model_path)
    MODEL_DEPTH =  args.maxlayer   # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {12,16, 20, 24, 30}
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = '/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth', f'/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt): print("var not exist")
    if not osp.exists(var_ckpt): print("var not exist")
    # if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    # if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False
        )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=False)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')
    model = var
    tokenizer = None

    # # model.seqlen = args.seqlen
    model.eval()
    
    args.minlayer = max(args.minlayer, 0)
    args.maxlayer = min(args.maxlayer,36)

    if args.non_uniform:
        assert 0 <= args.min_sparsity <= args.max_sparsity < 1        
        if args.non_uniform_strategy in ('log_increase', 'log_decrease'):
            linear_space = np.arange(0, args.maxlayer - args.minlayer)
            args.sparsity = args.min_sparsity + (args.max_sparsity - args.min_sparsity) / np.log(32) * np.log(1 + linear_space)
            args.sparsity = [0] * args.minlayer + list(args.sparsity)
            if args.non_uniform_strategy == 'log_decrease':
                args.sparsity = args.sparsity[::-1]
        elif args.non_uniform_strategy in ('linear_increase', 'linear_decrease'):
            sparsity_grad = (args.max_sparsity - args.min_sparsity) / (args.maxlayer - 1 - args.minlayer)
            args.sparsity = [(i - args.minlayer) * sparsity_grad + args.min_sparsity for i in range(args.minlayer, args.maxlayer)]
            args.sparsity = [0] * args.minlayer + args.sparsity
            if args.non_uniform_strategy == 'linear_decrease':
                args.sparsity = args.sparsity[::-1]
        else:
            raise Exception

    state_dict = model.state_dict()
    # print(state_dict.keys())
    layer_params = round(sum(v.numel() for k,v in state_dict.items() if k not in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    extra_params = round(sum(v.numel() for k,v in state_dict.items() if k in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    # sparsity_avg = sum(args.sparsity) / len(args.sparsity) if isinstance(args.sparsity, list) else args.sparsity
    print(f'all params: {layer_params + extra_params} B\t layer params: {layer_params} B\t extra params: {extra_params} B')

    print('load dataset...')
    # dataloader = get_loaders(
    #     args.dataset, 
    #     num_samples=args.num_samples, 
    #     seqlen=model.seqlen,
    #     tokenizer=tokenizer
    # )
    dataloader = torch.arange(0,args.num_samples).cuda()
        # 用 torch.arange 生成数据
    dataset = torch.arange(0, args.num_samples)
    # 包装成 TensorDataset
    tensor_dataset = TensorDataset(dataset)
    # 创建 DataLoader，每次 batch_size=10，按顺序不打乱
    dataloader = DataLoader(tensor_dataset, batch_size=10, shuffle=False)

    # dataloader = torch.randint(0, 1000, (args.num_samples,)).cuda()

    num_samples = len(dataloader)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')

    def _build_tap_regex(args):
        if args.curv_tap == "proj":
            return r".*attn\.proj$"
        if args.curv_tap == "fc2":
            return r".*ffn\.fc2$"
        return args.curv_tap_regex

    def _parse_scales(s: str):
        return [int(x) for x in s.split(",") if x.strip()]

    def _get_weight_fn(var_model, tap: str):
        if tap == "proj":
            def get_w(lid: int):
                try:    return var_model.blocks[lid].attn.proj.weight.data
                except: return None
            return get_w, 64
        elif tap == "fc2":
            def get_w(lid: int):
                try:    return var_model.blocks[lid].ffn.fc2.weight.data
                except: return None
            return get_w, 1
        else:
            # 默认当 proj 处理
            def get_w(lid: int):
                try:    return var_model.blocks[lid].attn.proj.weight.data
                except: return None
            return get_w, 64

    def run_curv_monitor(var_model, dataloader, args):
        tap_regex = _build_tap_regex(args)
        scales_order = _parse_scales(args.curv_scales)
        stat_dtype = parse_dtype(args.curv_stat_dtype)
        monitor = CurvatureMonitor(
            model=var_model,
            tap_regex=tap_regex,
            regex_layer=args.curv_regex_layer,
            include_types=(nn.Linear,),
            max_full_dim=args.curv_max_full_dim,
            scale_tokens_order=scales_order,
            key_prefix="T",
            store_device=args.curv_store_device,
            work_device=args.curv_work_device,
            stat_dtype=stat_dtype,
            stream_chunk=args.curv_stream_chunk,
        )
        # 采集（可与后续剪枝使用相同 dataloader；推荐 inference_mode+autocast 省显存）
        for step, batch in enumerate(dataloader, 1):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                _ = var_model(batch)
            if step % max(1, args.curv_ckpt_every) == 0:
                monitor.checkpoint()

        os.makedirs(args.curv_outdir, exist_ok=True)
        monitor.save_layer_scale_heatmap(out_dir=args.curv_outdir)
        monitor.dump_metrics_json(os.path.join(args.curv_outdir, "metrics_layer_scale.json"))

        get_w_fn, default_head = _get_weight_fn(var_model, args.curv_tap)
        headsz = args.curv_headsize if args.curv_headsize>0 else default_head

        results = monitor.layer_importance(
            get_weight_fn=get_w_fn,
            headsize=headsz,
            damping_perc=args.curv_damping,
            fair_how=args.curv_fair_how,
            fair_norm=args.curv_fair_norm,
            diag_fallback=True
        )
        monitor.save_layer_importance_bar(results, out_dir=args.curv_outdir, key="curvature_density")
        # 你也可以把 per_head/col 的打分保存一份：
        with open(os.path.join(args.curv_outdir, "layer_importance.json"), "w") as f:
            json.dump(results, f, indent=2)
            # 额外三张图
        if args.curv_plot_convergence:
            monitor.save_convergence_trace(out_dir=args.curv_outdir, fname="convergence_trace.png")
        if args.curv_plot_diag_heatmap:
            monitor.save_diag_heatmap(out_dir=args.curv_outdir,
                              fname="diag_heatmap.png",
                              mode=args.curv_diag_mode)
        if args.curv_plot_damping:
            percs = tuple(float(x) for x in args.curv_damping_sweep.split(",") if x.strip())
            monitor.save_damping_lcurve(
                layer_id=args.curv_damping_layer,
                get_weight_fn=get_w_fn,
                fair_how=args.curv_fair_how,
                fair_norm=args.curv_fair_norm,
                percs=percs,
                out_dir=args.curv_outdir,
                fname=args.curv_damping_fname
            )

        monitor.close()
        return results

    if args.curv_enable:
        _ = run_curv_monitor(model, dataloader, args)


    if isinstance(args.sparsity, list) or args.sparsity >= 0 :

        print('start slimming...')
        tick = time.time()
        model= model_slimming(model, dataloader, args)
        print(time.time() - tick)
        
    
    print("*"*30)
    sparsity_ratio = check_sparsity(model)

    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    # print(model)
    state_dict = model.state_dict()
    layer_params = round(sum(v.numel() for k,v in state_dict.items() if k not in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    extra_params = round(sum(v.numel() for k,v in state_dict.items() if k in ('model.embed_tokens.weight','lm_head.weight')) / 10**9, 2)
    # sparsity_avg = sum(args.sparsity) / len(args.sparsity) if isinstance(args.sparsity, list) else args.sparsity
    print(f'all params: {layer_params + extra_params} B\t layer params: {layer_params} B\t extra params: {extra_params} B')
    example_input = torch.tensor([0]).to(device)
    for b in model.blocks: b.attn.kv_caching(True)
    torch.cuda.synchronize()
    zero = time.time()
    result = var(example_input)
    torch.cuda.synchronize()
    end = time.time()
    print("all:{}",end-zero)

    print(model)
    macs, nparams = tp.utils.count_ops_and_params(model, example_input, layer_wise=False)

    print(model(example_input).shape)
    print(
        "  Params: %.2f M => %.2f M"
        % ( nparams / 1e6, nparams / 1e6)
    )
    print(
        "   MACs: %.2f G => %.2f G"
        % ( macs / 1e9, macs / 1e9)
    )


    # save_dir = "/home/wangzefang/edgevar/EdgeVAR/VAR_FIDtest/output/FID_test/d24_test_0.2_200i_temporary"
    # os.makedirs(save_dir,exist_ok=True)
    save_model = "/home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/sparsity_model"
    os.makedirs(save_model,exist_ok=True)
    save_path = os.path.join(save_model, args.model_name)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, 
        default="/home/sumingluo/Model_weight/meta/llama-2-7b-hf", help="model to load"
    )
    parser.add_argument(
        "--dataset", type=str,default="wikitext2",
        choices=["wikitext2", "c4", "alpaca", "gpt4_alpaca"],
        help="Where to extract calibration data from.",
    )
    
    parser.add_argument(
        "--num_samples", type=int, default=1024, 
        help="Number of calibration data samples."
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048, 
        help="Sequence length for the calibration data."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.2, 
        help="Target pruning ratio, which does not take effect when non_uniform is True"
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, 
        help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=32, 
        help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--cache_dev", type=str, default="cuda", 
        help="Defaults to `cuda`. When the GPU memory is insufficient, you can set `cache_dev` to `cpu`, but the trade-off is slower pruning speed."
    )
    parser.add_argument(
        "--batch_samples", type=int, default=128, 
        help="Works when `cache_dev=cpu`. The number of samples loaded onto the GPU each time."
    )
    parser.add_argument(
        "--skip_evaluate", action="store_true",
        help="When set to True, skip the evaluation on Wikitext-2 after the pruning is complete.",
    )
    parser.add_argument(
        "--save_pruned_weights", action="store_true",
        help="Whether save the checkpoint after removing the zeroed-out parameters.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="", 
        help="Path to saved model.",
    )

    # slimgpt & non_uniform config
    parser.add_argument(
        "--non_uniform", action="store_true",
        help="When set to True, use non-uniform pruning, and the parameter sparsity will be ineffective.",
    )
    parser.add_argument(
        "--non_uniform_strategy", type=str, default='log_increase', 
        choices=["log_increase", "log_decrease", "linear_increase", "linear_decrease"],
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--min_sparsity", type=float, default=0.06,
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--max_sparsity", type=float, default=0.3,
        help="Works when `non_uniform=True`",
    )
    parser.add_argument(
        "--no_compensate", action="store_true",
        help="Skip error compensation in SlimGPT",
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, 
        help="Seed for sampling the calibration data."
    ) 
    parser.add_argument(
        "--prune_method", type=str, default="slmipgpt", 
        help="method.",
    )
    parser.add_argument(
        "--specific_layer", type=int, default=0, 
        help="choose which layer to evaluate",
    )
#     parser.add_argument(
#     "--specific_layer", type=int, nargs='+', default=[0],
#     help="choose which layer(s) to evaluate, e.g. --specific_layer 0 1 2"
# )
    parser.add_argument(
        "--model_name", type=str, default="model", 
        help="choose which layer to evaluate",
    )
    parser.add_argument("--curv_enable", action="store_true", help="启用逐层×尺度曲率与重要性分析")
    parser.add_argument("--curv_tap", type=str, default="proj",
                        choices=["proj","fc2","regex"],
                        help="抓取位点：proj=attn.proj输入；fc2=ffn.fc2输入；regex=自定义正则")
    parser.add_argument("--curv_tap_regex", type=str, default=r".*attn\.proj$",
                        help="当 curv_tap=regex 时生效：匹配模块名的正则")
    parser.add_argument("--curv_regex_layer", type=str, default=r"blocks\.(\d+)\.",
                        help="从模块名解析层号的正则")
    parser.add_argument("--curv_max_full_dim", type=int, default=1024,
                        help="仅当输入维度<=该值时保存整块H（proj=1024；fc2=4096会被跳过）")
    parser.add_argument("--curv_store_device", type=str, default="cpu", choices=["cpu"],
                        help="统计存储设备（建议 cpu）")
    parser.add_argument("--curv_work_device", type=str, default="cuda", choices=["cuda","cpu"],
                        help="X^T X 的计算设备")
    parser.add_argument("--curv_stat_dtype", type=str, default="float32",
                        choices=["float32","bf16","float64"],
                        help="统计精度")
    parser.add_argument("--curv_scales", type=str, default="1,4,9,16,25,36,64,100,169,256",
                        help="尺度顺序（token数），逗号分隔")
    parser.add_argument("--curv_stream_chunk", type=int, default=8192,
                        help="流式分块大小（做 X^T X 时每次处理的条目数）")
    parser.add_argument("--curv_outdir", type=str, default="hcurv_out",
                        help="输出目录（图/JSON）")
    parser.add_argument("--curv_ckpt_every", type=int, default=5,
                        help="每多少步做一次曲率历史checkpoint")
    parser.add_argument("--curv_fair_how", type=str, default="equal_scales",
                        choices=["equal_scales","equal_tokens","unit_trace"],
                        help="跨尺度合成策略")
    parser.add_argument("--curv_fair_norm", type=str, default="none",
                        choices=["none","corr"],
                        help="是否对合成前的 H̄_s 做相关化（对角白化）")
    parser.add_argument("--curv_headsize", type=int, default=64,
                        help="head聚合大小（proj=64；fc2一般用1）")
    parser.add_argument("--curv_damping", type=float, default=0.02,
                        help="OBS 阻尼比例（按 mean(diag(H)) 缩放）")
    ############# 曲率分析图保存选项 #############
    parser.add_argument("--curv_plot_convergence", action="store_true",
                    help="保存 convergence_trace.png")
    parser.add_argument("--curv_plot_diag_heatmap", action="store_true",
                        help="保存 diag_heatmap.png（mean(diag)/token）")
    parser.add_argument("--curv_plot_damping", action="store_true",
                        help="保存 damping_lcurve.png（需该层有整块H）")
    parser.add_argument("--curv_damping_layer", type=int, default=0,
                        help="做 L-curve 的层号（默认 L0）")
    parser.add_argument("--curv_damping_sweep", type=str,
                        default="0,0.005,0.01,0.02,0.05,0.1",
                        help="阻尼比例 sweep，逗号分隔（按 mean(diag(H)) 缩放）")
    parser.add_argument("--curv_damping_fname", type=str, default="damping_lcurve.png",
                        help="L-curve 输出文件名")
    parser.add_argument("--curv_diag_mode", type=str, default="cv",
                    choices=["mean","cv","srdiag"],
                    help="diag_heatmap 的度量：mean / cv（默认）/ srdiag(对角稳定秩proxy)")
        
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    main(args)
