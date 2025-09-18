# ===== curvature_monitor.py =====
import os, json, re
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    # [B,T,C] or [N,C] -> [N,C]
    if x.dim() == 3:
        return x.flatten(0, 1)
    return x

class CurvatureMonitor:
    """
    钩在 attn.mat_qkv（或你指定的模块）输入处，按尺度(token数)累计:
      - trace_sum = Σ ||X||_F^2
      - diag_sum  = Σ diag(X^T X)
      - tokens    = Σ N
      - H         = Σ X^T X  (可选：当 C <= max_full_dim 时)
    还支持：
      - checkpoint(): 记录一次曲线点，用于收敛判定与曲线绘制
      - is_converged(): 判断收敛/停采
      - 可视化：柱状图、热力图、收敛曲线、阻尼 L-curve
    """
    def __init__(self,
                 model: nn.Module,
                 tap_regex: str = r'.*attn\.mat_qkv.*',   # 你也可改成 (q|k|v) 单独的层名
                 include_types: Tuple[type, ...] = (nn.Linear,),
                 max_full_dim: int = 1024,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64):
        self.model = model
        self.tap_re = re.compile(tap_regex)
        self.include_types = include_types
        self.max_full_dim = max_full_dim
        self.device = device or next(model.parameters()).device
        self.dtype = dtype

        # 累计量（按尺度 scale_key='T{tokens}'）
        self.trace_sum: Dict[str, float] = {}
        self.tokens: Dict[str, int] = {}
        self.diag_sum: Dict[str, torch.Tensor] = {}
        self.H_sum: Dict[str, torch.Tensor] = {}     # 可选

        # 曲线历史：记录每次 checkpoint 的 (tokens_total, trace/token, lambda_max 估计)
        self.history: Dict[str, List[Dict[str, float]]] = {}

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _register_hooks(self):
        for name, mod in self.model.named_modules():
            if not isinstance(mod, self.include_types):   # 线性层
                continue
            if self.tap_re.match(name):
                # 在输入处截获
                h = mod.register_forward_pre_hook(lambda m, inp, n=name: self._on_inp(n, m, inp))
                self.hooks.append(h)

    def _scale_key_from_x(self, x: torch.Tensor) -> str:
        # 以 token 数作为尺度标签，例如 T256 / T169 ...
        if x.dim() >= 2:
            T = int(x.shape[1])
        else:
            T = 1
        return f"T{T}"

    def _ensure_scale(self, key: str, C: int, want_full: bool):
        if key not in self.trace_sum:
            self.trace_sum[key] = 0.0
            self.tokens[key] = 0
            self.history.setdefault(key, [])
        if key not in self.diag_sum:
            self.diag_sum[key] = torch.zeros(C, device=self.device, dtype=self.dtype)
        if want_full and key not in self.H_sum:
            self.H_sum[key] = torch.zeros(C, C, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _on_inp(self, name: str, mod: nn.Module, inputs):
        x = inputs[0]   # [B,T,C] (VAR 的 mat_qkv 输入)
        X = _flatten_bt(x).to(self.device, self.dtype)  # [N,C]
        if X.numel() == 0 or X.dim() != 2:
            return
        N, C = X.shape
        key = self._scale_key_from_x(x)

        want_full = (C <= self.max_full_dim)
        self._ensure_scale(key, C, want_full)

        self.trace_sum[key] += float((X*X).sum().item())
        self.tokens[key]    += int(N)
        self.diag_sum[key]  += (X*X).sum(dim=0)

        if want_full:
            self.H_sum[key] += X.t().matmul(X)  # X^T X

    # ---------- 统计 / 指标 ----------
    @torch.no_grad()
    def metrics_now(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k in sorted(self.trace_sum.keys()):
            N = max(1, self.tokens[k])
            tr = self.trace_sum[k]
            dim = int(self.diag_sum[k].numel())
            m = {
                "tokens": float(N),
                "trace_sum": float(tr),
                "trace_per_token": float(tr / N),
                "dim": dim,
                "mean_diag": float(self.diag_sum[k].mean().item()),
            }
            # lambda_max / lambda_min (若有整 H)
            if k in self.H_sum:
                H = self.H_sum[k].detach().cpu()
                try:
                    evals = torch.linalg.eigvalsh(H)  # 对称 PSD
                    lmax = float(evals.max().item())
                    lmin = float(max(evals.min().item(), 1e-12))
                    m["lambda_max"] = lmax
                    m["lambda_min"] = lmin
                    m["cond"] = float(lmax / max(lmin, 1e-12))
                except Exception:
                    pass
            out[k] = m
        return out

    @torch.no_grad()
    def checkpoint(self):
        # 记录一次曲线点（用于收敛曲线）
        m = self.metrics_now()
        for k, v in m.items():
            self.history[k].append({
                "tokens": v["tokens"],
                "trace_per_token": v["trace_per_token"],
                "lambda_max": v.get("lambda_max", float('nan')),
            })

    @torch.no_grad()
    def is_converged(self,
                     eps_trace: float = 0.01,        # 1% 相对变化
                     eps_lambda: float = 0.05,       # 5% 相对变化
                     patience: int = 3,              # 连续几次达标
                     min_tokens_per_scale: int = 5000) -> Tuple[bool, Dict[str, bool]]:
        """
        收敛判定：每个尺度最近 patience 个 checkpoint 的 trace/token 与 lambda_max 的
        最大相对变化都低于阈值，且 tokens >= 最小值。
        返回 (all_ok, per_scale_ok)
        """
        per_ok = {}
        for k, hist in self.history.items():
            if self.tokens.get(k, 0) < min_tokens_per_scale or len(hist) < patience:
                per_ok[k] = False; continue
            tail = hist[-patience:]
            tr_vals = [h["trace_per_token"] for h in tail]
            lm_vals = [h["lambda_max"] for h in tail if not np.isnan(h["lambda_max"])]
            def _stable(vals, eps):
                vmin, vmax = min(vals), max(vals)
                base = max(1e-12, abs(np.mean(vals)))
                return (vmax - vmin) / base <= eps
            ok_trace = _stable(tr_vals, eps_trace)
            ok_lm    = True if len(lm_vals)==0 else _stable(lm_vals, eps_lambda)
            per_ok[k] = ok_trace and ok_lm
        all_ok = len(per_ok)>0 and all(per_ok.values())
        return all_ok, per_ok

    # ---------- 可视化 ----------
    def save_curvature_report(self, out_dir="hessian_curv", heatmap_width=512) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        m = self.metrics_now()
        scales = list(sorted(m.keys()))
        if len(scales)==0: return {}

        # 1) 柱状图：trace/token
        x = np.arange(len(scales))
        y = [m[s]["trace_per_token"] for s in scales]
        plt.figure()
        plt.bar(x, y)
        plt.xticks(x, scales, rotation=45, ha='right')
        plt.ylabel("trace(X^T X) / token")
        plt.title("Per-scale curvature")
        plt.tight_layout()
        p1 = os.path.join(out_dir, "scale_trace_per_token.png")
        plt.savefig(p1, dpi=150); plt.close()

        # 2) 热力图：diag(H)（按公共宽度重采样）
        mats = []
        for s in scales:
            v = self.diag_sum[s].detach().cpu().float().numpy().reshape(-1)
            xo = np.linspace(0,1,len(v)); xn = np.linspace(0,1,heatmap_width)
            vr = np.interp(xn, xo, v)
            vr = np.log1p(vr - vr.min() + 1e-12)
            mats.append(vr)
        M = np.stack(mats, 0)
        plt.figure()
        plt.imshow(M, aspect='auto')
        plt.yticks(np.arange(len(scales)), scales)
        plt.xlabel("feature (resampled)"); plt.ylabel("scale")
        plt.title("diag(X^T X) (log1p)")
        plt.tight_layout()
        p2 = os.path.join(out_dir, "diag_heatmap.png")
        plt.savefig(p2, dpi=150); plt.close()

        # 3) 收敛曲线：trace/token vs tokens
        plt.figure()
        for s in scales:
            hist = self.history.get(s, [])
            if len(hist)==0: continue
            xs = [h["tokens"] for h in hist]
            ys = [h["trace_per_token"] for h in hist]
            plt.plot(xs, ys, marker='o', label=s)
        plt.xlabel("tokens seen"); plt.ylabel("trace/token")
        plt.title("Convergence of curvature per scale")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        p3 = os.path.join(out_dir, "convergence_trace.png")
        plt.savefig(p3, dpi=150); plt.close()

        # 4) 阻尼 L-curve：cond(H+λI) 随 λ 变化（仅对有整 H 的尺度）
        lambdas = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        if any(s in self.H_sum for s in scales):
            plt.figure()
            for s in scales:
                if s not in self.H_sum: continue
                H = self.H_sum[s].detach().cpu()
                conds = []
                for lam in lambdas:
                    Hlam = H.clone()
                    idx = torch.arange(Hlam.shape[0])
                    Hlam[idx, idx] += lam * float(torch.diag(H).mean().item())
                    try:
                        evals = torch.linalg.eigvalsh(Hlam)
                        lmax = float(evals.max().item()); lmin = float(max(evals.min().item(), 1e-12))
                        conds.append(lmax / max(lmin,1e-12))
                    except Exception:
                        conds.append(np.nan)
                plt.plot(lambdas, conds, marker='o', label=s)
            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("damping λ (× mean diag)"); plt.ylabel("cond(H+λI)")
            plt.title("Damping L-curve")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            p4 = os.path.join(out_dir, "damping_lcurve.png")
            plt.savefig(p4, dpi=150); plt.close()
        else:
            p4 = ""

        # 导出数字
        with open(os.path.join(out_dir, "metrics_now.json"), "w") as f:
            json.dump(m, f, indent=2)
        return {"trace_bar": p1, "diag_heat": p2, "conv_curve": p3, "lcurve": p4}
# ===== end curvature_monitor.py =====






# 1) 初始化监控器（放在 model、dataloader 准备好之后）
from curvature_monitor import CurvatureMonitor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
curv = CurvatureMonitor(
    model=var,                      # 你的 VAR_d16 模型实例
    tap_regex=r".*attn\.mat_qkv.*", # 钩在 qkv 合并线性层的输入
    include_types=(nn.Linear,),
    max_full_dim=1024,              # d_model=1024 时会攒整 H
    device=device,
)

# 2) 逐 batch 前向并周期性 checkpoint + 收敛判定
CKPT_EVERY = 5            # 每多少个 batch 记录一次曲线点
PATIENCE   = 3            # 最近3次都稳定
EPS_TRACE  = 0.01         # trace/token 收敛阈值 1%
EPS_LMAX   = 0.05         # lambda_max 收敛阈值 5%
MIN_TOKENS = 5000         # 每个尺度至少见到这么多 token 才能谈收敛

for step, batch in enumerate(dataloader):
    with torch.inference_mode():
        _ = var(batch)    # 你的 dataloader 对 VAR 是类别/条件标签，和原流程一致

    if (step+1) % CKPT_EVERY == 0:
        curv.checkpoint()
        all_ok, per_ok = curv.is_converged(
            eps_trace=EPS_TRACE, eps_lambda=EPS_LMAX,
            patience=PATIENCE, min_tokens_per_scale=MIN_TOKENS
        )
        print(f"[curv] step={step+1} converged={all_ok} per_scale={per_ok}")
        if all_ok:
            print("[curv] early stop: curvature stabilized across scales.")
            break

# 3) 导出可视化报告
paths = curv.save_curvature_report(out_dir="hessian_curv_report")
print("Saved:", paths)

# 4) 用完记得卸钩
curv.close()
