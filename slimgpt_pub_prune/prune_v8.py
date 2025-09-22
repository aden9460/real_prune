###全尺度评估改为单个相乘 之后平均
##这里选择 进行1.曲率一致性可视化 2.数值稳定性分析 3.收敛/停采集
import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
from eval import eval_ppl
from DFOBS.slim_utils.slimgpt import SlimGPT
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
                 tap_regex: str = r'.*attn\.proj.*',   # 你也可改成 (q|k|v) 单独的层名
                 include_types: Tuple[type, ...] = (nn.Linear,),
                 max_full_dim: int = 1024,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float64,
                 scale_tokens_order: Optional[List[int]] = None,   # <== 新增
                 key_prefix: str = "T"  ):
        self.model = model
        self.tap_re = re.compile(tap_regex)
        self.include_types = include_types
        self.max_full_dim = max_full_dim
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
                             

        self.key_prefix = key_prefix
        # 显式顺序（例如 [1,4,9,16,25,36,64,100,169,256]）
        self._order_map = None
        if scale_tokens_order is not None:
            self._order_map = {f"{key_prefix}{t}": i for i, t in enumerate(scale_tokens_order)}

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
        T = int(x.shape[1]) if x.dim() >= 2 else 1
        key = f"{self.key_prefix}{T}"
        # 未在预设顺序中的新尺度，动态追加到末尾
        if self._order_map is not None and key not in self._order_map:
            self._order_map[key] = max(self._order_map.values(), default=-1) + 1
        return key

    def _sorted_scales(self) -> List[str]:
        keys = list(self.trace_sum.keys())
        if self._order_map is not None:
            return sorted(keys, key=lambda k: self._order_map.get(k, float('inf')))
        # 兜底：按数值 token 升序
        def _tok_num(k: str) -> int:
            m = re.search(r'(\d+)', k)
            return int(m.group(1)) if m else 10**9
        return sorted(keys, key=_tok_num)

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
        for k in self._sorted_scales():
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
                     # 计算对称正定矩阵的所有特征值
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
                     min_tokens_per_scale: int = 50) -> Tuple[bool, Dict[str, bool]]:
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
        scales = self._sorted_scales()
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


# def prepare_calibration_d16_last_input(args, model, dataloader,device):
    
#     label_B = dataloader.int()

#     layers = model.blocks

#     cache = []
#     outs = []
#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, x, **kwargs):  ##ori 只选取最后一层
#             # inps[cache["i"]] = inp  #
            
#             # cache["cond_BD"] = kwargs["cond_BD"]
#             # cache["attn_bias"] = kwargs["attn_bias"]
#             # inps.append(inp)
#             # print(x.shape)
#             # tokens = x.shape[1]
#             # if tokens ==args.specific_layer :

#             cache.append({
#             "i": len(cache),
#             "x": x,
#             "cond_BD": kwargs["cond_BD"],
#             })
#             outs.append({
#             "i": len(cache),
#             "x": 0,
#             "cond_BD": kwargs["cond_BD"],
#             })
#             raise ValueError

#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch)
#         except ValueError:
#             pass

#     layers[0] = layers[0].module

#     # outs = torch.zeros_like(inps)

#     # position_ids = None
#     cond_BD_or_gss = None
#     return cond_BD_or_gss, outs, cond_BD_or_gss, cache

# def prepare_calibration_d16_last_input(args, model, dataloader, device):
#     layers = model.blocks

#     # 为每一层准备 cache 和 outs
#     num_layers = len(layers)
#     num_batches = len(dataloader)
#     caches = [[None for _ in range(num_batches*10)] for _ in range(num_layers)]
#     outs = [[None for _ in range(num_batches*10)] for _ in range(num_layers)]
#     class Catcher(nn.Module):
#         def __init__(self, module, layer_idx,batch_idx):
#             super().__init__()
#             self.module = module
#             self.layer_idx = layer_idx
#             self.batch_idx = batch_idx
#         def forward(self, x, **kwargs):
            
#             caches[self.layer_idx][self.batch_idx] = {
#                 "i": self.batch_idx,
#                 "x": x,
#                 "cond_BD": kwargs.get("cond_BD", None),
#             }
#             # outs[self.layer_idx][self.batch_idx] = {
#             #     "i": self.batch_idx,
#             #     "x": None,
#             #     "cond_BD": kwargs.get("cond_BD", None),
#             # }
#             self.batch_idx += 1

#             return self.module(x, **kwargs)

#     # 将 Catcher 应用到每一层
#     # 所有层都会被捕获
    
#     for layers_num in range(num_layers):
#         layers[layers_num] = Catcher(layers[layers_num], layers_num, batch_idx=0)
#         print("layers{}".format(layers_num))
#         try:
#             for batch_idx, batch in enumerate(dataloader):
#                 model(batch)
#         except ValueError:
#             pass
#         layers[layers_num] = layers[layers_num].module


#     cond_BD_or_gss = None
#     return cond_BD_or_gss, outs, cond_BD_or_gss, caches



# from typing import Any, Dict, List

# def prepare_calibration_d16_last_input(
#     args,
#     model: nn.Module,
#     dataloader,
#     device,
#     max_steps_per_layer: int = 10,   # 每层采集步数
# ):
#     layers = model.blocks
#     num_layers = len(layers)

#     # 与原函数一致的占位
#     cond_BD_or_gss = None

#     # 形状一致的二维结构：num_layers x max_steps_per_layer x batch
#     caches: List[List[Dict[str, Any]]] = [
#         [None for _ in range(max_steps_per_layer)] for _ in range(num_layers)
#     ]
#     outs: List[List[Dict[str, Any]]] = [
#         [None for _ in range(max_steps_per_layer)] for _ in range(num_layers)
#     ]

#     class Catcher(nn.Module):
#         def __init__(self, module: nn.Module, layer_idx: int):
#             super().__init__()
#             self.module = module
#             self.layer_idx = layer_idx
#             self.step_count = 0  # 已记录的步数

#         def forward(self, x, **kwargs):
#             # 先执行原模块，拿到输出（不改变计算图与数据流）
#             y = self.module(x, **kwargs)

#             # 仅在未达上限时记录
#             if self.step_count < max_steps_per_layer:
#                 idx = self.step_count
#                 # 与你之前结构保持一致：键为 "i"、"x"、"cond_BD"
#                 caches[self.layer_idx][idx] = {
#                     "i": idx,
#                     "x": x,  # 不做 detach / 不搬 CPU，严格保持输入格式
#                     "cond_BD": kwargs.get("cond_BD", None),
#                 }
#                 outs[self.layer_idx][idx] = {
#                     "i": idx,
#                     "x": y,  # 同步记录该步的输出
#                     "cond_BD": kwargs.get("cond_BD", None),
#                 }
#                 self.step_count += 1

#             # 返回原输出，保证后续层正常前向
#             return y

#     # 给所有层套上 Catcher
#     wrapped_layers = []
#     for li in range(num_layers):
#         layers[li] = Catcher(layers[li], li)
#         wrapped_layers.append(layers[li])

#     # 跑数据，直到所有层都采满或数据耗尽
#     try:
#         for batch in dataloader:
#             _ = model(batch)

#             # 全部层都达到上限则提前结束
#             if all(w.step_count >= max_steps_per_layer for w in wrapped_layers):
#                 break
#     finally:
#         # 还原为原始层
#         for li in range(num_layers):
#             layers[li] = layers[li].module

#     # 返回顺序与原函数保持一致
#     return cond_BD_or_gss, outs, cond_BD_or_gss, caches
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


    with torch.no_grad():

        t1 = time.time()

        num_batches = len(dataloader)
        print("pruning...")
                # ===== [NEW] 曲率预扫：可视化 + 数值稳定性 + 收敛/停采 =====
        if getattr(args, "curv_enable", False):
            curv = CurvatureMonitor(
                model=model,
                tap_regex=args.curv_tap_regex,
                include_types=(nn.Linear,),
                max_full_dim=args.curv_max_full_dim,
                device=dev,
                scale_tokens_order=[1,4,9,16,25,36,64,100,169,256],
            )
            # 预扫：不改你的 kv_caching，纯推理
            for step, batch_data in enumerate(dataloader, 1):
                with torch.inference_mode():
                    _ = model(batch_data)
                if step % max(1, args.curv_ckpt_every) == 0:
                    curv.checkpoint()
                    all_ok, per_ok = curv.is_converged(
                        eps_trace=args.curv_eps_trace,
                        eps_lambda=args.curv_eps_lmax,
                        patience=args.curv_patience,
                        min_tokens_per_scale=args.curv_min_tokens_scale
                    )
                    print(f"[curv] step={step} converged={all_ok} per_scale={per_ok}")
                    if all_ok:
                        print("[curv] early stop: curvature stabilized across scales.")
                        break

            paths = curv.save_curvature_report(out_dir=args.curv_outdir)
            print(f"[curv] saved report: {paths}")
            curv.close()
            torch.cuda.empty_cache()
        # ===== [END NEW] =====

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

    # dataloader = torch.randint(0, 1000, (args.num_samples,)).cuda()

    num_samples = len(dataloader)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')
    
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

    # # dot_path = os.path.join(save_model, '0.4var_1i_256input_computational_graph.dot')
    # # dot = make_dot(result, params=dict(model.named_parameters()))
    # # dot.save(dot_path)
    # # for class_num in range(5):
    # class_num = 0
    # class_labels = (980, 980, 437, 437, 22, 22, 562, 562,50,50)
    # # class_labels = (980,)
    # # class_labels  = torch.full((10,), class_num, dtype=torch.long).cuda()
    # label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    # with torch.inference_mode():
    #     with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
    #         for b in model.blocks: b.attn.kv_caching(True)
    #         recon_B3HW = model(label_B)
    #         for i in range(recon_B3HW.shape[0]):
    #             save_image(recon_B3HW[i], f'{save_dir}/{class_num:03d}_img_{i:03d}.png')

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)


    # with torch.inference_mode():
    #     B = len(class_labels)
    #     label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    #     with measure_peak_memory():
    #         with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
    #             for i in range(3):
    #                 start_event.record()
    #                 recon_B3HW = model(label_B)
    #                 end_event.record()
    #                 torch.cuda.synchronize()
    #                 # Calculation run time (milliseconds)
    #                 elapsed_time = start_event.elapsed_time(end_event)
    #                 print("running time:",int(elapsed_time),"ms", "batch size:",str(len(class_labels)))

    # if not args.skip_evaluate:
    #     print('start evaluate...')
    #     dataset = 'wikitext2'
    #     ppl = eval_ppl(model.cuda(), tokenizer, dataset, device=torch.device("cuda:0"))
    #     print(f"\nppl on {dataset}: {ppl}\n")
    #     ppl_metric(model.cuda().half(), tokenizer, ['wikitext2'], 128, 2)

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

    # ======= Curvature monitor args =======
    parser.add_argument("--curv_enable", action="store_true",
                        help="开启曲率可视化/数值稳定性与收敛判定（不影响后续剪枝）")
    parser.add_argument("--curv_outdir", type=str, default="/home/suanba/EdgeVAR/real_prune/slimgpt_pub_prune/hessian_curv_report",
                        help="曲率报告输出目录")
    parser.add_argument("--curv_ckpt_every", type=int, default=5,
                        help="每多少个 batch 记录一次收敛曲线点")
    parser.add_argument("--curv_patience", type=int, default=3,
                        help="最近多少次都满足阈值则判定收敛")
    parser.add_argument("--curv_eps_trace", type=float, default=0.01,
                        help="trace/token 相对变化的收敛阈值（默认1%）")
    parser.add_argument("--curv_eps_lmax", type=float, default=0.05,
                        help="lambda_max 相对变化的收敛阈值（默认5%）")
    parser.add_argument("--curv_min_tokens_scale", type=int, default=5000,
                        help="每个尺度用于收敛判定的最小token数")
    parser.add_argument("--curv_max_full_dim", type=int, default=1024,
                        help="<=该维度时累积完整H用于数值稳定性(L-curve)")
    parser.add_argument("--curv_tap_regex", type=str, default=r".*attn\.proj.*",
                        help="在哪个模块的输入处抓X，默认抓 qkv 合并层的输入")

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    main(args)
