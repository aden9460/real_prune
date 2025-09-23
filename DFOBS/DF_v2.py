import time
import os
import os.path as osp
import sys
import gc
import argparse
from dataclasses import dataclass
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed

# third-party / project deps
from torchvision.utils import save_image  # noqa: F401 (kept for compatibility)
from torchviz import make_dot  # noqa: F401 (kept for compatibility)

# local project utils
from slim_utils.slimgpt import SlimGPT, WoodTaylorSlim
from slim_utils.slim_dataset import get_loaders  # noqa: F401 (kept if you switch to real dataset)
from slim_utils.params_remove import LLaMAParamsPruner  # noqa: F401 (kept for compatibility)

# torch-pruning
sys.path.append("/home/waas/EdgeVAR/Torch-Pruning")

import torch_pruning as tp
import torch_pruning.pruner.function as tfun  # noqa: F401 (kept for compatibility)

# models (VAR / VQVAE)
from models import VQVAE, build_vae_var

# optionally extend PYTHONPATH


# speedup: disable default parameter init
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


# ======================
# Utilities
# ======================

@dataclass
class VarTraceConfig:
    gamma: float = 0.5
    ema_rho: float = 0.95
    use_exact_per_scale: bool = True
    damping_mu: float = 0.0
    hutchinson_K: int = 4
    fisher_weight_per_token: bool = True
    device: Optional[torch.device] = None


class FisherTraceEMA:
    def __init__(self, num_layers: int, rho: float = 0.95, device: Optional[torch.device] = None):
        self.ema = torch.zeros(num_layers, device=device)
        self.rho = rho

    def update(self, current: torch.Tensor):
        self.ema = self.rho * self.ema + (1.0 - self.rho) * current
        return self.ema

    def value(self) -> torch.Tensor:
        return self.ema


def measure_peak_memory():
    class _Ctx:
        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
            return self

        def __exit__(self, exc_type, exc, tb):
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f'memory consumption: {peak:.2f} MB')
    return _Ctx()


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=''):
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def check_sparsity(model):
    """注意：结构化剪枝改变的是形状，不一定产生0值；此函数更适合非结构化稀疏。"""
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
        print(f"layer {i} sparsity (zeros/total) {float(sub_count)/max(1,sub_params):.6f}")
    return float(count) / max(1, total_params)


def get_module_by_name(layer, name: str):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module


def _zero_grads(parameters: Optional[Sequence[nn.Parameter]]):
    if parameters is None:
        return
    for p in parameters:
        if p.grad is not None:
            p.grad.zero_()


def squared_grad_per_layer_from_loss(
    loss: torch.Tensor,
    modules_per_layer: List[List[nn.Parameter]],
    retain_graph: bool,
) -> torch.Tensor:
    flat_params: List[nn.Parameter] = [p for group in modules_per_layer for p in group]
    _zero_grads(flat_params)
    loss.backward(retain_graph=retain_graph)
    sq = []
    for layer_params in modules_per_layer:
        s = 0.0
        for p in layer_params:
            if p.grad is not None:
                s += (p.grad.detach() ** 2).sum()
        sq.append(torch.as_tensor(s, device=loss.device))
    return torch.stack(sq)


def allocate_pruning_rates(
    traces: torch.Tensor,
    dims: torch.Tensor,
    global_keep_ratio: float,
    gamma: float = 0.5,
    min_alpha: float = 0.05,
    max_alpha: float = 1.0,
    protect_layers: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    eps = 1e-12
    score = traces.clamp_min(eps).pow(gamma) / dims.clamp_min(1.0)
    alpha = score / score.sum() * (global_keep_ratio * len(traces))
    alpha = alpha.clamp(min=min_alpha, max=max_alpha)
    if protect_layers:
        for li in protect_layers:
            alpha[li] = 1.0
    return alpha


def _prepare_batch(batch, device):
    # dataloader yields (tensor,) for TensorDataset
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Empty batch received from dataloader")
        batch = batch[0]
    return batch.to(device)


# ======================
# Fisher per-layer computation
# ======================

def compute_layerwise_fisher(
    model: nn.Module,
    dataloader: DataLoader,
    args,
    device: torch.device,
) -> List[float]:
    num_layers = len(model.blocks)
    modules_per_layer: List[List[nn.Parameter]] = []
    protect_layers: List[int] = []
    for i, block in enumerate(model.blocks):
        layer_params: List[nn.Parameter] = []
        if hasattr(block, "attn") and hasattr(block.attn, "proj"):
            layer_params.extend(list(block.attn.proj.parameters()))
        if hasattr(block, "ffn") and hasattr(block.ffn, "fc2"):
            layer_params.extend(list(block.ffn.fc2.parameters()))
        modules_per_layer.append(layer_params)
        if not (args.minlayer <= i < args.maxlayer):
            protect_layers.append(i)

    if not any(len(x) for x in modules_per_layer):
        # no params to compute fisher; return passthrough sparsity vector
        base = args.sparsity if not isinstance(args.sparsity, list) else 0.0
        return [base] * num_layers

    # open grads only for modules we track
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    for params in modules_per_layer:
        for p in params:
            p.requires_grad_(True)
            p.grad = None

    dims = torch.tensor([sum(p.numel() for p in params) for params in modules_per_layer],
                        device=device, dtype=torch.float)

    cfg = VarTraceConfig(gamma=0.5, ema_rho=0.95, use_exact_per_scale=True, device=device)
    fisher_sum = torch.zeros(len(modules_per_layer), device=device)
    fisher_ema = FisherTraceEMA(len(modules_per_layer), rho=cfg.ema_rho, device=device)
    total_weight = 0.0

    model.eval()
    for batch in dataloader:
        inputs = _prepare_batch(batch, device)
        with torch.autograd.enable_grad():
            outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                tensor_outputs = [o for o in outputs if torch.is_tensor(o)]
                if len(tensor_outputs) == 0:
                    raise RuntimeError("Model outputs contain no tensors to build a loss from")
                loss = sum(t.float().pow(2).mean() for t in tensor_outputs) / len(tensor_outputs)
            else:
                if not torch.is_tensor(outputs):
                    raise RuntimeError("Model output is not a tensor and cannot be used to compute Fisher information")
                loss = outputs.float().pow(2).mean()

        tokens = inputs.shape[0] if inputs.ndim > 0 else 1
        if cfg.fisher_weight_per_token and tokens > 0:
            loss = loss / float(tokens)

        sq = squared_grad_per_layer_from_loss(loss, modules_per_layer, retain_graph=False)
        fisher_sum += sq
        total_weight += 1.0

    if total_weight == 0:
        raise RuntimeError("No data available to compute Fisher information")

    fisher_now = fisher_sum / total_weight
    fisher_smoothed = fisher_ema.update(fisher_now)

    if isinstance(args.sparsity, list):
        avg_sp = float(sum(args.sparsity) / max(len(args.sparsity), 1))
        target_keep_ratio = 1.0 - avg_sp
    else:
        target_keep_ratio = 1.0 - float(args.sparsity)

    alpha = allocate_pruning_rates(
        traces=fisher_smoothed,
        dims=dims,
        global_keep_ratio=target_keep_ratio,
        gamma=cfg.gamma,
        min_alpha=0.05,
        max_alpha=1.0,
        protect_layers=protect_layers,
    )
    sparsity_per_layer = [1.0 - a.item() for a in alpha]

    # close grads
    for params in modules_per_layer:
        for p in params:
            p.requires_grad_(False)
    for p in model.parameters():
        p.grad = None

    return sparsity_per_layer


# ======================
# Slimming / Pruning
# ======================

def model_slimming(model, dataloader, args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(dev)
    layers = model.blocks

    def _clear_all_kv_caches(m):
        for b in m.blocks:
            attn = getattr(b, "attn", None)
            if attn is None:
                continue
            if hasattr(attn, "cached_k"): attn.cached_k = None
            if hasattr(attn, "cached_v"): attn.cached_v = None
            if hasattr(attn, "kv_len"):   attn.kv_len = 0
            if hasattr(attn, "cache_len"): attn.cache_len = 0

    # Fisher-based per-layer sparsity allocation
    if getattr(args, "use_fisher_sparsity", False):
        args.sparsity = compute_layerwise_fisher(model, dataloader, args, device_obj)

    t1 = time.time()
    print("pruning...")
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        if not (args.minlayer <= i < args.maxlayer):
            del layer
            continue

        all_module_dict = find_layers(layer)
        sequential = [
            ["attn.proj"],
            ["ffn.fc2"],
        ]
        for names in sequential:
            module_dict = {name: all_module_dict[name] for name in names}
            pruner_dict = {}
            for name in module_dict:
                if args.prune_method == "woodtaylor":
                    pruner_dict[name] = WoodTaylorSlim(module_dict[name], i, args)
                else:
                    pruner_dict[name] = SlimGPT(module_dict[name], i, args)

            # === XTX collection (only when H_mode == 'xtx') ===
            collect_xtx = (getattr(args, "H_mode", "xtx") == "xtx")
            if collect_xtx:
                def add_batch(tag):
                    def func(_, inp, out):
                        pruner_dict[tag].add_batch(inp[0].data, out.data)
                    return func
                handles = []
                for tag in module_dict:
                    handles.append(module_dict[tag].register_forward_hook(add_batch(tag)))
                for b in model.blocks:
                    if hasattr(b, "attn") and hasattr(b.attn, "kv_caching"):
                        b.attn.kv_caching(True)
                with torch.no_grad():
                    for _, batch in enumerate(dataloader):
                        inputs = _prepare_batch(batch, device_obj)
                        model(inputs)
                for b in model.blocks:
                    if hasattr(b, "attn") and hasattr(b.attn, "kv_caching"):
                        b.attn.kv_caching(False)
                for h in handles:
                    h.remove()

            # === Fisher / First-order pass (H_mode == 'fisher' or woodtaylor) ===
            need_grad_pass = (getattr(args, "H_mode", "xtx") == "fisher") or (args.prune_method == "woodtaylor")
            if need_grad_pass:
                use_batches = getattr(args, "fisher_batches", 1)

                # Disable all grads first
                for p in model.parameters():
                    p.requires_grad_(False)
                    p.grad = None
                # Enable only current modules' params
                for mname in module_dict:
                    for p in module_dict[mname].parameters():
                        p.requires_grad_(True)
                        p.grad = None

                used = 0
                for _, batch in enumerate(dataloader):
                    if used >= use_batches:
                        break
                    used += 1
                    inputs = _prepare_batch(batch, device_obj)

                    outputs = model(inputs)
                    if isinstance(outputs, (list, tuple)):
                        ts = [o for o in outputs if torch.is_tensor(o)]
                        if len(ts) == 0:
                            raise RuntimeError("No tensor outputs for Fisher loss")
                        loss = sum(t.float().pow(2).mean() for t in ts) / max(1, len(ts))
                    else:
                        if not torch.is_tensor(outputs):
                            raise RuntimeError("Non-tensor output for Fisher loss")
                        loss = outputs.float().pow(2).mean()

                    if inputs.ndim > 0 and getattr(args, "fisher_weight_per_token", True):
                        loss = loss / float(inputs.shape[0])

                    loss.backward()

                # collect grad^2 and hand to pruner
                for mname in module_dict:
                    H_chunks = []
                    for p in module_dict[mname].parameters():
                        if p.grad is not None:
                            H_chunks.append((p.grad.detach() ** 2).reshape(-1))
                    if len(H_chunks):
                        H_diag = torch.cat(H_chunks)
                        if hasattr(pruner_dict[mname], "set_H_diag"):
                            pruner_dict[mname].set_H_diag(H_diag)
                        else:
                            pruner_dict[mname].H_diag = H_diag  # fallback

                    if hasattr(pruner_dict[mname], "cache_grad"):
                        pruner_dict[mname].cache_grad()

                # cleanup grads and flags
                for p in model.parameters():
                    p.grad = None
                    p.requires_grad_(False)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # === Score & prune ===
            with torch.no_grad():
                for name in module_dict:
                    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                    print(f"layer {i}: {name} sparsity {sparsity}")
                    if args.prune_method in ["slimgpt", "woodtaylor"] or getattr(args, "use_fisher_sparsity", False):
                        idx = pruner_dict[name].struct_prune(
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "magnitude":
                        idx = pruner_dict[name].magnitude_prune(
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    elif args.prune_method == "taylor":
                        idx = pruner_dict[name].taylor_prune(
                            sparsity=sparsity,
                            percdamp=args.percdamp,
                            headsize=64 if name == "attn.proj" else 1,
                            layer_idx=i,
                        )
                    else:
                        raise ValueError(f"Unknown prune_method: {args.prune_method}")

                    pruner_dict[name].free()
                    target_layer = get_module_by_name(model.blocks[i], name)

                    # apply structural prune to graph
                    if name == "ffn.fc2":
                        target_layer_b = get_module_by_name(model.blocks[i], "ffn.fc1")
                        idx_list = idx.tolist()
                        tp.prune_linear_in_channels(target_layer, idx_list)   # remove columns in fc2
                        tp.prune_linear_out_channels(target_layer_b, idx_list)  # remove rows in fc1

                    elif name == "attn.proj":
                        # adjust num_heads based on *this block*'s heads
                        sparsity_blk = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                        orig_heads = int(getattr(model.blocks[i].attn, "num_heads"))
                        keep_heads = max(1, int(round(orig_heads * (1 - sparsity_blk))))
                        model.blocks[i].attn.num_heads = keep_heads

                        idx_m = idx.to(dtype=torch.long)
                        idx_list = idx.tolist()
                        keep_idxs = list(set(range(target_layer.in_features)) - set(idx_list))

                        # sync biases/scales that align with per-head channels
                        if hasattr(model.blocks[i].attn, "q_bias"):
                            model.blocks[i].attn.q_bias = nn.Parameter(model.blocks[i].attn.q_bias.data[keep_idxs])
                        if hasattr(model.blocks[i].attn, "zero_k_bias"):
                            zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
                            model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
                        if hasattr(model.blocks[i].attn, "v_bias"):
                            model.blocks[i].attn.v_bias = nn.Parameter(model.blocks[i].attn.v_bias.data[keep_idxs])
                        if hasattr(model.blocks[i].attn, "scale_mul_1H11"):
                            model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
                                torch.full(
                                    size=(1, keep_heads, 1, 1),
                                    fill_value=4.0,
                                    device=device_obj,
                                ).log(),
                                requires_grad=True,
                            )

                        target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")

                        # hidden is the *original* proj.in_features
                        hidden = target_layer.in_features

                        # prune proj input channels (columns)
                        tp.prune_linear_in_channels(target_layer, idx_list)

                        # map removed head channels to qkv out-channels: [Q, K, V] concatenated
                        rm_feat_q = idx_m
                        rm_qkv = torch.cat([rm_feat_q, rm_feat_q + hidden, rm_feat_q + 2 * hidden], dim=0)
                        rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
                        tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

                _clear_all_kv_caches(model)

            del pruner_dict

        # simple sanity print of shapes after pruning a block
        print(model.blocks[i].ffn.fc1.weight.shape)
        print(model.blocks[i].ffn.fc2.weight.shape)
        print(model.blocks[i].attn.mat_qkv.weight.shape)
        print(model.blocks[i].attn.proj.weight.shape)

        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("本次评估用用时为 {:.3f}s".format(time.time() - t1))
    return model


# ======================
# Main
# ======================

def main(args):
    print('load model...')

    # === decouple model depth (for building weights) from prune ranges ===
    MODEL_DEPTH = args.model_depth
    assert MODEL_DEPTH in {12, 16, 20, 24, 30}

    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt = '/home/waas/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = f'/home/waas/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/var_d{MODEL_DEPTH}.pth'
    if not osp.exists(vae_ckpt): print("vae checkpoint not exist:", vae_ckpt)
    if not osp.exists(var_ckpt): print("var checkpoint not exist:", var_ckpt)
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

    # 不需要让全模型默认 requires_grad=True，保持推理友好
    for p in vae.parameters(): p.requires_grad_(True)
    for p in var.parameters(): p.requires_grad_(False)

    print('prepare finished.')
    model = var
    tokenizer = None  # placeholder if you switch to text datasets

    model.eval()

    # clamp prune ranges
    args.minlayer = max(args.minlayer, 0)
    # var depth is MODEL_DEPTH; blocks length should match
    args.maxlayer = min(args.maxlayer, len(model.blocks))

    # non-uniform scheduling (optional)
    if args.non_uniform:
        assert 0 <= args.min_sparsity <= args.max_sparsity < 1
        if args.non_uniform_strategy in ('log_increase', 'log_decrease'):
            linear_space = np.arange(0, args.maxlayer - args.minlayer)
            args.sparsity = args.min_sparsity + (args.max_sparsity - args.min_sparsity) / np.log(32) * np.log(1 + linear_space)
            args.sparsity = [0] * args.minlayer + list(args.sparsity)
            if args.non_uniform_strategy == 'log_decrease':
                args.sparsity = args.sparsity[::-1]
        elif args.non_uniform_strategy in ('linear_increase', 'linear_decrease'):
            denom = max(1, args.maxlayer - 1 - args.minlayer)
            sparsity_grad = (args.max_sparsity - args.min_sparsity) / denom
            args.sparsity = [(i - args.minlayer) * sparsity_grad + args.min_sparsity for i in range(args.minlayer, args.maxlayer)]
            args.sparsity = [0] * args.minlayer + args.sparsity
            if args.non_uniform_strategy == 'linear_decrease':
                args.sparsity = args.sparsity[::-1]
        else:
            raise ValueError(f"Unknown non_uniform_strategy: {args.non_uniform_strategy}")

    state_dict = model.state_dict()
    layer_params = round(sum(v.numel() for k, v in state_dict.items() if k not in ('model.embed_tokens.weight', 'lm_head.weight')) / 1e9, 2)
    extra_params = round(sum(v.numel() for k, v in state_dict.items() if k in ('model.embed_tokens.weight', 'lm_head.weight')) / 1e9, 2)
    print(f'all params: {layer_params + extra_params} B\t layer params: {layer_params} B\t extra params: {extra_params} B')

    print('load dataset...')
    # 简易占位数据（按需替换为真实校准数据）
    dataset = torch.arange(0, args.num_samples)
    tensor_dataset = TensorDataset(dataset)
    dataloader = DataLoader(tensor_dataset, batch_size=10, shuffle=False)

    num_samples = len(dataset)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')

    if isinstance(args.sparsity, list) or args.sparsity >= 0:
        print('start slimming...')
        tick = time.time()
        model = model_slimming(model, dataloader, args)
        print("slimming time: {:.3f}s".format(time.time() - tick))

    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check (zero-ratio) {sparsity_ratio:.4f}")
    print("*" * 30)

    example_input = torch.tensor([0]).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    zero = time.time()
    result = model(example_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    print(f"inference time (1 step): {end - zero:.4f}s")

    print(model)
    macs, nparams = tp.utils.count_ops_and_params(model, example_input, layer_wise=False)

    out = model(example_input)
    try:
        print(out.shape)
    except Exception:
        pass

    print("  Params: %.2f M => %.2f M" % (nparams / 1e6, nparams / 1e6))
    print("   MACs: %.2f G => %.2f G" % (macs / 1e9, macs / 1e9))

    save_model = args.save_dir
    os.makedirs(save_model, exist_ok=True)
    save_path = os.path.join(save_model, args.model_name)
    torch.save(model.state_dict(), save_path)
    print("saved to:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,
                        default="/home/sumingluo/Model_weight/meta/llama-2-7b-hf", help="model to load")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4", "alpaca", "gpt4_alpaca"],
                        help="Where to extract calibration data from.")
    parser.add_argument("--num_samples", type=int, default=1024,
                        help="Number of calibration data samples.")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for the calibration data.")
    parser.add_argument("--sparsity", type=float, default=0.2,
                        help="Target pruning ratio, ineffective when non_uniform is True")
    parser.add_argument("--minlayer", type=int, default=0,
                        help="Prune all layers with id >= this.")
    parser.add_argument("--maxlayer", type=int, default=32,
                        help="Prune all layers with id < this.")
    parser.add_argument("--cache_dev", type=str, default="cuda",
                        help="Defaults to `cuda`. Set to `cpu` when GPU mem is insufficient.")
    parser.add_argument("--batch_samples", type=int, default=128,
                        help="Works when `cache_dev=cpu`. #samples loaded onto GPU each time.")
    parser.add_argument("--skip_evaluate", action="store_true",
                        help="Skip evaluation after pruning.")
    parser.add_argument("--save_pruned_weights", action="store_true",
                        help="Whether save the checkpoint after removing zeroed-out parameters.")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Path to saved model.")
    parser.add_argument("--non_uniform", action="store_true",
                        help="When set to True, use non-uniform pruning; `sparsity` ignored.")
    parser.add_argument("--non_uniform_strategy", type=str, default='log_increase',
                        choices=["log_increase", "log_decrease", "linear_increase", "linear_decrease"],
                        help="Works when `non_uniform=True`")
    parser.add_argument("--min_sparsity", type=float, default=0.06,
                        help="Works when `non_uniform=True`")
    parser.add_argument("--max_sparsity", type=float, default=0.3,
                        help="Works when `non_uniform=True`")
    parser.add_argument("--no_compensate", action="store_true",
                        help="Skip error compensation in SlimGPT")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Percent of average Hessian diagonal used for damping.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--specific_layer", type=int, default=0,
                        help="choose which layer to evaluate")
    parser.add_argument("--model_name", type=str, default="model", help="")
    parser.add_argument("--prune_method", type=str, default="slimgpt",
                        choices=["slimgpt", "woodtaylor", "magnitude", "taylor"],
                        help="层内剪枝方法：slimgpt=纯二阶(OBS)，woodtaylor=二阶+一阶补偿")
    parser.add_argument("--scale_update", type=float, default=1.0,
                        help="WoodTaylor 补偿步幅：W[:,keep] -= scale*(G H^{-1})")
    parser.add_argument("--use_fisher_sparsity", action="store_true",
                        help="是否用 Fisher 迹自动分配每层剪枝率（调用 compute_layerwise_fisher）。")
    parser.add_argument("--H_mode", type=str, default="xtx", choices=["xtx", "fisher"],
                        help="H 的来源：xtx=输入协方差；fisher=经验 Fisher（由 grad^2 构成对角 H）。")
    parser.add_argument("--fisher_batches", type=int, default=20,
                        help="当 H_mode=fisher 或 prune_method=woodtaylor 时，用多少个 batch 做轻量反传以累计 grad/grad^2。")
    parser.add_argument("--model_depth", type=int, required=True, choices=[12, 16, 20, 24, 30],
                        help="VAR 模型深度，只用于构建/加载，与剪枝范围无关。")
    # 可选：若想切换是否按 token 归一 Fisher loss，使用 getattr(args, 'fisher_weight_per_token', True)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    main(args)