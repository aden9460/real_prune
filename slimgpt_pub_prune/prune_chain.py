#!/usr/bin/env python3
"""
VAR Model Chain Pruning with KFAC-OBS
Based on prune_v6.py but implements the Chain Pruning + KFAC method

Created: 2025-11-05
Method: Chain Pruning + Full-KFAC-OBS + Multi-Scale Processing
Core Innovation: Using pseudo-loss to construct true Fisher information matrix
Based on: prune_v6.py (SlimGPT) with improvements
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
from typing import List, Dict, Tuple
from types import SimpleNamespace

# Import from existing modules
from slim_utils.slimgpt import SlimGPT
from slim_utils.slim_dataset import get_loaders
from slim_utils.params_remove import LLaMAParamsPruner
from ppl_eval.ppl_eval import ppl_metric
from torchvision.utils import save_image
import sys
sys.path.append("/home/suanba/EdgeVAR/Torch-Pruning")
from importlib.metadata import version
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import torch_pruning as tp
from torchviz import make_dot
import torch_pruning.pruner.function as tfun
from models import VQVAE, build_vae_var
import gc
from contextlib import contextmanager
import math

# Memory monitoring context manager (from prune_v6.py)
@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """Find layers of specific types (from prune_v6.py)"""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    """Check model sparsity (from prune_v6.py)"""
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

    return float(count) / total_params

# ========================================
# ChainKFACPruner: Core Innovation Class
# ========================================

class ChainKFACPruner:
    """Chain Pruning + KFAC Implementation"""

    def __init__(
        self,
        layer: nn.Linear,
        layer_idx: int,
        scale_weight_strategy: str = 'natural',
        percdamp: float = 0.01
    ):
        self.layer = layer
        self.layer_idx = layer_idx
        self.scale_weight_strategy = scale_weight_strategy
        self.percdamp = percdamp

        self.device = layer.weight.device
        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # Initialize A and G matrices
        self.A = torch.zeros(self.in_features, self.in_features, device=self.device)
        self.G = torch.zeros(self.out_features, self.out_features, device=self.device)

        # Scale weights for VAR's 10 scales
        self.scale_weights = self._compute_scale_weights()
        self.collected_samples = 0

    def _compute_scale_weights(self) -> List[float]:
        """Compute scale weights for VAR's 680 tokens"""
        token_counts = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]  # 680 total

        if self.scale_weight_strategy == 'natural':
            weights = [n / 680 for n in token_counts]
        elif self.scale_weight_strategy == 'equal':
            weights = [1.0 / 10 for _ in range(10)]
        elif self.scale_weight_strategy == 'sqrt':
            sqrt_counts = [n**0.5 for n in token_counts]
            total_sqrt = sum(sqrt_counts)
            weights = [s / total_sqrt for s in sqrt_counts]
        else:
            raise ValueError(f"Unknown strategy: {self.scale_weight_strategy}")

        return weights

    def add_batch_multiscale(
        self,
        inputs_10scales: List[torch.Tensor],
        original_outputs_10scales: List[torch.Tensor]
    ):
        """
        Add a multi-scale batch to update A and G matrices

        Args:
            inputs_10scales: 10 scale inputs [List[Tensor[B, L_s, in_dim]]]
            original_outputs_10scales: 10 scale original outputs [List[Tensor[B, L_s, out_dim]]]
        """
        A_batch = torch.zeros_like(self.A)
        G_batch = torch.zeros_like(self.G)
        total_weight = 0

        # Register hooks to collect activations and gradients
        activations = {}
        gradients = {}

        def hook_forward(module, inp, out):
            activations['current'] = inp[0].detach()

        def hook_backward(module, grad_in, grad_out):
            if grad_out[0] is not None:
                gradients['current'] = grad_out[0].detach()

        handle_fwd = self.layer.register_forward_hook(hook_forward)
        handle_bwd = self.layer.register_backward_hook(hook_backward)

        try:
            # Process each scale
            for scale_s in range(10):
                inp_s = inputs_10scales[scale_s]  # [B, L_s, in_dim]
                target_s = original_outputs_10scales[scale_s]  # [B, L_s, out_dim]

                activations.clear()
                gradients.clear()

                # Forward pass
                inp_s = inp_s.requires_grad_(True)
                out_s = self.layer(inp_s)

                # Ensure target has no gradients (detached)
                target_s = target_s.detach()

                # Pseudo-loss
                loss_s = F.mse_loss(out_s, target_s)

                # Backward pass
                loss_s.backward(retain_graph=True)

                # Collect A_s (input covariance)
                if 'current' not in activations:
                    print(f"    Warning: No activations collected for scale {scale_s}")
                    continue

                X_s = activations['current'].reshape(-1, self.in_features).t()  # [in_dim, B*L_s]
                A_s = X_s @ X_s.t()  # [in_dim, in_dim]

                # Collect G_s (gradient covariance)
                if 'current' not in gradients:
                    print(f"    Warning: No gradients collected for scale {scale_s}")
                    continue

                G_s_mat = gradients['current'].reshape(-1, self.out_features).t()  # [out_dim, B*L_s]
                G_s = G_s_mat @ G_s_mat.t()  # [out_dim, out_dim]

                # Normalize
                n_samples_s = X_s.shape[1]
                A_s = A_s / n_samples_s
                G_s = G_s / n_samples_s

                # Weighted accumulation
                weight_s = self.scale_weights[scale_s]
                A_batch += weight_s * A_s
                G_batch += weight_s * G_s
                total_weight += weight_s

        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        # Normalize and accumulate to global A, G
        if total_weight > 0:
            A_batch = A_batch / total_weight
            G_batch = G_batch / total_weight

            # Calculate actual sample count (all scale tokens)
            actual_samples = sum(
                inputs_10scales[s].shape[0] * inputs_10scales[s].shape[1]
                for s in range(10)
            )

            # EMA update (based on actual sample count, not batch count)
            if not hasattr(self, 'total_samples_seen'):
                self.total_samples_seen = 0

            self.total_samples_seen += actual_samples
            alpha = actual_samples / self.total_samples_seen

            self.A = (1 - alpha) * self.A + alpha * A_batch
            self.G = (1 - alpha) * self.G + alpha * G_batch

            self.collected_samples += 1

    def struct_prune(self, sparsity: float, headsize: int = 64) -> torch.Tensor:
        """
        Perform structured pruning using KFAC-OBS

        Args:
            sparsity: sparsity ratio
            headsize: head size

        Returns:
            pruned_indices: deleted column indices
        """
        W = self.layer.weight.data.clone().float()
        num_heads = self.in_features // headsize

        print(f"    Layer {self.layer_idx}: Collected {self.collected_samples} batches")
        print(f"    Scale weights: {[f'{w:.3f}' for w in self.scale_weights]}")

        # ========================================
        # 1. Compute Fisher matrix inverse diagonal
        # ========================================
        A_inv, G_inv = self._compute_inverses()
        F_inv_diag = self._compute_fisher_inverse_diag(A_inv, G_inv)

        # ========================================
        # 2. Compute input dimension importance
        # ========================================
        importance_element = W ** 2 / F_inv_diag  # [out_dim, in_dim]
        importance_column = importance_element.sum(dim=0)  # [in_dim]

        # Head-wise importance
        head_importance = importance_column.view(num_heads, headsize).sum(1)  # [num_heads]

        print(f"    Head importances: {head_importance.tolist()}")

        # ========================================
        # 3. Select heads to prune
        # ========================================
        num_prune = int(num_heads * sparsity)
        prune_heads = torch.argsort(head_importance)[:num_prune]

        print(f"    Pruning {num_prune}/{num_heads} heads: {prune_heads.tolist()}")

        # ========================================
        # 4. OBS surgery (using A_inv)
        # ========================================
        self._perform_obs_surgery(W, A_inv, prune_heads, headsize)

        # Update weights
        self.layer.weight.data = W.to(self.layer.weight.data.dtype)

        # Return deleted column indices
        prune_indices = []
        for head_idx in prune_heads:
            start = head_idx * headsize
            prune_indices.extend(range(start, start + headsize))

        return torch.tensor(prune_indices, device=self.device)

    def _compute_inverses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute A and G inverses"""
        # Add damping
        damp_A = self.percdamp * torch.mean(torch.diag(self.A))
        damp_G = self.percdamp * torch.mean(torch.diag(self.G))

        A_damped = self.A + damp_A * torch.eye(self.A.shape[0], device=self.device)
        G_damped = self.G + damp_G * torch.eye(self.G.shape[0], device=self.device)

        # Cholesky decomposition for inverse
        try:
            A_inv = torch.cholesky_inverse(torch.linalg.cholesky(A_damped))
            G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G_damped))
        except:
            print(f"    Warning: Cholesky failed, using eigenvalue decomposition")
            # Fallback to eigenvalue decomposition
            eigvals_A, eigvecs_A = torch.linalg.eigh(A_damped)
            eigvals_A = torch.clamp(eigvals_A, min=1e-6)
            A_inv = eigvecs_A @ torch.diag(1.0 / eigvals_A) @ eigvecs_A.t()

            eigvals_G, eigvecs_G = torch.linalg.eigh(G_damped)
            eigvals_G = torch.clamp(eigvals_G, min=1e-6)
            G_inv = eigvecs_G @ torch.diag(1.0 / eigvals_G) @ eigvecs_G.t()

        return A_inv, G_inv

    def _compute_fisher_inverse_diag(self, A_inv: torch.Tensor, G_inv: torch.Tensor) -> torch.Tensor:
        """Compute Fisher matrix inverse diagonal elements F = G ⊗ A (standard KFAC)"""
        A_inv_diag = torch.diag(A_inv)  # [in_dim]
        G_inv_diag = torch.diag(G_inv)  # [out_dim]

        # For F = G ⊗ A, diagonal elements are:
        # F^{-1}[i*in_dim + j, i*in_dim + j] = G^{-1}[i,i] * A^{-1}[j,j]
        F_inv_diag = G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0)  # [out_dim, in_dim]

        return F_inv_diag

    def _perform_obs_surgery(
        self,
        W: torch.Tensor,
        A_inv: torch.Tensor,
        prune_heads: torch.Tensor,
        headsize: int
    ):
        """Perform OBS surgery"""
        A_inv_upper = torch.linalg.cholesky(A_inv, upper=True)

        for head_idx in prune_heads:
            start = head_idx * headsize
            end = start + headsize

            # Compute error matrix
            Err = torch.zeros(W.shape[0], headsize, device=W.device)

            for i in range(headsize):
                col_idx = start + i

                # Error
                Err[:, i] = W[:, col_idx] / A_inv_upper[col_idx, col_idx]

                # Local update (within head)
                if i < headsize - 1:
                    W[:, col_idx+1:end] -= Err[:, i:i+1] @ A_inv_upper[col_idx:col_idx+1, col_idx+1:end]

            # Global update (other columns)
            if end < W.shape[1]:
                W[:, end:] -= Err @ A_inv_upper[start:end, end:]

            # Delete this head
            W[:, start:end] = 0

# ========================================
# Helper Functions for VAR Multi-Scale Processing
# ========================================

def extract_layer_inputs(var_model, layer_idx: int, batch, device: str = 'cuda') -> List[torch.Tensor]:
    """
    Extract specified layer's 10-scale inputs from VAR model

    Args:
        var_model: VAR model
        layer_idx: layer index
        batch: single input batch
        device: device

    Returns:
        inputs_10scales: 10 scale inputs [List[Tensor[B, L_s, hidden_dim]]]
    """
    var_model.eval()
    target_layer = var_model.blocks[layer_idx].attn.proj

    inputs_10scales = []

    def input_hook(_, inp, out):
        """Hook to collect inputs"""
        inputs_10scales.append(inp[0].detach().clone())

    # Register hook
    handle = target_layer.register_forward_hook(input_hook)

    try:
        with torch.no_grad():
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

            # VAR forward pass triggers hook
            _ = var_model(batch)

    finally:
        handle.remove()

    return inputs_10scales

def extract_layer_outputs(var_model, layer_idx: int, batch, device: str = 'cuda') -> List[torch.Tensor]:
    """
    Extract specified layer's 10-scale outputs from VAR model

    Args:
        var_model: VAR model
        layer_idx: layer index
        batch: single input batch
        device: device

    Returns:
        outputs_10scales: 10 scale outputs [List[Tensor[B, L_s, hidden_dim]]]
    """
    var_model.eval()
    target_layer = var_model.blocks[layer_idx].attn.proj

    outputs_10scales = []

    def output_hook(_, inp, out):
        """Hook to collect outputs"""
        outputs_10scales.append(out.detach().clone())

    # Register hook
    handle = target_layer.register_forward_hook(output_hook)

    try:
        with torch.no_grad():
            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

            # VAR forward pass triggers hook
            _ = var_model(batch)

    finally:
        handle.remove()

    return outputs_10scales

# ========================================
# Main Pruning Function (Chain Pruning + KFAC Integration)
# ========================================

def get_module_by_name(layer, name):
    """Get module by dot-separated name (from prune_v6.py)"""
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module

def model_slimming(model, dataloader, args):
    """
    Main model slimming function with Chain Pruning + KFAC
    Adapts prune_v6.py structure for chain pruning methodology
    """

    dev = "cuda" if torch.cuda.is_available() else 'cpu'
    layers = model.blocks
    num_batches = len(dataloader)

    print("Chain Pruning: Starting model slimming...")
    print(f"  Strategy: Layer 0 = SlimGPT, Layers 1+ = Chain KFAC")
    print(f"  Scale weighting: {args.scale_weight_strategy}")
    print(f"  Total samples: {num_batches}")

    with torch.no_grad():
        for i in range(len(layers)):
            if not (args.minlayer <= i < args.maxlayer):
                continue

            print(f"\n{'='*60}")
            print(f"Processing Layer {i}/{len(layers)}")
            print(f"{'='*60}")

            # Clear any lingering computation graphs
            torch.cuda.empty_cache()

            layer = layers[i].to(dev)
            all_module_dict = find_layers(layer)

            # Define sequential pruning targets (only attention for now)
            sequential = [
                ["attn.proj"],
                # ["ffn.fc2"],  # 暂时不剪枝FFN
            ]

            for names in sequential:
                module_dict = {name: all_module_dict[name] for name in names}
                pruner_dict = {}

                # ========================================
                # KEY INNOVATION: Conditional Pruner Selection
                # ========================================
                for name in module_dict:
                    if name == "attn.proj" and i == 0:
                        # First layer: Use SlimGPT
                        print(f"  {name}: Using SlimGPT (first layer)")
                        pruner_dict[name] = SlimGPT(module_dict[name], i, args)
                    elif name == "attn.proj" and i > 0:
                        # Subsequent layers: Use Chain KFAC
                        print(f"  {name}: Using Chain Pruning + KFAC")
                        pruner_dict[name] = ChainKFACPruner(
                            layer=module_dict[name],
                            layer_idx=i,
                            scale_weight_strategy=args.scale_weight_strategy,
                            percdamp=args.percdamp
                        )
                    else:
                        # FFN layers: Use SlimGPT
                        print(f"  {name}: Using SlimGPT (FFN layer)")
                        pruner_dict[name] = SlimGPT(module_dict[name], i, args)

                # ========================================
                # Data Collection: Multi-scale Hook System
                # ========================================
                _cache_dict = {}

                def add_batch_chain(name):
                    """Modified hook for chain pruning data collection"""
                    def func(_, inp, out):
                        # Initialize cache
                        if name not in _cache_dict:
                            _cache_dict[name] = []

                        # Add to cache (detach instead of .data)
                        _cache_dict[name].append((inp[0].detach(), out.detach()))

                        # When we have 10 scales, just store them - don't process yet
                        if len(_cache_dict[name]) >= 10:
                            # For Chain KFAC: store 10-scale data for later processing
                            if isinstance(pruner_dict[name], ChainKFACPruner):
                                inputs_10scales = [p[0] for p in _cache_dict[name]]
                                outputs_10scales = [p[1] for p in _cache_dict[name]]

                                # Store for later KFAC processing
                                if not hasattr(pruner_dict[name], '_cached_data'):
                                    pruner_dict[name]._cached_data = []
                                pruner_dict[name]._cached_data.append((inputs_10scales, outputs_10scales))
                            else:
                                # SlimGPT: concatenate and use normal add_batch
                                inps = [p[0] for p in _cache_dict[name]]
                                outs = [p[1] for p in _cache_dict[name]]
                                inp_cat = torch.cat(inps, dim=1)
                                out_cat = torch.cat(outs, dim=1)
                                pruner_dict[name].add_batch(inp_cat, out_cat)

                            _cache_dict[name] = []  # Clear cache
                    return func

                # Register hooks
                handles = []
                for name in module_dict:
                    handles.append(module_dict[name].register_forward_hook(add_batch_chain(name)))

                # Enable KV caching and process data
                for b in model.blocks: b.attn.kv_caching(True)

                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= args.num_samples:  # Limit processing
                        break
                    try:
                        model(batch)
                    except Exception as e:
                        print(f"    Warning: Batch {batch_idx} failed: {e}")
                        continue

                    if (batch_idx + 1) % 10 == 0:
                        print(f"    Processed {batch_idx + 1}/{min(len(dataloader), args.num_samples)} batches")

                # Cleanup
                for b in model.blocks: b.attn.kv_caching(False)
                for h in handles:
                    h.remove()

                # ========================================
                # Process cached KFAC data (避免递归调用)
                # ========================================
                for name in module_dict:
                    if isinstance(pruner_dict[name], ChainKFACPruner):
                        if hasattr(pruner_dict[name], '_cached_data'):
                            print(f"    Processing {len(pruner_dict[name]._cached_data)} cached batches for {name}")
                            # KFAC需要梯度计算，暂时退出no_grad模式
                            for inputs_10scales, outputs_10scales in pruner_dict[name]._cached_data:
                                # Enable gradients for KFAC computation
                                with torch.enable_grad():
                                    pruner_dict[name].add_batch_multiscale(inputs_10scales, outputs_10scales)
                        else:
                            print(f"    Warning: No cached data for {name}")

                # ========================================
                # Pruning Execution
                # ========================================
                for name in module_dict:
                    sparsity = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                    print(f"  Layer {i}: {name} sparsity {sparsity:.3f}")

                    # Check if sufficient data was collected
                    if isinstance(pruner_dict[name], ChainKFACPruner):
                        if pruner_dict[name].collected_samples == 0:
                            print(f"    Error: No data collected for {name}. Skipping layer {i}.")
                            continue

                    try:
                        target_layer = get_module_by_name(model.blocks[i], name)

                        # For attn.proj, calculate sparsity to match formula-based num_heads
                        if name == "attn.proj":
                            head_dim = 64
                            original_num_heads = 16
                            sparsity_val = args.sparsity[i] if isinstance(args.sparsity, list) else args.sparsity
                            new_num_heads = round(original_num_heads * (1 - sparsity_val))
                            target_keep_features = new_num_heads * head_dim
                            # Recalculate sparsity to prune exact number of features
                            adjusted_sparsity = 1 - (target_keep_features / target_layer.in_features)
                        else:
                            adjusted_sparsity = sparsity

                        # Execute pruning
                        idx = pruner_dict[name].struct_prune(
                            sparsity=adjusted_sparsity,
                            headsize=64 if name == "attn.proj" else 1,
                        )

                        # Free pruner memory
                        if hasattr(pruner_dict[name], 'free'):
                            pruner_dict[name].free()

                        # ========================================
                        # Physical Pruning (Same as prune_v6.py)
                        # ========================================

                        if name == "ffn.fc2":
                            target_layer_b = get_module_by_name(model.blocks[i], "ffn.fc1")
                            idx = idx.tolist()
                            tp.prune_linear_in_channels(target_layer, idx)
                            tp.prune_linear_out_channels(target_layer_b, idx)

                        elif name == "attn.proj":
                            # QKV Synchronization (same as prune_v6.py)
                            # num_heads was already calculated above
                            idx_m = idx.to(dtype=torch.long)
                            idx = idx.tolist()
                            keep_idxs = list(set(range(target_layer.in_features)) - set(idx))

                            model.blocks[i].attn.num_heads = new_num_heads

                            # Update attention parameters
                            model.blocks[i].attn.q_bias = nn.Parameter(model.blocks[i].attn.q_bias.data[keep_idxs])
                            zero_k_bias = model.blocks[i].attn.zero_k_bias.data[keep_idxs]
                            model.blocks[i].attn.register_buffer('zero_k_bias', zero_k_bias)
                            model.blocks[i].attn.v_bias = nn.Parameter(model.blocks[i].attn.v_bias.data[keep_idxs])
                            model.blocks[i].attn.scale_mul_1H11 = nn.Parameter(
                                torch.full(size=(1, new_num_heads, 1, 1), fill_value=4.0, device='cuda').log(), requires_grad=True
                            )

                            # Prune proj input channels
                            target_layer_b = get_module_by_name(model.blocks[i], "attn.mat_qkv")
                            tp.prune_linear_in_channels(target_layer, idx)

                            # Synchronize QKV pruning
                            hidden = original_num_heads * head_dim  # Use original dimensions
                            rm_feat_q = idx_m
                            rm_qkv = torch.cat([rm_feat_q,
                                              rm_feat_q + hidden,
                                              rm_feat_q + 2*hidden], dim=0)

                            rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
                            tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

                    except Exception as e:
                        print(f"    Error pruning {name}: {e}")
                        continue

                del pruner_dict
                print(f"    Layer {i} complete: {model.blocks[i].ffn.fc1.weight.shape}")

                # Memory cleanup
                torch.cuda.empty_cache()
                gc.collect()

    return model

# ========================================
# Main Function with Full Argument Parsing
# ========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAR Chain Pruning with KFAC-OBS")

    # Model and data arguments (from prune_v6.py)
    parser.add_argument(
        "--model_path", type=str,
        default="/home/sumingluo/Model_weight/meta/llama-2-7b-hf",
        help="model to load"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext2",
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

    # Pruning configuration
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

    # Cache and performance
    parser.add_argument(
        "--cache_dev", type=str, default="cuda",
        help="Defaults to `cuda`. When the GPU memory is insufficient, you can set `cache_dev` to `cpu`, but the trade-off is slower pruning speed."
    )
    parser.add_argument(
        "--batch_samples", type=int, default=128,
        help="Works when `cache_dev=cpu`. The number of samples loaded onto the GPU each time."
    )

    # Evaluation and saving
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

    # Non-uniform sparsity configuration (from prune_v6.py)
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

    # SlimGPT and KFAC parameters
    parser.add_argument(
        "--no_compensate", action="store_true",
        help="Skip error compensation in SlimGPT",
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )

    # Chain Pruning specific arguments (NEW)
    parser.add_argument(
        "--scale_weight_strategy", type=str, default='natural',
        choices=['natural', 'equal', 'sqrt'],
        help="Multi-scale weighting strategy for VAR. 'natural': token-proportional, 'equal': uniform, 'sqrt': square-root balanced"
    )
    parser.add_argument(
        "--chain_pruning_mode", action="store_true",
        help="Enable chain pruning mode (default: enabled for layers > 0)"
    )

    # Other
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--prune_method", type=str, default="chain_kfac",
        help="Pruning method: 'chain_kfac' (recommended), 'slimgpt' (fallback)",
    )
    parser.add_argument(
        "--model_name", type=str, default="",
        help="Custom model name for saving",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print("="*80)
    print("VAR Chain Pruning with KFAC-OBS")
    print(f"  Method: Chain Pruning + Full-KFAC-OBS")
    print(f"  Scale weighting: {args.scale_weight_strategy}")
    print(f"  Target sparsity: {args.sparsity:.1%}")
    print("="*80)

    # Model loading (adapted from prune_v6.py)
    print('Loading VAR model...')
    MODEL_DEPTH = args.maxlayer
    assert MODEL_DEPTH in {12, 16, 20, 24, 30}

    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt = '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = f'/home/project/daily/AR/model_zoo/var_d{MODEL_DEPTH}.pth'

    if not osp.exists(vae_ckpt):
        print("Warning: VAE checkpoint not found")
    if not osp.exists(var_ckpt):
        print("Warning: VAR checkpoint not found")

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

    print(f'Model preparation finished.')
    model = var
    model.eval()

    # Set layer range
    args.minlayer = max(args.minlayer, 0)
    args.maxlayer = min(args.maxlayer, 36)

    # Non-uniform sparsity setup (from prune_v6.py)
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

    # Data preparation
    print('Preparing calibration data...')
    dataloader = torch.arange(0, args.num_samples).cuda()
    num_samples = len(dataloader)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')

    # Model statistics before pruning
    state_dict = model.state_dict()
    layer_params = round(sum(v.numel() for k, v in state_dict.items() if k not in ('model.embed_tokens.weight', 'lm_head.weight')) / 10**9, 2)
    extra_params = round(sum(v.numel() for k, v in state_dict.items() if k in ('model.embed_tokens.weight', 'lm_head.weight')) / 10**9, 2)
    print(f'Before pruning - All params: {layer_params + extra_params:.2f}B\t Layer params: {layer_params:.2f}B\t Extra params: {extra_params:.2f}B')

    # Execute pruning
    if isinstance(args.sparsity, list) or args.sparsity >= 0:
        print('Starting chain pruning...')
        tick = time.time()

        with measure_peak_memory():
            model = model_slimming(model, dataloader, args)

        print(f'Pruning completed in {time.time() - tick:.2f} seconds')

    # Post-pruning analysis
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"Final sparsity: {sparsity_ratio:.4f}")
    print("*" * 30)

    # Model statistics after pruning
    state_dict = model.state_dict()
    layer_params = round(sum(v.numel() for k, v in state_dict.items() if k not in ('model.embed_tokens.weight', 'lm_head.weight')) / 10**9, 2)
    extra_params = round(sum(v.numel() for k, v in state_dict.items() if k in ('model.embed_tokens.weight', 'lm_head.weight')) / 10**9, 2)
    print(f'After pruning - All params: {layer_params + extra_params:.2f}B\t Layer params: {layer_params:.2f}B\t Extra params: {extra_params:.2f}B')

    # Performance test
    example_input = torch.tensor([0]).to(device)
    for b in model.blocks: b.attn.kv_caching(True)

    torch.cuda.synchronize()
    start_time = time.time()
    result = model(example_input)
    torch.cuda.synchronize()
    end_time = time.time()

    print(f'Inference time: {end_time - start_time:.4f} seconds')

    # Save model if requested
    if args.save_pruned_weights and args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            # Use custom model name if provided, otherwise use default format
            if hasattr(args, 'model_name') and args.model_name:
                save_path = osp.join(args.save_dir, args.model_name)
            else:
                save_path = osp.join(args.save_dir, f'var_d{MODEL_DEPTH}_chain_pruned_{args.sparsity:.2f}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Pruned model saved to: {save_path}')
        except Exception as e:
            print(f'Warning: Failed to save model: {e}')

    print("Chain pruning completed successfully!")
