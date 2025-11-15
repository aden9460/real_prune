"""
FastOBA + Attention + SlimGPT Integration

This module combines:
- FastOBA's automatic differentiation for Hessian computation
- SlimGPT's Cholesky decomposition and OBS pruning
- Attention-specific layer-internal Hessian

Key features:
- Computes Hessian w.r.t. Attention OUTPUT (not final loss)
- Supports block-diagonal approximation for multi-head attention
- VAR-aware: scale-by-scale token input handling
- Per-stage analysis for understanding scale-specific pruning preferences
"""

import torch
import torch.nn as nn
import math
import warnings
from typing import Dict, Optional, Tuple, List
import numpy as np


class FastOBAAttentionSlimGPT:
    """
    Attention-specific pruner combining FastOBA Hessian with SlimGPT's OBS framework

    Unlike standard OBS that computes Hessian w.r.t. final loss,
    this computes Hessian w.r.t. **Attention layer output** (layer-internal OBS).

    Args:
        attention_module: Complete Attention module (with qkv_proj and out_proj)
        layer_idx: Layer index for logging
        args: Configuration object with:
            - hessian_mode: 'slimgpt' or 'block_diagonal'
            - head_importance_mode: 'block_mean' or 'slimgpt_mean'
            - prune_mode: 'head_dims', 'num_heads', or 'both'
            - use_compensation: bool
            - fastoba_order: int (default 2)
            - fastoba_delta: float (default 1.0)
            - hessian_accumulate_freq: int (default 10)
    """

    def __init__(self, attention_module, layer_idx: int, args):
        # Extract out_proj layer (this is what we'll prune)
        if hasattr(attention_module, 'out_proj'):
            self.layer = attention_module.out_proj
        elif hasattr(attention_module, 'proj'):
            self.layer = attention_module.proj
        else:
            raise ValueError("Attention module must have 'out_proj' or 'proj' attribute")

        self.attention_module = attention_module
        self.layer_idx = layer_idx

        # Configuration
        self.hessian_mode = getattr(args, 'hessian_mode', 'block_diagonal')
        self.head_importance_mode = getattr(args, 'head_importance_mode', 'block_mean')
        self.prune_mode = getattr(args, 'prune_mode', 'head_dims')
        self.use_compensation = getattr(args, 'use_compensation', True)

        # FastOBA parameters
        self.order = getattr(args, 'fastoba_order', 2)
        self.delta = getattr(args, 'fastoba_delta', 1.0)
        self.hessian_accumulate_freq = getattr(args, 'hessian_accumulate_freq', 10)

        # Attention structure
        if hasattr(attention_module, 'num_heads'):
            self.num_heads = attention_module.num_heads
            self.embed_dim = attention_module.embed_dim
            self.head_dim = self.embed_dim // self.num_heads
        else:
            # Infer from out_proj weight shape
            self.embed_dim = self.layer.weight.shape[0]
            # Assume standard multi-head (need to be provided externally if different)
            self.num_heads = getattr(args, 'num_heads', 12)
            self.head_dim = self.embed_dim // self.num_heads

        # Hessian matrix: [in_features, in_features]
        self.H = torch.zeros(
            self.layer.weight.shape[1],
            self.layer.weight.shape[1],
            dtype=torch.float32
        )
        self.nsamples = 0

        # Caching for FastOBA computation
        self.inp_cache = []
        self.out_cache = []

        # VAR-specific: scale-aware caching (based on add_batch_v7)
        self._var_cache = []
        self._var_cache_limit = self.hessian_accumulate_freq
        self._var_equalize_seq = True  # 按 seqlen 归一化
        self._var_gamma = None  # 权重；None 表示均匀

        # Per-stage analysis caching
        self._per_stage_cache = {s: [] for s in range(10)}  # VAR has 10 stages
        self._per_stage_H = {s: None for s in range(10)}

        self.debug = getattr(args, 'debug', False)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Main entry point for adding batch data

        Routes to appropriate method based on hessian_mode:
        - 'slimgpt': Use H = XX^T (first-order statistics)
        - 'block_diagonal': Use FastOBA automatic differentiation

        Args:
            inp: Attention input [batch, seq_len, embed_dim]
            out: Attention output [batch, seq_len, embed_dim]
        """
        if self.hessian_mode == 'slimgpt':
            self._add_batch_slimgpt(inp, out)
        elif self.hessian_mode == 'block_diagonal':
            self.add_batch_fastoba(inp, out)
        else:
            raise ValueError(f"Unknown hessian_mode: {self.hessian_mode}")

    def _add_batch_slimgpt(self, inp: torch.Tensor, out: torch.Tensor):
        """
        SlimGPT's original H = XX^T method

        This computes first-order input covariance, not true Hessian.
        """
        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])  # [batch*seq, embed_dim]

        inp = inp.t()  # [embed_dim, batch*seq]

        # EMA update
        tmp = inp.shape[1]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        # Compute local H = XX^T
        inp_normalized = math.sqrt(2 / self.nsamples) * inp.float()
        H_local = inp_normalized @ inp_normalized.t()

        self.H += H_local.cpu()

    def add_batch_fastoba(self, inp: torch.Tensor, out: torch.Tensor,
                          stage_id: Optional[int] = None):
        """
        FastOBA-based Hessian computation using automatic differentiation

        Computes Hessian w.r.t. Attention OUTPUT (pseudo-loss = ||attn_out||^2),
        not the final model loss. This is layer-internal OBS.

        Args:
            inp: Attention input [batch, seq_len, embed_dim]
            out: Attention output [batch, seq_len, embed_dim]
            stage_id: Optional stage ID for VAR multi-scale (0-9)
        """
        # Cache current input
        self.inp_cache.append(inp.detach())
        self.out_cache.append(out.detach())

        # Check if we should compute Hessian
        if len(self.inp_cache) < self.hessian_accumulate_freq:
            return

        # --- Start Hessian computation ---
        device = self.layer.weight.device

        # 1. Define pseudo-loss (based on Attention output)
        def compute_pseudo_loss():
            total_loss = 0
            for inp_batch in self.inp_cache:
                inp_batch = inp_batch.to(device).requires_grad_(True)

                # Forward through Attention module
                out_batch = self.attention_module(inp_batch)

                # Pseudo-loss: minimize Attention output change
                loss_batch = out_batch.pow(2).sum()
                total_loss = total_loss + loss_batch

            return total_loss / len(self.inp_cache)

        # 2. Compute Hessian using FastOBA's automatic differentiation
        loss = compute_pseudo_loss()

        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],
            order=self.order
        )

        # 3. Build H matrix from weight Hessian
        weight_hessian = hessian_grads[0].abs()  # [out_features, in_features]

        if self.num_heads > 1:
            # Block-diagonal approximation (for multi-head attention)
            H_new = self.compute_block_diagonal_hessian(
                weight_hessian, self.num_heads, self.head_dim
            )
        else:
            # Diagonal approximation (for single-head or fallback)
            H_diag = weight_hessian.sum(dim=0)  # [in_features]
            H_new = torch.diag(H_diag)

        # 4. EMA update
        tmp = len(self.inp_cache)
        self.H = self.H.to(device)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        scale = math.sqrt(2 / self.nsamples)
        self.H += scale * H_new
        self.H = self.H.cpu()

        # Clear cache
        self.inp_cache.clear()
        self.out_cache.clear()

    def add_batch_v7_fastoba(self, inp: torch.Tensor, out: torch.Tensor,
                             stage_id: Optional[int] = None):
        """
        VAR-aware FastOBA Hessian computation (mimics add_batch_v7)

        Features:
        - Caches last N inputs (each may be from different VAR scales)
        - Scale-independent statistics: H_local = Σ_s gamma_s * ((X_s @ X_s^T) / seqlen_s)
        - Supports per-scale analysis via dual-caching

        Args:
            inp: Attention input [batch, seq_len, embed_dim]
            out: Attention output [batch, seq_len, embed_dim]
            stage_id: VAR stage ID (0-9), inferred from seq_len if None
        """
        # Infer stage ID from sequence length
        if stage_id is None:
            L = inp.shape[1]
            patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            stage_id = next((i for i, pn in enumerate(patch_nums) if pn * pn == L), None)
            if stage_id is None:
                # Cumulative tokens (multi-scale concatenated), use global mode
                stage_id = -1

        # Cache entry with metadata
        cache_entry = {
            'inp': inp.detach(),
            'out': out.detach(),
            'stage_id': stage_id,
            'seqlen': inp.shape[1]
        }

        self._var_cache.append(cache_entry)

        # Also cache to per-stage buffer (for comparison analysis)
        if 0 <= stage_id < 10:
            self._per_stage_cache[stage_id].append(cache_entry)

        # Check if we should compute Hessian
        if len(self._var_cache) < self._var_cache_limit:
            return

        # --- Start Hessian computation ---
        xs = self._var_cache
        self._var_cache = []  # Clear for next window

        device = self.layer.weight.device
        dtype = torch.float32
        S = len(xs)

        # Compute weights gamma_s
        if self._var_gamma is None:
            gammas = torch.full((S,), 1.0 / S, device=device, dtype=dtype)
        else:
            g = torch.as_tensor(self._var_gamma, device=device, dtype=dtype)
            gammas = g / (g.sum() + 1e-12)

        # Define weighted pseudo-loss
        def compute_weighted_pseudo_loss():
            total_loss = 0
            for s, entry in enumerate(xs):
                inp_batch = entry['inp'].to(device).requires_grad_(True)
                out_batch = self.attention_module(inp_batch)

                # Scale-independent normalization
                seqlen = entry['seqlen']
                norm = (1.0 / seqlen) if self._var_equalize_seq else 1.0

                # Weighted loss
                loss_batch = gammas[s] * norm * out_batch.pow(2).sum()
                total_loss = total_loss + loss_batch

            return total_loss

        # Compute Hessian
        loss = compute_weighted_pseudo_loss()
        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],
            order=self.order
        )

        # Build block-diagonal H
        weight_hessian = hessian_grads[0].abs()
        H_local = self.compute_block_diagonal_hessian(
            weight_hessian, self.num_heads, self.head_dim
        )

        # EMA update
        tmp_total = sum(1 if entry['inp'].dim() == 2 else entry['inp'].shape[0]
                       for entry in xs)
        self.H = self.H.to(device)
        self.H *= self.nsamples / (self.nsamples + tmp_total)
        self.nsamples += tmp_total
        scale = 2.0 / self.nsamples
        self.H += scale * H_local
        self.H = self.H.cpu()

    def compute_block_diagonal_hessian(self, weight_hessian: torch.Tensor,
                                       num_heads: int, head_dim: int) -> torch.Tensor:
        """
        Build block-diagonal H matrix for multi-head attention

        Assumes different heads are independent, but dimensions within each head
        are correlated. This is a reasonable approximation for multi-head attention.

        Structure:
            H = [H_0   0    0    ...  0   ]  ← Head 0 (64×64)
                [0     H_1  0    ...  0   ]  ← Head 1 (64×64)
                [0     0    H_2  ...  0   ]
                [...   ...  ...  ...  ... ]
                [0     0    0    ...  H_11]  ← Head 11 (64×64)

        Args:
            weight_hessian: [out_features, in_features], e.g., [768, 768]
            num_heads: Number of attention heads (e.g., 12)
            head_dim: Dimension per head (e.g., 64)

        Returns:
            H: [in_features, in_features] block-diagonal matrix
        """
        in_features = weight_hessian.shape[1]
        device = weight_hessian.device

        H = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)

        for head_id in range(num_heads):
            start = head_id * head_dim
            end = (head_id + 1) * head_dim

            # Extract Hessian for this head's input dimensions
            head_hessian = weight_hessian[:, start:end]  # [out_features, head_dim]

            # Build covariance matrix: H_block[j,k] = Σ_i (∂²L/∂W[i,j]) · (∂²L/∂W[i,k])
            # This is A^T A form, which is positive semi-definite
            H_block = head_hessian.t() @ head_hessian  # [head_dim, head_dim]

            # Normalize to avoid numerical overflow
            H_block = H_block / weight_hessian.shape[0]

            # Fill into H's block-diagonal position
            H[start:end, start:end] = H_block

        return H

    def any_order_differentiation(self, loss: torch.Tensor, delta: float,
                                   parameters: List[torch.Tensor],
                                   order: int) -> List[torch.Tensor]:
        """
        FastOBA's automatic differentiation for k-th order Taylor expansion

        Computes: grad^(k) = ∂^k L / ∂θ^k

        For order=2: returns Hessian-vector product × θ × δ²

        Args:
            loss: Scalar loss tensor
            delta: Delta parameter for Taylor expansion
            parameters: List of parameter tensors
            order: Order of differentiation (1 for gradient, 2 for Hessian, etc.)

        Returns:
            List of k-th order gradients for each parameter
        """
        grads = [torch.zeros_like(param) for param in parameters]

        for current_order in range(1, order + 1):
            if current_order == 1:
                # First-order: ∂L/∂θ
                if current_order == order:
                    current_grad = torch.autograd.grad(loss, parameters)
                else:
                    current_grad = torch.autograd.grad(loss, parameters, create_graph=True)
            else:
                # k-th order: ∂(grad^(k-1))/∂θ
                grad_outputs = [param * delta for param in parameters]
                if current_order == order:
                    current_grad = torch.autograd.grad(
                        current_grad, parameters,
                        grad_outputs=grad_outputs
                    )
                else:
                    current_grad = torch.autograd.grad(
                        current_grad, parameters,
                        grad_outputs=grad_outputs,
                        create_graph=True
                    )

        # Final k-th order term: grad^(k) × θ × δ^k
        grads = [grad.detach() * param.data * delta
                for grad, param in zip(current_grad, parameters)]

        return grads

    def compute_per_stage_hessian(self, stage_id: int,
                                   flush_cache: bool = True) -> Optional[torch.Tensor]:
        """
        Compute independent Hessian for a specific VAR stage

        Used for per-stage comparison analysis to understand how different
        scales affect pruning decisions.

        Args:
            stage_id: Stage ID (0-9 for VAR's 10 scales)
            flush_cache: Whether to clear the cache after computation

        Returns:
            H_stage: [in_features, in_features] Hessian for this stage,
                    or None if no data cached
        """
        if stage_id not in self._per_stage_cache:
            return None

        xs = self._per_stage_cache[stage_id]
        if len(xs) == 0:
            return None

        device = self.layer.weight.device

        # Define stage-specific pseudo-loss
        def compute_stage_loss():
            total_loss = 0
            for entry in xs:
                inp_batch = entry['inp'].to(device).requires_grad_(True)
                out_batch = self.attention_module(inp_batch)
                loss_batch = out_batch.pow(2).sum()
                total_loss = total_loss + loss_batch
            return total_loss / len(xs)

        # Compute Hessian
        loss = compute_stage_loss()
        hessian_grads = self.any_order_differentiation(
            loss=loss,
            delta=self.delta,
            parameters=[self.layer.weight],
            order=self.order
        )

        # Build block-diagonal H
        weight_hessian = hessian_grads[0].abs()
        H_stage = self.compute_block_diagonal_hessian(
            weight_hessian, self.num_heads, self.head_dim
        )

        # Cache result
        self._per_stage_H[stage_id] = H_stage.cpu()

        if flush_cache:
            self._per_stage_cache[stage_id] = []

        return H_stage.cpu()

    def compute_per_stage_head_importance(self, stage_id: int) -> Optional[torch.Tensor]:
        """
        Compute head importance for a specific scale

        Uses block-diagonal mean: avg(diag(H_block)) for each head

        Args:
            stage_id: Stage ID (0-9)

        Returns:
            head_imp: [num_heads] importance scores, or None if no data
        """
        H_stage = self._per_stage_H.get(stage_id)
        if H_stage is None:
            H_stage = self.compute_per_stage_hessian(stage_id, flush_cache=False)

        if H_stage is None:
            return None

        head_imp = torch.zeros(self.num_heads)
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = (h + 1) * self.head_dim
            H_block = H_stage[start:end, start:end]
            head_imp[h] = torch.diag(H_block).mean()

        return head_imp

    def compute_per_stage_head_dim_importance(self, stage_id: int) -> Optional[torch.Tensor]:
        """
        Compute dimension importance within each head for a specific scale

        Uses OBS formula: importance = W² / [H^-1]_diag

        Args:
            stage_id: Stage ID (0-9)

        Returns:
            dim_imp: [num_heads, head_dim] importance scores, or None if no data
        """
        H_stage = self._per_stage_H.get(stage_id)
        if H_stage is None:
            H_stage = self.compute_per_stage_hessian(stage_id, flush_cache=False)

        if H_stage is None:
            return None

        device = H_stage.device
        H_stage = H_stage.to(device)

        # Add dampening for numerical stability
        damp = 0.01 * torch.diag(H_stage).mean()
        diag_indices = torch.arange(H_stage.shape[0], device=device)
        H_stage[diag_indices, diag_indices] += damp

        # Compute H^-1
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H_stage))
            Hinv_diag = torch.diag(Hinv)
        except RuntimeError as e:
            warnings.warn(f"Cholesky failed for stage {stage_id}: {e}. Using diagonal approximation.")
            Hinv_diag = 1.0 / (torch.diag(H_stage) + 1e-8)

        # OBS importance
        importance = (self.layer.weight.to(device) ** 2).sum(0) / (Hinv_diag + 1e-8)

        # Reshape to [num_heads, head_dim]
        dim_imp = importance.view(self.num_heads, self.head_dim)

        return dim_imp.cpu()

    def struct_prune(self, sparsity: float, headsize: int = 1,
                    percdamp: float = 0.01) -> torch.Tensor:
        """
        Structured pruning using OBS framework

        This is a placeholder that delegates to the appropriate pruning mode.
        The actual implementation should be integrated with torch_pruning's
        dependency graph.

        Args:
            sparsity: Target sparsity ratio (default: 0.4 for 40%)
            headsize: Head dimension (64 for head-wise, 1 for channel-wise)
            percdamp: Dampening percentage for numerical stability

        Returns:
            pruned_indices: Indices of pruned channels/dimensions
        """
        device = self.layer.weight.device
        H = self.H.to(device)
        W = self.layer.weight.data

        # Add dampening
        if percdamp > 0:
            damp = percdamp * torch.diag(H).mean()
            diag_indices = torch.arange(H.shape[0], device=device)
            H[diag_indices, diag_indices] += damp

        # Compute H^-1
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError as e:
            warnings.warn(f"Cholesky decomposition failed: {e}. Using diagonal approximation.")
            Hinv = torch.diag(1.0 / (torch.diag(H) + 1e-8))

        # Compute importance based on mode
        if self.prune_mode == 'head_dims':
            # Prune dimensions within each head independently
            importance = self._compute_channel_importance(W, Hinv, headsize)
            pruned_indices = self._select_pruning_indices_per_head(
                importance, sparsity, headsize
            )

        elif self.prune_mode == 'num_heads':
            # Prune entire heads
            head_importance = self._compute_head_importance(W, Hinv)
            n_pruned_heads = int(self.num_heads * sparsity)
            pruned_head_ids = torch.argsort(head_importance)[:n_pruned_heads]

            # Convert head IDs to dimension indices
            pruned_indices = []
            for head_id in pruned_head_ids:
                start = head_id * self.head_dim
                end = (head_id + 1) * self.head_dim
                pruned_indices.extend(range(start, end))
            pruned_indices = torch.tensor(pruned_indices, device=device)

        elif self.prune_mode == 'both':
            # Two-stage: first prune heads, then prune dims in remaining heads
            # TODO: Implement two-stage pruning
            raise NotImplementedError("'both' pruning mode not yet implemented")

        else:
            raise ValueError(f"Unknown prune_mode: {self.prune_mode}")

        return pruned_indices

    def _compute_channel_importance(self, W: torch.Tensor, Hinv: torch.Tensor,
                                   headsize: int) -> torch.Tensor:
        """
        Compute OBS importance for each channel/dimension

        Formula: importance = W² / [H^-1]_diag

        For head-wise pruning (headsize > 1), uses block Cholesky decomposition.
        """
        if headsize > 1:
            # Head-wise: block-diagonal Cholesky (SlimGPT's approach)
            Hinv_diag = torch.stack([
                Hinv[i:i+headsize, i:i+headsize]
                for i in range(0, W.shape[1], headsize)
            ])  # [num_heads, head_dim, head_dim]

            Hinv_diag = torch.diagonal(
                torch.linalg.cholesky(Hinv_diag),
                dim1=-2, dim2=-1
            ).reshape(-1)  # [in_features]

            Hinv_diag = Hinv_diag ** 2
        else:
            # Channel-wise: simple diagonal
            Hinv_diag = torch.diag(Hinv)

        # OBS importance
        importance = (W ** 2).sum(dim=0) / (Hinv_diag + 1e-8)

        return importance

    def _compute_head_importance(self, W: torch.Tensor,
                                Hinv: torch.Tensor) -> torch.Tensor:
        """
        Compute importance for each attention head

        Uses the method specified by head_importance_mode:
        - 'block_mean': Average of block-diagonal elements
        - 'slimgpt_mean': Average of per-channel OBS importance
        """
        if self.head_importance_mode == 'block_mean':
            # Use block-diagonal H matrix
            head_imp = torch.zeros(self.num_heads, device=W.device)
            for h in range(self.num_heads):
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                H_block = self.H[start:end, start:end].to(W.device)
                head_imp[h] = torch.diag(H_block).mean()

        elif self.head_importance_mode == 'slimgpt_mean':
            # Use OBS importance for each channel, then average per head
            Hinv_diag = torch.diag(Hinv)
            importance = (W ** 2).sum(dim=0) / (Hinv_diag + 1e-8)
            head_imp = importance.view(self.num_heads, self.head_dim).mean(1)

        else:
            raise ValueError(f"Unknown head_importance_mode: {self.head_importance_mode}")

        return head_imp

    def _select_pruning_indices_per_head(self, importance: torch.Tensor,
                                         sparsity: float,
                                         headsize: int) -> torch.Tensor:
        """
        Select pruning indices independently for each head

        Args:
            importance: [in_features] importance scores
            sparsity: Target sparsity ratio
            headsize: Head dimension

        Returns:
            pruned_indices: [n_pruned] indices to prune
        """
        device = importance.device
        n_pruned_per_head = int(self.head_dim * sparsity)

        pruning_indices = []
        for h in range(self.num_heads):
            start = h * headsize
            end = (h + 1) * headsize

            head_importance = importance[start:end]
            head_pruning_idxs = torch.argsort(head_importance)[:n_pruned_per_head]
            head_pruning_idxs = head_pruning_idxs + start  # Add offset

            pruning_indices.append(head_pruning_idxs)

        pruning_indices = torch.cat(pruning_indices)
        return pruning_indices
