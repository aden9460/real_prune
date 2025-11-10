"""
FastOBA + Attention + SlimGPT Integration for VAR Models

This module combines:
- FastOBA's automatic differentiation for Hessian computation
- SlimGPT's Cholesky decomposition and OBS pruning
- Attention-specific layer-internal Hessian
- VAR-aware multi-scale token handling

Key features:
- Computes Hessian w.r.t. Attention OUTPUT (not final loss)
- Supports block-diagonal approximation for multi-head attention
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
        attention_module: Complete Attention module (with qkv and proj)
        layer_idx: Layer index for logging
        num_heads: Number of attention heads
        embed_dim: Embedding dimension
        hessian_mode: 'slimgpt' or 'block_diagonal'
        head_importance_mode: 'block_mean' or 'slimgpt_mean'
        prune_mode: 'head_dims', 'num_heads', or 'both'
        use_compensation: Whether to use OBS weight compensation
        fastoba_order: Order of Taylor expansion (default 2)
        fastoba_delta: Delta parameter for FastOBA (default 1.0)
        hessian_accumulate_freq: Compute Hessian every N batches (default 10)
    """

    def __init__(self,
                 attention_module,
                 layer_idx: int,
                 num_heads: int = 12,
                 embed_dim: int = 768,
                 hessian_mode: str = 'block_diagonal',
                 head_importance_mode: str = 'block_mean',
                 prune_mode: str = 'head_dims',
                 use_compensation: bool = True,
                 fastoba_order: int = 2,
                 fastoba_delta: float = 1.0,
                 hessian_accumulate_freq: int = 10,
                 use_exact_hessian: bool = False,  # 新参数：是否使用精确Hessian
                 debug: bool = False):

        # Extract out_proj layer (this is what we'll prune)
        if hasattr(attention_module, 'proj'):
            self.layer = attention_module.proj
        elif hasattr(attention_module, 'out_proj'):
            self.layer = attention_module.out_proj
        else:
            raise ValueError("Attention module must have 'proj' or 'out_proj' attribute")

        self.attention_module = attention_module
        self.layer_idx = layer_idx

        # Configuration
        self.hessian_mode = hessian_mode
        self.head_importance_mode = head_importance_mode
        self.prune_mode = prune_mode
        self.use_compensation = use_compensation

        # FastOBA parameters
        self.order = fastoba_order
        self.delta = fastoba_delta
        self.hessian_accumulate_freq = hessian_accumulate_freq
        self.use_exact_hessian = use_exact_hessian  # 新增：控制Hessian计算方式
        self.debug = debug

        # Attention structure
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        # Hessian matrix: [in_features, in_features]
        in_features = self.layer.weight.shape[1]
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32)
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
            """
            伪损失：简单的L2范数

            loss = ||attn_out||²

            注：这是最简单的形式，虽然理论上不完美，但实践中可能更稳定
            """
            total_loss = 0
            for inp_batch in self.inp_cache:
                inp_batch = inp_batch.to(device).requires_grad_(True)

                # Forward through Attention module
                out_batch = self.attention_module(inp_batch)

                # Pseudo-loss: L2 norm of attention output
                loss_batch = out_batch.pow(2).sum()

                total_loss = total_loss + loss_batch

            return total_loss / len(self.inp_cache)

        # 2. Compute Hessian using FastOBA's automatic differentiation
        loss = compute_pseudo_loss()

        if self.use_exact_hessian:
            # 【新方法】计算精确的完整Hessian（慢但准确）
            print(f"[FastOBA Layer {self.layer_idx}] Computing exact Hessian using basis vectors...")
            H_new = self.compute_full_block_diagonal_hessian(
                self.num_heads, self.head_dim, use_approximation=False
            )
        else:
            # 【原方法】使用A^T A近似（快但不够准确）
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

        # 【关键修复】与SlimGPT的add_batch_v7保持一致
        # 原问题: 使用 sqrt(2/n) 导致H矩阵数值偏大约3.16倍（当n=20时）
        # SlimGPT逻辑: inp *= sqrt(2/n), 然后 H += inp @ inp.t()
        #             相当于 H += (2/n) * X@X^T
        # 因此FastOBA也应该用 2/n 而不是 sqrt(2/n)
        scale = 2.0 / self.nsamples  # 修正: 2/n 而非 sqrt(2/n)
        self.H += scale * H_new
        self.H = self.H.cpu()

        # 【诊断】输出H矩阵统计信息，验证修复效果
        if self.debug and self.nsamples <= self.hessian_accumulate_freq * 2:
            print(f"[FastOBA] nsamples={self.nsamples}, scale={scale:.6f}")
            print(f"  H diagonal mean: {torch.diag(self.H).mean():.6e}")
            print(f"  H max: {self.H.max():.6e}")

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
        - Supports per-stage analysis via dual-caching

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

    def compute_full_hessian_per_head(self, head_id: int, head_dim: int) -> torch.Tensor:
        """
        【新方法】计算单个head的完整Hessian矩阵（不使用A^T A近似）

        使用标准基向量法：对每个输入维度计算Hessian-vector product
        H[:, j] = Hessian @ e_j，其中e_j是第j个标准基向量

        这比A^T A近似更准确，但需要O(head_dim)次反向传播。
        对于head_dim=64，需要64次反向传播，仍然可行。

        Args:
            head_id: Head索引 (0 to num_heads-1)
            head_dim: 每个head的维度 (如64)

        Returns:
            H_head: [head_dim, head_dim] 该head的完整Hessian矩阵
        """
        device = self.layer.weight.device
        H_head = torch.zeros(head_dim, head_dim, device=device, dtype=torch.float32)

        # 计算该head在权重矩阵中的列范围
        start_col = head_id * head_dim
        end_col = (head_id + 1) * head_dim

        # 对每个维度j计算Hessian的第j列
        for j in range(head_dim):
            # 构造标准基向量 e_j (只有第j个位置为1)
            v = torch.zeros_like(self.layer.weight)
            v[:, start_col + j] = 1.0  # 该head的第j个输入维度

            # 定义伪损失（与add_batch_fastoba中相同）
            def compute_pseudo_loss_with_v():
                total_loss = 0
                for inp_entry in self.inp_cache:
                    inp_batch = inp_entry.to(device).requires_grad_(True)
                    out_batch = self.attention_module(inp_batch)
                    loss_batch = out_batch.pow(2).sum()
                    total_loss = total_loss + loss_batch
                return total_loss / len(self.inp_cache)

            loss = compute_pseudo_loss_with_v()

            # 计算一阶梯度
            first_grad = torch.autograd.grad(loss, self.layer.weight, create_graph=True)[0]

            # 计算Hessian-vector product: H @ v
            # 这给出Hessian的第j列（对应该head的维度j）
            hvp = torch.autograd.grad(
                first_grad,
                self.layer.weight,
                grad_outputs=v,
                retain_graph=False
            )[0]

            # 提取该head对应的Hessian列
            # hvp的形状是[out_features, in_features]
            # 我们需要该head对应的输入维度部分
            H_column = hvp[:, start_col:end_col].sum(dim=0)  # [head_dim]

            # 填充到Hessian矩阵的第j列
            H_head[:, j] = H_column

        # 对称化（理论上Hessian应该对称，但数值误差可能导致轻微不对称）
        H_head = (H_head + H_head.t()) / 2.0

        return H_head

    def compute_full_block_diagonal_hessian(self, num_heads: int, head_dim: int,
                                             use_approximation: bool = True) -> torch.Tensor:
        """
        【新方法】计算块对角Hessian，可选择是否使用近似

        Args:
            num_heads: 头数量
            head_dim: 每个头的维度
            use_approximation:
                - True: 使用快速的A^T A近似（当前方法）
                - False: 计算完整Hessian（慢，但准确）

        Returns:
            H: [in_features, in_features] 块对角Hessian矩阵
        """
        in_features = num_heads * head_dim
        device = self.layer.weight.device
        H = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)

        if use_approximation:
            # 快速近似方法（原有逻辑）
            # 需要先通过any_order_differentiation获取weight_hessian
            print("[Info] Using fast A^T A approximation for Hessian")
            # 这部分在add_batch_fastoba中调用
            return None  # 需要weight_hessian作为输入
        else:
            # 精确方法：逐个head计算完整Hessian
            print(f"[Info] Computing exact Hessian for {num_heads} heads (this may take a while)...")
            for head_id in range(num_heads):
                print(f"  Computing head {head_id+1}/{num_heads}...", end='\r')

                # 计算该head的完整Hessian
                H_head = self.compute_full_hessian_per_head(head_id, head_dim)

                # 填充到块对角位置
                start = head_id * head_dim
                end = (head_id + 1) * head_dim
                H[start:end, start:end] = H_head

            print(f"\n[Info] Exact Hessian computation complete!")

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

    def get_pruning_importance(self, percdamp: float = 0.01) -> torch.Tensor:
        """
        Get channel/dimension importance scores for pruning

        Returns importance based on current H matrix and configured mode.

        Args:
            percdamp: Dampening percentage for numerical stability

        Returns:
            importance: [in_features] importance scores (higher = more important)
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

        # Compute importance based on head_importance_mode
        if self.head_importance_mode == 'block_mean' and self.num_heads > 1:
            # Block-diagonal Cholesky (SlimGPT's head-wise approach)
            Hinv_diag = torch.stack([
                Hinv[i:i+self.head_dim, i:i+self.head_dim]
                for i in range(0, W.shape[1], self.head_dim)
            ])  # [num_heads, head_dim, head_dim]

            Hinv_diag = torch.diagonal(
                torch.linalg.cholesky(Hinv_diag),
                dim1=-2, dim2=-1
            ).reshape(-1)  # [in_features]

            Hinv_diag = Hinv_diag ** 2
        else:
            # Simple diagonal
            Hinv_diag = torch.diag(Hinv)

        # OBS importance: W² / [H^-1]_diag
        importance = (W ** 2).sum(dim=0) / (Hinv_diag + 1e-8)

        return importance.cpu()

    def struct_prune(self, sparsity: float = 0.4, headsize: int = 1,
                     percdamp: float = 0.01, blocksize: int = 128) -> torch.Tensor:
        """
        Structured pruning with OBS weight compensation

        This implements the full SlimGPT pruning algorithm with two modes:
        1. Channel-wise (headsize=1): Prune individual dimensions
        2. Head-wise (headsize=head_dim): Prune entire attention heads

        Args:
            sparsity: Target sparsity ratio (default: 0.4 for 40%)
            headsize: Pruning granularity (1 for channel-wise, head_dim for head-wise)
            percdamp: Dampening percentage for numerical stability
            blocksize: Block size for iterative pruning

        Returns:
            pruned_indices: Indices of pruned channels/dimensions
        """
        device = self.layer.weight.device
        W = self.layer.weight.data.clone().float()

        # Prepare H matrix
        H = self.H.to(device)

        # Add dampening
        if percdamp > 0:
            damp = percdamp * torch.diag(H).mean()
            diag_indices = torch.arange(H.shape[0], device=device)
            H[diag_indices, diag_indices] += damp

        # Compute H^-1
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError as e:
            warnings.warn(f"Cholesky failed: {e}. Using diagonal approximation.")
            Hinv = torch.diag(1.0 / (torch.diag(H) + 1e-8))

        # Iterative pruning with compensation
        columns = W.shape[1]
        target_columns = int(columns * sparsity)
        pruned_columns = 0
        column_mask = torch.zeros(columns, dtype=torch.bool, device=device)

        if self.debug:
            print(f"\nStarting pruning: target={target_columns}/{columns} ({sparsity*100:.1f}%)")
            print(f"W.shape: {W.shape}")
            print(f"H.shape: {H.shape}")
            print(f"Hinv.shape: {Hinv.shape}")

        while pruned_columns < target_columns:
            # Compute pruning error for each column
            if headsize > 1:
                # Head-wise: block-diagonal Cholesky
                Hinv_diag = torch.stack([
                    Hinv[i:i+headsize, i:i+headsize]
                    for i in range(0, columns, headsize)
                ])
                Hinv_diag = torch.diagonal(
                    torch.linalg.cholesky(Hinv_diag),
                    dim1=-2, dim2=-1
                ).reshape(-1)
                Hinv_diag = Hinv_diag ** 2
            else:
                # Channel-wise: simple diagonal
                Hinv_diag = torch.diag(Hinv)

            if self.debug:
                print(f"headsize: {headsize}, Hinv_diag.shape: {Hinv_diag.shape}, W.shape: {W.shape}")

            # OBS error: W² / [H^-1]_diag
            error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
            error[column_mask] = torch.inf

            # Select columns to prune
            if headsize > 1:
                # Head-wise: sort by head error
                head_sort_idx = error.view(-1, headsize).sum(1).argsort()
                column_sort_idx = torch.hstack([
                    torch.arange(x * headsize, x * headsize + headsize)
                    for x in head_sort_idx
                ])
                cnt = headsize
            else:
                # Channel-wise: sort by column error
                column_sort_idx = error.argsort()
                cnt = min(target_columns - pruned_columns, blocksize)

            # Reorder for efficient processing
            W = W[:, column_sort_idx]
            Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]

            # 【重要修复】与原始SlimGPT的关键区别：
            # 原始SlimGPT代码: Hinv = torch.linalg.cholesky(Hinv, upper=True)[:cnt]
            # 问题: 直接切片[:cnt]会将Hinv从[768,768]方阵变成[cnt,768]非方阵，
            #      导致下一次迭代时torch.diag(Hinv)只能提取cnt个元素，引发维度不匹配错误
            # 修复: 保持Hinv完整性，将Cholesky结果存到新变量Hinv_chol中，
            #      后续只在需要的地方使用Hinv_chol的子块
            Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)

            # Prepare for compensation
            W1 = W[:, :cnt].clone()
            # 【修改】从完整的Hinv_chol中提取需要的cnt×cnt块，而不是使用被破坏的Hinv
            Hinv1 = Hinv_chol[:cnt, :cnt]
            Err1 = torch.zeros_like(W1)

            # Local compensation (within pruned block)
            for i in range(cnt):
                Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
                if self.use_compensation:
                    W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

            # Zero out pruned columns
            W[:, :cnt] = 0

            # Global compensation (remaining columns)
            # 【注意】FastOBA使用block-diagonal Hessian，不同head间off-diagonal块为0
            # 因此全局补偿仅在headsize=1时有效（细粒度剪枝）
            # 对于head-wise或head-dim剪枝，全局补偿效果有限，但保留以兼容SlimGPT接口
            if self.use_compensation:
                end = columns - pruned_columns
                # 【修复】使用完整的Hinv_chol而不是被截断的Hinv
                # 原始代码: W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])
                # 问题: Hinv已被截断为[cnt, columns]，导致下次迭代维度错误
                W[:, cnt:end] -= Err1.matmul(Hinv_chol[:cnt, cnt:end])

            # Restore original order
            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W = W[:, column_sort_idx_inv]

            # Update masks
            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = H[:, pruned_idx] = 0
            H[pruned_idx, pruned_idx] = 1
            column_mask[pruned_idx] = 1
            pruned_columns += cnt

            if self.debug:
                print(f"  Iteration: pruned {pruned_columns}/{target_columns}")

        # Update layer weights
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        pruned_indices = torch.where(column_mask)[0]

        if self.debug:
            print(f"Pruning complete: {len(pruned_indices)} channels pruned")

        return pruned_indices.cpu()

    def struct_prune_head_dims(self, sparsity: float = 0.4,
                                percdamp: float = 0.01,
                                blocksize: int = 16) -> torch.Tensor:
        """
        Head-internal dimension pruning with block-diagonal compensation

        This prunes dimensions WITHIN each head independently, using block-diagonal
        Hessian structure. The compensation is head-specific: pruning dim_i in head_j
        only affects other dimensions in head_j, not other heads.

        Args:
            sparsity: Target sparsity ratio per head (default: 0.4 for 40%)
            percdamp: Dampening percentage for numerical stability
            blocksize: Block size for iterative pruning within each head

        Returns:
            pruned_indices: Global indices of pruned dimensions
        """
        device = self.layer.weight.device
        W = self.layer.weight.data.clone().float()

        # Prepare H matrix
        H = self.H.to(device)

        # Add dampening
        if percdamp > 0:
            damp = percdamp * torch.diag(H).mean()
            diag_indices = torch.arange(H.shape[0], device=device)
            H[diag_indices, diag_indices] += damp

        # Global pruned indices
        all_pruned_indices = []

        # Process each head independently
        for head_id in range(self.num_heads):
            start_idx = head_id * self.head_dim
            end_idx = (head_id + 1) * self.head_dim

            # Extract head-specific weights and Hessian
            W_head = W[:, start_idx:end_idx].clone()  # [out_features, head_dim]
            H_head = H[start_idx:end_idx, start_idx:end_idx].clone()  # [head_dim, head_dim]

            # Compute H^-1 for this head
            try:
                Hinv_head = torch.cholesky_inverse(torch.linalg.cholesky(H_head))
            except RuntimeError as e:
                warnings.warn(f"Cholesky failed for head {head_id}: {e}. Using diagonal.")
                Hinv_head = torch.diag(1.0 / (torch.diag(H_head) + 1e-8))

            # Iterative pruning within this head
            target_dims = int(self.head_dim * sparsity)
            pruned_dims = 0
            dim_mask = torch.zeros(self.head_dim, dtype=torch.bool, device=device)

            if self.debug:
                print(f"\nHead {head_id}: pruning {target_dims}/{self.head_dim} dims")

            while pruned_dims < target_dims:
                # Compute OBS error for each dimension
                Hinv_diag = torch.diag(Hinv_head)
                error = torch.sum(W_head ** 2 / Hinv_diag.unsqueeze(0), dim=0)
                error[dim_mask] = torch.inf

                # Select dimensions to prune in this iteration
                dim_sort_idx = error.argsort()
                cnt = min(target_dims - pruned_dims, blocksize, self.head_dim)

                # Reorder for efficient processing
                W_head = W_head[:, dim_sort_idx]
                Hinv_head = Hinv_head[dim_sort_idx, :][:, dim_sort_idx]
                # 【修复】保持Hinv_chol完整性，不能直接切片[:cnt]
                # 问题: [:cnt]会将方阵变成非方阵，导致迭代时维度不匹配
                Hinv_chol = torch.linalg.cholesky(Hinv_head, upper=True)

                # Prepare for compensation
                W1 = W_head[:, :cnt].clone()
                # 【修复】从完整的Hinv_chol中提取需要的cnt×cnt块
                Hinv1 = Hinv_chol[:cnt, :cnt]
                Err1 = torch.zeros_like(W1)

                # Local compensation (within pruned dims)
                for i in range(cnt):
                    Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
                    if self.use_compensation:
                        W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

                # Zero out pruned dimensions
                W_head[:, :cnt] = 0

                # Global compensation (remaining dims in this head)
                if self.use_compensation:
                    end = self.head_dim - pruned_dims
                    # 【修复】使用完整Hinv_chol的子块，而不是被截断的矩阵
                    W_head[:, cnt:end] -= Err1.matmul(Hinv_chol[:cnt, cnt:end])

                # Restore original order
                dim_sort_idx_inv = torch.argsort(dim_sort_idx)
                W_head = W_head[:, dim_sort_idx_inv]

                # Update masks
                pruned_local_idx = dim_sort_idx[:cnt]
                H_head[pruned_local_idx, :] = H_head[:, pruned_local_idx] = 0
                H_head[pruned_local_idx, pruned_local_idx] = 1
                dim_mask[pruned_local_idx] = 1
                pruned_dims += cnt

                if self.debug:
                    print(f"  Head {head_id} iteration: {pruned_dims}/{target_dims} pruned")

            # Store pruned head weights back
            W[:, start_idx:end_idx] = W_head

            # Convert local indices to global indices
            global_pruned = torch.where(dim_mask)[0] + start_idx
            all_pruned_indices.append(global_pruned)

            # Update global H matrix
            H[start_idx:end_idx, start_idx:end_idx] = H_head

        # Update layer weights
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Combine all pruned indices
        pruned_indices = torch.cat(all_pruned_indices)

        if self.debug:
            print(f"\nPruning complete: {len(pruned_indices)} total dims pruned across {self.num_heads} heads")

        return pruned_indices.cpu()
