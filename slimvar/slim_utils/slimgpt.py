import math
import time
import os
import torch
import torch.nn as nn
import transformers

import matplotlib.pyplot as plt

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SlimGPT(object):
    def __init__(self, layer, layer_idx, args):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.args = args
        self.no_compensate = args.no_compensate

        # Taylor方法所需的梯度累积器
        self.grad_first = None   # 一阶梯度（最后一次backward）
        self.grad_second = None  # 二阶梯度累积（多次backward的grad²）
        self.taylor_samples = 0  # 已处理的样本数

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out                

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # [hsize, seqlen]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def struct_prune(
        self, sparsity, headsize=1, percdamp=0.0, layer_idx=None, 
    ):
        assert self.columns % headsize == 0

        tick = time.time()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev) # 1 for remove
        pruned_columns = column_mask.count_nonzero()
        target_columns = round(self.columns // headsize * sparsity) * headsize

        if headsize > 1:
            pass
        else:
            blocksize = (target_columns - 512) // 2

        while pruned_columns < target_columns:     
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            if headsize > 1:
                Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize] for i in range(0, self.columns, headsize)])
                Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag), dim1=-2, dim2=-1).reshape(-1)
                Hinv_diag = Hinv_diag ** 2
            else:
                Hinv_diag = Hinv.diag()

            error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
            error[column_mask] = torch.inf
            if headsize > 1:
                head_sort_idx = error.view(-1, headsize).sum(1).argsort()
                column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize) for x in head_sort_idx])
                cnt = headsize
            else:
                column_sort_idx = error.argsort()
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)

            W = W[:, column_sort_idx]
            Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
            Hinv = torch.linalg.cholesky(Hinv, upper=True)[:cnt]
            
            W1 = W[:, :cnt].clone()
            Hinv1 = Hinv[:, :cnt]
            Err1 = torch.zeros_like(W1)

            for i in range(cnt):
                Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
                if not self.no_compensate:
                    W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])  # local update

            W[:, :cnt] = 0
            if not self.no_compensate:
                end = self.columns - pruned_columns
                W[:, cnt:end] -= Err1.matmul(Hinv[:, cnt:end])  # global update

            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W = W[:, column_sort_idx_inv]

            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = H[:, pruned_idx] = 0
            H[pruned_idx, pruned_idx] = 1
            column_mask[pruned_idx] = 1
            pruned_columns += cnt

            if headsize > 1:
                pass
            else:
                blocksize = (blocksize - 512) // 2

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # print('time %.2f' % (time.time() - tick), flush=True)
        print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)), flush=True)

        if DEBUG:
            out_gap = torch.mean((self.layer(self.inp1) - self.out1) ** 2).item()
            out = torch.mean(self.out1 ** 2).item()
            print('output_gap:', out_gap, flush=True)
            print('output:', out, flush=True)
            print('output_gap / output:', out_gap / out, flush=True)

        # Return the indices of pruned columns for Torch-Pruning integration
        pruned_indices = column_mask.nonzero(as_tuple=True)[0]
        return pruned_indices

    def accumulate_hessian_diag(self):
        """
        累积Hessian对角线近似（二阶梯度 = grad²）

        调用时机: 每次loss.backward()后，zero_grad()前
        """
        if self.layer.weight.grad is None:
            print(f"    WARNING: gradient is None for layer {self.layer.__class__.__name__}, skipping accumulation")
            return

        grad = self.layer.weight.grad.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()

        # 验证梯度
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        grad_max = grad.abs().max().item()

        if grad_max < 1e-8:
            print(f"    WARNING: very small gradients (max={grad_max:.2e}), this may affect pruning quality")

        # 梯度平方
        grad_squared = grad ** 2

        # 累积
        if self.grad_second is None:
            self.grad_second = grad_squared
        else:
            self.grad_second += grad_squared

        self.taylor_samples += 1

        # 每5次样本打印统计
        if self.taylor_samples % 5 == 0:
            print(f"    Taylor sample {self.taylor_samples}: grad mean={grad_mean:.2e}, std={grad_std:.2e}, max={grad_max:.2e}")

    def finalize_hessian_diag(self):
        """
        归一化累积的Hessian对角线

        调用时机: 所有样本处理完后
        """
        if self.grad_second is not None and self.taylor_samples > 0:
            self.grad_second /= self.taylor_samples

    def capture_first_order_grad(self):
        """
        捕获一阶梯度（用于param_first和param_mix）

        调用时机: 最后一次loss.backward()后
        """
        if self.layer.weight.grad is None:
            print(f"    WARNING: gradient is None for layer {self.layer.__class__.__name__}, cannot capture first-order gradient")
            return

        grad = self.layer.weight.grad.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()

        # 验证梯度
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        grad_max = grad.abs().max().item()

        print(f"    First-order grad: mean={grad_mean:.2e}, std={grad_std:.2e}, max={grad_max:.2e}")

        self.grad_first = grad

    def taylor_prune_llm(self, sparsity, headsize=64, percdamp=0.01,
                         layer_idx=0, taylor_type='param_mix'):
        """
        基于LLM-Pruner Taylor重要性的结构化剪枝

        支持三种Taylor变体:
        - param_first: I = |w · ∂L/∂w| (一阶)
        - param_second: I = |w · H_ii · w| (纯二阶)
        - param_mix: I = |w · ∂L/∂w - 0.5 · w · H_ii · w| (混合，推荐)

        Args:
            sparsity: 剪枝率 (0-1)
            headsize: head大小 (VAR固定64)
            percdamp: 保留参数（与Taylor无关）
            layer_idx: 层索引（用于日志）
            taylor_type: 'param_first', 'param_second', 'param_mix'

        Returns:
            prune_indices: 要剪枝的column索引 [Tensor]
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # 计算salience（显著性）
        if taylor_type == 'param_first':
            # 一阶: S = w · ∂L/∂w
            if self.grad_first is None:
                raise ValueError("First order gradient not captured. Call capture_first_order_grad() first.")
            salience = W * self.grad_first

        elif taylor_type == 'param_second':
            # 纯二阶: S = w · H_ii · w
            if self.grad_second is None:
                raise ValueError("Second order gradient not accumulated. Call accumulate_hessian_diag() during training.")
            salience = W * self.grad_second * W

        elif taylor_type == 'param_mix':
            # 混合: S = w · ∂L/∂w - 0.5 · w · H_ii · w
            if self.grad_first is None or self.grad_second is None:
                raise ValueError("Both first and second order gradients required for param_mix.")
            salience = W * self.grad_first - 0.5 * W * self.grad_second * W

        else:
            raise ValueError(f"Unknown taylor_type: {taylor_type}. Must be 'param_first', 'param_second', or 'param_mix'.")

        # 聚合到输出通道 (sum across input dimension)
        importance = salience.abs().sum(dim=1)  # [out_channels]

        # 按head分组（VAR特定）
        if headsize > 1:
            num_heads = importance.shape[0] // headsize
            assert importance.shape[0] % headsize == 0, f"out_channels={importance.shape[0]} must be divisible by headsize={headsize}"

            head_importance = importance.view(num_heads, headsize).sum(dim=1)  # [num_heads]

            # 选择要剪枝的heads（重要性最低的）- 修复：使用round而非int
            num_prune_heads = round(num_heads * sparsity)
            if num_prune_heads == 0:
                print(f"    Taylor {taylor_type}: Layer {layer_idx}, sparsity too low, no heads pruned")
                return torch.tensor([], dtype=torch.long, device=W.device)

            if num_prune_heads >= num_heads:
                print(f"    Taylor {taylor_type}: Layer {layer_idx}, sparsity too high, pruning {num_heads-1}/{num_heads} heads")
                num_prune_heads = num_heads - 1  # 至少保留1个head

            prune_head_indices = head_importance.argsort()[:num_prune_heads]

            # 转换为channel索引
            prune_indices = []
            for head_idx in prune_head_indices:
                prune_indices.extend(range(head_idx * headsize, (head_idx + 1) * headsize))
            prune_indices = torch.tensor(prune_indices, dtype=torch.long, device=W.device)

            # 详细日志
            actual_sparsity = num_prune_heads / num_heads
            print(f"    Taylor {taylor_type}: Layer {layer_idx}")
            print(f"      Requested sparsity: {sparsity:.3f} ({sparsity*num_heads:.1f} heads)")
            print(f"      Actual sparsity: {actual_sparsity:.3f} ({num_prune_heads}/{num_heads} heads)")
            print(f"      Head importance range: [{head_importance.min().item():.6f}, {head_importance.max().item():.6f}]")
            print(f"      Pruned heads: {prune_head_indices.tolist()}")
            print(f"      Kept heads: {sorted(set(range(num_heads)) - set(prune_head_indices.tolist()))}")

        else:
            # headsize=1, 直接按column剪枝
            num_prune = round(self.columns * sparsity)  # 修复：使用round而非int
            if num_prune == 0:
                return torch.tensor([], dtype=torch.long, device=W.device)

            prune_indices = importance.argsort()[:num_prune]
            print(f"    Taylor {taylor_type}: Layer {layer_idx}, pruned {num_prune}/{self.columns} columns")

        # 执行剪枝（零化）
        W[:, prune_indices] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        print(f"    Pruned columns {(self.layer.weight.sum(0) == 0).sum().item()}/{self.layer.weight.size(1)}")

        return prune_indices

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        # 释放Taylor相关内存
        self.grad_first = None
        self.grad_second = None
        torch.cuda.empty_cache()

    def struct_prune_with_indices(self, prune_columns, percdamp=0.0, headsize=1):
        """
        Prune exactly the specified input columns with SlimGPT global compensation.

        Args:
            prune_columns: 1D torch.LongTensor/list of column indices to remove (input features)
            percdamp: damping ratio
            headsize: kept for API symmetry; group-size >1 not supported here (expects 1)

        Returns:
            pruned_indices: torch.LongTensor of pruned columns (same as input set)
        """
        assert headsize == 1, "struct_prune_with_indices currently supports headsize=1 (linear columns)"

        dev = self.layer.weight.device
        # Prepare W in 2D [rows, columns]
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Prepare H
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=dev)
            H[diag, diag] += damp

        # Build reorder index: [pruned..., kept...]
        if not torch.is_tensor(prune_columns):
            prune_columns = torch.tensor(prune_columns, dtype=torch.long, device=dev)
        else:
            prune_columns = prune_columns.to(device=dev, dtype=torch.long)

        # unique and sorted
        prune_columns = torch.unique(prune_columns)
        cnt = prune_columns.numel()
        all_idx = torch.arange(self.columns, device=dev)
        keep_mask = torch.ones(self.columns, dtype=torch.bool, device=dev)
        keep_mask[prune_columns] = False
        keep_columns = all_idx[keep_mask]
        column_sort_idx = torch.cat([prune_columns, keep_columns], dim=0)

        # Compute Hinv and its cholesky factor in reordered basis
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
        Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)[:cnt]

        # Local elim + global update (same as iterative but in one block)
        W = W[:, column_sort_idx]
        W1 = W[:, :cnt].clone()
        Hinv1 = Hinv_chol[:, :cnt]
        Err1 = torch.zeros_like(W1)

        for i in range(cnt):
            Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
            if not self.no_compensate:
                W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

        # Zero pruned block
        W[:, :cnt] = 0
        if not self.no_compensate:
            end = self.columns
            # Use the cholesky factor (shape: cnt x columns) to match dimensions
            W[:, cnt:end] -= Err1.matmul(Hinv_chol[:, cnt:end])

        # Restore original column order
        inv_perm = torch.argsort(column_sort_idx)
        W = W[:, inv_perm]

        # Write back
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        return prune_columns

    def struct_prune_with_indices_iterative(self, prune_columns, percdamp=0.0, headsize=1, chunk_size=64):
        """
        Iteratively prune the specified input columns in chunks, updating H between chunks,
        following the same local+global compensation scheme as struct_prune.

        Args:
            prune_columns: 1D tensor/list of column indices (input features) to remove
            percdamp: damping ratio applied to H diagonal
            headsize: kept for API symmetry; only headsize=1 supported
            chunk_size: number of columns to prune per iteration (e.g., head_dim for one head)

        Returns:
            pruned_indices: torch.LongTensor of pruned columns
        """
        assert headsize == 1, "iterative indices pruning supports headsize=1 (linear columns)"

        dev = self.layer.weight.device
        # Prepare W in 2D [rows, columns]
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Prepare H (keep for iterative updates)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if not torch.is_tensor(prune_columns):
            prune_columns = torch.tensor(prune_columns, dtype=torch.long, device=dev)
        else:
            prune_columns = prune_columns.to(device=dev, dtype=torch.long)
        prune_columns = torch.unique(prune_columns)

        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=dev)
        pruned_columns = 0

        remaining = prune_columns.tolist()
        while remaining:
            # current chunk
            cnt = min(chunk_size, len(remaining))
            chunk = torch.tensor(remaining[:cnt], dtype=torch.long, device=dev)

            # Compute Hinv and reorder with chunk at front, pruned columns at tail
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            all_idx = torch.arange(self.columns, device=dev)
            pruned_tail = all_idx[column_mask]
            keep_mask = torch.ones(self.columns, dtype=torch.bool, device=dev)
            keep_mask[chunk] = False
            keep_mask[column_mask] = False
            keep_alive = all_idx[keep_mask]
            column_sort_idx = torch.cat([chunk, keep_alive, pruned_tail], dim=0)

            Hinv = Hinv[column_sort_idx, :][:, column_sort_idx]
            Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)[:cnt]

            # Local elim + global update for this chunk
            W = W[:, column_sort_idx]
            W1 = W[:, :cnt].clone()
            Hinv1 = Hinv_chol[:, :cnt]
            Err1 = torch.zeros_like(W1)
            for i in range(cnt):
                Err1[:, i:i+1] = W1[:, i:i+1] / Hinv1[i, i]
                if not self.no_compensate:
                    W1[:, i:] -= Err1[:, i:i+1].matmul(Hinv1[i:i+1, i:])

            W[:, :cnt] = 0
            if not self.no_compensate:
                end = self.columns - pruned_columns
                W[:, cnt:end] -= Err1.matmul(Hinv_chol[:, cnt:end])

            # Restore order
            inv_perm = torch.argsort(column_sort_idx)
            W = W[:, inv_perm]

            # Update H mask to simulate removal
            pruned_idx = chunk
            H[pruned_idx, :] = 0
            H[:, pruned_idx] = 0
            H[pruned_idx, pruned_idx] = 1
            column_mask[pruned_idx] = 1
            pruned_columns += cnt
            remaining = remaining[cnt:]

        # Write back
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        pruned_indices = column_mask.nonzero(as_tuple=True)[0]
        return pruned_indices

    def magnitude_prune(self, sparsity, percdamp, headsize, layer_idx):
        """
        Magnitude-based pruning: prune channels/heads with smallest weight magnitude.

        Args:
            sparsity: target pruning ratio
            percdamp: not used (kept for API consistency)
            headsize: if > 1, prune by heads; if == 1, prune by columns
            layer_idx: not used (kept for API consistency)

        Returns:
            prune_col_idx: torch.LongTensor of pruned column indices
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if headsize > 1:
            num_heads = W.shape[1] // headsize
            assert W.shape[1] % headsize == 0, "Column count must be divisible by headsize"
            # Calculate number of heads to prune
            target_heads = round(num_heads * sparsity)
            # Calculate head scores (sum of absolute values per head)
            head_scores = W.abs().reshape(W.shape[0], num_heads, headsize).sum(dim=(0, 2))  # [num_heads]
            prune_head_idx = torch.argsort(head_scores)[:target_heads]  # heads to prune
            # Get column indices for all pruned heads
            prune_col_idx = []
            for h in prune_head_idx:
                prune_col_idx.extend(range(h * headsize, (h + 1) * headsize))
            prune_col_idx = torch.tensor(prune_col_idx, device=W.device)
        else:
            # headsize=1, prune by columns directly
            num_prune = round(W.shape[1] * sparsity)
            col_scores = W.abs().sum(dim=0)
            prune_col_idx = torch.argsort(col_scores)[:num_prune]

        # Prune (zero out)
        W[:, prune_col_idx] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)), flush=True)
        return prune_col_idx

    def taylor_prune(self, sparsity, percdamp, headsize, layer_idx):
        """
        Taylor-based pruning: prune channels/heads with smallest |W * grad| score.

        Args:
            sparsity: target pruning ratio
            percdamp: not used (kept for API consistency)
            headsize: if > 1, prune by heads; if == 1, prune by columns
            layer_idx: not used (kept for API consistency)

        Returns:
            prune_col_idx: torch.LongTensor of pruned column indices
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Need gradients
        if self.layer.weight.grad is None:
            raise RuntimeError("Taylor pruning requires gradients from backward pass")
        grad = self.layer.weight.grad.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()
        grad = grad.float()

        if headsize > 1:
            num_heads = W.shape[1] // headsize
            assert W.shape[1] % headsize == 0, "Column count must be divisible by headsize"
            target_heads = round(num_heads * sparsity)
            # Calculate Taylor scores per head
            taylor_scores = (W * grad).abs().reshape(W.shape[0], num_heads, headsize).sum(dim=(0, 2))  # [num_heads]
            prune_head_idx = torch.argsort(taylor_scores)[:target_heads]
            prune_col_idx = []
            for h in prune_head_idx:
                prune_col_idx.extend(range(h * headsize, (h + 1) * headsize))
            prune_col_idx = torch.tensor(prune_col_idx, device=W.device)
        else:
            num_prune = round(W.shape[1] * sparsity)
            taylor_scores = (W * grad).abs().sum(dim=0)
            prune_col_idx = torch.argsort(taylor_scores)[:num_prune]

        # Prune (zero out)
        W[:, prune_col_idx] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        print('pruned columns %d/%d' % ((self.layer.weight.sum(0) == 0).sum().item(), self.layer.weight.size(1)), flush=True)
        return prune_col_idx
