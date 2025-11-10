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

    def head_dim_prune(self, sparsity, headsize=64, percdamp=0.0, layer_idx=None):
        """
        Head维度剪枝：对每个head独立评估，删除相同数量的维度

        与struct_prune的区别：
        - struct_prune: 删除完整的head（12 heads × 64 dim → 9 heads × 64 dim）
        - head_dim_prune: 减少每个head的维度（12 heads × 64 dim → 12 heads × 48 dim）

        优势：
        - 每个head独立评估，保留各自最重要的维度（个性化）
        - 需要Global Update来补偿所有其他列（包括其他head）
        - 保持多头结构（reshape兼容）

        Args:
            sparsity: 稀疏度（例如0.25表示每个head删除25%维度）
            headsize: 每个head的维度（默认64）
            percdamp: Hessian对角线阻尼系数
            layer_idx: 层索引（用于日志）

        Returns:
            pruned_indices: 被删除的列索引（全局索引）
        """
        assert self.columns % headsize == 0, \
            f"columns ({self.columns}) must be divisible by headsize ({headsize})"

        num_heads = self.columns // headsize
        dims_to_remove_per_head = round(headsize * sparsity)

        if dims_to_remove_per_head == 0:
            print(f"Warning: sparsity too low, no dimensions to remove. sparsity={sparsity}, headsize={headsize}")
            return torch.tensor([], dtype=torch.long, device=self.dev)

        tick = time.time()

        # 准备权重和Hessian
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        del self.H

        # 处理死节点
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # 添加阻尼
        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        # 用于记录所有被删除的维度（全局索引）
        all_pruned_indices = []

        # 逐个head处理
        for head_idx in range(num_heads):
            start_col = head_idx * headsize
            end_col = start_col + headsize

            # 1. 每次重新计算完整Hinv（因为H在变化）
            try:
                Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            except RuntimeError as e:
                print(f"Warning: Cholesky failed for complete H at head {head_idx}: {e}")
                continue

            # 2. 提取head子矩阵，计算head内误差
            W_head = W[:, start_col:end_col]
            H_head = H[start_col:end_col, start_col:end_col]
            Hinv_head = Hinv[start_col:end_col, start_col:end_col]

            try:
                Hinv_diag_head = torch.diagonal(torch.linalg.cholesky(Hinv_head)) ** 2
            except RuntimeError:
                print(f"Warning: Cholesky failed for head {head_idx}, skipping")
                continue

            error_head = torch.sum(W_head ** 2 / Hinv_diag_head.unsqueeze(0), dim=0)

            # 3. 选择要删除的维度（head内相对索引）
            dim_sort_idx = error_head.argsort()
            dims_to_remove = dim_sort_idx[:dims_to_remove_per_head]
            keep_dims = dim_sort_idx[dims_to_remove_per_head:]

            # 转换为全局索引
            global_pruned_idx = start_col + dims_to_remove
            all_pruned_indices.append(global_pruned_idx)

            # 4. 重排：将待删除维度移到head最前面
            reorder_idx_head = torch.cat([dims_to_remove, keep_dims])

            # 全局重排索引
            reorder_idx_global = torch.arange(self.columns, device=self.dev)
            reorder_idx_global[start_col:end_col] = start_col + reorder_idx_head

            # 重排整个W和Hinv
            W = W[:, reorder_idx_global]
            Hinv_reordered = Hinv[reorder_idx_global, :][:, reorder_idx_global]

            # 5. 对重排后的矩阵做Cholesky分解（上三角）
            try:
                Hinv_chol = torch.linalg.cholesky(Hinv_reordered, upper=True)[
                    start_col:start_col+dims_to_remove_per_head
                ]
            except RuntimeError:
                print(f"Warning: Cholesky failed for head {head_idx} reordered, skipping compensation")
                # 直接清零，不做补偿
                W[:, start_col:start_col+dims_to_remove_per_head] = 0
                # 恢复原始顺序
                reorder_idx_inv = torch.argsort(reorder_idx_global)
                W = W[:, reorder_idx_inv]
                # 更新H
                for idx in global_pruned_idx:
                    H[idx, :] = H[:, idx] = 0
                    H[idx, idx] = 1
                continue

            # 6. Local Update（在head内）
            if not self.no_compensate:
                W1 = W[:, start_col:start_col+dims_to_remove_per_head].clone()
                Hinv1 = Hinv_chol[:, start_col:end_col]

                for i in range(dims_to_remove_per_head):
                    col_i = start_col + i
                    Err_i = W1[:, i:i+1] / Hinv_chol[i, col_i]
                    W1[:, i:] -= Err_i.matmul(Hinv1[i:i+1, i:])

                # 7. Global Update（补偿所有其他列）
                W[:, start_col:start_col+dims_to_remove_per_head] = 0
                W[:, start_col+dims_to_remove_per_head:] -= W1.matmul(
                    Hinv_chol[:, start_col+dims_to_remove_per_head:]
                )
            else:
                # 不补偿，直接清零
                W[:, start_col:start_col+dims_to_remove_per_head] = 0

            # 8. 恢复原始顺序
            reorder_idx_inv = torch.argsort(reorder_idx_global)
            W = W[:, reorder_idx_inv]

            # 9. 更新H（清零已删除维度）
            for idx in global_pruned_idx:
                H[idx, :] = H[:, idx] = 0
                H[idx, idx] = 1

        # 写回
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if layer_idx is not None:
            print(f'Layer {layer_idx}: head_dim_prune completed in {time.time() - tick:.2f}s, '
                  f'removed {len(all_pruned_indices) * dims_to_remove_per_head} dims total '
                  f'({dims_to_remove_per_head} dims per head)', flush=True)

        return torch.cat(all_pruned_indices) if all_pruned_indices else torch.tensor([], dtype=torch.long, device=self.dev)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
