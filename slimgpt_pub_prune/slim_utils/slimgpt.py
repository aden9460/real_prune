import math
import time
import os
import torch
import torch.nn as nn
import transformers

# import matplotlib.pyplot as plt

DEBUG = True 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import numpy as np
import matplotlib.pyplot as plt

def plot_3d(w,title,abs_flag=True):

    w = w.detach()
    w = w.cpu().numpy()
    rows, cols = w.shape
    X = np.arange(cols)
    Y = np.arange(rows)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    if abs_flag:
        ax1.plot_surface(X, Y, abs(w),  cmap='CMRmap_r')
    else:
        ax1.plot_surface(X, Y, w,  cmap='CMRmap_r')

    ax1.set_ylabel('Output Channel',fontsize=10)
    ax1.set_xlabel("Input Channel",fontsize=10)
    ax1.view_init(elev=60, azim=230)
    ax1.set_title(f"{title}",y=-0.3, fontsize=12)
    ax1.tick_params(axis='x', labelsize=8)  # 设置 x 轴刻度标签大小
    ax1.tick_params(axis='y', labelsize=8)  # 设置 y 轴刻度标签大小
    ax1.tick_params(axis='z', labelsize=8)  # 设置 z 轴刻度标签大小

    plt.savefig(f"{title}.png",dpi=300)

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
        # if DEBUG:
        # self.inp1 = inp
        # self.out1 = out   
        # plot_3d(inp[0],"inps1")             
        # inp: batch,seq,hidden
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # [hsize, seqlen]
        self.H *= self.nsamples / (self.nsamples + tmp)  #缩放
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float() #缩放
        self.H += inp.matmul(inp.t())
    
    def add_batch_v7(self, inp, out):
        """
        逐分辨率独立统计 + 加权/平均：
        - 缓存最近10次的 inp
        - 满10次时：H_local = Σ_s gamma_s * ( (X_s @ X_s^T) / seqlen_s )
        （注意：没有跨分辨率的交叉项）
        - 然后用原来的 EMA + sqrt(2/nsamples) 规则把 H_local 融入 self.H
        """

        # --------- 懒初始化缓存/参数（只改这个方法就能用） ---------
        if not hasattr(self, "_var_cache"):
            self._var_cache = []              # 存最近10次的 inp（未改形状）
        if not hasattr(self, "_var_cache_limit"):
            self._var_cache_limit = 10        # 每10次触发一次融合
        if not hasattr(self, "_var_equalize_seq"):
            self._var_equalize_seq = True     # 是否按 seqlen 做平均（分辨率同权）
        if not hasattr(self, "_var_gamma"):   # 每次flush的权重；None表示均匀
            self._var_gamma = None

        # --------- 先把本次输入放进缓存，然后看是否满足触发 ---------
        self._var_cache.append(inp.detach())

        if len(self._var_cache) < self._var_cache_limit:
            return  # 不足10次先不更新 H

        # ---------- 满10次：构建 H_local（分辨率独立统计 + 加权/平均） ----------
        xs = self._var_cache
        self._var_cache = []  # 清空缓存，准备下一个窗口

        device = self.H.device
        dtype  = torch.float32
        S = len(xs)

        # 权重 γ_s：默认均匀同权；如需自定义，可在外部设置 self._var_gamma 为长度为S的list/tuple
        if self._var_gamma is None:
            gammas = torch.full((S,), 1.0 / S, device=device, dtype=dtype)
        else:
            g = torch.as_tensor(self._var_gamma, device=device, dtype=dtype)
            gammas = g / (g.sum() + 1e-12)

        H_local = torch.zeros_like(self.H, dtype=dtype, device=device)

        # 为了保持与原逻辑一致，nsamples使用“批大小”的累计
        tmp_total = 0

        for s, x in enumerate(xs):
            # 统计批大小（与原逻辑一致：2D视作B=1）
            tmp_s = 1 if x.dim() == 2 else x.shape[0]
            tmp_total += tmp_s

            x = x.to(device=device, dtype=dtype)

            # 与你原始逻辑等价的形状处理，得到 X_s: [hsize, seqlen]
            if x.dim() == 2:
                x = x.unsqueeze(0)  # -> [1, L, h]
            if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])  # [B*L, h]
                Xs = x.t()  # [h, seqlen]
            else:
                # 兜底：把最后一维当 hsize，其余展平到 seq
                if x.dim() >= 2:
                    x = x.reshape(-1, x.shape[-1])  # [N, h]
                    Xs = x.t()                      # [h, N]
                else:
                    raise ValueError("Unexpected input rank for VAR Hessian builder.")

            seqlen = max(Xs.shape[1], 1)
            # 分辨率同权：按 seqlen 做平均；若不想平均，把下一行改为 norm = 1.0
            norm = (1.0 / seqlen) if self._var_equalize_seq else 1.0

            # 本分辨率二阶项（无交叉项）
            H_local = H_local + gammas[s] * norm * (Xs @ Xs.t())

        # ---------- 按你原式子的 EMA + sqrt(2/nsamples) 融合进 self.H ----------
        # 原式：
        # self.H *= self.nsamples / (self.nsamples + tmp)
        # self.nsamples += tmp
        # inp = sqrt(2 / self.nsamples) * inp
        # self.H += inp @ inp.t()
        #
        # 现在把 inp@inp^T 换成 H_local，并配套 (2 / nsamples) 的缩放
        self.H *= self.nsamples / (self.nsamples + tmp_total)
        self.nsamples += tmp_total
        scale = 2.0 / self.nsamples
        self.H += scale * H_local

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
        # print(torch.linalg.matrix_rank(self.H))
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
            blocksize = 1
        #     blocksize = (target_columns - 512) // 2
        
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
                blocksize = 1
                # blocksize = (blocksize - 512) // 2

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        
        # print('time %.2f' % (time.time() - tick), flush=True)
        # count = (W == 0).sum().item()
        # total_params = W.numel()
        # print(float(count) / total_params)
        # out_gap = torch.mean((self.layer(self.inp1) - self.out1) ** 2).item()
        # out = torch.mean(self.out1 ** 2).item()
        # print('output_gap:', out_gap, flush=True)
        # print('output:', out, flush=True)
        # print('output_gap / output:', out_gap / out, flush=True)

        pruned_indices = torch.where(column_mask)[0]

        return pruned_indices
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

    def magnitude_prune(self, sparsity, percdamp,headsize,layer_idx):
        """
        按权重绝对值剪枝，headsize=64时以head为单位剪除，headsize=1时直接按列剪除
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if headsize > 1:
            num_heads = W.shape[1] // headsize
            assert W.shape[1] % headsize == 0, "列数必须能被headsize整除"
            # 计算要剪掉的head数量
            target_heads = round(num_heads * sparsity)
            # 计算每个head的分数
            head_scores = W.abs().reshape(W.shape[0], num_heads, headsize).sum(dim=(0,2))  # [num_heads]
            prune_head_idx = torch.argsort(head_scores)[:target_heads]  # 要剪掉的head编号
            # 得到要剪掉的所有列的索引
            prune_col_idx = []
            for h in prune_head_idx:
                prune_col_idx.extend(range(h*headsize, (h+1)*headsize))
            prune_col_idx = torch.tensor(prune_col_idx, device=W.device)
        else:
            # headsize=1，直接按列剪
            num_prune = round(W.shape[1] * sparsity)
            col_scores = W.abs().sum(dim=0)
            prune_col_idx = torch.argsort(col_scores)[:num_prune]

        # 剪枝
        W[:, prune_col_idx] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        return prune_col_idx

    def taylor_prune(self, sparsity, percdamp, headsize, layer_idx):
        """
        按一阶Taylor分数剪枝，headsize=64时以head为单位剪除，headsize=1时直接按列剪除
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # 需要有梯度
        if self.layer.weight.grad is None:
            raise RuntimeError("Taylor剪枝需要先反向传播获得梯度")
        grad = self.layer.weight.grad.clone()
        if isinstance(self.layer, nn.Conv2d):
            grad = grad.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            grad = grad.t()
        grad = grad.float()

        if headsize > 1:
            num_heads = W.shape[1] // headsize
            assert W.shape[1] % headsize == 0, "列数必须能被headsize整除"
            target_heads = round(num_heads * sparsity)
            # 计算每个head的taylor分数
            taylor_scores = (W * grad).abs().reshape(W.shape[0], num_heads, headsize).sum(dim=(0,2))  # [num_heads]
            prune_head_idx = torch.argsort(taylor_scores)[:target_heads]
            prune_col_idx = []
            for h in prune_head_idx:
                prune_col_idx.extend(range(h*headsize, (h+1)*headsize))
            prune_col_idx = torch.tensor(prune_col_idx, device=W.device)
        else:
            num_prune = round(W.shape[1] * sparsity)
            taylor_scores = (W * grad).abs().sum(dim=0)
            prune_col_idx = torch.argsort(taylor_scores)[:num_prune]

        # 剪枝
        W[:, prune_col_idx] = 0
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        return prune_col_idx