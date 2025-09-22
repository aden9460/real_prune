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
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # ==== 新增：H 来源控制（xtx/fisher）====
        self.H_mode = getattr(args, "H_mode", "xtx")  # 'xtx' 或 'fisher'

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.args = args
        self.no_compensate = args.no_compensate

        # ==== 新增：一阶梯度缓存（woodtaylor 打分会用；fisher H 也需要 grad^2）====
        self.G = torch.zeros((self.rows, self.columns), device=self.dev)
        self.grad_acc_steps = 0

    def add_batch(self, inp, out):
        """
        统计 H 的入口：
          - xtx: 累计输入协方差 X^T X（与原代码一致）
          - fisher: 这里跳过（由 cache_grad 在反传后用 grad^2 填 H 的对角）
        """
        if self.H_mode != "xtx":
            return  # fisher 模式下，此处不做 X^T X，以节省显存

        # ====== 原始 SlimGPT 的 X^T X 逻辑 ======
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # [hsize, seqlen]
        self.H *= self.nsamples / (self.nsamples + tmp)  # 缩放
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float() # 缩放
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def cache_grad(self):
        """
        在外部做过一次 backward 后调用：
          - fisher: 用 grad^2 沿输出维度求和，填入 H 的对角（对角 Fisher）
          - 同时缓存一阶梯度 G（woodtaylor 打分用）
        """
        g = self.layer.weight.grad
        if g is None:
            return
        if isinstance(self.layer, transformers.Conv1D):
            g = g.t()
        g = g.float()

        # 累加一阶梯度（供 woodtaylor 使用）
        self.G += g
        self.grad_acc_steps += 1

        if self.H_mode == "fisher":
            # 将 grad^2 沿着行维（输出通道）求和，得到每一列（输入通道）的 Fisher 对角
            g2_col = (g ** 2).sum(dim=0)  # [columns]
            diag_idx = torch.arange(self.columns, device=self.dev)
            self.H[diag_idx, diag_idx] += g2_col

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
            blocksize = 1

        while pruned_columns < target_columns:     
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            if headsize > 1:
                Hinv_diag = torch.stack([Hinv[i:i+headsize, i:i+headsize] for i in range(0, self.columns, headsize)])
                Hinv_diag = torch.diagonal(torch.linalg.cholesky(Hinv_diag), dim1=-2, dim2=-1).reshape(-1)
                Hinv_diag = Hinv_diag ** 2
            else:
                Hinv_diag = Hinv.diag()

            # ====== 原始 SlimGPT 的纯二阶“误差”打分 ======
            error = torch.sum(W ** 2 / Hinv_diag.unsqueeze(0), dim=0)
            
            # 屏蔽已剪列
            error[column_mask] = torch.inf
            # 组排序
            if headsize > 1:
                head_sort_idx = error.view(-1, headsize).sum(1).argsort()
                column_sort_idx = torch.hstack([torch.arange(x * headsize, x * headsize + headsize, device=self.dev) for x in head_sort_idx])
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

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        pruned_indices = torch.where(column_mask)[0]
        return pruned_indices

    def free(self):
        self.H = None
        self.G = None
        torch.cuda.empty_cache()

class WoodTaylorSlim(SlimGPT):
    """
    仅改变列/组“排序打分”为 WoodTaylor (二阶+一阶)，
    其它 Cholesky/补偿/置零流程与 SlimGPT 保持一致，确保最小侵入。
    """
    @torch.no_grad()
    def struct_prune(self, sparsity, headsize=1, percdamp=0.0, layer_idx=None):
        assert self.columns % headsize == 0

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

        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
        pruned_columns = 0
        target_columns = round(self.columns // headsize * sparsity) * headsize

        # 预计算（用于一阶项）
        G = self.G / max(1, self.grad_acc_steps) if self.grad_acc_steps > 0 else torch.zeros_like(W)

        blocksize = 1
        while pruned_columns < target_columns:
            # Hinv 仅用于后续补偿（保持与你原代码一致）
            Hinv_full = torch.cholesky_inverse(torch.linalg.cholesky(H))

            # === WoodTaylor 列打分：second + first ===
            # second = 0.5 * H_jj * ||w_j||^2
            H_diag = torch.diag(H)
            second = 0.5 * (H_diag * (W * W).sum(dim=0))     # [columns]
            first  = (G * W).sum(dim=0)                       # [columns]
            col_score = second + first                        # 越小越先剪
            col_score[column_mask] = torch.inf               # 屏蔽已剪列

            if headsize > 1:
                group_score = col_score.view(-1, headsize).sum(1)
                head_sort_idx = torch.argsort(group_score)
                column_sort_idx = torch.hstack([
                    torch.arange(x * headsize, x * headsize + headsize, device=self.dev)
                    for x in head_sort_idx
                ])
                cnt = headsize
            else:
                column_sort_idx = torch.argsort(col_score)
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)

            # 置换到“最优列在前”
            W = W[:, column_sort_idx]
            Hinv = Hinv_full[column_sort_idx, :][:, column_sort_idx]
            # 先求 H 的逆，再做排序
            Hinv_full = torch.cholesky_inverse(torch.linalg.cholesky(H))
            Hinv_full = Hinv_full[column_sort_idx, :][:, column_sort_idx]

            # 1) 先对 H 做逆（用阻尼的 H 更稳）
            Hinv_full = torch.cholesky_inverse(torch.linalg.cholesky(H))
            # 2) 先重排，再做 Cholesky（上三角）
            Hinv_full = Hinv_full[column_sort_idx.to(self.dev), :][:, column_sort_idx.to(self.dev)]
            U = torch.linalg.cholesky(Hinv_full, upper=True)  # 形状 (C, C)

            # 3) 只取左上角块参与“局部补偿”
            W1 = W[:, :cnt].clone()            # (rows, cnt)
            U_block = U[:cnt, :cnt]            # (cnt,  cnt)  ← 局部更新只在这个块里
            Err1 = torch.zeros_like(W1)        # (rows, cnt)

            for i in range(cnt):
                # 标量对角
                diag = U_block[i, i]
                Err1[:, i:i+1] = W1[:, i:i+1] / diag
                if not self.no_compensate:
                    # 右侧也只用 U_block 的上三角切片 (1, cnt-i)
                    W1[:, i:] -= Err1[:, i:i+1].matmul(U_block[i:i+1, i:])

            # 4) 置零已剪列
            W[:, :cnt] = 0

            # 5) 全局补偿用“右上角块”（前 cnt 行、后半列）
            if not self.no_compensate:
                end = self.columns - pruned_columns
                U_top_right = U[:cnt, cnt:end]     # (cnt, end-cnt)
                W[:, cnt:end] -= Err1.matmul(U_top_right)  # (rows,cnt) @ (cnt,rest) -> (rows,rest)
            # 复位
            column_sort_idx_inv = torch.argsort(column_sort_idx)
            W = W[:, column_sort_idx_inv]

            # 标记 & 更新 H（保持稀疏图）
            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = H[:, pruned_idx] = 0
            H[pruned_idx, pruned_idx] = 1
            column_mask[pruned_idx] = 1
            pruned_columns += cnt

            blocksize = 1

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        pruned_indices = torch.where(column_mask)[0]
        return pruned_indices