import math
import time
import torch
import torch.nn as nn
import transformers
import numpy as np
# import matplotlib.pyplot as plt  # 如需可视化再打开

DEBUG = True

# 关闭 TF32 以避免数值差异（按需打开）
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _cuda_empty_cache_safe():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===== 可选：3D 可视化，调试用 =====
def plot_3d(w, title, abs_flag=True):
    import matplotlib.pyplot as plt
    w = w.detach().cpu().numpy()
    rows, cols = w.shape
    X = np.arange(cols)
    Y = np.arange(rows)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    if abs_flag:
        ax1.plot_surface(X, Y, np.abs(w), cmap='CMRmap_r')
    else:
        ax1.plot_surface(X, Y, w, cmap='CMRmap_r')
    ax1.set_ylabel('Output Channel', fontsize=10)
    ax1.set_xlabel("Input Channel", fontsize=10)
    ax1.view_init(elev=60, azim=230)
    ax1.set_title(f"{title}", y=-0.3, fontsize=12)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.tick_params(axis='z', labelsize=8)
    plt.savefig(f"{title}.png", dpi=300)


class SlimGPT(object):
    """
    结构化列/组剪枝器（OBS 二阶补偿版本）
    - H_mode = 'xtx'：用输入协方差 X^T X（需要前向收集）
    - H_mode = 'fisher'：用 grad^2 构造对角 Fisher（需要反传后 cache_grad 或外部 set_H_diag）
    """
    def __init__(self, layer, layer_idx, args):
        self.layer = layer
        self.layer_idx = layer_idx
        self.dev = self.layer.weight.device

        # 统一成 [rows, columns]
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # ==== H 的来源 ====
        self.H_mode = getattr(args, "H_mode", "xtx")  # 'xtx' 或 'fisher'
        # XTX: 全矩阵；Fisher: 仅用对角时也存成对角矩阵，便于统一补偿流程
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

        self.args = args
        self.no_compensate = getattr(args, "no_compensate", False)

        # 一阶梯度累积（WoodTaylor 打分用；Fisher 统计也会用到 grad^2）
        self.G = torch.zeros((self.rows, self.columns), device=self.dev, dtype=torch.float32)
        self.grad_acc_steps = 0

    # ---------- H 的三种设置路径 ----------
    def add_batch(self, inp, out):
        """
        前向统计入口（仅 xtx 模式）：
          H += X^T X 的在线估计；Fisher 模式下此处直接返回（不做无用统计）
        """
        if self.H_mode != "xtx":
            return  # fisher 模式下，此处不做 X^T X

        # 保证维度
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        # 线性/Conv1D 的输入最后一维是 in_features
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))  # [B*T, C]
            inp = inp.t().float()  # [C, B*T]

        # 在线缩放更新
        self.H.mul_(self.nsamples / (self.nsamples + tmp))
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        # X^T X 累加
        self.H.add_(inp @ inp.t())

    @torch.no_grad()
    def cache_grad(self):
        """
        反传之后调用：
          - Fisher: 用 grad^2 沿行求和，形成对角 Fisher，叠加到 H 的对角
          - 同时把一阶梯度累加到 G（WoodTaylor 打分用）
        """
        g = self.layer.weight.grad
        if g is None:
            return
        if isinstance(self.layer, transformers.Conv1D):
            g = g.t()
        g = g.float()

        # 一阶累加（取平均时用 grad_acc_steps）
        self.G.add_(g)
        self.grad_acc_steps += 1

        if self.H_mode == "fisher":
            # grad^2 沿输出维（行）求和 -> 每一列（输入通道）的 Fisher 对角
            g2_col = (g ** 2).sum(dim=0)  # [columns]
            diag_idx = torch.arange(self.columns, device=self.dev)
            self.H[diag_idx, diag_idx] += g2_col

    @torch.no_grad()
    def set_H_diag(self, H_diag: torch.Tensor):
        """
        从外部直接注入对角 H（常用于在循环外把 grad^2 拼好后统一设置）
        """
        H_diag = H_diag.to(self.dev).flatten()
        assert H_diag.numel() == self.columns, "H_diag 尺寸必须等于列数（输入通道数）"
        self.H.zero_()
        idx = torch.arange(self.columns, device=self.dev)
        self.H[idx, idx] = H_diag

    # ---------- 核心：结构化剪枝 ----------
    @torch.no_grad()
    def struct_prune(self, sparsity, headsize=1, percdamp=0.0, layer_idx=None):
        """
        OBS 二阶补偿的结构化列/组剪枝
          - 按列或按组（headsize>1）进行选择与补偿
          - H 可以是全矩阵（xtx）或对角矩阵（fisher）
        返回：被剪掉的列索引（1D LongTensor）
        """
        assert self.columns % headsize == 0, "列数必须能被 headsize 整除（组剪）"

        tick = time.time()

        # 权重到 [rows, columns]
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H  # [C, C]

        # 防止奇异：对角为 0 的列置 1，且权重置 0（不可用通道）
        diagH = torch.diag(H)
        dead = (diagH == 0)
        if dead.any():
            H[dead, dead] = 1.0
            W[:, dead] = 0.0

        # 阻尼（percdamp * 平均对角）
        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        # 目标剪枝列数（向 headsize 对齐）
        target_columns = round(self.columns // headsize * float(sparsity)) * headsize
        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)  # True 表示已剪
        pruned_columns = 0

        # 分批剪（每次最多剪 64/1024 列，避免一次太大）
        blocksize = 1
        while pruned_columns < target_columns:
            # 先求 H^{-1}，再做排序用的上三角 U（cholesky of Hinv）
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            # 纯二阶 OBS 的列打分：error_j = ||w_j||^2 / (Hinv_jj)
            Hinv_diag = torch.diag(Hinv)  # [C]
            error = (W * W).sum(dim=0) / Hinv_diag.clamp_min(1e-12)  # [C]
            error[column_mask] = torch.inf  # 屏蔽已剪列

            # 组/列排序
            if headsize > 1:
                # 组分数 = 组内列分数之和（越小越先剪）
                group_score = error.view(-1, headsize).sum(1)
                head_sort_idx = torch.argsort(group_score)  # [num_groups]
                column_sort_idx = torch.hstack([
                    torch.arange(g * headsize, g * headsize + headsize, device=self.dev)
                    for g in head_sort_idx
                ])
                cnt = headsize
            else:
                column_sort_idx = torch.argsort(error)  # 升序
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)

            # 统一置换到“待剪列在前”
            W = W[:, column_sort_idx]
            Hinv_sorted = Hinv[column_sort_idx, :][:, column_sort_idx]

            # 上三角 U（Hinv 的 Cholesky），只取前 cnt 行用于局部更新
            U = torch.linalg.cholesky(Hinv_sorted, upper=True)[:cnt, :]  # [cnt, C]
            W1 = W[:, :cnt].clone()                 # 待剪块
            U_block = U[:, :cnt]                    # [cnt, cnt]
            Err1 = torch.zeros_like(W1)             # 局部误差/补偿项

            # 局部补偿（逐列）
            for i in range(cnt):
                diag_scalar = U_block[i, i]
                Err1[:, i:i+1] = W1[:, i:i+1] / diag_scalar
                if not self.no_compensate:
                    # 只在右侧上三角内传播
                    W1[:, i:] -= Err1[:, i:i+1].matmul(U_block[i:i+1, i:])

            # 剪掉目标列（置零）
            W[:, :cnt] = 0.0

            # 全局补偿（传播到未剪列）
            if not self.no_compensate:
                end = self.columns - pruned_columns
                U_top_right = U[:, cnt:end]  # [cnt, rest]
                if U_top_right.numel() > 0:
                    W[:, cnt:end] -= Err1.matmul(U_top_right)

            # 逆置换还原列顺序
            inv_idx = torch.argsort(column_sort_idx)
            W = W[:, inv_idx]

            # 更新 H：把已剪列“屏蔽”（对角=1，其它=0），便于下轮迭代
            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = 0.0
            H[:, pruned_idx] = 0.0
            H[pruned_idx, pruned_idx] = 1.0
            column_mask[pruned_idx] = True
            pruned_columns += cnt

            blocksize = 1  # 可保留为动态策略

        # 回写权重形状
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(f"[SlimGPT] layer {self.layer_idx} pruned {pruned_columns}/{self.columns} cols "
                  f"in {time.time() - tick:.3f}s")

        pruned_indices = torch.where(column_mask)[0]
        return pruned_indices

    def free(self):
        self.H = None
        self.G = None
        _cuda_empty_cache_safe()


class WoodTaylorSlim(SlimGPT):
    """
    WoodTaylor 版本：
      - 仅改变“列/组排序打分”为： second + first
        second = 0.5 * H_jj * ||w_j||^2
        first  = <g_j, w_j>  （g、w 逐元素乘积后沿行求和）
      - 补偿/置零/传播流程与 SlimGPT 保持一致（使用 Hinv 的 Cholesky 上三角 U）
      - 如需额外一阶修正步（W[:,keep] -= scale*(G H^{-1})），可在剪枝循环后按需加一小步
    """
    @torch.no_grad()
    def struct_prune(self, sparsity, headsize=1, percdamp=0.0, layer_idx=None):
        assert self.columns % headsize == 0, "列数必须能被 headsize 整除（组剪）"

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        H = self.H

        # 奇异/不可用通道处理
        diagH = torch.diag(H)
        dead = (diagH == 0)
        if dead.any():
            H[dead, dead] = 1.0
            W[:, dead] = 0.0

        # 阻尼（稳定数值）
        if percdamp > 0:
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(H.size(0), device=self.dev)
            H[diag, diag] += damp

        # 一阶项（用平均梯度，避免 batch 偏置）
        G_mean = self.G / max(1, self.grad_acc_steps) if self.grad_acc_steps > 0 else torch.zeros_like(W)

        target_columns = round(self.columns // headsize * float(sparsity)) * headsize
        column_mask = torch.zeros(self.columns, dtype=torch.bool, device=self.dev)
        pruned_columns = 0

        blocksize = 1
        while pruned_columns < target_columns:
            # H^{-1} & 其上三角 U
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

            # WoodTaylor 排序打分
            H_diag = torch.diag(H)  # 注意这里用 H 的对角（二阶项）
            second = 0.5 * H_diag * (W * W).sum(dim=0)  # [C]
            first = (G_mean * W).sum(dim=0)             # [C]
            score = second + first                      # 越小越优先剪
            score[column_mask] = torch.inf

            # 组/列排序
            if headsize > 1:
                group_score = score.view(-1, headsize).sum(1)
                head_sort_idx = torch.argsort(group_score)
                column_sort_idx = torch.hstack([
                    torch.arange(g * headsize, g * headsize + headsize, device=self.dev)
                    for g in head_sort_idx
                ])
                cnt = headsize
            else:
                column_sort_idx = torch.argsort(score)
                cnt = min(target_columns - pruned_columns, max(blocksize, 64), 1024)

            # 置换
            W = W[:, column_sort_idx]
            Hinv_sorted = Hinv[column_sort_idx, :][:, column_sort_idx]

            # U：Hinv 的上三角 Cholesky
            U = torch.linalg.cholesky(Hinv_sorted, upper=True)[:cnt, :]
            W1 = W[:, :cnt].clone()
            U_block = U[:, :cnt]
            Err1 = torch.zeros_like(W1)

            # 局部补偿
            for i in range(cnt):
                diag_scalar = U_block[i, i]
                Err1[:, i:i+1] = W1[:, i:i+1] / diag_scalar
                if not self.no_compensate:
                    W1[:, i:] -= Err1[:, i:i+1].matmul(U_block[i:i+1, i:])

            # 置零待剪列
            W[:, :cnt] = 0.0

            # 全局补偿
            if not self.no_compensate:
                end = self.columns - pruned_columns
                U_top_right = U[:, cnt:end]
                if U_top_right.numel() > 0:
                    W[:, cnt:end] -= Err1.matmul(U_top_right)

            # 逆置换
            inv_idx = torch.argsort(column_sort_idx)
            W = W[:, inv_idx]

            # 更新 H 屏蔽掉已剪列
            pruned_idx = column_sort_idx[:cnt]
            H[pruned_idx, :] = 0.0
            H[:, pruned_idx] = 0.0
            H[pruned_idx, pruned_idx] = 1.0
            column_mask[pruned_idx] = True
            pruned_columns += cnt

            blocksize = 1

        # （可选）额外一阶修正步：W[:, keep] -= scale*(G_mean * Hinv) 的列方向投影
        # scale = getattr(self.args, "scale_update", 0.0)
        # if scale > 0:
        #     keep_mask = ~column_mask
        #     Hinv_full = torch.cholesky_inverse(torch.linalg.cholesky(self.H + 1e-6 * torch.eye(self.columns, device=self.dev)))
        #     # 仅对保留列做一阶修正（简单示意，可按需改进）
        #     delta = (G_mean @ Hinv_full)[:, keep_mask]
        #     W[:, keep_mask] -= scale * delta

        # 回写
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(f"[WoodTaylor] layer {self.layer_idx} pruned {pruned_columns}/{self.columns} cols")

        pruned_indices = torch.where(column_mask)[0]
        return pruned_indices