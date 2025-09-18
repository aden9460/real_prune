import torch
from torch import nn as nn
from torch.nn import functional as F


# def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
#     B, l, V = logits_BlV.shape
#     if top_k > 0:
#         idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=True, dim=-1)[0].amin(dim=-1, keepdim=True)
#         logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
#     if top_p > 0:
#         sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
#         sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)
#         sorted_idx_to_remove[..., -1:] = False
#         logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
#     # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
#     replacement = num_samples >= 0
#     num_samples = abs(num_samples)
#     return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sample_with_top_k_top_p_(logits_BlV: torch.Tensor,
                             top_k: int = 0,
                             top_p: float = 0.0,
                             rng=None,               # 保留以兼容调用处，但不再传入 multinomial
                             num_samples: int = 1) -> torch.Tensor:
    """
    CoreML-safe 版本：
      - 无原地操作
      - 无 bool 目的张量写入
      - 不使用 generator=...
      - 支持同时 top-k 与 top-p
    返回: 采样到的 idx，形状 (B, l, num_samples)
    """
    B, l, V = logits_BlV.shape
    x = logits_BlV  # 不做原地修改；如需完全隔离可 x = logits_BlV.clone()
    NEG = x.new_tensor(-1e9)

    # ---- top-k（sorted=True，兼容 iOS15/16 的限制）----
    if top_k and top_k > 0:
        # k-th 最大值的阈值：[..., -1:] 是第 k 个最大值
        kth = x.topk(top_k, dim=-1, largest=True, sorted=True).values[..., -1:]
        mask_k = x < kth                                 # bool，仅作为读取用
        x = x + mask_k.to(x.dtype) * NEG                # 非原地屏蔽

    # ---- top-p（nucleus）----
    if top_p and top_p > 0.0:
        # 先按概率从大到小排序
        sorted_logits, sorted_idx = torch.sort(x, dim=-1, descending=True)
        probs_sorted = F.softmax(sorted_logits, dim=-1)
        cdf = probs_sorted.cumsum(dim=-1)               # 非原地 cumsum

        # 移除 mask：cdf > top_p 的位置需要被屏蔽
        remove = cdf > top_p                            # bool（读）
        # 右移一位，确保至少保留一个 token（避免 ~全屏蔽）
        remove = torch.cat(
            [torch.zeros_like(remove[..., :1], dtype=torch.bool), remove[..., :-1]],
            dim=-1
        )
        # 用数值方式屏蔽（避免 masked_fill_ / 布尔索引写）
        sorted_logits_filtered = sorted_logits + remove.to(sorted_logits.dtype) * NEG

        # 还原到原顺序：用 scatter（非原地）
        x = torch.empty_like(x).scatter(dim=-1, index=sorted_idx, src=sorted_logits_filtered)

    # ---- multinomial 采样（不传 generator=，避免 'generator' op）----
    probs = F.softmax(x, dim=-1)
    replacement = num_samples >= 0
    ns = abs(num_samples)
    idx = torch.multinomial(probs.view(-1, V), num_samples=ns, replacement=replacement) \
               .view(B, l, ns)
    
    return idx

def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'
