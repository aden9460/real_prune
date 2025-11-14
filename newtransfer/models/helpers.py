import torch
from torch import nn as nn
from torch.nn import functional as F
import time

# def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
#     B, l, V = logits_BlV.shape
#     if top_k > 0:
#         idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
#         logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
#     if top_p > 0:
#         sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
#         sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
#         sorted_idx_to_remove[..., -1:] = False
#         logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
#     # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
#     replacement = num_samples >= 0
#     num_samples = abs(num_samples)
#     return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = (sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)).float()
        sorted_idx_to_remove[..., -1:] = 0.0
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove).bool(), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


# def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:
#     t0=time.time()
#     B, l, V = logits_BlV.shape
#     if top_k > 0:
#         idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
#         logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
#     if top_p > 0:
#         sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
#         probs = sorted_logits.softmax(dim=-1)
#         cum_probs = torch.cumsum(probs, dim=-1)  # 使用非 in-place 版本，避免潜在问题
#         sorted_idx_to_remove = (cum_probs <= (1 - top_p)).float()  # 转换为 fp32 (0.0 或 1.0)，支持 MIL dtype
#         sorted_idx_to_remove[..., -1:] = 0.0  # in-place 赋值，现在是 fp32，支持
#         # scatter 生成 unsorted 掩码（fp32），然后 cast 到 bool 用于 masked_fill_
#         unsorted_remove = sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove)
#         logits_BlV.masked_fill_(unsorted_remove.bool(), -torch.inf)  # cast 回 bool，仅用于掩码
#     replacement = num_samples >= 0
#     num_samples = abs(num_samples)
#     t1=time.time()
#     print(f"modified result is {t1-t0}")
#     return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

# import torch
# import time

# def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:
#     t0 = time.time()
#     B, l, V = logits_BlV.shape
#     if top_k > 0:
#         idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
#         logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
#     if top_p > 0:
#         sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
#         probs = sorted_logits.softmax(dim=-1)
#         probs = torch.cumsum(probs, dim=-1)
#         sorted_idx_to_remove = (probs <= (1 - top_p))  # 直接bool比较，无需float转换
#         sorted_idx_to_remove[..., -1:] = False  # in-place bool赋值
#         # scatter 生成 unsorted 掩码（bool），直接用于masked_fill_
#         unsorted_remove = sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove)
#         logits_BlV.masked_fill_(unsorted_remove, -torch.inf)  # 直接用bool掩码
#     replacement = num_samples >= 0
#     num_samples = abs(num_samples)
#     t1 = time.time()
#     print(f"new modified result is {t1 - t0}")
#     return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


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
