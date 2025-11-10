# VAR模型推理流程深入分析报告

## 1. autoregressive_infer_cfg 方法分析

### 1.1 方法签名与位置

**文件路径**: `/home/wangzefang/Project/AR/VAR_real/models/var.py`
**行号**: 127-190

```python
@torch.no_grad()
def autoregressive_infer_cfg(
    self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
    more_smooth=False,
) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
```

### 1.2 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| B | int | 批大小(batch size) |
| label_B | Optional[Union[int, torch.LongTensor]] | ImageNet标签，若为None则随机采样 |
| g_seed | Optional[int] | 随机种子 |
| cfg | float | 分类器自由指导(Classifier-Free Guidance)的强度系数，默认1.5 |
| top_k | int | Top-K采样的K值 |
| top_p | float | Top-P核采样的概率阈值 |
| more_smooth | bool | 使用Gumbel Softmax进行平滑，仅用于可视化，不用于FID/IS评估 |

### 1.3 关键观察

**重要发现**: 该方法**不接受patch_num或单尺度参数**。推理始终遍历所有预设的patch_nums。

---

## 2. 多尺度处理机制详解

### 2.1 patch_nums定义

**在 `__init__` 中**（第27行）:

```python
patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
```

这表示10个逐步扩大的分辨率尺度：
- 尺度0: 1×1 = 1个token
- 尺度1: 2×2 = 4个token
- 尺度2: 3×3 = 9个token
- ...
- 尺度9: 16×16 = 256个token

### 2.2 核心数据结构初始化（__init__, 第39-48行）

```python
self.patch_nums: Tuple[int] = patch_nums
self.L = sum(pn ** 2 for pn in self.patch_nums)        # 总token数 = 1+4+9+...+256 = 1496
self.first_l = self.patch_nums[0] ** 2                 # 首尺度token数 = 1
self.begin_ends = []
cur = 0
for i, pn in enumerate(self.patch_nums):
    self.begin_ends.append((cur, cur+pn ** 2))
    cur += pn ** 2
```

`begin_ends`列表示例：
```
[
  (0, 1),      # 尺度0: 位置0-1
  (1, 5),      # 尺度1: 位置1-5
  (5, 14),     # 尺度2: 位置5-14
  ...
  (1240, 1496) # 尺度9: 位置1240-1496
]
```

### 2.3 多尺度循环的核心实现（第160-187行）

```python
for si, pn in enumerate(self.patch_nums):   # si: i-th segment
    ratio = si / self.num_stages_minus_1    # 当前进度比 [0, 1]
    cur_L += pn*pn
    
    cond_BD_or_gss = self.shared_ada_lin(cond_BD)  # 条件编码
    x = next_token_map                              # 当前输入
    
    # 关键：所有30层Transformer blocks在每个尺度下执行一遍
    for b in self.blocks:
        x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
    
    logits_BlV = self.get_logits(x, cond_BD)
    
    # CFG指导
    t = cfg * ratio
    logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
    
    # 采样
    idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
    
    # VAE量化到embedding
    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # (B, l, Cvae)
    
    # 重塑为(B, Cvae, pn, pn)
    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
    
    # 获取下一尺度输入
    f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
        si, len(self.patch_nums), f_hat, h_BChw
    )
    
    if si != self.num_stages_minus_1:   # 非最后阶段
        next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
        next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
        next_token_map = next_token_map.repeat(2, 1, 1)   # 因为CFG要double batch size
```

---

## 3. Transformer调用流程分析

### 3.1 Transformer块的属性结构

每个block是`AdaLNSelfAttn`的实例（来自`basic_var.py`第128-162行）:

```python
class AdaLNSelfAttn(nn.Module):
    def __init__(self, block_idx, last_drop_p, embed_dim, cond_dim, ...):
        super().__init__()
        self.attn = SelfAttention(...)                    # Attention模块
        self.ffn = FFN(...)                                # Feed-Forward网络
        self.ln_wo_grad = norm_layer(embed_dim, ...)       # LayerNorm
        self.drop_path = DropPath(...) if drop_path > 0 else nn.Identity()
        
        if self.shared_aln:
            self.ada_gss = nn.Parameter(...)              # 共享的AdaLN参数
        else:
            self.ada_lin = nn.Sequential(nn.SiLU(...), nn.Linear(...))
```

### 3.2 每个尺度下Transformer的执行方式

**关键事实**：在每个尺度循环中，所有30层都会被执行：

```python
for b in self.blocks:  # self.blocks 包含所有30个AdaLNSelfAttn块
    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
```

**执行次数计算**：
- 尺度数: 10
- 每个尺度执行的层数: depth (通常16-30)
- **总Transformer块执行次数**: 10 × depth

### 3.3 每个块的Forward流程

来自`basic_var.py`第152-159行：

```python
def forward(self, x, cond_BD, attn_bias):
    if self.shared_aln:
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
    else:
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
    
    # Self-Attention + Residual
    x = x + self.drop_path(
        self.attn(
            self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), 
            attn_bias=attn_bias
        ).mul_(gamma1)
    )
    
    # FFN + Residual
    x = x + self.drop_path(
        self.ffn(
            self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ).mul(gamma2)
    )
    return x
```

### 3.4 Attention模块详解

`SelfAttention`类（`basic_var.py`第58-125行）:

```python
class SelfAttention(nn.Module):
    def __init__(self, block_idx, embed_dim=768, num_heads=12, ...):
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)       # 输出投影
        self.proj_drop = nn.Dropout(...)
        
        self.caching = False                               # KV缓存标志
        self.cached_k = None
        self.cached_v = None
```

**关键特性**：
1. **KV缓存在推理时启用**（第159行）：`for b in self.blocks: b.attn.kv_caching(True)`
2. **无attention_bias**（推理时）：因为启用了KV缓存，attn_bias参数为None

### 3.5 FFN模块详解

`FFN`类（`basic_var.py`第33-55行）:

```python
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ...):
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(...)
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
```

---

## 4. 激活形状与数据流追踪

### 4.1 初始化阶段

```
batch_size = B
cond_BD: (2*B, D)           # D = embed_dim = 1024
sos (start-of-sequence): (2*B, D)

next_token_map: (2*B, first_l, D)  # (2*B, 1, 1024)
               = sos.unsqueeze(1) + pos_start + lvl_pos
```

### 4.2 每个尺度的数据流

```
尺度 si (pn = patch_nums[si]):

输入:
  x: (2*B, cur_L, D)
     其中cur_L = sum(patch_nums[0:si+1]^2)

Transformer循环:
  for layer_idx in range(depth):
    x = block[layer_idx](x, cond_BD, attn_bias=None)
    # x形状保持: (2*B, cur_L, D)

输出logits:
  logits_BlV: (2*B, cur_L, V)
             其中V = vocab_size = 4096

采样:
  idx_Bl: (2*B, cur_L)  ->  (2*B, pn*pn) 提取当前尺度的索引

VAE embedding:
  h_BChw: (B, Cvae, pn, pn)  
         其中Cvae = 32

下一尺度准备:
  if si != 9:
    next_token_map: (2*B, pn_next*pn_next, D)
```

### 4.3 关键形状变化示例（depth=16）

```
阶段0 (pn=1):  x: (2B, 1, 1024)   -> 16层处理 -> logits: (2B, 1, 4096)
阶段1 (pn=2):  x: (2B, 5, 1024)   -> 16层处理 -> logits: (2B, 5, 4096)
阶段2 (pn=3):  x: (2B, 14, 1024)  -> 16层处理 -> logits: (2B, 14, 4096)
...
阶段9 (pn=16): x: (2B, 1496, 1024) -> 16层处理 -> logits: (2B, 1496, 4096)
```

---

## 5. Transformer组件属性名确认

### 5.1 块内属性名称

```python
# 在每个 block (AdaLNSelfAttn) 中:

block.attn          # SelfAttention模块
  ├── .mat_qkv      # Q,K,V投影 (Linear)
  ├── .proj          # 输出投影 (Linear)
  ├── .proj_drop     # 输出Dropout
  ├── .q_bias        # Q的bias
  ├── .v_bias        # V的bias
  ├── .cached_k      # KV缓存 (推理用)
  └── .cached_v

block.ffn           # FFN模块
  ├── .fc1           # 第一个线性层 (in_features -> hidden_features)
  ├── .act           # GELU激活
  ├── .fc2           # 第二个线性层 (hidden_features -> out_features)
  ├── .drop          # Dropout
  └── .fused_mlp_func # 融合MLP函数（如果可用）

block.ln_wo_grad    # 无梯度的LayerNorm
block.drop_path     # DropPath或Identity
block.attn.using_flash  # Flash Attention标志
block.ffn.fused_mlp_func  # 融合MLP标志
```

### 5.2 验证脚本片段

```python
# 从var.py第99行的打印可以看到：
print(
    f'[constructor] ==== flash_if_available={flash_if_available} '
    f'({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), '
    f'fused_if_available={fused_if_available} '
    f'(fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, '
    f'fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth})'
)
```

---

## 6. 多尺度循环的完整执行流程图

```
┌─────────────────────────────────────────────────────┐
│        autoregressive_infer_cfg 开始               │
│   创建条件向量 cond_BD (B, D)                       │
│   启用所有blocks的KV缓存                            │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────┐
    │ 对于每个尺度 si (0到9)：                 │
    │   pn = patch_nums[si]                   │
    │   cur_L += pn*pn                        │
    └────────────┬────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 1. 条件编码：cond_BD_or_gss = shared_ada_lin(cond_BD)
    │                                                 │
    │ 2. 获取当前输入：x = next_token_map (2B, L, D) │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 3. Transformer处理（核心）：                     │
    │                                                 │
    │   for layer_idx in range(depth):  # 16或更多  │
    │     block = self.blocks[layer_idx]             │
    │                                                 │
    │     ① 自注意力 (带AdaLN):                      │
    │        norm_x = ln_wo_grad(x)                  │
    │        norm_x = norm_x * (scale1 + 1) + shift1│
    │        attn_out = block.attn(norm_x)           │
    │        x = x + drop_path(attn_out * gamma1)    │
    │                                                 │
    │     ② FFN (带AdaLN):                           │
    │        norm_x = ln_wo_grad(x)                  │
    │        norm_x = norm_x * (scale2 + 1) + shift2│
    │        ffn_out = fc2(GELU(fc1(norm_x)))        │
    │        x = x + drop_path(ffn_out * gamma2)     │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 4. 获取logits：logits_BlV = head(head_nm(x))    │
    │    形状：(2B, L, V) 其中V=4096                  │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 5. CFG指导：                                   │
    │    t = cfg * (si / num_stages_minus_1)         │
    │    logits = (1+t)*logits[:B] - t*logits[B:]    │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 6. 采样当前尺度tokens：                         │
    │    idx_Bl = sample_with_top_k_top_p_(logits)   │
    │    形状：(B, pn*pn)                            │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 7. VAE映射：                                   │
    │    h_BChw = vae.embedding(idx_Bl)              │
    │    形状：(B, pn*pn, Cvae) -> (B, Cvae, pn, pn)│
    │                                                 │
    │    f_hat, next_token_map = vae.                │
    │        get_next_autoregressive_input(          │
    │            si, 10, f_hat, h_BChw               │
    │        )                                        │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 8. 是否最后尺度？                               │
    │    if si != 9:                                 │
    │        next_token_map = embed + position      │
    │        next_token_map = next_token_map.repeat(2, 1, 1)  # CFG双batch
    │    else:                                       │
    │        break                                   │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │ 9. 返回最终图像：                               │
    │    img = vae.fhat_to_img(f_hat)                │
    │    img = img.add_(1).mul_(0.5)  # [-1,1] -> [0,1]
    └─────────────────────────────────────────────────┘
```

---

## 7. VectorQuantizer2 的 get_next_autoregressive_input

**位置**: `/home/wangzefang/Project/AR/VAR_real/models/quant.py`（第186-196行）

```python
def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    HW = self.v_patch_nums[-1]  # 最大尺度 = 16
    
    if si != SN-1:  # 非最后尺度 (si = 0到8)
        # 当前尺度的特征进行上采样到最大分辨率，经过残差模块
        h = self.quant_resi[si/(SN-1)](
            F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')
        )
        f_hat.add_(h)  # 累积
        
        # 下采样到下一尺度的分辨率
        return f_hat, F.interpolate(
            f_hat, 
            size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), 
            mode='area'
        )
    else:  # 最后尺度 (si = 9)
        h = self.quant_resi[si/(SN-1)](h_BChw)
        f_hat.add_(h)
        return f_hat, f_hat
```

**关键作用**:
1. 当前尺度的embedding上采样到最大分辨率(16×16)
2. 通过残差模块处理
3. 累积到f_hat
4. 为下一尺度下采样到合适分辨率

---

## 8. 完整调用链总结

```
demo.py:
  var.autoregressive_infer_cfg(B=1, label_B=0, cfg=4, top_k=900, top_p=0.95)
    │
    ├─ 初始化：
    │  ├─ 标签embedding: class_emb(label_B) -> (2B, D)
    │  ├─ 位置embedding + level embedding
    │  ├─ 启用KV缓存: for b in blocks: b.attn.kv_caching(True)
    │  └─ 初始f_hat: zeros(B, Cvae, 16, 16)
    │
    ├─ 多尺度循环 (si=0到9):
    │  ├─ 条件处理: shared_ada_lin(cond_BD)
    │  │
    │  ├─ Transformer前向 (核心):
    │  │  └─ for block in self.blocks:  # 深度16-30
    │  │     ├─ 自注意力(含KV缓存)
    │  │     ├─ FFN
    │  │     └─ x形状保持(2B, cur_L, D)
    │  │
    │  ├─ 输出头: get_logits -> (2B, cur_L, V)
    │  ├─ CFG应用: logits调整
    │  ├─ 采样: top-k/top-p -> (B, pn*pn)
    │  ├─ VAE映射: embedding(idx) -> (B, Cvae, pn, pn)
    │  └─ 多尺度融合: get_next_autoregressive_input
    │
    ├─ 禁用KV缓存: for b in blocks: b.attn.kv_caching(False)
    │
    └─ 返回图像:
       vae.fhat_to_img(f_hat) -> (B, 3, 256, 256) in [0, 1]
```

---

## 9. 重要发现总结

### 9.1 所有30层都在每个尺度执行
- **不是**第1-10层处理尺度1，11-20层处理尺度2的方式
- **而是**所有30层在尺度1上执行一遍，再在尺度2上执行一遍

### 9.2 Attention属性清单
✓ `attn.mat_qkv` - Q,K,V投影  
✓ `attn.proj` - 输出投影  
✓ `attn.q_bias`, `attn.v_bias` - 偏置  
✓ `attn.cached_k`, `attn.cached_v` - KV缓存  
✓ `attn.using_flash` - Flash Attention标志  

### 9.3 FFN属性清单
✓ `ffn.fc1` - 第一层  
✓ `ffn.fc2` - 第二层  
✓ `ffn.act` - GELU激活  
✓ `ffn.drop` - Dropout  
✓ `ffn.fused_mlp_func` - 融合MLP函数  

### 9.4 方法的单尺度局限性
- `autoregressive_infer_cfg` **不支持**单尺度推理
- 若需单尺度，需要：
  1. 修改方法添加`target_patch_num`参数
  2. 在循环中根据条件break/continue
  3. 调整输出处理逻辑

### 9.5 性能特征
- **KV缓存**: 推理时使用，减少内存和计算
- **CFG强度随进度衰减**: `t = cfg * (si / num_stages_minus_1)`
- **批大小翻倍**: CFG需要保存无条件分支，batch从B变为2B

---

## 10. 代码位置索引

| 概念 | 文件 | 行号 |
|-----|------|------|
| autoregressive_infer_cfg | models/var.py | 127-190 |
| AdaLNSelfAttn | models/basic_var.py | 128-162 |
| SelfAttention | models/basic_var.py | 58-125 |
| FFN | models/basic_var.py | 33-55 |
| get_next_autoregressive_input | models/quant.py | 186-196 |
| 初始化(patch_nums处理) | models/var.py | 39-48 |
| 块列表 | models/var.py | 85-94 |
| KV缓存管理 | models/basic_var.py | 87-109 |

