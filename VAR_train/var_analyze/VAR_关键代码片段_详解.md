# VAR模型推理流程 - 关键代码片段详解

## A. autoregressive_infer_cfg 完整代码（models/var.py 127-190）

```python
@torch.no_grad()
def autoregressive_infer_cfg(
    self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
    more_smooth=False,
) -> torch.Tensor:
    """
    仅用于推理，自回归模式
    :param B: batch大小
    :param label_B: imagenet标签；若为None，随机采样
    :param g_seed: 随机种子
    :param cfg: 分类器自由指导比例
    :param top_k: top-k采样
    :param top_p: top-p采样
    :param more_smooth: 使用gumbel softmax平滑；仅用于可视化
    :return: 重建图像 (B, 3, H, W) in [0, 1]
    """
    # ========== 第1部分：初始化 ==========
    if g_seed is None: rng = None
    else: self.rng.manual_seed(g_seed); rng = self.rng
    
    # 处理标签
    if label_B is None:
        label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
    
    # 标签到embedding：(B,) -> (2*B, D)
    # 前B是有条件，后B是无条件（用于CFG）
    sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
    
    # 位置和级别embedding
    lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
    
    # 初始token map：(2*B, first_l, D) = (2*B, 1, 1024)
    next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
    
    # VAE特征图：累积特征
    cur_L = 0
    f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
    
    # ========== 第2部分：启用KV缓存 ==========
    for b in self.blocks: b.attn.kv_caching(True)
    
    # ========== 第3部分：多尺度自回归生成循环 ==========
    for si, pn in enumerate(self.patch_nums):   # si: i-th segment (0到9)
        ratio = si / self.num_stages_minus_1     # CFG强度调度：[0, 1]
        # last_L = cur_L
        cur_L += pn*pn                           # 累积当前位置
        
        # ①条件编码
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map                        # 当前输入 (2*B, cur_L, D)
        
        # ②Transformer前向：所有深度层都执行一遍
        AdaLNSelfAttn.forward  # 这是一个占位符，实际上下面执行
        for b in self.blocks:  # self.blocks 有 16/20/24/30 个块
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        
        # ③获取logits
        logits_BlV = self.get_logits(x, cond_BD)  # (2*B, cur_L, 4096)
        
        # ④CFG应用
        t = cfg * ratio
        logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]  # (B, cur_L, 4096)
        
        # ⑤采样token
        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]  # (B, cur_L)
        
        # ⑥VAE映射
        if not more_smooth:  # 默认路径
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # (B, cur_L, Cvae=32)
        else:   # 平滑路径（仅可视化）
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
        
        # ⑦重塑为图像形式
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)  # (B, 32, pn, pn)
        
        # ⑧多尺度融合与下一尺度准备
        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
        
        if si != self.num_stages_minus_1:   # 非最后阶段
            # 准备下一尺度的输入
            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)  # (B, L_next, Cvae)
            next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]  # 加embedding和position
            next_token_map = next_token_map.repeat(2, 1, 1)   # CFG翻倍：(B, L_next, D) -> (2*B, L_next, D)
    
    # ========== 第4部分：禁用缓存并返回 ==========
    for b in self.blocks: b.attn.kv_caching(False)
    return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # [-1, 1] -> [0, 1]
```

---

## B. AdaLNSelfAttn 块结构与Forward（models/basic_var.py 128-162）

### B.1 类定义与初始化

```python
class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        
        # 残差连接的路径衰减
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 自注意力模块（包含KV缓存）
        self.attn = SelfAttention(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, 
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, 
            flash_if_available=flash_if_available
        )
        
        # FFN模块
        self.ffn = FFN(
            in_features=embed_dim, 
            hidden_features=round(embed_dim * mlp_ratio), 
            drop=drop, 
            fused_if_available=fused_if_available
        )
        
        # 归一化（无学习参数）
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        
        # 自适应层归一化参数
        self.shared_aln = shared_aln
        if self.shared_aln:
            # 共享方式：统一的ada参数
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            # 非共享方式：每个块学习自己的ada参数
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
```

### B.2 Forward方法

```python
def forward(self, x, cond_BD, attn_bias):
    """
    :param x: 输入特征 (B, L, C)
    :param cond_BD: 条件向量 (B, D)
    :param attn_bias: 注意力偏置，推理时为None（使用KV缓存）
    :return: 输出特征 (B, L, C)
    """
    # 获取AdaLN参数：6个，分别用于2个前归一化
    # (gamma1, gamma2, scale1, scale2, shift1, shift2)
    if self.shared_aln:
        # 共享参数：从ada_gss和条件向量组合
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        # cond_BD: (B, 1, 6, C) 来自 shared_ada_lin
        # unbind(2) 在6维度解包得到6个 (B, 1, C) 张量
    else:
        # 非共享参数：从 ada_lin 学习
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        # cond_BD: (B, D) -> ada_lin -> (B, 6*C) -> view(B, 1, 6, C) -> unbind(2)
    
    # ========== 自注意力分支 ==========
    # 前归一化 + 自适应参数缩放
    x = x + self.drop_path(
        self.attn(
            self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),  # AdaLN: LN后缩放平移
            attn_bias=attn_bias  # 推理时为None
        ).mul_(gamma1)  # 输出缩放（残差系数）
    )
    
    # ========== FFN分支 ==========
    # 前归一化 + 自适应参数缩放
    x = x + self.drop_path(
        self.ffn(
            self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)  # AdaLN
        ).mul(gamma2)  # 输出缩放
    )
    
    return x
```

---

## C. SelfAttention 详解（models/basic_var.py 58-125）

### C.1 初始化

```python
class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        # 缩放因子
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)  # 默认：1/sqrt(head_dim)
        
        # Q,K,V投影：合并为一个大的线性层后分割
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)  # 无偏置在Q,K,V
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))               # Q偏置
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))               # V偏置
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))      # K无偏置
        
        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        
        # 加速选项
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # ========== KV缓存（仅推理时使用） ==========
        self.caching, self.cached_k, self.cached_v = False, None, None
```

### C.2 KV缓存控制与Forward

```python
def kv_caching(self, enable: bool):
    """启用/禁用KV缓存"""
    self.caching, self.cached_k, self.cached_v = enable, None, None

def forward(self, x, attn_bias):
    """
    :param x: 输入 (B, L, C)
    :param attn_bias: 注意力掩码，推理时为None
    :return: 输出 (B, L, C)
    """
    B, L, C = x.shape
    
    # ========== 第1步：计算Q,K,V ==========
    # 使用自定义偏置：Q和V有偏置，K无偏置
    qkv = F.linear(
        input=x, 
        weight=self.mat_qkv.weight, 
        bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
    ).view(B, L, 3, self.num_heads, self.head_dim)
    # qkv: (B, L, 3, H, c) 其中 c = head_dim = C / num_heads
    
    main_type = qkv.dtype
    
    # ========== 第2步：选择注意力实现 ==========
    using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
    
    if using_flash or self.using_xform:
        q, k, v = qkv.unbind(dim=2)  # 分别为 (B, L, H, c)
        dim_cat = 1  # 在L维度连接缓存
    else:
        # 标准实现：转换为 (B, H, L, c)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        dim_cat = 2  # 在L维度连接缓存
    
    # ========== 第3步：L2归一化（可选） ==========
    if self.attn_l2_norm:
        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        if using_flash or self.using_xform:
            scale_mul = scale_mul.transpose(1, 2)  # 1H11 -> 11H1
        q = F.normalize(q, dim=-1).mul(scale_mul)
        k = F.normalize(k, dim=-1)
    
    # ========== 第4步：KV缓存处理 ==========
    if self.caching:
        if self.cached_k is None:
            # 第一次：初始化缓存
            self.cached_k = k
            self.cached_v = v
        else:
            # 后续：拼接新的K,V到缓存
            k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
            v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
    
    # ========== 第5步：计算注意力 ==========
    dropout_p = self.attn_drop if self.training else 0.0
    
    if using_flash:
        # Flash Attention：高效实现
        oup = flash_attn_func(
            q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
            dropout_p=dropout_p, softmax_scale=self.scale
        ).view(B, L, C)
    elif self.using_xform:
        # xFormers 内存高效注意力
        oup = memory_efficient_attention(
            q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
            attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), 
            p=dropout_p, scale=self.scale
        ).view(B, L, C)
    else:
        # 标准实现
        oup = slow_attn(
            query=q, key=k, value=v, 
            scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p
        ).transpose(1, 2).reshape(B, L, C)
    
    # ========== 第6步：输出投影 ==========
    return self.proj_drop(self.proj(oup))
```

---

## D. FFN 实现（models/basic_var.py 33-55）

```python
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        """
        标准 FFN: Linear -> GELU -> Dropout -> Linear
        """
        if self.fused_mlp_func is not None:
            # 使用融合版本（如果可用）
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, 
                bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, 
                return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            # 标准实现
            return self.drop(self.fc2(self.act(self.fc1(x))))
```

---

## E. get_next_autoregressive_input（models/quant.py 186-196）

```python
def get_next_autoregressive_input(
    self, si: int, SN: int, 
    f_hat: torch.Tensor, h_BChw: torch.Tensor
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    多尺度自回归推理中的特征融合与下采样
    
    :param si: 当前尺度索引 (0-9)
    :param SN: 总尺度数 (10)
    :param f_hat: 累积特征图 (B, Cvae, 16, 16)
    :param h_BChw: 当前尺度的embedding (B, Cvae, pn, pn)
    :return: (更新后的f_hat, 下一尺度的输入)
    """
    HW = self.v_patch_nums[-1]  # 最大尺度 = 16
    
    if si != SN-1:  # 非最后尺度
        # 将当前尺度特征上采样到最大分辨率
        h = self.quant_resi[si/(SN-1)](
            F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')
        )
        
        # 累积到特征图
        f_hat.add_(h)
        
        # 为下一尺度准备：下采样到下一尺度的分辨率
        return f_hat, F.interpolate(
            f_hat, 
            size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), 
            mode='area'
        )
    else:  # 最后尺度
        h = self.quant_resi[si/(SN-1)](h_BChw)
        f_hat.add_(h)
        return f_hat, f_hat  # 返回最终特征，不需要进一步处理
```

---

## F. 关键数据结构初始化（models/var.py 27-48）

```python
def __init__(
    self, vae_local: VQVAE,
    num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
    norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
    attn_l2_norm=False,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 stages
    flash_if_available=True, fused_if_available=True,
):
    super().__init__()
    
    # ========== 基本超参数 ==========
    assert embed_dim % num_heads == 0
    self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
    self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
    
    self.cond_drop_rate = cond_drop_rate
    self.prog_si = -1   # 渐进式训练（目前未使用）
    
    # ========== 多尺度配置 ==========
    self.patch_nums: Tuple[int] = patch_nums  # (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    self.L = sum(pn ** 2 for pn in self.patch_nums)  # 总token数：1496
    self.first_l = self.patch_nums[0] ** 2  # 第一个尺度token数：1
    
    # 每个尺度的位置范围
    self.begin_ends = []
    cur = 0
    for i, pn in enumerate(self.patch_nums):
        self.begin_ends.append((cur, cur+pn ** 2))
        cur += pn ** 2
    # begin_ends = [(0,1), (1,5), (5,14), (14,30), (30,55), (55,91), 
    #               (91,155), (155,255), (255,424), (424,680), (680,1496)]
    
    self.num_stages_minus_1 = len(self.patch_nums) - 1  # 9
    
    # ========== 1. 输入embedding ==========
    quant: VectorQuantizer2 = vae_local.quantize
    self.vae_proxy: Tuple[VQVAE] = (vae_local,)
    self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
    self.word_embed = nn.Linear(self.Cvae, self.C)  # VAE特征到token embedding
    
    # ========== 2. 标签embedding ==========
    init_std = math.sqrt(1 / self.C / 3)
    self.num_classes = num_classes
    self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
    self.class_emb = nn.Embedding(self.num_classes + 1, self.C)  # +1用于无条件
    nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
    
    # ========== 3. 位置embedding ==========
    self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))  # 第一个尺度的位置
    nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
    
    # 所有尺度的位置embedding
    pos_1LC = []
    for i, pn in enumerate(self.patch_nums):
        pe = torch.empty(1, pn*pn, self.C)
        nn.init.trunc_normal_(pe, mean=0, std=init_std)
        pos_1LC.append(pe)
    pos_1LC = torch.cat(pos_1LC, dim=1)     # (1, L, C)
    assert tuple(pos_1LC.shape) == (1, self.L, self.C)
    self.pos_1LC = nn.Parameter(pos_1LC)
    
    # 级别embedding（用于区分不同尺度）
    self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
    nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
    
    # ========== 4. Transformer块 ==========
    self.shared_ada_lin = nn.Sequential(
        nn.SiLU(inplace=False), 
        SharedAdaLin(self.D, 6*self.C)
    ) if shared_aln else nn.Identity()
    
    norm_layer = partial(nn.LayerNorm, eps=norm_eps)
    self.drop_path_rate = drop_path_rate
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    
    self.blocks = nn.ModuleList([
        AdaLNSelfAttn(
            cond_dim=self.D, shared_aln=shared_aln,
            block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, 
            num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], 
            last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available, 
            fused_if_available=fused_if_available,
        )
        for block_idx in range(depth)
    ])
    
    # ========== 5. 注意力掩码（用于训练） ==========
    # 推理时不使用，因为启用了KV缓存
    d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
    dT = d.transpose(1, 2)
    lvl_1L = dT[:, 0].contiguous()
    self.register_buffer('lvl_1L', lvl_1L)
    attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
    self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
    
    # ========== 6. 输出头 ==========
    self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
    self.head = nn.Linear(self.C, self.V)
```

---

## G. 属性访问验证代码

```python
# 访问Transformer块的属性
for layer_idx, block in enumerate(var.blocks):
    print(f"Layer {layer_idx}:")
    
    # Attention组件
    print(f"  attn.mat_qkv: {block.attn.mat_qkv}")           # QKV投影
    print(f"  attn.proj: {block.attn.proj}")                 # 输出投影
    print(f"  attn.q_bias.shape: {block.attn.q_bias.shape}") # (1024,)
    print(f"  attn.v_bias.shape: {block.attn.v_bias.shape}") # (1024,)
    print(f"  attn.cached_k: {block.attn.cached_k}")         # None或缓存
    print(f"  attn.cached_v: {block.attn.cached_v}")         # None或缓存
    print(f"  attn.using_flash: {block.attn.using_flash}")   # Flash Attention标志
    
    # FFN组件
    print(f"  ffn.fc1: {block.ffn.fc1}")                      # fc1
    print(f"  ffn.act: {block.ffn.act}")                      # GELU
    print(f"  ffn.fc2: {block.ffn.fc2}")                      # fc2
    print(f"  ffn.drop: {block.ffn.drop}")                    # Dropout
    print(f"  ffn.fused_mlp_func: {block.ffn.fused_mlp_func}") # 融合函数或None
    
    # AdaLN
    if hasattr(block, 'ada_lin'):
        print(f"  ada_lin: {block.ada_lin}")
    elif hasattr(block, 'ada_gss'):
        print(f"  ada_gss: {block.ada_gss}")
    
    # 其他
    print(f"  ln_wo_grad: {block.ln_wo_grad}")
    print(f"  drop_path: {block.drop_path}")
```

