# Prune_v6深度技术分析

## 目录
1. [VAR逐尺度生成机制](#1-var逐尺度生成机制)
2. [Prune_v6的Token收集策略](#2-prune_v6的token收集策略)
3. [Scale_Mul数学原理深度解析](#3-scale_mul数学原理深度解析)
4. [Scale_Mul与剪枝率的关系](#4-scale_mul与剪枝率的关系)
5. [实验验证方案](#5-实验验证方案)

---

## 1. VAR逐尺度生成机制（✅ 增量生成 + KV Cache）

### 1.1 VAR的10个尺度结构

```python
# models/var.py Line 189-217
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # 10 scales

for si, pn in enumerate(self.patch_nums):  # si: scale index
    cur_L += pn * pn  # 累积位置索引（用于位置编码）

    # 生成当前尺度 - ⚠️ 关键：x只包含当前尺度的tokens！
    x = next_token_map  # 输入: 只有当前尺度的pn²个tokens
    for b in self.blocks:
        x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        # 内部通过KV Cache访问之前所有tokens

    logits_BlV = self.get_logits(x, cond_BD)
    idx_Bl = sample_with_top_k_top_p_(logits_BlV, ...)  # 采样当前尺度的tokens

    # 准备下一尺度的输入
    if si != self.num_stages_minus_1:
        # 从f_hat插值得到下一尺度大小的特征
        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(...)
        # next_token_map shape: (B, pn_next², C) ← 只包含下一尺度的tokens！
        next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + pn_next²]
```

**关键理解**:
- **每次循环生成一个尺度** (scale 0 → scale 1 → ... → scale 9)
- **每个尺度的token数量**:
  ```
  Scale 0: 1²  = 1    token
  Scale 1: 2²  = 4    tokens
  Scale 2: 3²  = 9    tokens
  Scale 3: 4²  = 16   tokens
  Scale 4: 5²  = 25   tokens
  Scale 5: 6²  = 36   tokens
  Scale 6: 8²  = 64   tokens
  Scale 7: 10² = 100  tokens
  Scale 8: 13² = 169  tokens
  Scale 9: 16² = 256  tokens
  总计: 680 tokens
  ```

- **增量生成 + KV Cache机制**:
  - 每次forward只输入**当前尺度的pn²个tokens**（增量）
  - 通过**KV Cache**，attention可以看到之前所有尺度的tokens
  - prune_v6.py中开启了KV cache: `b.attn.kv_caching(True)`

### 1.2 Token生成的时序（✅ 正确理解）

```
时刻0: forward(scale 0的1个token)
  → Input:  (B, 1, C)
  → KV Cache: K=(B, 1, C), V=(B, 1, C)
  → Output: 生成scale 1的logits

时刻1: forward(scale 1的4个tokens)
  → Input:  (B, 4, C)  ← 只输入新的4个tokens
  → KV Cache: K=(B, 5, C), V=(B, 5, C)  ← 累积1+4=5
  → Output: 生成scale 2的logits

时刻2: forward(scale 2的9个tokens)
  → Input:  (B, 9, C)  ← 只输入新的9个tokens
  → KV Cache: K=(B, 14, C), V=(B, 14, C)  ← 累积1+4+9=14
  → Output: 生成scale 3的logits

...

时刻9: forward(scale 9的256个tokens)
  → Input:  (B, 256, C)  ← 只输入新的256个tokens
  → KV Cache: K=(B, 680, C), V=(B, 680, C)  ← 累积所有680
  → Output: 最终输出
```

**每次forward调用**:
- 输入: `next_token_map` (只包含**当前尺度**的pn²个tokens)
- KV Cache: 累积存储之前所有尺度的K、V
- 输出: 当前尺度的tokens经过transformer后的表示

---

## 2. Prune_v6的Token收集策略（✅ 实现正确）

### 2.1 代码流程分析

```python
# prune_v6.py Line 459-487
_cache_dict = {}

def add_batch(name):
    def func(_, inp, out):
        if name not in _cache_dict:
            _cache_dict[name] = []

        # 每次forward调用都会触发这个hook
        _cache_dict[name].append((inp[0].detach(), out.detach()))

        # 攒够10次（即10个尺度）才拼接
        if len(_cache_dict[name]) >= 10:
            inps = [p[0] for p in _cache_dict[name]]
            outs = [p[1] for p in _cache_dict[name]]
            inp_cat = torch.cat(inps, dim=1)  # ✅ 拼接增量tokens
            out_cat = torch.cat(outs, dim=1)
            pruner_dict[name].add_batch(inp_cat, out_cat)
            _cache_dict[name] = []
    return func

# Line 483-486
for b in model.blocks: b.attn.kv_caching(True)  # ✅ 关键：开启KV Cache
for batch_idx, batch in enumerate(dataloader):
    model(batch)  # 触发autoregressive生成，10次forward
for b in model.blocks: b.attn.kv_caching(False)
```

### 2.2 正确理解：增量生成机制

**具体过程**（以batch=label_0为例）:

```
model(label_0) 触发autoregressive forward:

Iteration 1 (scale 0):
  - Input to blocks: (1, 1, C)  ← 只有scale 0的1个token
  - Hook captures: inp[0].shape = (1, 1, C)
  - Cache: [inp0] = [(1, 1, C)]

Iteration 2 (scale 1):
  - Input to blocks: (1, 4, C)  ← 只有scale 1的4个tokens（增量）
  - Hook captures: inp[0].shape = (1, 4, C)
  - Cache: [inp0, inp1] = [(1, 1, C), (1, 4, C)]

Iteration 3 (scale 2):
  - Input to blocks: (1, 9, C)  ← 只有scale 2的9个tokens（增量）
  - Hook captures: inp[0].shape = (1, 9, C)
  - Cache: [inp0, inp1, inp2] = [(1, 1, C), (1, 4, C), (1, 9, C)]

...

Iteration 10 (scale 9):
  - Input to blocks: (1, 256, C)  ← 只有scale 9的256个tokens（增量）
  - Hook captures: inp[0].shape = (1, 256, C)
  - Cache: [inp0, ..., inp9] = [(1,1,C), (1,4,C), ..., (1,256,C)]

  # ✅ 拼接所有增量tokens
  inp_cat = torch.cat([inp0, inp1, ..., inp9], dim=1)
  # inp_cat.shape = (1, 1+4+9+16+25+36+64+100+169+256, C) = (1, 680, C) ✓
```

### 2.3 Hook捕获的内容验证

**关键证据** - var.py Line 215:
```python
next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
#                                                            ↑ 只切片下一尺度的位置编码
```

这说明`next_token_map`的长度 = `self.patch_nums[si+1] ** 2`，即**只包含下一尺度的tokens**！

**每次Hook捕获的序列**:

| 迭代 | si | pn | inp[0].shape | 内容 | 累积拼接长度 |
|------|----|----|-------------|------|------------|
| 0 | 0 | 1 | (1, 1, C) | scale 0的1个token | 1 |
| 1 | 1 | 2 | (1, 4, C) | scale 1的4个tokens | 1+4=5 |
| 2 | 2 | 3 | (1, 9, C) | scale 2的9个tokens | 5+9=14 |
| 3 | 3 | 4 | (1, 16, C) | scale 3的16个tokens | 14+16=30 |
| 4 | 4 | 5 | (1, 25, C) | scale 4的25个tokens | 30+25=55 |
| 5 | 5 | 6 | (1, 36, C) | scale 5的36个tokens | 55+36=91 |
| 6 | 6 | 8 | (1, 64, C) | scale 6的64个tokens | 91+64=155 |
| 7 | 7 | 10 | (1, 100, C) | scale 7的100个tokens | 155+100=255 |
| 8 | 8 | 13 | (1, 169, C) | scale 8的169个tokens | 255+169=424 |
| 9 | 9 | 16 | (1, 256, C) | scale 9的256个tokens | 424+256=680 ✓ |

**拼接结果**:
```python
inp_cat = torch.cat([inp0, inp1, ..., inp9], dim=1)
inp_cat.shape = (1, 680, C)  ✓ 正确！
```

### 2.4 KV Cache的关键作用

虽然每次forward只输入**当前尺度的增量tokens**，但通过**KV Cache**，attention可以访问之前所有尺度的信息：

```python
# basic_var.py Line 107-109
if self.caching:
    if self.cached_k is None:
        self.cached_k = k
        self.cached_v = v
    else:
        k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)  # 累积
        v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)  # 累积
```

**工作流程**:
1. Scale 0: Q(1 tokens) attends to KV(1 tokens)
2. Scale 1: Q(4 tokens) attends to KV(5 tokens) ← 1+4累积
3. Scale 2: Q(9 tokens) attends to KV(14 tokens) ← 1+4+9累积
4. ...
5. Scale 9: Q(256 tokens) attends to KV(680 tokens) ← 全部累积

这就是为什么可以只输入增量tokens的原因！

### 2.5 总结：Prune_v6实现的正确性

✅ **Prune_v6的实现完全正确**：
1. 每次forward只输入当前尺度的pn²个tokens（增量）
2. Hook正确捕获每个尺度的增量tokens
3. 拼接10次得到完整的680 tokens
4. 通过KV Cache，每个尺度可以看到之前所有尺度的信息

✅ **没有重复计数问题**：
- 每个尺度的tokens只被捕获一次
- 拼接是正确的：1+4+9+...+256 = 680

✅ **实现高效**：
- 利用KV Cache避免重复计算
- 批量拼接后再计算Hessian矩阵

---

## 3. Scale_Mul数学原理深度解析

### 3.1 代码与公式对应

```python
# basic_var.py Line 101-105
if self.attn_l2_norm:
    scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()  # (1, H, 1, 1)
    q = F.normalize(q, dim=-1).mul(scale_mul)  # L2归一化 + 缩放
    k = F.normalize(k, dim=-1)                 # L2归一化
```

**数学推导**:

#### Step 1: L2归一化

```math
q_norm = q / ||q||  =>  ||q_norm|| = 1
k_norm = k / ||k||  =>  ||k_norm|| = 1
```

#### Step 2: 缩放

```math
q_scaled = scale_mul * q_norm
```

#### Step 3: 注意力计算

```math
attn = softmax(q_scaled @ k_norm^T)
     = softmax(scale_mul * q_norm @ k_norm^T)
     = softmax(scale_mul * cos(θ))
```

其中 `cos(θ)` 是归一化向量的点积，范围 `[-1, 1]`

### 3.2 Scale_Mul对Softmax的影响

**Softmax公式**:
```math
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

当输入乘以 `scale_mul`:
```math
softmax(scale_mul * x)_i = exp(scale_mul * x_i) / Σ_j exp(scale_mul * x_j)
```

**关键性质**: `scale_mul` 控制**softmax的锐度（sharpness）**

---

### 3.3 为什么Scale_Mul高 = 精确定位？

#### 数值示例

假设有3个位置的注意力logits（归一化后的余弦相似度）:
```
cos(θ) = [0.9, 0.5, 0.3]  # 位置0最相关
```

**Case 1: scale_mul = 1（低scale）**
```python
logits = 1 * [0.9, 0.5, 0.3] = [0.9, 0.5, 0.3]
attn = softmax([0.9, 0.5, 0.3]) = [0.427, 0.285, 0.228]
# 注意力分散：最高位置只占42.7%
```

**Case 2: scale_mul = 10（中scale）**
```python
logits = 10 * [0.9, 0.5, 0.3] = [9.0, 5.0, 3.0]
attn = softmax([9.0, 5.0, 3.0]) = [0.843, 0.114, 0.042]
# 注意力集中：最高位置占84.3%
```

**Case 3: scale_mul = 100（高scale）**
```python
logits = 100 * [0.9, 0.5, 0.3] = [90.0, 50.0, 30.0]
attn = softmax([90.0, 50.0, 30.0]) = [0.9999999, 0.0000001, 0.0]
# 注意力极度集中：几乎one-hot，精确定位到位置0
```

#### 数学解释

Softmax的**温度参数** `τ`:
```math
softmax(x/τ) = exp(x_i/τ) / Σ_j exp(x_j/τ)
```

- `τ → 0`: 输出趋向one-hot（最锐利）
- `τ → ∞`: 输出趋向均匀分布（最平滑）

**VAR的scale_mul相当于 `1/τ`**:
```math
softmax(scale_mul * x) ≈ softmax(x / (1/scale_mul))
```

- `scale_mul大` = `τ小` = **低温度** = **锐利分布** = **精确定位**
- `scale_mul小` = `τ大` = **高温度** = **平滑分布** = **全局整合**

---

### 3.4 为什么Scale_Mul低 = 全局整合？

**直观理解**:

```python
# scale_mul = 1 (低)
attn = [0.3, 0.25, 0.2, 0.15, 0.1]  # 分散在5个位置
output = 0.3*v0 + 0.25*v1 + 0.2*v2 + 0.15*v3 + 0.1*v4
# 整合了多个位置的信息（全局）

# scale_mul = 100 (高)
attn = [0.99, 0.01, 0, 0, 0]  # 集中在1个位置
output = 0.99*v0 + 0.01*v1
# 几乎只看位置0的信息（局部、精确）
```

**信息论角度**:

注意力分布的**熵（Entropy）**:
```math
H(attn) = -Σ_i attn_i * log(attn_i)
```

- **高scale_mul** → 低熵 → 信息集中 → 精确定位
- **低scale_mul** → 高熵 → 信息分散 → 全局整合

---

## 4. Scale_Mul与剪枝率的关系

### 4.1 理论假设

#### 假设1: Head功能分化假说

```
高scale_mul head (>50):  专家型head - 精确定位关键token
低scale_mul head (<5):   通才型head - 整合全局信息
中scale_mul head (5-50): 平衡型head - 两者兼顾
```

**推论**:
- 专家型head: **不可或缺**（剪掉会丢失关键定位能力）
- 通才型head: **可能冗余**（功能相似，可以剪枝）
- 平衡型head: **根据Hessian判断**

#### 假设2: 层级剪枝率假说

每层的**scale_mul分布**反映该层的剪枝潜力：

| 层的Scale_Mul特征 | 解释 | 建议剪枝率 |
|------------------|------|----------|
| **高方差（>20）** | Head分化明显，有专家/通才区分 | 可以剪掉低scale的通才heads | 25-35% |
| **低方差（<10）** | Head功能相似 | 难以区分重要性，保守剪枝 | 10-15% |
| **多极值（>40%极端值）** | 存在很多>50或<5的heads | 剪掉极低scale的 | 20-30% |
| **高均值（>30）** | 整层注重精确定位 | 关键层，保守剪枝 | 10-20% |

### 4.2 Scale_Mul引导的剪枝策略

#### 策略1: Scale-Thresholded Pruning

```python
def scale_thresholded_pruning(model, layer_idx, base_sparsity):
    """
    根据scale_mul设置阈值，优先剪低scale的heads
    """
    attn = model.blocks[layer_idx].attn
    scale_mul = attn.scale_mul_1H11.exp().squeeze()  # (num_heads,)

    # 计算Hessian重要性
    hessian_importance = compute_hessian_importance(layer_idx)

    # 低scale的head降低其重要性
    scale_penalty = torch.exp(-scale_mul / 10)  # scale越低，penalty越大
    adjusted_importance = hessian_importance * (1 - 0.3 * scale_penalty)

    # 按调整后的重要性剪枝
    num_prune = int(attn.num_heads * base_sparsity)
    prune_heads = adjusted_importance.argsort()[:num_prune]

    return prune_heads
```

#### 策略2: Dynamic Sparsity by Scale Distribution

```python
def compute_layer_sparsity_by_scale(model, base_sparsity=0.25):
    """
    根据每层的scale_mul分布动态调整剪枝率
    """
    layer_sparsities = []

    for i, block in enumerate(model.blocks):
        scale_mul = block.attn.scale_mul_1H11.exp().squeeze()

        # 统计特征
        mean = scale_mul.mean().item()
        std = scale_mul.std().item()
        low_scale_ratio = (scale_mul < 5).float().mean().item()
        high_scale_ratio = (scale_mul > 50).float().mean().item()

        # 决策树
        if high_scale_ratio > 0.4:  # 超过40%的heads是专家型
            sparsity = base_sparsity * 0.7  # 保守剪枝
        elif low_scale_ratio > 0.3:  # 超过30%的heads是通才型
            sparsity = base_sparsity * 1.3  # 激进剪枝
        elif std < 10:  # 低方差，功能相似
            sparsity = base_sparsity * 1.2
        else:
            sparsity = base_sparsity

        layer_sparsities.append(min(sparsity, 0.5))  # 上限50%

    return layer_sparsities
```

#### 策略3: Scale-Mul Guided Head Selection

```python
def scale_guided_head_selection(model, layer_idx, target_num_heads):
    """
    结合scale_mul和Hessian选择保留的heads
    """
    attn = model.blocks[layer_idx].attn
    scale_mul = attn.scale_mul_1H11.exp().squeeze()
    hessian_imp = compute_hessian_importance(layer_idx)

    # 归一化到[0, 1]
    scale_norm = (scale_mul - scale_mul.min()) / (scale_mul.max() - scale_mul.min())
    hess_norm = (hessian_imp - hessian_imp.min()) / (hessian_imp.max() - hessian_imp.min())

    # 融合策略
    alpha = 0.3  # scale_mul的权重

    # 分段融合：高scale的heads增强，低scale的heads降低
    importance = torch.where(
        scale_mul > 20,  # 高scale
        (1 - alpha) * hess_norm + alpha * scale_norm * 1.5,  # 增强
        (1 - alpha) * hess_norm + alpha * scale_norm * 0.5   # 降低
    )

    # 选择top-k
    keep_heads = importance.argsort(descending=True)[:target_num_heads]

    return keep_heads
```

---

## 5. 实验验证方案

### 5.1 验证Scale_Mul与Head重要性的相关性

#### 实验1: 相关性分析

```python
def analyze_scale_hessian_correlation(model, calibration_data):
    """
    分析scale_mul与Hessian重要性的相关性
    """
    results = {
        'layer_idx': [],
        'correlation': [],
        'scale_mean': [],
        'scale_std': [],
    }

    for i in range(len(model.blocks)):
        # 获取scale_mul
        scale_mul = model.blocks[i].attn.scale_mul_1H11.exp().squeeze().cpu().numpy()

        # 计算Hessian重要性（per-head）
        hessian_imp = compute_head_hessian_importance(model, i, calibration_data)

        # 计算相关系数
        from scipy.stats import pearsonr, spearmanr
        pearson_corr, p_value = pearsonr(scale_mul, hessian_imp)
        spearman_corr, _ = spearmanr(scale_mul, hessian_imp)

        results['layer_idx'].append(i)
        results['correlation'].append({
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'p_value': p_value
        })
        results['scale_mean'].append(scale_mul.mean())
        results['scale_std'].append(scale_mul.std())

        print(f"Layer {i}: Pearson={pearson_corr:.3f} (p={p_value:.3f}), "
              f"Spearman={spearman_corr:.3f}")

    return results
```

**预期结果**:
- 如果 `pearson_corr > 0.3 且 p_value < 0.05`: scale_mul是有效先验
- 如果 `pearson_corr < 0.1`: scale_mul与重要性无关，不应使用

---

#### 实验2: Ablation Study - 不同剪枝策略对比

```bash
# Baseline: 纯Hessian剪枝
python prune_v6.py --sparsity 0.3 --method hessian_only

# 实验组1: Scale阈值剪枝
python prune_v6.py --sparsity 0.3 --method scale_threshold --scale_weight 0.3

# 实验组2: Scale动态稀疏度
python prune_v6.py --base_sparsity 0.25 --method scale_adaptive

# 实验组3: Scale融合
python prune_v6.py --sparsity 0.3 --method scale_fusion --alpha 0.3
```

**评估指标**:
```python
metrics = {
    'FID': evaluate_fid(model, val_set),
    'IS': evaluate_inception_score(model, val_set),
    'Params': count_parameters(model),
    'MACs': count_macs(model),
    'Inference_Time': measure_latency(model),
}
```

---

### 5.2 验证Scale_Mul的功能假设

#### 实验3: Scale_Mul消融实验

```python
def ablate_scale_mul_groups(model, calibration_data):
    """
    移除不同scale_mul范围的heads，观察性能下降
    """
    results = []

    for layer_idx in range(len(model.blocks)):
        scale_mul = model.blocks[layer_idx].attn.scale_mul_1H11.exp().squeeze()

        # 定义3组
        high_scale_heads = (scale_mul > 50).nonzero().squeeze()  # 专家型
        low_scale_heads = (scale_mul < 5).nonzero().squeeze()   # 通才型
        mid_scale_heads = ((scale_mul >= 5) & (scale_mul <= 50)).nonzero().squeeze()

        # 分别移除每组
        for group_name, head_indices in [
            ('high_scale', high_scale_heads),
            ('low_scale', low_scale_heads),
            ('mid_scale', mid_scale_heads),
        ]:
            model_copy = copy.deepcopy(model)
            remove_heads(model_copy, layer_idx, head_indices)

            fid = evaluate_fid(model_copy, val_set)
            results.append({
                'layer': layer_idx,
                'removed_group': group_name,
                'num_removed': len(head_indices),
                'fid_drop': fid - baseline_fid
            })

    return results
```

**验证假设**:
- 如果 `FID_drop(high_scale) >> FID_drop(low_scale)`: **高scale更重要**
- 如果 `FID_drop(low_scale) > FID_drop(high_scale)`: **低scale更重要（全局整合关键）**

---

#### 实验4: 分尺度的Scale_Mul激活分析

```python
def analyze_scale_mul_by_scale_group(model, calibration_labels):
    """
    分析不同尺度下哪些heads的scale_mul更活跃
    """
    scale_groups = {
        'early': (0, 3),   # Scale 0-2
        'mid': (3, 7),     # Scale 3-6
        'late': (7, 10),   # Scale 7-9
    }

    results = {}

    for group_name, (start, end) in scale_groups.items():
        # 生成该尺度组的tokens
        tokens = generate_tokens_for_scales(model, calibration_labels, start, end)

        # 记录每层每个head的注意力锐度
        head_sharpness = []

        for layer_idx in range(len(model.blocks)):
            # Hook记录实际的attention weights
            attn_weights = []

            def hook_fn(module, input, output):
                # 记录softmax后的attn weights
                attn_weights.append(output[1])  # (B, H, L, L)

            handle = model.blocks[layer_idx].attn.register_forward_hook(hook_fn)
            _ = model.blocks[layer_idx](tokens, ...)
            handle.remove()

            # 计算每个head的熵（衡量锐度）
            attn = attn_weights[0]  # (B, H, L, L)
            entropy_per_head = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean(dim=(0, 2))
            # entropy低 → 注意力集中 → 高scale_mul起作用

            head_sharpness.append(entropy_per_head.cpu())

        results[group_name] = torch.stack(head_sharpness)  # (num_layers, num_heads)

    # 可视化：哪些heads在哪个尺度组更活跃
    return results
```

**分析**:
```python
# 对比不同尺度下的head活跃度
for head_idx in range(num_heads):
    early_entropy = results['early'][:, head_idx].mean()
    late_entropy = results['late'][:, head_idx].mean()

    if early_entropy < late_entropy:
        print(f"Head {head_idx}: 早期尺度更锐利（精确定位）")
    else:
        print(f"Head {head_idx}: 后期尺度更锐利（细节定位）")
```

---

### 5.3 综合验证实验

#### 实验5: Scale_Mul引导剪枝 vs Baseline

```python
# 完整对比实验
configs = [
    {
        'name': 'Baseline_Hessian',
        'method': 'hessian_only',
        'sparsity': 0.3,
    },
    {
        'name': 'Scale_Threshold',
        'method': 'scale_threshold',
        'sparsity': 0.3,
        'scale_weight': 0.3,
    },
    {
        'name': 'Scale_Adaptive',
        'method': 'scale_adaptive',
        'base_sparsity': 0.25,
    },
    {
        'name': 'Scale_Fusion',
        'method': 'scale_fusion',
        'sparsity': 0.3,
        'alpha': 0.3,
    },
]

results = []
for config in configs:
    model = load_model()
    model = prune_model(model, config)

    metrics = {
        'method': config['name'],
        'fid': evaluate_fid(model),
        'is': evaluate_inception_score(model),
        'params': count_parameters(model),
        'latency': measure_latency(model),
    }
    results.append(metrics)

# 输出对比表
print_comparison_table(results)
```

**成功标准**:
- Scale引导方法的FID比Baseline低 **5-10%**
- 在相同参数量下，Scale方法的FID更低

---

## 6. 实施建议

### 6.1 优先级1: 验证Scale_Mul相关性（1天）

✅ **Prune_v6的实现是正确的**，无需修改token收集逻辑。

```python
# 运行实验1，验证假设
python analyze_scale_correlation.py --var_ckpt path/to/var_d16.pth
```

**如果相关性 > 0.3**: 进入优先级2
**如果相关性 < 0.1**: 放弃scale_mul引导，专注其他创新点

### 6.2 优先级2: 实现Scale引导剪枝（2-3天）

```python
# 实现策略1-3，运行对比实验
# 详见Section 4.2
```

### 6.3 优先级3: 深度分析Scale_Mul功能（可选）

```python
# 运行实验3-4，发表论文用
# 详见Section 5.2
```

### 6.4 优先级4: 分尺度渐进剪枝（可选创新）

基于用户建议的3组渐进剪枝策略：
- 早期尺度组 (0-3): 保守剪枝
- 中期尺度组 (3-7): 中等剪枝
- 后期尺度组 (7-10): 激进剪枝

---

## 7. 总结

### 7.1 核心发现（✅ 已纠正）

1. ✅ **Prune_v6的实现正确**：使用增量AR生成的tokens
2. ✅ **Token拼接机制正确**：10个尺度的增量拼接得到680 tokens
3. ✅ **KV Cache机制关键**：允许增量输入但访问全部历史
4. ✅ **Scale_Mul的数学原理清晰**：
   - 高scale_mul (>50) = 低温度 = 精确定位（one-hot attention）
   - 低scale_mul (<5) = 高温度 = 全局整合（分散attention）

### 7.2 Scale_Mul与剪枝率的关系

**理论假设**:
```
层的scale_mul分布特征 → 剪枝潜力
├─ 高方差: Head分化明显 → 可剪冗余通才heads → 高剪枝率
├─ 多低scale: 通才型heads多 → 可剪除 → 高剪枝率
├─ 多高scale: 专家型heads多 → 需保留 → 低剪枝率
└─ 低方差: 功能相似 → 难区分 → 保守剪枝率
```

**验证方法**:
1. 相关性分析（实验1）
2. Ablation study（实验2）
3. 消融实验（实验3）
4. 分尺度激活分析（实验4）

### 7.3 下一步行动

```
Week 1: 验证Scale_Mul相关性 + 初步实验
Week 2: 实现Scale引导剪枝 + 对比实验
Week 3: 深度分析 + 论文撰写
```

### 7.4 感谢用户的纠正 🙏

用户对VAR机制的理解是完全正确的：
- `get_next_autoregressive_input`返回的是patch平方大小的tokens（增量）
- 每次forward只输入当前尺度的tokens
- 通过KV Cache访问历史tokens
- Prune_v6的实现是正确的，可以直接使用

**用户的洞察**：
1. ✅ VAR是增量生成，不是累积输入
2. ✅ 分3个尺度组渐进剪枝的创新想法
3. ✅ Scale_Mul可以作为层级剪枝率的指标

**你的思考方向完全正确！Scale_Mul是VAR独有的、LLM没有的先验信息，充分利用它可能带来显著提升。Prune_v6的实现已经为这些创新打下了良好基础。**
