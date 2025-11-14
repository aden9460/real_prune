#!/usr/bin/env python3
"""
验证Scale_mul与Attention集中度的关系 (Verification 1)
Verify relationship between scale_mul and attention score concentration

核心假设:
- 高scale_mul → 注意力分布尖锐/集中（低熵、小span、高Gini）
- 低scale_mul → 注意力分布平滑/分散（高熵、大span、低Gini）

理论基础: Softmax温度效应
attention = softmax(scale_mul * (Q @ K^T))
高scale_mul = 低温度 → 分布尖锐
低scale_mul = 高温度 → 分布平滑

成功标准:
- Pearson(scale_mul, entropy) < -0.5, p < 0.01 (强负相关)
- Pearson(scale_mul, span) < -0.5, p < 0.01 (强负相关)
- Pearson(scale_mul, Gini) > 0.5, p < 0.01 (强正相关)
- 至少4/6层达到标准

创建日期: 2025-11-11
作者: 基于用户创新3思路
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# 添加VAR路径
sys.path.append("VAR/")
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


# =====================================================================
# 真实图片数据加载 (从 model_slimming_basic.py 复用)
# =====================================================================

def prepare_calibration_data(vae, num_samples, imagenet_dir,
                            final_reso=256, device='cuda'):
    """
    从ImageNet加载真实图片并编码为tokens

    Args:
        vae: VAE模型
        num_samples: 样本数量
        imagenet_dir: ImageNet根目录（包含train/和val/子目录）
        final_reso: 图像分辨率（默认256）
        device: 设备

    Returns:
        calibration_labels: (num_samples,) - 类别标签 [0-999]
        calibration_tokens: (num_samples, 679, 32) - VAE编码的tokens
    """
    from VAR.utils.data import build_dataset

    print(f"\nLoading {num_samples} real images from ImageNet...")
    print(f"  ImageNet directory: {imagenet_dir}")

    # 1. 加载ImageNet数据集
    try:
        num_classes, train_set, val_set = build_dataset(
            data_path=imagenet_dir,
            final_reso=final_reso,
            hflip=False,      # 验证集不使用翻转
            mid_reso=1.125    # 中间分辨率倍数
        )
        dataset = val_set
        print(f"  ✓ Loaded ImageNet validation set: {len(dataset)} images")
    except Exception as e:
        print(f"  ✗ Failed to load ImageNet: {e}")
        print(f"  Please check if {imagenet_dir}/val/ exists")
        raise

    # 2. 均匀采样（覆盖所有类别）
    if num_samples <= len(dataset):
        step = len(dataset) // num_samples
        indices = torch.arange(0, len(dataset), step)[:num_samples]
    else:
        print(f"  Warning: Requested {num_samples} samples but dataset has only {len(dataset)}")
        indices = torch.arange(len(dataset))
        num_samples = len(dataset)

    print(f"  Sampling strategy: uniform stride (step={step if num_samples <= len(dataset) else 1})")

    # 3. 批处理编码
    calibration_labels = []
    calibration_tokens = []

    batch_size = 8
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"\n  Encoding {num_samples} images to tokens...")
    for i in tqdm(range(num_batches), desc="  Processing"):
        batch_indices = indices[i*batch_size:min((i+1)*batch_size, num_samples)]
        images = []
        labels = []

        # 从数据集加载batch
        for idx in batch_indices:
            try:
                img, label = dataset[int(idx)]
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"\n  Warning: Failed to load image at index {idx}: {e}")
                continue

        if len(images) == 0:
            continue

        images = torch.stack(images).to(device)  # (B, 3, 256, 256)
        labels = torch.tensor(labels).to(device)  # (B,)

        with torch.no_grad():
            # VAE编码：图像 → token IDs
            gt_idx_Bl = vae.img_to_idxBl(images)
            # Token IDs → VAR输入格式 (679 tokens)
            x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)

            calibration_labels.append(labels.cpu())
            calibration_tokens.append(x_BLCv.cpu())  # 移到CPU节省显存

    # 4. 合并所有batches
    calibration_labels = torch.cat(calibration_labels, dim=0)
    calibration_tokens = torch.cat(calibration_tokens, dim=0)

    print(f"\n  ✓ Successfully encoded {len(calibration_labels)} images")
    print(f"    Labels shape: {calibration_labels.shape}")
    print(f"    Tokens shape: {calibration_tokens.shape}")
    print(f"    Class distribution: {len(torch.unique(calibration_labels))} unique classes")

    return calibration_labels, calibration_tokens


# =====================================================================
# 模型加载 (复用已有函数)
# =====================================================================

def load_var_model(model_depth, vae_ckpt_path, var_ckpt_path, device='cuda'):
    """加载VAR模型"""
    from VAR.models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=False
    )

    vae.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt_path, map_location='cpu'), strict=False)

    vae.eval()
    var.eval()

    print(f'✓ VAR-d{model_depth} model loaded successfully.')
    return vae, var


def extract_scale_mul(model):
    """
    提取所有层的scale_mul参数

    Returns:
        scale_mul_matrix: (num_layers, num_heads) 实际scale值
    """
    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads

    scale_mul_matrix = np.zeros((num_layers, num_heads))

    for i, block in enumerate(model.blocks):
        # scale_mul_1H11: (1, num_heads, 1, 1)
        raw_values = block.attn.scale_mul_1H11.data.cpu().squeeze().numpy()  # (num_heads,)
        scale_values = np.exp(raw_values)  # 转换为实际scale
        scale_mul_matrix[i] = scale_values

    return scale_mul_matrix


# =====================================================================
# Attention Map提取器 (复用并简化)
# =====================================================================

class AttentionMapExtractor:
    """
    Hook VAR模型的attention层，提取attention weights
    专门用于集中度分析
    """

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.attention_maps = []
        self.hook_handle = None

        # 获取目标层
        self.target_block = model.blocks[layer_idx]
        self.target_attn = self.target_block.attn

        # 保存原始设置
        self.original_using_flash = self.target_attn.using_flash
        self.original_using_xform = self.target_attn.using_xform

        # 强制使用slow attention以便hook
        self.target_attn.using_flash = False
        self.target_attn.using_xform = False

    def _compute_slow_attn_with_capture(self, query, key, value, scale, attn_mask=None):
        """
        复制slow_attn的逻辑，但捕获attention weights
        """
        # 计算attention scores: [B, H, L, L]
        attn_scores = query.mul(scale) @ key.transpose(-2, -1)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Softmax得到attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 收集attention weights (全序列)
        L = attn_weights.shape[2]

        # 保存所有尺度的attention weights
        # Note: VAR有多个尺度 (first_l=1) + 679 tokens → 680 total
        if L >= 256:  # 至少包含最大尺度
            # 保存完整的attention weights [B, H, L, L]
            self.attention_maps.append(attn_weights.detach().cpu())

        # 计算输出
        output = attn_weights @ value
        return output

    def extract(self, inputs, use_teacher_forcing=False, input_tokens=None):
        """
        提取attention maps

        支持两种模式：
        1. Autoregressive生成模式（默认，使用随机标签）
        2. Teacher forcing模式（使用真实图片编码的tokens）

        Args:
            inputs: class labels [B]
            use_teacher_forcing: 是否使用teacher forcing模式
            input_tokens: 预编码的tokens (B, 679, 32)，仅在teacher forcing模式需要

        Returns:
            attention_maps: [B, num_heads, L, L]
        """
        self.attention_maps = []

        # Monkey patch slow_attn函数
        original_forward = self.target_attn.forward

        def patched_forward(x, attn_bias):
            """Patched forward函数"""
            B, L, C = x.shape

            # QKV projection
            qkv = F.linear(
                input=x,
                weight=self.target_attn.mat_qkv.weight,
                bias=torch.cat((
                    self.target_attn.q_bias,
                    self.target_attn.zero_k_bias,
                    self.target_attn.v_bias
                ))
            ).view(B, L, 3, self.target_attn.num_heads, 64)

            # 分离q, k, v
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # [B, H, L, 64]

            # L2 normalize if needed
            if self.target_attn.attn_l2_norm:
                scale_mul = self.target_attn.scale_mul_1H11.clamp_max(
                    self.target_attn.max_scale_mul
                ).exp()
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)

            # 使用我们的捕获版本slow_attn
            oup = self._compute_slow_attn_with_capture(
                query=q, key=k, value=v,
                scale=self.target_attn.scale,
                attn_mask=attn_bias
            ).transpose(1, 2).reshape(B, L, C)

            # Projection
            return self.target_attn.proj_drop(self.target_attn.proj(oup))

        # 临时替换forward
        self.target_attn.forward = patched_forward

        try:
            # 执行forward pass
            with torch.no_grad():
                B = inputs.shape[0]

                if use_teacher_forcing:
                    # ========== 模式1: Teacher Forcing（真实图片tokens） ==========
                    assert input_tokens is not None, "Teacher forcing mode requires input_tokens"

                    device = next(self.model.parameters()).device

                    # 逐样本处理
                    for b in range(B):
                        label = inputs[b:b+1].to(device)  # (1,)
                        tokens = input_tokens[b:b+1].to(device)  # (1, 679, 32)

                        # 1. 获取class embedding and process
                        cond_BD = self.model.class_emb(label)  # (1, C)

                        # 2. Create start-of-sequence token (first_l positions)
                        sos = cond_BD.unsqueeze(1).expand(1, self.model.first_l, -1) + \
                              self.model.pos_start.expand(1, self.model.first_l, -1)

                        # 3. Embed image tokens
                        x_tokens = self.model.word_embed(tokens.float())  # (1, 679, C)

                        # 4. 拼接 sos tokens 和 image tokens
                        x = torch.cat([sos, x_tokens], dim=1)  # (1, first_l+679, C)

                        # 5. Add level and position embeddings
                        seq_len = x.shape[1]
                        x += self.model.lvl_embed(self.model.lvl_1L[:, :seq_len].expand(1, -1))
                        x += self.model.pos_1LC[:, :seq_len]

                        # 6. Process cond_BD for blocks
                        cond_BD_or_gss = self.model.shared_ada_lin(cond_BD)

                        # 7. 逐层前向传播到目标层
                        for layer_idx in range(self.layer_idx + 1):
                            # 获取attention bias（causal mask）
                            seq_len = x.shape[1]
                            attn_bias = self.model.attn_bias_for_masking[:, :, :seq_len, :seq_len]

                            # 前向传播
                            if layer_idx < self.layer_idx:
                                # 非目标层：正常前向
                                x = self.model.blocks[layer_idx](x, cond_BD_or_gss, attn_bias)
                            else:
                                # 目标层：会触发attention捕获
                                x = self.target_block(x, cond_BD_or_gss, attn_bias)
                else:
                    # ========== 模式2: Autoregressive生成（随机标签） ==========
                    # Process each sample
                    for b in range(B):
                        label = inputs[b:b+1]  # [1]
                        # Run autoregressive inference
                        _ = self.model.autoregressive_infer_cfg(
                            B=1,
                            label_B=label,
                        cfg=1.5,
                        top_p=0.96,
                        g_seed=None,
                    )
        finally:
            # 恢复原始forward
            self.target_attn.forward = original_forward

        # 合并所有batch的attention maps
        if len(self.attention_maps) == 0:
            raise ValueError(f"No attention maps captured for layer {self.layer_idx}")

        attention_maps = torch.cat(self.attention_maps, dim=0)  # [total_B, H, L, L]
        return attention_maps

    def restore(self):
        """恢复原始设置"""
        self.target_attn.using_flash = self.original_using_flash
        self.target_attn.using_xform = self.original_using_xform


# =====================================================================
# 集中度度量函数
# =====================================================================

def compute_attention_entropy(attention_weights):
    """
    计算attention分布的Shannon熵

    Args:
        attention_weights: [batch, num_heads, seq, seq]

    Returns:
        entropy_per_head: [num_heads] 每个head的平均熵

    解释:
        - 熵 = 0: 完全集中在一个位置（极端尖锐）
        - 熵 = log(seq_len): 完全均匀分布（最平滑）
        - 对于seq=256: 最大熵 ≈ 5.54
    """
    batch, num_heads, seq, _ = attention_weights.shape
    entropies = []

    for head_idx in range(num_heads):
        head_attn = attention_weights[:, head_idx, :, :]  # [batch, seq, seq]

        # 计算每个query position的熵
        entropies_per_query = []
        for b in range(batch):
            for q in range(seq):
                attn_dist = head_attn[b, q, :]  # [seq] - 某个query的attention分布

                # Shannon entropy: -Σ p*log(p)
                # 添加小的epsilon避免log(0)
                entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10))
                entropies_per_query.append(entropy.item())

        # 平均熵
        avg_entropy = np.mean(entropies_per_query)
        entropies.append(avg_entropy)

    return torch.tensor(entropies)


def compute_effective_span(attention_weights, threshold=0.9):
    """
    计算有效关注范围（Effective Attention Span）

    覆盖threshold概率质量所需的位置数

    Args:
        attention_weights: [batch, num_heads, seq, seq]
        threshold: 概率阈值（默认0.9）

    Returns:
        spans: [num_heads] 每个head的平均有效span

    解释:
        - Span越小 = 越集中（关注少数位置）
        - Span越大 = 越分散（关注多个位置）
        - 最小值: 1 (完全集中)
        - 最大值: seq_len (完全均匀)
    """
    batch, num_heads, seq, _ = attention_weights.shape
    spans = []

    for head_idx in range(num_heads):
        head_attn = attention_weights[:, head_idx, :, :]  # [batch, seq, seq]

        spans_per_query = []
        for b in range(batch):
            for q in range(seq):
                attn_dist = head_attn[b, q, :]  # [seq]

                # 排序
                sorted_attn, _ = torch.sort(attn_dist, descending=True)

                # 累积和达到threshold的位置数
                cumsum = torch.cumsum(sorted_attn, dim=0)
                span = torch.sum(cumsum < threshold).item() + 1

                spans_per_query.append(span)

        avg_span = np.mean(spans_per_query)
        spans.append(avg_span)

    return torch.tensor(spans)


def compute_gini_coefficient(attention_weights):
    """
    计算Gini系数：衡量分布不均匀程度

    Args:
        attention_weights: [batch, num_heads, seq, seq]

    Returns:
        gini_coeffs: [num_heads] 每个head的平均Gini系数

    解释:
        - Gini = 0: 完全均匀分布
        - Gini = 1: 完全不均匀（集中）
        - 高Gini = 注意力集中在少数位置
        - 低Gini = 注意力均匀分布
    """
    batch, num_heads, seq, _ = attention_weights.shape
    gini_coeffs = []

    for head_idx in range(num_heads):
        head_attn = attention_weights[:, head_idx, :, :]  # [batch, seq, seq]

        ginis = []
        for b in range(batch):
            for q in range(seq):
                attn_dist = head_attn[b, q, :].cpu().numpy()

                # 排序
                sorted_attn = np.sort(attn_dist)
                n = len(sorted_attn)

                # Gini系数计算
                cumsum = np.cumsum(sorted_attn)
                gini = (n + 1 - 2 * np.sum((n - np.arange(n)) * sorted_attn) / cumsum[-1]) / n
                ginis.append(gini)

        avg_gini = np.mean(ginis)
        gini_coeffs.append(avg_gini)

    return torch.tensor(gini_coeffs)


# =====================================================================
# 相关性分析函数
# =====================================================================

def analyze_concentration_correlations(scale_mul, entropies, spans, ginis):
    """
    分析scale_mul与各种集中度指标的相关性

    Args:
        scale_mul: [num_heads] scale_mul values
        entropies: [num_heads] Shannon entropies
        spans: [num_heads] Effective spans
        ginis: [num_heads] Gini coefficients

    Returns:
        correlations: dict with correlation results
    """
    results = {}

    # Entropy vs Scale (期望：强负相关)
    r_entropy, p_entropy = pearsonr(scale_mul.cpu().numpy(), entropies.cpu().numpy())
    results['entropy'] = {
        'pearson_r': r_entropy,
        'p_value': p_entropy,
        'expected': 'negative',
        'pass': r_entropy < -0.5 and p_entropy < 0.01
    }

    # Span vs Scale (期望：强负相关)
    r_span, p_span = pearsonr(scale_mul.cpu().numpy(), spans.cpu().numpy())
    results['span'] = {
        'pearson_r': r_span,
        'p_value': p_span,
        'expected': 'negative',
        'pass': r_span < -0.5 and p_span < 0.01
    }

    # Gini vs Scale (期望：强正相关)
    r_gini, p_gini = pearsonr(scale_mul.cpu().numpy(), ginis.cpu().numpy())
    results['gini'] = {
        'pearson_r': r_gini,
        'p_value': p_gini,
        'expected': 'positive',
        'pass': r_gini > 0.5 and p_gini < 0.01
    }

    return results


def plot_scale_vs_concentration(layer_idx, scale_mul, entropies, spans, ginis,
                               correlations, save_dir):
    """
    可视化scale_mul与集中度指标的关系
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Layer {layer_idx}: Scale_mul vs Attention Concentration',
                 fontsize=16, fontweight='bold')

    # 转换为numpy
    scale_np = scale_mul.cpu().numpy()
    entropy_np = entropies.cpu().numpy()
    span_np = spans.cpu().numpy()
    gini_np = ginis.cpu().numpy()

    # Plot 1: Scale vs Entropy
    ax = axes[0, 0]
    ax.scatter(scale_np, entropy_np, alpha=0.7, s=60, c='blue', edgecolors='black')
    z = np.polyfit(scale_np, entropy_np, 1)
    p = np.poly1d(z)
    x_line = np.linspace(scale_np.min(), scale_np.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel('Scale_mul')
    ax.set_ylabel('Shannon Entropy')
    ax.set_title('Scale vs Entropy (Expected: Negative)')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    r, p_val = correlations['entropy']['pearson_r'], correlations['entropy']['p_value']
    status = "✓ PASS" if correlations['entropy']['pass'] else "✗ FAIL"
    color = 'green' if correlations['entropy']['pass'] else 'red'
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}\n{status}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Plot 2: Scale vs Effective Span
    ax = axes[0, 1]
    ax.scatter(scale_np, span_np, alpha=0.7, s=60, c='orange', edgecolors='black')
    z = np.polyfit(scale_np, span_np, 1)
    p = np.poly1d(z)
    x_line = np.linspace(scale_np.min(), scale_np.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel('Scale_mul')
    ax.set_ylabel('Effective Span')
    ax.set_title('Scale vs Span (Expected: Negative)')
    ax.grid(True, alpha=0.3)

    r, p_val = correlations['span']['pearson_r'], correlations['span']['p_value']
    status = "✓ PASS" if correlations['span']['pass'] else "✗ FAIL"
    color = 'green' if correlations['span']['pass'] else 'red'
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}\n{status}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Plot 3: Scale vs Gini
    ax = axes[1, 0]
    ax.scatter(scale_np, gini_np, alpha=0.7, s=60, c='green', edgecolors='black')
    z = np.polyfit(scale_np, gini_np, 1)
    p = np.poly1d(z)
    x_line = np.linspace(scale_np.min(), scale_np.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel('Scale_mul')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Scale vs Gini (Expected: Positive)')
    ax.grid(True, alpha=0.3)

    r, p_val = correlations['gini']['pearson_r'], correlations['gini']['p_value']
    status = "✓ PASS" if correlations['gini']['pass'] else "✗ FAIL"
    color = 'green' if correlations['gini']['pass'] else 'red'
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}\n{status}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Plot 4: Summary metrics
    ax = axes[1, 1]
    ax.axis('off')

    # 汇总信息
    summary_text = f"Layer {layer_idx} Summary:\n\n"
    summary_text += f"Scale_mul range: [{scale_np.min():.1f}, {scale_np.max():.1f}]\n"
    summary_text += f"Scale_mul variance: {scale_np.var():.1f}\n\n"
    summary_text += "Concentration Metrics:\n"
    summary_text += f"  Entropy: [{entropy_np.min():.2f}, {entropy_np.max():.2f}]\n"
    summary_text += f"  Span: [{span_np.min():.1f}, {span_np.max():.1f}]\n"
    summary_text += f"  Gini: [{gini_np.min():.3f}, {gini_np.max():.3f}]\n\n"

    # 成功统计
    passes = sum([correlations[k]['pass'] for k in correlations])
    summary_text += f"Verification Results: {passes}/3 PASS\n"

    overall_pass = passes >= 2  # 至少2/3指标通过
    if overall_pass:
        summary_text += "✓ OVERALL: PASS\n"
        summary_text += "Strong evidence for scale-concentration relationship!"
        bbox_color = 'lightgreen'
    else:
        summary_text += "✗ OVERALL: FAIL\n"
        summary_text += "Insufficient evidence for scale-concentration relationship."
        bbox_color = 'lightcoral'

    ax.text(0.05, 0.95, summary_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.7),
            fontsize=11, fontfamily='monospace')

    plt.tight_layout()
    save_path = f'{save_dir}/scale_concentration_layer{layer_idx}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved visualization: {save_path}")


# =====================================================================
# 主验证函数
# =====================================================================

def verify_scale_concentration_relationship(model, calibration_data, layers=[0, 2, 5, 7, 11, 12],
                                           use_teacher_forcing=False, calibration_tokens=None):
    """
    验证scale_mul与attention集中度的关系

    对每一层：
    1. 提取所有heads的attention weights
    2. 计算scale_mul
    3. 计算集中度指标（熵、span、Gini）
    4. 相关性分析

    Args:
        model: VAR模型
        calibration_data: 校准数据（class labels）
        layers: 要测试的层索引列表
        use_teacher_forcing: 是否使用teacher forcing模式（真实图片tokens）
        calibration_tokens: 预编码的tokens (num_samples, 679, 32)，teacher forcing模式需要

    Returns:
        results: dict 包含所有层的验证结果
    """
    results = {}

    print(f"\n{'='*80}")
    print(" Verification 1: Scale_mul vs Attention Concentration ".center(80, '='))
    print(f"{'='*80}\n")

    # 创建输出目录
    output_dir = 'scale_concentration_verification'
    os.makedirs(output_dir, exist_ok=True)

    for layer_idx in layers:
        print(f"\n{'-'*60}")
        print(f" Analyzing Layer {layer_idx} ".center(60, '-'))
        print(f"{'-'*60}\n")

        # 1. 获取attention weights
        print("Step 1: Extracting attention weights...")
        if use_teacher_forcing:
            print("  Mode: Teacher Forcing (using real image tokens)")
        else:
            print("  Mode: Autoregressive Generation (using random labels)")

        try:
            extractor = AttentionMapExtractor(model, layer_idx)
            attention_weights = extractor.extract(
                calibration_data,
                use_teacher_forcing=use_teacher_forcing,
                input_tokens=calibration_tokens
            )  # [batch, num_heads, seq, seq]
            extractor.restore()
            print(f"  ✓ Extracted attention maps: {attention_weights.shape}")
        except Exception as e:
            print(f"  ✗ Failed to extract attention for layer {layer_idx}: {e}")
            continue

        # 2. 获取scale_mul
        print("Step 2: Extracting scale_mul...")
        scale_mul_matrix = extract_scale_mul(model)
        scale_mul = torch.tensor(scale_mul_matrix[layer_idx])  # [num_heads]
        print(f"  ✓ Scale_mul range: [{scale_mul.min():.2f}, {scale_mul.max():.2f}]")
        print(f"  ✓ Scale_mul variance: {scale_mul.var():.2f}")

        # 3. 计算集中度指标
        print("Step 3: Computing concentration metrics...")
        entropies = compute_attention_entropy(attention_weights)
        spans = compute_effective_span(attention_weights)
        ginis = compute_gini_coefficient(attention_weights)

        print(f"  ✓ Entropy range: [{entropies.min():.2f}, {entropies.max():.2f}]")
        print(f"  ✓ Span range: [{spans.min():.1f}, {spans.max():.1f}]")
        print(f"  ✓ Gini range: [{ginis.min():.3f}, {ginis.max():.3f}]")

        # 4. 相关性分析
        print("Step 4: Correlation analysis...")
        correlations = analyze_concentration_correlations(scale_mul, entropies, spans, ginis)

        # 打印结果
        print("\n  Correlation Results:")
        for metric, result in correlations.items():
            r = result['pearson_r']
            p = result['p_value']
            status = "✓ PASS" if result['pass'] else "✗ FAIL"
            print(f"    {metric.capitalize():10s}: r={r:6.3f}, p={p:.4f}  {status}")

        # 综合判断
        passes = sum([correlations[k]['pass'] for k in correlations])
        overall_pass = passes >= 2  # 至少2/3指标通过

        print(f"\n  Layer {layer_idx} Summary: {passes}/3 metrics pass")
        if overall_pass:
            print("  ✓ OVERALL: PASS - Strong evidence for scale-concentration relationship")
        else:
            print("  ✗ OVERALL: FAIL - Insufficient evidence")

        # 5. 可视化
        print("\nStep 5: Generating visualization...")
        plot_scale_vs_concentration(layer_idx, scale_mul, entropies, spans, ginis,
                                   correlations, output_dir)

        # 保存详细结果
        layer_results = {
            'layer_idx': layer_idx,
            'scale_mul': {
                'mean': float(scale_mul.mean()),
                'std': float(scale_mul.std()),
                'min': float(scale_mul.min()),
                'max': float(scale_mul.max()),
                'values': scale_mul.tolist()
            },
            'concentration_metrics': {
                'entropies': entropies.tolist(),
                'spans': spans.tolist(),
                'ginis': ginis.tolist()
            },
            'correlations': {
                k: {
                    'pearson_r': float(v['pearson_r']),
                    'p_value': float(v['p_value']),
                    'expected': v['expected'],
                    'pass': bool(v['pass'])
                } for k, v in correlations.items()
            },
            'overall': {
                'passes': int(passes),
                'total': 3,
                'pass': bool(overall_pass)
            }
        }

        results[layer_idx] = layer_results

        # 保存单层结果
        with open(f'{output_dir}/layer_{layer_idx}_results.json', 'w') as f:
            json.dump(layer_results, f, indent=2)

    # 保存综合结果
    with open(f'{output_dir}/verification_1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 最终汇总
    print(f"\n{'='*80}")
    print(" VERIFICATION 1 SUMMARY ".center(80, '='))
    print(f"{'='*80}\n")

    total_layers = len(layers)
    passing_layers = sum([results[layer]['overall']['pass'] for layer in layers if layer in results])

    print(f"Tested layers: {total_layers}")
    print(f"Passing layers: {passing_layers}")
    print(f"Success rate: {passing_layers}/{total_layers} ({100*passing_layers/total_layers:.1f}%)")

    if passing_layers >= 4:  # 至少4/6层通过
        print("\n✅ VERIFICATION 1: SUCCESS!")
        print("Strong evidence for scale_mul-concentration relationship")
        print("→ Proceed to Verification 2 (Layer Classification)")
    else:
        print("\n❌ VERIFICATION 1: FAILED")
        print("Insufficient evidence for scale_mul-concentration relationship")
        print("→ Scale-based strategies may not be effective")

    print(f"\nResults saved to: {output_dir}/")
    print(f"{'='*80}")

    return results


# =====================================================================
# 主函数
# =====================================================================

def main(args):
    print("\n" + "="*80)
    print(" Scale_mul vs Attention Concentration Verification ".center(80, "="))
    print("="*80 + "\n")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # 加载模型
    print("Loading VAR model...")
    vae_ckpt = args.vae_ckpt if args.vae_ckpt else '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt if args.var_ckpt else f'/home/project/daily/AR/model_zoo/var_d{args.model_depth}.pth'

    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device=device)

    # 准备校准数据
    if args.use_images:
        # 模式1: 使用真实ImageNet图片
        print(f"\n{'='*60}")
        print(" Using REAL ImageNet Images ".center(60, '='))
        print(f"{'='*60}")

        calibration_labels, calibration_tokens = prepare_calibration_data(
            vae,
            num_samples=args.num_samples,
            imagenet_dir=args.imagenet_dir,
            device=device
        )

        use_teacher_forcing = True
        print("\n✓ Using teacher forcing mode with pre-encoded tokens")
    else:
        # 模式2: 使用随机生成（原有方式）
        print(f"\n{'='*60}")
        print(" Using Random Generation ".center(60, '='))
        print(f"{'='*60}")
        print(f"\nPreparing {args.num_samples} calibration samples...")

        calibration_labels = torch.randint(0, 1000, (args.num_samples,)).to(device)
        calibration_tokens = None
        use_teacher_forcing = False
        print("✓ Using autoregressive generation mode")

    # 解析要测试的层
    if args.layers == 'default':
        test_layers = [0, 2, 5, 7, 11, 12]  # 推荐的代表性层
    else:
        test_layers = [int(x.strip()) for x in args.layers.split(',')]

    print(f"\nTesting layers: {test_layers}")
    print(f"Calibration samples: {len(calibration_labels)}")

    # 验证
    results = verify_scale_concentration_relationship(
        model=var,
        calibration_data=calibration_labels,
        layers=test_layers,
        use_teacher_forcing=use_teacher_forcing,
        calibration_tokens=calibration_tokens
    )

    print("="*80)
    print(" Verification 1 Complete! ".center(80, "="))
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify scale_mul vs attention concentration relationship (Verification 1)"
    )

    parser.add_argument(
        "--model_depth", type=int, default=16,
        help="VAR model depth (default: 16)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of calibration samples (default: 50)"
    )
    parser.add_argument(
        "--layers", type=str, default="default",
        help="Comma-separated layer indices or 'default' for [0,2,5,7,11,12]"
    )
    parser.add_argument(
        "--vae_ckpt", type=str, default="",
        help="Path to VQVAE checkpoint"
    )
    parser.add_argument(
        "--var_ckpt", type=str, default="",
        help="Path to VAR checkpoint"
    )
    parser.add_argument(
        "--use_images", action="store_true",
        help="Use real ImageNet images instead of random generation"
    )
    parser.add_argument(
        "--imagenet_dir", type=str, default="/home/project/ImageNet-1K",
        help="Path to ImageNet root directory (default: /home/project/ImageNet-1K)"
    )

    args = parser.parse_args()
    main(args)