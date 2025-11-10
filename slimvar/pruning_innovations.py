#!/usr/bin/env python3
"""
VAR Pruning Innovation Features
创新功能模块：分尺度分析、QKV/FC1补偿、渐进式剪枝

This module implements advanced pruning features on top of the basic pruning:
1. Scale-wise importance analysis
2. QKV/FC1 compensation pruning
3. Progressive pruning with lightweight evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


# ============================================================================
# 1. Scale-wise Importance Analysis
# ============================================================================

def compute_scale_ranges(patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)):
    """
    计算每个尺度的token位置范围

    Returns:
        {1: (0, 1), 2: (1, 5), 3: (5, 14), ..., 10: (424, 680)}
    """
    scale_ranges = {}
    start = 0
    for scale_idx, pn in enumerate(patch_nums, 1):
        num_tokens = pn * pn
        end = start + num_tokens
        scale_ranges[scale_idx] = (start, end)
        start = end
    return scale_ranges


@torch.no_grad()
def collect_activations_by_scale(var_model, calibration_loader, num_samples=256):
    """
    分尺度收集激活值

    Args:
        var_model: VAR模型
        calibration_loader: 校准数据 (calibration_labels, calibration_tokens)
        num_samples: 收集的样本数量

    Returns:
        activations_by_scale: {
            'scale_1': {'layer_0': tensor([N, 1, 1024]), ...},
            'scale_2': {'layer_0': tensor([N, 4, 1024]), ...},
            ...
        }
    """
    calibration_labels, calibration_tokens = calibration_loader

    scale_ranges = compute_scale_ranges()
    num_scales = len(scale_ranges)
    num_layers = len(var_model.blocks)

    # 初始化存储
    activations_by_scale = {
        f'scale_{i}': {f'layer_{j}': [] for j in range(num_layers)}
        for i in range(1, num_scales + 1)
    }

    # 注册hooks收集每层的输入
    layer_inputs = [[] for _ in range(num_layers)]

    def make_hook(layer_idx):
        def hook(module, inp, out):
            layer_inputs[layer_idx].append(inp[0].detach().cpu())
        return hook

    handles = []
    for idx, block in enumerate(var_model.blocks):
        handle = block.register_forward_hook(make_hook(idx))
        handles.append(handle)

    # 启用KV-cache加速（仅在label-only模式下）
    if calibration_tokens is None:
        for block in var_model.blocks:
            block.attn.kv_caching(True)

    # 前向传播收集数据
    print("开始分尺度收集激活值...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16

    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_labels = calibration_labels[i:end_idx].to(device)

        if calibration_tokens is not None:
            # 使用预编码的tokens（teacher forcing模式，不需要KV-cache）
            batch_tokens = calibration_tokens[i:end_idx].to(device)
            _ = var_model(batch_labels, batch_tokens)
        else:
            # 使用类别标签（需要KV-cache）
            _ = var_model(batch_labels)

        if i % 64 == 0:
            print(f"  已收集 {i}/{num_samples} 个样本")

    # 清理
    if calibration_tokens is None:
        for block in var_model.blocks:
            block.attn.kv_caching(False)
    for handle in handles:
        handle.remove()

    # 合并所有批次并按尺度分割
    print("正在按尺度分割激活值...")
    for layer_idx in range(num_layers):
        # 合并该层所有批次的输入
        layer_activations = torch.cat(layer_inputs[layer_idx], dim=0)  # (N, 680, 1024)

        # 按尺度分割
        for scale_idx, (start, end) in scale_ranges.items():
            scale_activation = layer_activations[:, start:end, :]  # (N, scale_tokens, 1024)
            activations_by_scale[f'scale_{scale_idx}'][f'layer_{layer_idx}'] = scale_activation

    print(f"✓ 分尺度收集完成")
    return activations_by_scale


def evaluate_head_importance_by_scale(var_model, activations_by_scale, args):
    """
    对每个尺度，使用SlimGPT的OBS方法评估每个attention head的重要性

    Args:
        var_model: VAR模型
        activations_by_scale: 分尺度收集的激活值
        args: 包含percdamp等参数

    Returns:
        head_importance: shape (num_layers, num_heads, num_scales)
            每个head在每个尺度下的重要性分数（OBS分数）
    """
    from slim_utils.slimgpt import SlimGPT

    num_layers = len(var_model.blocks)
    num_heads = var_model.num_heads
    num_scales = 10
    head_dim = 64

    head_importance = np.zeros((num_layers, num_heads, num_scales))

    for scale_idx in range(1, num_scales + 1):
        scale_key = f'scale_{scale_idx}'
        print(f"评估 Scale {scale_idx}...")

        for layer_idx in range(num_layers):
            layer_key = f'layer_{layer_idx}'
            layer_activations = activations_by_scale[scale_key][layer_key]  # (N, L, 1024)

            block = var_model.blocks[layer_idx]

            # 使用SlimGPT评估attn.proj的输入重要性
            pruner = SlimGPT(block.attn.proj, layer_idx, args)

            # 需要收集proj层的输入和输出
            inputs_proj = []
            outputs_proj = []

            def hook_fn(module, inp, out):
                inputs_proj.append(inp[0].detach())
                outputs_proj.append(out.detach())

            handle = block.attn.proj.register_forward_hook(hook_fn)

            # 批量前向传播该层
            N = layer_activations.shape[0]
            batch_size = 8
            for i in range(0, N, batch_size):
                batch = layer_activations[i:i+batch_size].cuda()
                with torch.no_grad():
                    _ = block.attn(batch, attn_bias=None)

            handle.remove()

            # 合并所有批次的输入输出
            inp_all = torch.cat(inputs_proj, dim=0)  # (N, L, 1024)
            out_all = torch.cat(outputs_proj, dim=0)

            # 添加到pruner计算Hessian
            pruner.add_batch(inp_all, out_all)

            # 使用SlimGPT的OBS方法计算每个通道的重要性
            W = pruner.layer.weight.data.clone()
            H = pruner.H
            damp = args.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(pruner.columns, device=pruner.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            # OBS重要性分数：W_i^2 / [H^{-1}]_{ii}
            importance_scores = torch.zeros(pruner.columns, device=pruner.dev)
            for i in range(pruner.columns):
                w_i = W[:, i]
                h_ii = Hinv[i, i]
                importance_scores[i] = (w_i ** 2).sum() / (h_ii + 1e-8)

            # 将通道重要性映射到head重要性
            for head_idx in range(num_heads):
                start_ch = head_idx * head_dim
                end_ch = (head_idx + 1) * head_dim
                head_importance[layer_idx, head_idx, scale_idx - 1] = \
                    importance_scores[start_ch:end_ch].mean().item()

            pruner.free()

    return head_importance  # (num_layers, num_heads, 10)


def analyze_importance_patterns(head_importance, save_path=None):
    """
    分析跨尺度的重要性规律

    Args:
        head_importance: (num_layers, num_heads, num_scales)
        save_path: 保存分析结果和可视化的路径

    Returns:
        analysis_results: 包含各种分析结果的字典
    """
    num_layers, num_heads, num_scales = head_importance.shape

    print("\n" + "="*60)
    print("重要性分析报告")
    print("="*60)

    # 1. 计算每个head的平均重要性（跨所有尺度）
    head_avg_importance = head_importance.mean(axis=2)  # (num_layers, num_heads)

    # 2. 找出平均重要性最低的head
    unimportant_heads = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            avg_imp = head_avg_importance[layer_idx, head_idx]
            unimportant_heads.append({
                'layer': layer_idx,
                'head': head_idx,
                'importance': avg_imp,
                'channels': list(range(head_idx * 64, (head_idx + 1) * 64))
            })

    # 按重要性排序
    unimportant_heads.sort(key=lambda x: x['importance'])

    print(f"\n【最不重要的20个head】")
    for i, head_info in enumerate(unimportant_heads[:20], 1):
        print(f"{i:2d}. Layer {head_info['layer']:2d}, Head {head_info['head']:2d}: {head_info['importance']:.6f}")

    # 3. 分析尺度差异
    scale_importance = head_importance.mean(axis=(0, 1))  # (10,)
    print(f"\n【各尺度的平均重要性】")
    for scale_idx, imp in enumerate(scale_importance, 1):
        print(f"Scale {scale_idx:2d}: {imp:.6f}")

    # 4. 分析层差异
    layer_importance = head_importance.mean(axis=(1, 2))  # (num_layers,)
    print(f"\n【各层的平均重要性】")
    for layer_idx, imp in enumerate(layer_importance):
        print(f"Layer {layer_idx:2d}: {imp:.6f}")

    # 5. 可视化
    if save_path:
        visualize_importance_heatmap(head_importance, save_path)

        # 保存分析结果
        np.savez(
            save_path + '.npz',
            head_importance=head_importance,
            head_avg_importance=head_avg_importance,
            scale_importance=scale_importance,
            layer_importance=layer_importance
        )
        print(f"\n✓ 分析结果已保存到: {save_path}.npz")

    return {
        'head_avg_importance': head_avg_importance,
        'unimportant_heads': unimportant_heads,
        'scale_importance': scale_importance,
        'layer_importance': layer_importance
    }


def visualize_importance_heatmap(head_importance, save_path):
    """绘制重要性热力图"""
    num_layers, num_heads, num_scales = head_importance.shape

    # 每层一个子图
    rows = (num_layers + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4*rows))
    axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]

        # 绘制该层的head重要性热力图（横轴=尺度，纵轴=head）
        im = ax.imshow(head_importance[layer_idx, :, :], cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Head')
        ax.set_xticks(range(num_scales))
        ax.set_xticklabels(range(1, num_scales + 1))
        ax.set_yticks(range(num_heads))
        plt.colorbar(im, ax=ax)

    # 隐藏多余的子图
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path + '_heatmap.png', dpi=150)
    print(f"✓ 重要性热力图已保存到: {save_path}_heatmap.png")
    plt.close()


# ============================================================================
# 2. QKV/FC1 Compensation Pruning
# ============================================================================

def compensate_qkv_cosine_similarity(mat_qkv, channels_to_prune):
    """
    方法1：基于余弦相似度的QKV补偿

    将被剪通道补偿到最相似的保留通道

    Args:
        mat_qkv: QKV线性层
        channels_to_prune: 要剪枝的通道索引列表

    Returns:
        None (in-place修改mat_qkv的权重)
    """
    W_qkv = mat_qkv.weight.data  # (3072, 1024)

    all_channels = set(range(1024))
    keep_channels = list(all_channels - set(channels_to_prune))

    hidden = 1024  # 原始hidden size

    for prune_ch in channels_to_prune:
        # Q部分的权重向量
        q_prune = W_qkv[prune_ch, :]  # (1024,)

        # 找到Q中最相似的保留通道
        max_sim = -1
        most_similar_ch = None
        for keep_ch in keep_channels:
            q_keep = W_qkv[keep_ch, :]
            sim = F.cosine_similarity(q_prune.unsqueeze(0), q_keep.unsqueeze(0), dim=1).item()
            if sim > max_sim:
                max_sim = sim
                most_similar_ch = keep_ch

        # 补偿权重 = 相似度
        alpha = max(max_sim, 0.0)  # 确保非负

        # Q,K,V都补偿到最相似的通道
        W_qkv[most_similar_ch, :] += alpha * W_qkv[prune_ch, :]
        W_qkv[most_similar_ch + hidden, :] += alpha * W_qkv[prune_ch + hidden, :]
        W_qkv[most_similar_ch + 2*hidden, :] += alpha * W_qkv[prune_ch + 2*hidden, :]

    mat_qkv.weight.data = W_qkv
    print(f"  ✓ QKV补偿完成（余弦相似度方法）：{len(channels_to_prune)} 个通道")


def compensate_qkv_optimal(mat_qkv, proj, channels_to_prune, calibration_data):
    """
    方法2：最优补偿权重

    通过最小二乘法计算最优补偿权重
    目标：min ||proj(qkv(X)) - proj(qkv'(X) + compensation)||^2

    Args:
        mat_qkv: QKV线性层
        proj: projection线性层
        channels_to_prune: 要剪枝的通道索引列表
        calibration_data: 校准数据（用于计算最优alpha）

    Returns:
        None (in-place修改mat_qkv的权重)
    """
    # TODO: 实现更准确但计算成本更高的最优补偿方法
    # 目前先使用简化版本
    print("  ⚠ 最优补偿方法尚未完全实现，退回到余弦相似度方法")
    compensate_qkv_cosine_similarity(mat_qkv, channels_to_prune)


def compensate_fc1(fc1, fc2, channels_to_prune, method='cosine_similarity'):
    """
    FC1补偿，与QKV类似

    Args:
        fc1: FFN的fc1层
        fc2: FFN的fc2层
        channels_to_prune: 要剪枝的通道索引列表
        method: 补偿方法

    Returns:
        None (in-place修改fc1的权重)
    """
    W_fc1 = fc1.weight.data  # (4096, 1024)

    all_channels = set(range(W_fc1.shape[0]))
    keep_channels = list(all_channels - set(channels_to_prune))

    for prune_ch in channels_to_prune:
        fc1_col = W_fc1[prune_ch, :]

        # 找到最相似的保留通道
        max_sim = -1
        most_similar_ch = None
        for keep_ch in keep_channels:
            fc1_keep = W_fc1[keep_ch, :]
            sim = F.cosine_similarity(fc1_col.unsqueeze(0), fc1_keep.unsqueeze(0), dim=1).item()
            if sim > max_sim:
                max_sim = sim
                most_similar_ch = keep_ch

        alpha = max(max_sim, 0.0) if method == 'cosine_similarity' else 1.0
        W_fc1[most_similar_ch, :] += alpha * W_fc1[prune_ch, :]

    fc1.weight.data = W_fc1
    print(f"  ✓ FC1补偿完成：{len(channels_to_prune)} 个通道")


# ============================================================================
# 3. Progressive Pruning
# ============================================================================

def progressive_pruning_pipeline(var_model, calibration_data, args):
    """
    渐进式剪枝主流程

    Args:
        var_model: VAR模型
        calibration_data: 校准数据
        args.progressive_stages: 阶段数（0=关闭）
        args.stage_sparsity: 各阶段稀疏度，如 "0.1,0.2,0.3"
        args.lightweight_eval_metric: 轻量级评估指标

    Returns:
        pruned_model: 剪枝后的模型
    """
    if args.progressive_stages == 0:
        print("渐进式剪枝已关闭")
        return None

    # 解析阶段稀疏度
    if hasattr(args, 'stage_sparsity') and args.stage_sparsity:
        sparsity_schedule = [float(s) for s in args.stage_sparsity.split(',')]
    else:
        # 默认：均匀分布到目标稀疏度
        sparsity_schedule = np.linspace(0, args.sparsity, args.progressive_stages + 1)[1:].tolist()

    print(f"\n{'='*60}")
    print(f"渐进式剪枝：{args.progressive_stages} 个阶段")
    print(f"稀疏度计划：{[f'{s:.2%}' for s in sparsity_schedule]}")
    print(f"轻量级评估：{args.lightweight_eval_metric}")
    print(f"{'='*60}\n")

    current_model = var_model
    previous_model = None
    baseline_metric = None

    for stage, target_sparsity in enumerate(sparsity_schedule):
        print(f"\n{'='*60}")
        print(f"阶段 {stage + 1}/{len(sparsity_schedule)}: 目标稀疏度 {target_sparsity:.1%}")
        print(f"{'='*60}")

        # 计算增量稀疏度
        current_sparsity = sparsity_schedule[stage - 1] if stage > 0 else 0
        incremental_sparsity = (target_sparsity - current_sparsity) / (1 - current_sparsity) if current_sparsity < 1 else 0
        print(f"增量剪枝比例: {incremental_sparsity:.2%}")

        # 执行本阶段的剪枝（需要调用基础剪枝函数）
        # 这里只是框架，具体实现需要与基础剪枝集成
        print(f"⚠  渐进式剪枝需要与基础剪枝函数集成")

        # 轻量级评估
        print(f"轻量级评估...")
        eval_result = lightweight_evaluation(
            current_model,
            calibration_data,
            args.lightweight_eval_metric
        )
        print(f"评估结果: {eval_result:.6f}")

        if stage == 0:
            baseline_metric = eval_result

        # 早停机制
        if stage > 0 and hasattr(args, 'enable_early_stop') and args.enable_early_stop:
            threshold = getattr(args, 'early_stop_threshold', 1.5)
            if eval_result > baseline_metric * threshold:
                print(f"\n⚠️  质量下降过多，停止剪枝")
                print(f"回退到阶段 {stage} 的模型")
                current_model = previous_model
                break

        previous_model = copy.deepcopy(current_model)

    return current_model


def lightweight_evaluation(model, calibration_data, metric='reconstruction_loss'):
    """
    轻量级评估指标

    Args:
        model: VAR模型
        calibration_data: 校准数据
        metric: 'reconstruction_loss' | 'kl_div' | 'sample_quality'

    Returns:
        score: 评估分数（越小越好）
    """
    if metric == 'reconstruction_loss':
        # 简化版：计算平均loss
        total_loss = 0
        count = 0

        with torch.no_grad():
            for b in model.blocks:
                b.attn.kv_caching(True)

            batch_size = 16
            for i in range(0, len(calibration_data), batch_size):
                batch = calibration_data[i:min(i+batch_size, len(calibration_data))]
                try:
                    outputs = model(batch)
                    # 简化：不计算真实loss，只看输出的方差作为代理
                    loss = outputs.var().item()
                    total_loss += loss
                    count += 1
                except:
                    pass

            for b in model.blocks:
                b.attn.kv_caching(False)

        return total_loss / max(count, 1)

    elif metric == 'kl_div':
        # TODO: 实现KL散度评估
        print("  ⚠ KL散度评估尚未实现")
        return 0.0

    elif metric == 'sample_quality':
        # TODO: 实现样本质量评估
        print("  ⚠ 样本质量评估尚未实现")
        return 0.0

    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# Utility Functions
# ============================================================================

def get_module_by_name(layer, name):
    """根据名称获取模块"""
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module


if __name__ == "__main__":
    print("VAR Pruning Innovation Features Module")
    print("This module provides advanced pruning features.")
    print("Import and use the functions in your main pruning script.")
