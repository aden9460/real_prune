#!/usr/bin/env python3
"""
VAR OBA Pruning Script
使用Optimal Brain Apoptosis (OBA)对VAR模型进行结构化剪枝

OBA核心优势：
1. 全局重要性排序（vs SlimGPT的逐层贪心）
2. 完整Hessian-vector积（vs Fisher对角近似）
3. 连接性建模：upward/downward/parallel三种依赖关系

目标：40%稀疏度，剪枝ffn.fc2输入和attn.proj输入及其耦合层
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
import torch_pruning as tp
import sys
from copy import deepcopy
import gc

# 添加VAR路径
sys.path.append("../slimvar/VAR/")
sys.path.append("torch_pruning/")

# 加速：禁用默认参数初始化
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

# 导入SlimGPT的工具函数（复用经过验证的代码）
sys.path.append("../slimvar/")
from model_slimming_basic_v1 import (
    load_var_model, find_layers, get_module_by_name,
    check_sparsity, get_imagenet_data_for_var
)

# OBA imports
from torch_pruning.pruner.algorithms.oba_pruner import OBAPruner
from torch_pruning.pruner.oba_importance import HessianImportance


def compute_oba_global_importance(model, calibration_loader, target_layers,
                                   delta=1.0, upward_delta=1.0, downward_delta=1.0, parallel_delta=1.0):
    """
    计算OBA全局重要性分数

    Args:
        model: VAR模型
        calibration_loader: 校准数据
        target_layers: 目标剪枝层 ['ffn.fc2', 'attn.proj']
        delta weights: OBA连接性权重

    Returns:
        importance_dict: {layer_name: importance_scores}
    """
    print("🔥 Computing OBA global importance scores...")

    model.eval()
    importance_accumulator = {}

    # 初始化重要性累加器
    for layer_idx in range(model.depth):
        for layer_name in target_layers:
            key = f'layer_{layer_idx}.{layer_name}'
            importance_accumulator[key] = []

    # 配置example inputs用于dependency graph
    example_label = torch.zeros(1, dtype=torch.long).cuda()
    example_tokens = torch.randn(1, 679, 32).cuda()
    example_inputs = (example_label, example_tokens)

    # 初始化OBA importance计算器
    importance_fn = HessianImportance(
        normalizer="max",  # 重要性归一化方式
        multivariable=True  # 多变量重要性
    )

    # 设置忽略的层（不剪枝）
    ignored_layers = [
        model.class_emb,
        model.pos_1LC,
        model.lvl_embed,
        model.head,
        model.head_nm,
    ]

    # 初始化OBA pruner（仅用于重要性计算）
    oba_pruner = OBAPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance_fn,
        global_pruning=True,
        pruning_ratio=0.0,  # 仅计算重要性，暂不剪枝
        delta=delta,
        upward_delta=upward_delta,
        downward_delta=downward_delta,
        parallel_delta=parallel_delta,
        self_unit_weight=False,
        other_unit_weight=False,
        normalizer='max',
        ignored_layers=ignored_layers,
    )

    # 注册hooks
    oba_pruner.register_hooks()

    batch_count = 0
    for batch_idx, (labels, tokens) in enumerate(calibration_loader):
        labels = labels.cuda()
        tokens = tokens.cuda()

        # VAR前向传播
        try:
            logits = model(labels, tokens[:, :-1])  # 自回归：predict下一个token

            # 计算损失（自回归语言建模损失）
            targets = tokens[:, 1:]  # shift right for next token prediction
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )

            # 使用OBA计算重要性
            oba_pruner.obtain_importance(loss)

            # 反向传播
            loss.backward()
            model.zero_grad()

            batch_count += 1
            print(f"  Processed batch {batch_count}/{len(calibration_loader)}, Loss: {loss.item():.4f}")

            # 内存管理
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ⚠️ Batch {batch_idx} failed: {e}")
            continue

    # 获取最终的重要性分数
    print("📊 Extracting importance scores from OBA...")

    final_importance = {}

    # 从OBA pruner中提取重要性分数
    for layer_idx in range(model.depth):
        block = model.blocks[layer_idx]

        for layer_name in target_layers:
            if layer_name == 'ffn.fc2':
                target_module = block.ffn.fc2
            elif layer_name == 'attn.proj':
                target_module = block.attn.proj
            else:
                continue

            # 获取该层的重要性分数（输入channel维度）
            if hasattr(target_module, 'importance') and target_module.importance is not None:
                # OBA计算的重要性分数
                importance = target_module.importance.detach().cpu()
                key = f'layer_{layer_idx}.{layer_name}'
                final_importance[key] = importance
                print(f"  ✅ {key}: {importance.shape} -> [{importance.min():.4f}, {importance.max():.4f}]")
            else:
                print(f"  ⚠️ No importance found for layer_{layer_idx}.{layer_name}")

    # 移除hooks
    oba_pruner.remove_hooks()

    return final_importance


def make_global_pruning_decision(importance_scores, target_sparsity=0.4):
    """
    OBA的关键优势：全局优化剪枝决策

    Args:
        importance_scores: 各层重要性分数
        target_sparsity: 目标稀疏度

    Returns:
        pruning_plan: {layer_key: [channel_indices_to_prune]}
    """
    print(f"🌍 Making global pruning decisions (target sparsity: {target_sparsity})")

    # 收集所有参数的重要性
    all_importance = []
    param_info = []

    for layer_key, importance in importance_scores.items():
        layer_idx = int(layer_key.split('_')[1].split('.')[0])
        layer_type = layer_key.split('.')[1] + '.' + layer_key.split('.')[2]  # 'ffn.fc2' or 'attn.proj'

        for channel_idx, imp_score in enumerate(importance):
            all_importance.append(imp_score.item())
            param_info.append({
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'channel_idx': channel_idx,
                'importance': imp_score.item()
            })

    # 全局排序（升序：移除重要性最低的）
    sorted_indices = sorted(range(len(all_importance)), key=lambda x: all_importance[x])

    # 计算剪枝预算
    total_params = len(all_importance)
    num_to_prune = int(total_params * target_sparsity)

    print(f"  📈 Total parameters: {total_params}")
    print(f"  ✂️ Parameters to prune: {num_to_prune}")

    # 选择要剪枝的参数（全局最不重要的）
    prune_indices = sorted_indices[:num_to_prune]

    # 按层分组
    pruning_plan = {}
    layer_stats = {}

    for idx in prune_indices:
        info = param_info[idx]
        layer_key = f"layer_{info['layer_idx']}.{info['layer_type']}"

        if layer_key not in pruning_plan:
            pruning_plan[layer_key] = []
            layer_stats[layer_key] = []

        pruning_plan[layer_key].append(info['channel_idx'])
        layer_stats[layer_key].append(info['importance'])

    # 打印剪枝计划
    print(f"  📋 Pruning plan:")
    for layer_key, channel_indices in pruning_plan.items():
        avg_importance = np.mean(layer_stats[layer_key])
        sparsity = len(channel_indices) / len(importance_scores[layer_key])
        print(f"    {layer_key}: {len(channel_indices)} channels ({sparsity:.1%}), avg_imp={avg_importance:.4f}")

    return pruning_plan


def execute_pruning_with_slimgpt_method(model, pruning_plan):
    """
    复用SlimGPT验证过的剪枝执行代码
    """
    print("✂️ Executing pruning using SlimGPT method...")

    for layer_plan, channel_indices in pruning_plan.items():
        layer_idx = int(layer_plan.split('_')[1].split('.')[0])
        layer_type = layer_plan.split('.')[1] + '.' + layer_plan.split('.')[2]

        print(f"  🔧 Processing {layer_plan}: {len(channel_indices)} channels")

        if layer_type == 'ffn.fc2':
            # FFN剪枝：复用SlimGPT逻辑
            target_layer = model.blocks[layer_idx].ffn.fc2     # fc2输入
            target_layer_b = model.blocks[layer_idx].ffn.fc1   # fc1输出（耦合）

            tp.prune_linear_in_channels(target_layer, channel_indices)
            tp.prune_linear_out_channels(target_layer_b, channel_indices)
            print(f"    ✅ Pruned {len(channel_indices)} channels from fc2 input and fc1 output")

        elif layer_type == 'attn.proj':
            # Attention剪枝：复用SlimGPT复杂逻辑
            execute_attention_pruning_slimgpt(model.blocks[layer_idx], channel_indices)

        else:
            print(f"    ⚠️ Unknown layer type: {layer_type}")

        # 内存清理
        torch.cuda.empty_cache()

    return model


def execute_attention_pruning_slimgpt(block, channel_indices):
    """
    复用SlimGPT的attention剪枝逻辑（lines 431-486）
    这部分代码已经验证过，直接复用
    """
    target_layer = block.attn.proj
    sparsity = len(channel_indices) / target_layer.in_features

    # 更新头数
    block.attn.num_heads = torch.round(torch.tensor(16 * (1 - sparsity))).int()

    idx_m = torch.tensor(channel_indices, dtype=torch.long)
    keep_idxs = list(set(range(target_layer.in_features)) - set(channel_indices))

    # 更新biases (SlimGPT原始代码)
    block.attn.q_bias = nn.Parameter(block.attn.q_bias.data[keep_idxs])
    zero_k_bias = block.attn.zero_k_bias.data[keep_idxs]
    block.attn.register_buffer('zero_k_bias', zero_k_bias)
    block.attn.v_bias = nn.Parameter(block.attn.v_bias.data[keep_idxs])

    # 更新scale参数 - 保留学习到的值
    head_dim = 64
    old_num_heads = target_layer.in_features // head_dim

    # 计算保留的heads
    removed_heads = set((idx_m // head_dim).tolist())
    all_heads = set(range(old_num_heads))
    keep_heads = sorted(list(all_heads - removed_heads))

    # 保留学习到的scale_mul值
    old_scale_mul = block.attn.scale_mul_1H11.data  # (1, old_num_heads, 1, 1)
    new_scale_mul = old_scale_mul[0, keep_heads, 0, 0].view(1, -1, 1, 1)

    block.attn.scale_mul_1H11 = nn.Parameter(
        new_scale_mul.clone().cuda(),
        requires_grad=True
    )

    print(f"    ✅ Preserved scale_mul for heads {keep_heads}")
    print(f"      Removed heads: {sorted(removed_heads)}")
    print(f"      New head count: {block.attn.num_heads}")

    # 剪枝proj
    tp.prune_linear_in_channels(target_layer, channel_indices)

    # 剪枝mat_qkv（耦合层）
    target_layer_b = block.attn.mat_qkv
    hidden = 16 * 64  # original embed_dim

    # 构造Q/K/V的剪枝索引
    rm_feat_q = idx_m
    rm_qkv = torch.cat([
        rm_feat_q,                    # Q
        rm_feat_q + hidden,           # K
        rm_feat_q + 2*hidden          # V
    ], dim=0)

    rm_qkv_list = torch.unique(rm_qkv.to("cpu")).sort().values.tolist()
    tp.prune_linear_out_channels(target_layer_b, rm_qkv_list)

    print(f"    ✅ Pruned {len(channel_indices)} channels ({len(channel_indices)//64} heads)")


def save_pruned_model(model, pruning_plan, importance_scores, save_path):
    """保存剪枝后的模型和统计信息"""
    print(f"💾 Saving pruned model to {save_path}")

    # 计算压缩统计
    total_params = sum(p.numel() for p in model.parameters())

    # 保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'pruning_plan': pruning_plan,
        'importance_scores': {k: v.cpu() for k, v in importance_scores.items()},
        'total_params': total_params,
        'compression_info': f"OBA pruned model with {total_params:,} parameters"
    }, save_path)

    print(f"  ✅ Model saved with {total_params:,} parameters")


def main():
    parser = argparse.ArgumentParser(description='VAR OBA Pruning')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30],
                        help='VAR model depth')
    parser.add_argument('--var_ckpt', type=str, default=None,
                        help='VAR checkpoint path')
    parser.add_argument('--vae_ckpt', type=str, default=None,
                        help='VAE checkpoint path')
    parser.add_argument('--imagenet_dir', type=str, default='/home/project/ImageNet-1K',
                        help='ImageNet dataset directory')
    parser.add_argument('--target_sparsity', type=float, default=0.4,
                        help='Target sparsity ratio')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of calibration samples')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for calibration')

    # OBA specific parameters
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Self importance weight')
    parser.add_argument('--upward_delta', type=float, default=1.0,
                        help='Upward connectivity weight')
    parser.add_argument('--downward_delta', type=float, default=1.0,
                        help='Downward connectivity weight')
    parser.add_argument('--parallel_delta', type=float, default=1.0,
                        help='Parallel connectivity weight')

    parser.add_argument('--output_path', type=str, default='./var_d16_oba_pruned_0.4sparsity.pth',
                        help='Output pruned model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置默认路径
    vae_ckpt = args.vae_ckpt if args.vae_ckpt else f'/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt if args.var_ckpt else f'/home/project/daily/AR/model_zoo/var_d{args.model_depth}.pth'

    print("🚀 VAR OBA Pruning Started")
    print(f"  Model: VAR-d{args.model_depth}")
    print(f"  VAR checkpoint: {var_ckpt}")
    print(f"  VAE checkpoint: {vae_ckpt}")
    print(f"  Target sparsity: {args.target_sparsity}")
    print(f"  OBA weights: delta={args.delta}, upward={args.upward_delta}, downward={args.downward_delta}, parallel={args.parallel_delta}")

    # 1. 加载模型
    print("\n📦 Loading models...")
    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device='cuda')
    print(f"  ✅ Models loaded successfully")

    # 2. 准备校准数据
    print(f"\n📊 Preparing calibration data ({args.num_samples} samples)...")
    calibration_loader = get_imagenet_data_for_var(
        imagenet_dir=args.imagenet_dir,
        vae=vae,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_real_images=True
    )
    print(f"  ✅ Calibration data ready: {len(calibration_loader)} batches")

    # 3. 计算OBA重要性
    target_layers = ['ffn.fc2', 'attn.proj']
    importance_scores = compute_oba_global_importance(
        model=var,
        calibration_loader=calibration_loader,
        target_layers=target_layers,
        delta=args.delta,
        upward_delta=args.upward_delta,
        downward_delta=args.downward_delta,
        parallel_delta=args.parallel_delta
    )

    # 4. 全局剪枝决策
    pruning_plan = make_global_pruning_decision(importance_scores, args.target_sparsity)

    # 5. 执行剪枝
    var_pruned = execute_pruning_with_slimgpt_method(var, pruning_plan)

    # 6. 验证和保存
    final_sparsity = check_sparsity(var_pruned)
    print(f"\n📈 Final Results:")
    print(f"  Achieved sparsity: {final_sparsity:.2%}")
    print(f"  Target sparsity: {args.target_sparsity:.2%}")

    save_pruned_model(var_pruned, pruning_plan, importance_scores, args.output_path)

    print(f"\n🎉 OBA Pruning Completed!")
    print(f"  Pruned model saved to: {args.output_path}")
    print(f"  Ready for quality testing!")


if __name__ == "__main__":
    main()