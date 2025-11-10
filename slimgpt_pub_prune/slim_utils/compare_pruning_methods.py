"""
Head剪枝 vs Head_dim剪枝误差对比实验

对比两种剪枝方法在相同稀疏度下的效果：
1. struct_prune: 删除完整的head
2. head_dim_prune: 减少每个head的维度

评估指标：
- 输出误差（MSE）
- 权重变化
- 计算时间
- 理论FLOPs
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加sobs目录到路径以导入FastOBA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sobs'))

from slimgpt import SlimGPT
from fastoba_attention_slimgpt import FastOBAAttentionSlimGPT


class SimpleMultiHeadAttention(nn.Module):
    """
    简化的Multi-Head Attention实现，用于测试head剪枝
    支持返回中间结果用于剪枝
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Q, K, V投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, return_intermediate=False):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            return_intermediate: 是否返回O矩阵的输入（用于统计Hessian）
        Returns:
            output: [batch, seq_len, hidden_size]
            attn_concat (optional): [batch, seq_len, hidden_size] - O矩阵的输入
        """
        batch_size, seq_len, _ = x.shape

        # 投影
        Q = self.q_proj(x)  # [batch, seq_len, hidden_size]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape成multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在: [batch, num_heads, seq_len, head_dim]

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # 合并heads - 这是O矩阵的输入
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_concat = attn_output.view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.out_proj(attn_concat)

        if return_intermediate:
            return output, attn_concat
        return output

    def apply_structured_pruning(self, pruned_indices, headsize):
        """
        根据O矩阵的剪枝索引，同步修改Q、K、V的输出维度

        注意：O矩阵已经被OBS算法修改过（包括权重补偿），不需要再清零！
        这个函数只负责同步清零Q、K、V的对应输出维度。

        Args:
            pruned_indices: 被删除的列索引（O矩阵的输入维度）
            headsize: head的维度
        """
        # pruned_indices是O矩阵输入维度的索引，对应Q、K、V的输出维度

        # 只修改Q、K、V的输出维度（行）
        # O矩阵已经被struct_prune/head_dim_prune修改过了，不要再动它！
        with torch.no_grad():
            self.q_proj.weight.data[pruned_indices, :] = 0
            self.k_proj.weight.data[pruned_indices, :] = 0
            self.v_proj.weight.data[pruned_indices, :] = 0

            if self.q_proj.bias is not None:
                self.q_proj.bias.data[pruned_indices] = 0
            if self.k_proj.bias is not None:
                self.k_proj.bias.data[pruned_indices] = 0
            if self.v_proj.bias is not None:
                self.v_proj.bias.data[pruned_indices] = 0


class ComparisonExperiment:
    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        batch_size=32,
        seq_len=128,
        num_batches=20,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化对比实验

        Args:
            hidden_size: 隐藏层维度
            num_heads: head数量
            batch_size: batch大小
            seq_len: 序列长度
            num_batches: 用于统计Hessian的batch数
            device: 设备
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.device = device

        print(f"实验配置:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {self.head_dim}")
        print(f"  Device: {device}")
        print()

    def create_test_data(self):
        """创建测试数据"""
        # 用于统计Hessian的数据
        train_data = []
        for _ in range(self.num_batches):
            inp = torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_size,
                device=self.device
            )
            train_data.append(inp)

        # 用于评估的测试数据（固定）
        torch.manual_seed(42)
        test_data = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_size,
            device=self.device
        )

        return train_data, test_data

    def compute_output_error(self, layer_original, layer_pruned, test_data):
        """
        计算输出误差

        Args:
            layer_original: 原始层
            layer_pruned: 剪枝后的层
            test_data: 测试数据

        Returns:
            相对MSE误差
        """
        with torch.no_grad():
            out_original = layer_original(test_data)
            out_pruned = layer_pruned(test_data)

            mse = torch.mean((out_original - out_pruned) ** 2).item()
            output_norm = torch.mean(out_original ** 2).item()

            relative_error = mse / output_norm if output_norm > 0 else float('inf')

        return mse, relative_error

    def compute_weight_change(self, weight_original, weight_pruned):
        """
        计算权重变化

        Returns:
            Frobenius范数相对变化
        """
        diff = weight_original - weight_pruned
        diff_norm = torch.norm(diff, p='fro').item()
        original_norm = torch.norm(weight_original, p='fro').item()

        relative_change = diff_norm / original_norm if original_norm > 0 else 0

        return diff_norm, relative_change

    def compute_flops(self, in_features, out_features, seq_len, batch_size):
        """
        计算理论FLOPs（浮点运算数）

        对于线性层: FLOPs = 2 * batch_size * seq_len * in_features * out_features
        （每次乘加算2次运算）
        """
        flops = 2 * batch_size * seq_len * in_features * out_features
        return flops

    def run_comparison(self, sparsity=0.25, percdamp=0.01, test_type='linear', use_fastoba=False):
        """
        运行对比实验

        Args:
            sparsity: 稀疏度
            percdamp: 阻尼系数
            test_type: 'linear' 或 'attention' - 测试线性层还是attention层
            use_fastoba: 是否使用FastOBA方法进行对比

        Returns:
            results: 包含所有对比结果的字典
        """
        print("=" * 80)
        method_str = " + FastOBA" if use_fastoba else ""
        print(f"开始对比实验{method_str}（稀疏度: {sparsity:.1%}, 测试类型: {test_type}）")
        print("=" * 80)
        print()

        # 创建测试数据
        print("生成测试数据...")
        train_data, test_data = self.create_test_data()
        print(f"  训练数据: {self.num_batches} batches")
        print(f"  测试数据: {test_data.shape}")
        print()

        # 根据是否使用FastOBA选择对应的方法
        if use_fastoba:
            if test_type == 'attention':
                return self._run_attention_comparison_with_fastoba(train_data, test_data, sparsity, percdamp)
            elif test_type == 'linear':
                # 线性层暂时使用原有方法（FastOBA主要针对Attention）
                print("注意: 线性层使用原有SlimGPT对比方法")
                return self._run_linear_comparison(train_data, test_data, sparsity, percdamp)
            else:
                raise ValueError(f"Unknown test_type: {test_type}")
        else:
            # 原有逻辑
            if test_type == 'linear':
                return self._run_linear_comparison(train_data, test_data, sparsity, percdamp)
            elif test_type == 'attention':
                return self._run_attention_comparison(train_data, test_data, sparsity, percdamp)
            else:
                raise ValueError(f"Unknown test_type: {test_type}")

    def _run_linear_comparison(self, train_data, test_data, sparsity, percdamp):
        """对线性层进行对比测试"""
        # ========== 方法1: Head剪枝 (struct_prune) ==========
        print("=" * 80)
        print("方法1: Head剪枝 (struct_prune) - 线性层")
        print("=" * 80)

        # 创建层（方法1）
        layer_head_prune = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        weight_original_head = layer_head_prune.weight.data.clone()

        # 计算原始输出
        with torch.no_grad():
            output_original_head = layer_head_prune(test_data)

        # 创建pruner
        class Args:
            no_compensate = False
        args = Args()

        pruner_head = SlimGPT(layer_head_prune, layer_idx=0, args=args)

        # 统计Hessian
        print("统计Hessian矩阵...")
        for inp in train_data:
            out = layer_head_prune(inp)
            pruner_head.add_batch(inp, out)
        print("  完成！")
        print()

        # 执行剪枝
        print("执行head剪枝...")
        start_time = time.time()
        pruned_indices_head = pruner_head.struct_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp,
            layer_idx=0
        )
        time_head_prune = time.time() - start_time

        # 计算结果
        num_heads_removed = len(pruned_indices_head) // self.head_dim
        remaining_heads = self.num_heads - num_heads_removed

        mse_head, rel_error_head = self.compute_output_error(
            layer_head_prune, layer_head_prune, test_data
        )

        # 注意：这里比较的是剪枝后的输出和原始输出
        with torch.no_grad():
            output_pruned_head = layer_head_prune(test_data)
            mse_head = torch.mean((output_original_head - output_pruned_head) ** 2).item()
            output_norm_head = torch.mean(output_original_head ** 2).item()
            rel_error_head = mse_head / output_norm_head

        weight_change_head, rel_weight_change_head = self.compute_weight_change(
            weight_original_head, layer_head_prune.weight.data
        )

        # FLOPs计算（考虑剪枝后的维度）
        effective_dim_head = remaining_heads * self.head_dim
        flops_original = self.compute_flops(
            self.hidden_size, self.hidden_size, self.seq_len, self.batch_size
        )
        flops_head = self.compute_flops(
            self.hidden_size, effective_dim_head, self.seq_len, self.batch_size
        )
        flops_reduction_head = (flops_original - flops_head) / flops_original

        print(f"  删除head数: {num_heads_removed}")
        print(f"  剩余head数: {remaining_heads}")
        print(f"  有效维度: {effective_dim_head}")
        print(f"  输出MSE: {mse_head:.6e}")
        print(f"  输出相对误差: {rel_error_head:.6e}")
        print(f"  权重变化: {weight_change_head:.6f} (相对: {rel_weight_change_head:.2%})")
        print(f"  剪枝时间: {time_head_prune:.3f}s")
        print(f"  FLOPs减少: {flops_reduction_head:.2%}")
        print()

        # ========== 方法2: Head_dim剪枝 ==========
        print("=" * 80)
        print("方法2: Head维度剪枝 (head_dim_prune)")
        print("=" * 80)

        # 创建层（方法2）- 使用相同的初始权重
        layer_dim_prune = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        # 使用方法1保存的原始权重，确保两个方法从相同的权重开始
        layer_dim_prune.weight.data = weight_original_head.clone()
        if layer_head_prune.bias is not None:
            layer_dim_prune.bias.data = layer_head_prune.bias.data.clone()

        weight_original_dim = layer_dim_prune.weight.data.clone()

        # 计算原始输出
        with torch.no_grad():
            output_original_dim = layer_dim_prune(test_data)

        # 创建pruner
        pruner_dim = SlimGPT(layer_dim_prune, layer_idx=1, args=args)

        # 统计Hessian（使用相同的数据）
        print("统计Hessian矩阵...")
        for inp in train_data:
            out = layer_dim_prune(inp)
            pruner_dim.add_batch(inp, out)
        print("  完成！")
        print()

        # 执行剪枝
        print("执行head_dim剪枝...")
        start_time = time.time()
        pruned_indices_dim = pruner_dim.head_dim_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp,
            layer_idx=1
        )
        time_dim_prune = time.time() - start_time

        # 计算结果
        dims_removed_per_head = len(pruned_indices_dim) // self.num_heads
        remaining_dim_per_head = self.head_dim - dims_removed_per_head

        with torch.no_grad():
            output_pruned_dim = layer_dim_prune(test_data)
            mse_dim = torch.mean((output_original_dim - output_pruned_dim) ** 2).item()
            output_norm_dim = torch.mean(output_original_dim ** 2).item()
            rel_error_dim = mse_dim / output_norm_dim

        weight_change_dim, rel_weight_change_dim = self.compute_weight_change(
            weight_original_dim, layer_dim_prune.weight.data
        )

        # FLOPs计算
        effective_dim_dim = self.num_heads * remaining_dim_per_head
        flops_dim = self.compute_flops(
            self.hidden_size, effective_dim_dim, self.seq_len, self.batch_size
        )
        flops_reduction_dim = (flops_original - flops_dim) / flops_original

        print(f"  每head删除维度: {dims_removed_per_head}")
        print(f"  每head剩余维度: {remaining_dim_per_head}")
        print(f"  保持head数: {self.num_heads}")
        print(f"  有效维度: {effective_dim_dim}")
        print(f"  输出MSE: {mse_dim:.6e}")
        print(f"  输出相对误差: {rel_error_dim:.6e}")
        print(f"  权重变化: {weight_change_dim:.6f} (相对: {rel_weight_change_dim:.2%})")
        print(f"  剪枝时间: {time_dim_prune:.3f}s")
        print(f"  FLOPs减少: {flops_reduction_dim:.2%}")
        print()

        # ========== 对比总结 ==========
        print("=" * 80)
        print("对比总结")
        print("=" * 80)
        print()

        # 创建对比表
        print(f"{'指标':<25} {'Head剪枝':<25} {'Head_dim剪枝':<25} {'优势':<15}")
        print("-" * 90)

        # 输出误差
        winner_mse = "Head_dim ✓" if mse_dim < mse_head else "Head ✓"
        print(f"{'输出MSE':<25} {mse_head:<25.6e} {mse_dim:<25.6e} {winner_mse:<15}")

        winner_rel = "Head_dim ✓" if rel_error_dim < rel_error_head else "Head ✓"
        print(f"{'输出相对误差':<25} {rel_error_head:<25.6e} {rel_error_dim:<25.6e} {winner_rel:<15}")

        # 权重变化
        winner_weight = "Head_dim ✓" if weight_change_dim < weight_change_head else "Head ✓"
        print(f"{'权重变化':<25} {weight_change_head:<25.6f} {weight_change_dim:<25.6f} {winner_weight:<15}")

        # 时间
        winner_time = "Head_dim ✓" if time_dim_prune < time_head_prune else "Head ✓"
        print(f"{'剪枝时间(s)':<25} {time_head_prune:<25.3f} {time_dim_prune:<25.3f} {winner_time:<15}")

        # FLOPs
        print(f"{'FLOPs减少':<25} {flops_reduction_head:<25.2%} {flops_reduction_dim:<25.2%} {'相同':<15}")

        # 结构
        structure_head = f"{remaining_heads} heads × {self.head_dim} dim"
        structure_dim = f"{self.num_heads} heads × {remaining_dim_per_head} dim"
        print(f"{'最终结构':<25} {structure_head:<25} {structure_dim:<25} {'-':<15}")

        print()

        # 相对性能提升
        if rel_error_head > 0:
            improvement = (rel_error_head - rel_error_dim) / rel_error_head * 100
            print(f"Head_dim剪枝相对误差改进: {improvement:+.2f}%")

        if time_head_prune > 0:
            speedup = time_head_prune / time_dim_prune
            print(f"Head_dim剪枝速度提升: {speedup:.2f}x")

        print()

        # 返回结果
        results = {
            'sparsity': sparsity,
            'head_prune': {
                'mse': mse_head,
                'relative_error': rel_error_head,
                'weight_change': weight_change_head,
                'time': time_head_prune,
                'flops_reduction': flops_reduction_head,
                'structure': structure_head,
                'remaining_heads': remaining_heads,
            },
            'head_dim_prune': {
                'mse': mse_dim,
                'relative_error': rel_error_dim,
                'weight_change': weight_change_dim,
                'time': time_dim_prune,
                'flops_reduction': flops_reduction_dim,
                'structure': structure_dim,
                'remaining_dim_per_head': remaining_dim_per_head,
            }
        }

        return results

    def _run_attention_comparison(self, train_data, test_data, sparsity, percdamp):
        """对attention层进行对比测试 - 基于O矩阵的输入维度剪枝"""
        print("=" * 80)
        print("方法1: Head剪枝 (struct_prune) - 基于O矩阵输入维度")
        print("=" * 80)

        # 创建attention层
        attn_head = SimpleMultiHeadAttention(self.hidden_size, self.num_heads).to(self.device)
        attn_dim = SimpleMultiHeadAttention(self.hidden_size, self.num_heads).to(self.device)

        # 复制权重，确保两个模型从相同状态开始
        attn_dim.load_state_dict(attn_head.state_dict())

        # 保存原始O矩阵权重
        weight_original_head = attn_head.out_proj.weight.data.clone()
        weight_original_dim = attn_dim.out_proj.weight.data.clone()

        # 计算原始输出
        with torch.no_grad():
            output_original_head = attn_head(test_data)
            output_original_dim = attn_dim(test_data)

        # 创建Args对象
        class Args:
            no_compensate = False
        args = Args()

        # ========== 方法1: Head剪枝 ==========
        print("\n对O矩阵的输入维度应用Head剪枝...")

        pruner_head = SlimGPT(attn_head.out_proj, layer_idx=0, args=args)

        # 统计Hessian - 使用O矩阵的输入（attention concat的输出）
        print("统计Hessian矩阵...")
        for inp in train_data:
            # 计算attention的中间输出（O矩阵的输入）
            _, attn_concat = attn_head(inp, return_intermediate=True)
            # 将batch和seq维度合并
            attn_concat_flat = attn_concat.view(-1, self.hidden_size)
            out = attn_head.out_proj(attn_concat_flat)
            pruner_head.add_batch(attn_concat_flat, out)
        print("  完成！")

        # 执行剪枝 - 决定O矩阵输入维度的剪枝
        start_time = time.time()
        pruned_indices_head = pruner_head.struct_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp,
            layer_idx=0
        )
        time_head_prune = time.time() - start_time

        # 同步应用剪枝到Q、K、V、O
        attn_head.apply_structured_pruning(pruned_indices_head, self.head_dim)

        # 计算结果
        num_heads_removed = len(pruned_indices_head) // self.head_dim
        remaining_heads = self.num_heads - num_heads_removed

        # 计算剪枝后整个attention的输出误差
        with torch.no_grad():
            output_pruned_head = attn_head(test_data)
            mse_head = torch.mean((output_original_head - output_pruned_head) ** 2).item()
            output_norm_head = torch.mean(output_original_head ** 2).item()
            rel_error_head = mse_head / output_norm_head

        weight_change_head, rel_weight_change_head = self.compute_weight_change(
            weight_original_head, attn_head.out_proj.weight.data
        )

        effective_dim_head = remaining_heads * self.head_dim
        flops_original = self.compute_flops(
            self.hidden_size, self.hidden_size, self.seq_len, self.batch_size
        )
        flops_head = self.compute_flops(
            self.hidden_size, effective_dim_head, self.seq_len, self.batch_size
        )
        flops_reduction_head = (flops_original - flops_head) / flops_original

        print(f"\n  删除head数: {num_heads_removed}")
        print(f"  剩余head数: {remaining_heads}")
        print(f"  有效维度: {effective_dim_head}")
        print(f"  Attention输出MSE: {mse_head:.6e}")
        print(f"  Attention输出相对误差: {rel_error_head:.6e}")
        print(f"  O矩阵权重变化: {weight_change_head:.6f} (相对: {rel_weight_change_head:.2%})")
        print(f"  剪枝时间: {time_head_prune:.3f}s")
        print(f"  FLOPs减少: {flops_reduction_head:.2%}")

        # ========== 方法2: Head_dim剪枝 ==========
        print("\n" + "=" * 80)
        print("方法2: Head维度剪枝 (head_dim_prune) - 基于O矩阵输入维度")
        print("=" * 80)

        pruner_dim = SlimGPT(attn_dim.out_proj, layer_idx=1, args=args)

        # 统计Hessian
        print("统计Hessian矩阵...")
        for inp in train_data:
            _, attn_concat = attn_dim(inp, return_intermediate=True)
            attn_concat_flat = attn_concat.view(-1, self.hidden_size)
            out = attn_dim.out_proj(attn_concat_flat)
            pruner_dim.add_batch(attn_concat_flat, out)
        print("  完成！")

        # 执行剪枝
        start_time = time.time()
        pruned_indices_dim = pruner_dim.head_dim_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp,
            layer_idx=1
        )
        time_dim_prune = time.time() - start_time

        # 同步应用剪枝到Q、K、V、O
        attn_dim.apply_structured_pruning(pruned_indices_dim, self.head_dim)

        # 计算结果
        dims_removed_per_head = len(pruned_indices_dim) // self.num_heads
        remaining_dim_per_head = self.head_dim - dims_removed_per_head

        # 计算剪枝后整个attention的输出误差
        with torch.no_grad():
            output_pruned_dim = attn_dim(test_data)
            mse_dim = torch.mean((output_original_dim - output_pruned_dim) ** 2).item()
            output_norm_dim = torch.mean(output_original_dim ** 2).item()
            rel_error_dim = mse_dim / output_norm_dim

        weight_change_dim, rel_weight_change_dim = self.compute_weight_change(
            weight_original_dim, attn_dim.out_proj.weight.data
        )

        effective_dim_dim = self.num_heads * remaining_dim_per_head
        flops_dim = self.compute_flops(
            self.hidden_size, effective_dim_dim, self.seq_len, self.batch_size
        )
        flops_reduction_dim = (flops_original - flops_dim) / flops_original

        print(f"\n  每head删除维度: {dims_removed_per_head}")
        print(f"  每head剩余维度: {remaining_dim_per_head}")
        print(f"  保持head数: {self.num_heads}")
        print(f"  有效维度: {effective_dim_dim}")
        print(f"  Attention输出MSE: {mse_dim:.6e}")
        print(f"  Attention输出相对误差: {rel_error_dim:.6e}")
        print(f"  O矩阵权重变化: {weight_change_dim:.6f} (相对: {rel_weight_change_dim:.2%})")
        print(f"  剪枝时间: {time_dim_prune:.3f}s")
        print(f"  FLOPs减少: {flops_reduction_dim:.2%}")

        # ========== 对比总结 ==========
        print("\n" + "=" * 80)
        print("对比总结 (Attention层 - 基于O矩阵输入维度剪枝)")
        print("=" * 80)
        print()

        print(f"{'指标':<25} {'Head剪枝':<25} {'Head_dim剪枝':<25} {'优势':<15}")
        print("-" * 90)

        winner_mse = "Head_dim ✓" if mse_dim < mse_head else "Head ✓"
        print(f"{'Attention输出MSE':<25} {mse_head:<25.6e} {mse_dim:<25.6e} {winner_mse:<15}")

        winner_rel = "Head_dim ✓" if rel_error_dim < rel_error_head else "Head ✓"
        print(f"{'Attention相对误差':<25} {rel_error_head:<25.6e} {rel_error_dim:<25.6e} {winner_rel:<15}")

        winner_weight = "Head_dim ✓" if weight_change_dim < weight_change_head else "Head ✓"
        print(f"{'O矩阵权重变化':<25} {weight_change_head:<25.6f} {weight_change_dim:<25.6f} {winner_weight:<15}")

        winner_time = "Head_dim ✓" if time_dim_prune < time_head_prune else "Head ✓"
        print(f"{'剪枝时间(s)':<25} {time_head_prune:<25.3f} {time_dim_prune:<25.3f} {winner_time:<15}")

        print(f"{'FLOPs减少':<25} {flops_reduction_head:<25.2%} {flops_reduction_dim:<25.2%} {'相同':<15}")

        structure_head = f"{remaining_heads} heads × {self.head_dim} dim"
        structure_dim = f"{self.num_heads} heads × {remaining_dim_per_head} dim"
        print(f"{'最终结构':<25} {structure_head:<25} {structure_dim:<25} {'-':<15}")

        print()
        print("说明: 剪枝O矩阵输入维度后，同步修改了Q、K、V的输出维度")
        print()

        if rel_error_head > 0:
            improvement = (rel_error_head - rel_error_dim) / rel_error_head * 100
            print(f"Head_dim剪枝相对误差改进: {improvement:+.2f}%")

        if time_head_prune > 0:
            speedup = time_head_prune / time_dim_prune
            print(f"Head_dim剪枝速度提升: {speedup:.2f}x")

        print()

        # 返回结果
        results = {
            'sparsity': sparsity,
            'test_type': 'attention',
            'head_prune': {
                'mse': mse_head,
                'relative_error': rel_error_head,
                'weight_change': weight_change_head,
                'time': time_head_prune,
                'flops_reduction': flops_reduction_head,
                'structure': structure_head,
                'remaining_heads': remaining_heads,
            },
            'head_dim_prune': {
                'mse': mse_dim,
                'relative_error': rel_error_dim,
                'weight_change': weight_change_dim,
                'time': time_dim_prune,
                'flops_reduction': flops_reduction_dim,
                'structure': structure_dim,
                'remaining_dim_per_head': remaining_dim_per_head,
            }
        }

        return results

    def _run_attention_comparison_with_fastoba(self, train_data, test_data, sparsity, percdamp):
        """
        对attention层进行对比测试 - 添加FastOBA方法

        对比4种组合：
        1. SlimGPT + Head-wise pruning
        2. SlimGPT + Head-dim pruning
        3. FastOBA + Head-wise pruning
        4. FastOBA + Head-dim pruning
        """
        print("=" * 80)
        print("Attention层对比: SlimGPT vs FastOBA (2种Hessian × 2种策略 = 4个实验)")
        print("=" * 80)
        print()

        # 创建4个独立的attention实例（相同初始权重）
        attn_slimgpt_head = SimpleMultiHeadAttention(self.hidden_size, self.num_heads).to(self.device)
        attn_slimgpt_headdim = SimpleMultiHeadAttention(self.hidden_size, self.num_heads).to(self.device)
        attn_fastoba_head = SimpleMultiHeadAttention(self.hidden_size, self.num_heads).to(self.device)
        attn_fastoba_headdim = SimpleMultiHeadAttention(self.hidden_size, self.num_heads).to(self.device)

        # 确保所有模型从相同权重开始
        base_state = attn_slimgpt_head.state_dict()
        attn_slimgpt_headdim.load_state_dict(base_state)
        attn_fastoba_head.load_state_dict(base_state)
        attn_fastoba_headdim.load_state_dict(base_state)

        # 保存原始权重和输出
        weight_original = attn_slimgpt_head.out_proj.weight.data.clone()
        with torch.no_grad():
            output_original = attn_slimgpt_head(test_data)

        # Args配置
        class Args:
            no_compensate = False
        args = Args()

        # ========== 实验1: SlimGPT + Head-wise ==========
        print("\n" + "=" * 80)
        print("实验1: SlimGPT + Head-wise Pruning (H = XX^T)")
        print("=" * 80)

        pruner_slimgpt_head = SlimGPT(attn_slimgpt_head.out_proj, layer_idx=0, args=args)

        print("统计Hessian (XX^T方法)...")
        for inp in train_data:
            _, attn_concat = attn_slimgpt_head(inp, return_intermediate=True)
            attn_concat_flat = attn_concat.view(-1, self.hidden_size)
            out = attn_slimgpt_head.out_proj(attn_concat_flat)
            pruner_slimgpt_head.add_batch(attn_concat_flat, out)
        print("  完成！")

        start_time = time.time()
        pruned_indices_slimgpt_head = pruner_slimgpt_head.struct_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp,
            layer_idx=0
        )
        time_slimgpt_head = time.time() - start_time

        attn_slimgpt_head.apply_structured_pruning(pruned_indices_slimgpt_head, self.head_dim)

        with torch.no_grad():
            output_slimgpt_head = attn_slimgpt_head(test_data)
            mse_slimgpt_head = torch.mean((output_original - output_slimgpt_head) ** 2).item()
            output_norm = torch.mean(output_original ** 2).item()
            rel_error_slimgpt_head = mse_slimgpt_head / output_norm

        weight_change_slimgpt_head, rel_weight_slimgpt_head = self.compute_weight_change(
            weight_original, attn_slimgpt_head.out_proj.weight.data
        )

        num_heads_removed_slimgpt = len(pruned_indices_slimgpt_head) // self.head_dim
        remaining_heads_slimgpt = self.num_heads - num_heads_removed_slimgpt

        print(f"  删除head数: {num_heads_removed_slimgpt}")
        print(f"  剩余head数: {remaining_heads_slimgpt}")
        print(f"  输出MSE: {mse_slimgpt_head:.6e}")
        print(f"  相对误差: {rel_error_slimgpt_head:.6e}")
        print(f"  时间: {time_slimgpt_head:.3f}s")

        # ========== 实验2: SlimGPT + Head-dim ==========
        print("\n" + "=" * 80)
        print("实验2: SlimGPT + Head-dim Pruning (H = XX^T)")
        print("=" * 80)

        pruner_slimgpt_headdim = SlimGPT(attn_slimgpt_headdim.out_proj, layer_idx=1, args=args)

        print("统计Hessian (XX^T方法)...")
        for inp in train_data:
            _, attn_concat = attn_slimgpt_headdim(inp, return_intermediate=True)
            attn_concat_flat = attn_concat.view(-1, self.hidden_size)
            out = attn_slimgpt_headdim.out_proj(attn_concat_flat)
            pruner_slimgpt_headdim.add_batch(attn_concat_flat, out)
        print("  完成！")

        start_time = time.time()
        pruned_indices_slimgpt_headdim = pruner_slimgpt_headdim.head_dim_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp,
            layer_idx=1
        )
        time_slimgpt_headdim = time.time() - start_time

        attn_slimgpt_headdim.apply_structured_pruning(pruned_indices_slimgpt_headdim, self.head_dim)

        with torch.no_grad():
            output_slimgpt_headdim = attn_slimgpt_headdim(test_data)
            mse_slimgpt_headdim = torch.mean((output_original - output_slimgpt_headdim) ** 2).item()
            rel_error_slimgpt_headdim = mse_slimgpt_headdim / output_norm

        weight_change_slimgpt_headdim, rel_weight_slimgpt_headdim = self.compute_weight_change(
            weight_original, attn_slimgpt_headdim.out_proj.weight.data
        )

        dims_removed_slimgpt = len(pruned_indices_slimgpt_headdim) // self.num_heads
        remaining_dim_slimgpt = self.head_dim - dims_removed_slimgpt

        print(f"  每head删除维度: {dims_removed_slimgpt}")
        print(f"  每head剩余维度: {remaining_dim_slimgpt}")
        print(f"  输出MSE: {mse_slimgpt_headdim:.6e}")
        print(f"  相对误差: {rel_error_slimgpt_headdim:.6e}")
        print(f"  时间: {time_slimgpt_headdim:.3f}s")

        # ========== 实验3: FastOBA + Head-wise ==========
        print("\n" + "=" * 80)
        print("实验3: FastOBA + Head-wise Pruning (精确Hessian - 标准基向量法)")
        print("=" * 80)

        pruner_fastoba_head = FastOBAAttentionSlimGPT(
            attention_module=attn_fastoba_head,
            layer_idx=2,
            num_heads=self.num_heads,
            embed_dim=self.hidden_size,
            hessian_mode='block_diagonal',
            head_importance_mode='block_mean',
            use_compensation=True,
            fastoba_order=2,
            fastoba_delta=1.0,
            hessian_accumulate_freq=self.num_batches,  # 与batch数匹配
            use_exact_hessian=True,  # 【新增】使用精确Hessian计算
            debug=False
        )

        print("统计Hessian (FastOBA自动微分)...")
        for inp in train_data:
            out = attn_fastoba_head(inp)
            pruner_fastoba_head.add_batch(inp, out)
        print("  完成！")

        start_time = time.time()
        pruned_indices_fastoba_head = pruner_fastoba_head.struct_prune(
            sparsity=sparsity,
            headsize=self.head_dim,
            percdamp=percdamp
        )
        time_fastoba_head = time.time() - start_time

        attn_fastoba_head.apply_structured_pruning(pruned_indices_fastoba_head, self.head_dim)

        with torch.no_grad():
            output_fastoba_head = attn_fastoba_head(test_data)
            mse_fastoba_head = torch.mean((output_original - output_fastoba_head) ** 2).item()
            rel_error_fastoba_head = mse_fastoba_head / output_norm

        weight_change_fastoba_head, rel_weight_fastoba_head = self.compute_weight_change(
            weight_original, attn_fastoba_head.out_proj.weight.data
        )

        num_heads_removed_fastoba = len(pruned_indices_fastoba_head) // self.head_dim
        remaining_heads_fastoba = self.num_heads - num_heads_removed_fastoba

        print(f"  删除head数: {num_heads_removed_fastoba}")
        print(f"  剩余head数: {remaining_heads_fastoba}")
        print(f"  输出MSE: {mse_fastoba_head:.6e}")
        print(f"  相对误差: {rel_error_fastoba_head:.6e}")
        print(f"  时间: {time_fastoba_head:.3f}s")

        # ========== 实验4: FastOBA + Head-dim ==========
        print("\n" + "=" * 80)
        print("实验4: FastOBA + Head-dim Pruning (真实Hessian)")
        print("=" * 80)

        pruner_fastoba_headdim = FastOBAAttentionSlimGPT(
            attention_module=attn_fastoba_headdim,
            layer_idx=3,
            num_heads=self.num_heads,
            embed_dim=self.hidden_size,
            hessian_mode='block_diagonal',
            head_importance_mode='block_mean',
            use_compensation=True,
            fastoba_order=2,
            fastoba_delta=1.0,
            hessian_accumulate_freq=self.num_batches,
            use_exact_hessian=True,  # 【新增】使用精确Hessian计算
            debug=False
        )

        print("统计Hessian (FastOBA自动微分)...")
        for inp in train_data:
            out = attn_fastoba_headdim(inp)
            pruner_fastoba_headdim.add_batch(inp, out)
        print("  完成！")

        start_time = time.time()
        pruned_indices_fastoba_headdim = pruner_fastoba_headdim.struct_prune_head_dims(
            sparsity=sparsity,
            percdamp=percdamp
        )
        time_fastoba_headdim = time.time() - start_time

        attn_fastoba_headdim.apply_structured_pruning(pruned_indices_fastoba_headdim, self.head_dim)

        with torch.no_grad():
            output_fastoba_headdim = attn_fastoba_headdim(test_data)
            mse_fastoba_headdim = torch.mean((output_original - output_fastoba_headdim) ** 2).item()
            rel_error_fastoba_headdim = mse_fastoba_headdim / output_norm

        weight_change_fastoba_headdim, rel_weight_fastoba_headdim = self.compute_weight_change(
            weight_original, attn_fastoba_headdim.out_proj.weight.data
        )

        dims_removed_fastoba = len(pruned_indices_fastoba_headdim) // self.num_heads
        remaining_dim_fastoba = self.head_dim - dims_removed_fastoba

        print(f"  每head删除维度: {dims_removed_fastoba}")
        print(f"  每head剩余维度: {remaining_dim_fastoba}")
        print(f"  输出MSE: {mse_fastoba_headdim:.6e}")
        print(f"  相对误差: {rel_error_fastoba_headdim:.6e}")
        print(f"  时间: {time_fastoba_headdim:.3f}s")

        # ========== 对比总结 ==========
        print("\n" + "=" * 80)
        print("对比总结: SlimGPT vs FastOBA (Attention层)")
        print("=" * 80)
        print()

        print(f"{'方法':<30} {'输出MSE':<15} {'相对误差':<15} {'时间(s)':<10} {'优势':<10}")
        print("-" * 80)
        print(f"{'SlimGPT Head-wise':<30} {mse_slimgpt_head:<15.6e} {rel_error_slimgpt_head:<15.6e} {time_slimgpt_head:<10.3f} {'':<10}")
        print(f"{'SlimGPT Head-dim':<30} {mse_slimgpt_headdim:<15.6e} {rel_error_slimgpt_headdim:<15.6e} {time_slimgpt_headdim:<10.3f} {'':<10}")
        print(f"{'FastOBA Head-wise':<30} {mse_fastoba_head:<15.6e} {rel_error_fastoba_head:<15.6e} {time_fastoba_head:<10.3f} {'':<10}")
        print(f"{'FastOBA Head-dim':<30} {mse_fastoba_headdim:<15.6e} {rel_error_fastoba_headdim:<15.6e} {time_fastoba_headdim:<10.3f} {'':<10}")
        print()

        # 找出最佳方法
        errors = [rel_error_slimgpt_head, rel_error_slimgpt_headdim, rel_error_fastoba_head, rel_error_fastoba_headdim]
        methods = ['SlimGPT Head-wise', 'SlimGPT Head-dim', 'FastOBA Head-wise', 'FastOBA Head-dim']
        best_idx = errors.index(min(errors))
        print(f"最佳方法（最低误差）: {methods[best_idx]} (相对误差: {errors[best_idx]:.6e})")
        print()

        # 计算FlOPs
        flops_original = self.compute_flops(
            self.hidden_size, self.hidden_size, self.seq_len, self.batch_size
        )

        # 返回结果
        results = {
            'sparsity': sparsity,
            'test_type': 'attention_fastoba',
            'slimgpt_head': {
                'mse': mse_slimgpt_head,
                'relative_error': rel_error_slimgpt_head,
                'weight_change': weight_change_slimgpt_head,
                'time': time_slimgpt_head,
                'remaining_heads': remaining_heads_slimgpt,
                'structure': f"{remaining_heads_slimgpt}h×{self.head_dim}d"
            },
            'slimgpt_headdim': {
                'mse': mse_slimgpt_headdim,
                'relative_error': rel_error_slimgpt_headdim,
                'weight_change': weight_change_slimgpt_headdim,
                'time': time_slimgpt_headdim,
                'remaining_dim_per_head': remaining_dim_slimgpt,
                'structure': f"{self.num_heads}h×{remaining_dim_slimgpt}d"
            },
            'fastoba_head': {
                'mse': mse_fastoba_head,
                'relative_error': rel_error_fastoba_head,
                'weight_change': weight_change_fastoba_head,
                'time': time_fastoba_head,
                'remaining_heads': remaining_heads_fastoba,
                'structure': f"{remaining_heads_fastoba}h×{self.head_dim}d"
            },
            'fastoba_headdim': {
                'mse': mse_fastoba_headdim,
                'relative_error': rel_error_fastoba_headdim,
                'weight_change': weight_change_fastoba_headdim,
                'time': time_fastoba_headdim,
                'remaining_dim_per_head': remaining_dim_fastoba,
                'structure': f"{self.num_heads}h×{remaining_dim_fastoba}d"
            }
        }

        return results

    def run_multi_sparsity_comparison(self, sparsity_levels=[0.125, 0.25, 0.375, 0.5], test_type='linear', use_fastoba=False):
        """
        在多个稀疏度下运行对比实验

        Args:
            sparsity_levels: 稀疏度列表
            test_type: 'linear' 或 'attention'
            use_fastoba: 是否使用FastOBA方法进行对比

        Returns:
            all_results: 所有结果的列表
        """
        all_results = []

        for sparsity in sparsity_levels:
            results = self.run_comparison(sparsity=sparsity, test_type=test_type, use_fastoba=use_fastoba)
            all_results.append(results)
            print("\n" + "=" * 80 + "\n")

        # 绘制对比图
        self.plot_comparison(all_results, test_type=test_type, use_fastoba=use_fastoba)

        return all_results

    def plot_comparison(self, all_results, test_type='linear', use_fastoba=False):
        """
        绘制对比图表

        Args:
            all_results: 所有稀疏度下的结果
            test_type: 测试类型，用于标题
            use_fastoba: 是否包含FastOBA方法
        """
        sparsities = [r['sparsity'] for r in all_results]

        if use_fastoba and test_type == 'attention':
            # FastOBA模式：4种方法对比
            self._plot_fastoba_comparison(all_results, sparsities)
        else:
            # 原有模式：2种方法对比
            self._plot_original_comparison(all_results, sparsities, test_type)

    def _plot_original_comparison(self, all_results, sparsities, test_type):
        """原有的2种方法对比图"""
        # 提取数据
        mse_head = [r['head_prune']['mse'] for r in all_results]
        mse_dim = [r['head_dim_prune']['mse'] for r in all_results]

        rel_error_head = [r['head_prune']['relative_error'] for r in all_results]
        rel_error_dim = [r['head_dim_prune']['relative_error'] for r in all_results]

        time_head = [r['head_prune']['time'] for r in all_results]
        time_dim = [r['head_dim_prune']['time'] for r in all_results]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title_prefix = 'Attention Layer' if test_type == 'attention' else 'Linear Layer'
        fig.suptitle(f'Head Pruning vs Head_dim Pruning Comparison ({title_prefix})',
                     fontsize=16, fontweight='bold')

        # 图1: 输出MSE
        ax1 = axes[0, 0]
        ax1.plot(sparsities, mse_head, 'o-', label='Head Pruning', linewidth=2, markersize=8)
        ax1.plot(sparsities, mse_dim, 's-', label='Head_dim Pruning', linewidth=2, markersize=8)
        ax1.set_xlabel('Sparsity', fontsize=12)
        ax1.set_ylabel('Output MSE', fontsize=12)
        ax1.set_title('Output Mean Squared Error Comparison', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 图2: 相对误差
        ax2 = axes[0, 1]
        ax2.plot(sparsities, rel_error_head, 'o-', label='Head Pruning', linewidth=2, markersize=8)
        ax2.plot(sparsities, rel_error_dim, 's-', label='Head_dim Pruning', linewidth=2, markersize=8)
        ax2.set_xlabel('Sparsity', fontsize=12)
        ax2.set_ylabel('Relative Error', fontsize=12)
        ax2.set_title('Output Relative Error Comparison', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 图3: 剪枝时间
        ax3 = axes[1, 0]
        ax3.plot(sparsities, time_head, 'o-', label='Head Pruning', linewidth=2, markersize=8)
        ax3.plot(sparsities, time_dim, 's-', label='Head_dim Pruning', linewidth=2, markersize=8)
        ax3.set_xlabel('Sparsity', fontsize=12)
        ax3.set_ylabel('Time (seconds)', fontsize=12)
        ax3.set_title('Pruning Time Comparison', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # 图4: 误差改进百分比
        ax4 = axes[1, 1]
        improvements = [(h - d) / h * 100 if h > 0 else 0
                       for h, d in zip(rel_error_head, rel_error_dim)]
        ax4.bar(range(len(sparsities)), improvements, color='green', alpha=0.7)
        ax4.set_xlabel('Sparsity', fontsize=12)
        ax4.set_ylabel('Error Improvement (%)', fontsize=12)
        ax4.set_title('Head_dim Error Improvement over Head', fontsize=13, fontweight='bold')
        ax4.set_xticks(range(len(sparsities)))
        ax4.set_xticklabels([f'{s:.1%}' for s in sparsities])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)

        # 在柱状图上添加数值标签
        for i, v in enumerate(improvements):
            ax4.text(i, v + 0.5, f'{v:+.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = f'head_vs_headdim_comparison_{test_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存: {filename}")
        plt.show()

    def _plot_fastoba_comparison(self, all_results, sparsities):
        """FastOBA模式：4种方法对比图"""
        # 提取数据
        methods = ['slimgpt_head', 'slimgpt_headdim', 'fastoba_head', 'fastoba_headdim']
        labels = ['SlimGPT Head', 'SlimGPT Head-dim', 'FastOBA Head', 'FastOBA Head-dim']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        mses = {method: [r[method]['mse'] for r in all_results] for method in methods}
        rel_errors = {method: [r[method]['relative_error'] for r in all_results] for method in methods}
        times = {method: [r[method]['time'] for r in all_results] for method in methods}

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SlimGPT vs FastOBA: Attention Pruning Comparison (4 Methods)',
                     fontsize=16, fontweight='bold')

        # 图1: 输出MSE对比
        ax1 = axes[0, 0]
        for i, method in enumerate(methods):
            ax1.plot(sparsities, mses[method], marker=markers[i], label=labels[i],
                    linewidth=2, markersize=8, color=colors[i])
        ax1.set_xlabel('Sparsity', fontsize=12)
        ax1.set_ylabel('Output MSE', fontsize=12)
        ax1.set_title('Output MSE Comparison (4 Methods)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 图2: 相对误差对比
        ax2 = axes[0, 1]
        for i, method in enumerate(methods):
            ax2.plot(sparsities, rel_errors[method], marker=markers[i], label=labels[i],
                    linewidth=2, markersize=8, color=colors[i])
        ax2.set_xlabel('Sparsity', fontsize=12)
        ax2.set_ylabel('Relative Error', fontsize=12)
        ax2.set_title('Relative Error Comparison (4 Methods)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 图3: 剪枝时间对比
        ax3 = axes[1, 0]
        for i, method in enumerate(methods):
            ax3.plot(sparsities, times[method], marker=markers[i], label=labels[i],
                    linewidth=2, markersize=8, color=colors[i])
        ax3.set_xlabel('Sparsity', fontsize=12)
        ax3.set_ylabel('Time (seconds)', fontsize=12)
        ax3.set_title('Pruning Time Comparison (4 Methods)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 图4: 误差改进对比（以SlimGPT Head为基准）
        ax4 = axes[1, 1]
        baseline = rel_errors['slimgpt_head']
        improvements = {}
        for method in methods[1:]:  # 跳过baseline自己
            improvements[method] = [(b - e) / b * 100 if b > 0 else 0
                                   for b, e in zip(baseline, rel_errors[method])]

        x = np.arange(len(sparsities))
        width = 0.25
        for i, (method, label) in enumerate(zip(methods[1:], labels[1:])):
            ax4.bar(x + i * width, improvements[method], width, label=label,
                   alpha=0.7, color=colors[i+1])

        ax4.set_xlabel('Sparsity', fontsize=12)
        ax4.set_ylabel('Error Reduction vs SlimGPT Head (%)', fontsize=12)
        ax4.set_title('Error Improvement over SlimGPT Head', fontsize=13, fontweight='bold')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([f'{s:.1%}' for s in sparsities])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)

        plt.tight_layout()
        filename = 'slimgpt_vs_fastoba_attention_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"FastOBA对比图表已保存: {filename}")
        plt.show()


def main():
    """主函数"""
    print("=" * 80)
    print("Head剪枝 vs Head_dim剪枝 误差对比实验")
    print("=" * 80)
    print()

    # 创建实验
    experiment = ComparisonExperiment(
        hidden_size=768,
        num_heads=12,
        batch_size=32,
        seq_len=128,
        num_batches=20,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 运行多个稀疏度的对比
    sparsity_levels = [0.125, 0.25, 0.375, 0.5]

    print("将在以下稀疏度下进行对比:")
    for s in sparsity_levels:
        print(f"  - {s:.1%}")
    print()

    # 选择测试类型
    print("选择测试类型:")
    print("  1. 线性层测试 (linear)")
    print("  2. Attention层测试 (attention)")
    print("  3. 两者都测试 (both)")
    print()

    test_choice = input("请输入选项 (1/2/3，默认为3): ").strip() or '3'

    test_types = []
    if test_choice == '1':
        test_types = ['linear']
    elif test_choice == '2':
        test_types = ['attention']
    else:
        test_types = ['linear', 'attention']

    # 选择是否使用FastOBA
    print()
    print("选择Hessian计算方法:")
    print("  1. 仅SlimGPT (H = XX^T)")
    print("  2. 仅FastOBA (真实Hessian，仅Attention层)")
    print("  3. 两者都测试 (对比SlimGPT vs FastOBA)")
    print()

    method_choice = input("请输入选项 (1/2/3，默认为1): ").strip() or '1'

    use_fastoba_options = []
    if method_choice == '1':
        use_fastoba_options = [False]
    elif method_choice == '2':
        use_fastoba_options = [True]
    else:
        use_fastoba_options = [False, True]

    print()
    input("按Enter键开始实验...")
    print()

    all_results_dict = {}

    for use_fastoba in use_fastoba_options:
        method_name = "FastOBA" if use_fastoba else "SlimGPT"
        print("\n" + "=" * 80)
        print(f"开始测试方法: {method_name}")
        print("=" * 80 + "\n")

        for test_type in test_types:
            print("\n" + "=" * 80)
            print(f"测试类型: {test_type.upper()}")
            print("=" * 80 + "\n")

            all_results = experiment.run_multi_sparsity_comparison(
                sparsity_levels,
                test_type=test_type,
                use_fastoba=use_fastoba
            )

            key = f"{test_type}{'_fastoba' if use_fastoba else ''}"
            all_results_dict[key] = all_results

            # 保存结果
            import json
            filename = f'comparison_results_{key}.json'
            with open(filename, 'w') as f:
                # 转换tensor为列表
                results_serializable = []
                for r in all_results:
                    result_dict = {'sparsity': r['sparsity'], 'test_type': r.get('test_type', test_type)}

                    # 根据是否使用FastOBA保存不同的键
                    if use_fastoba and test_type == 'attention':
                        for method in ['slimgpt_head', 'slimgpt_headdim', 'fastoba_head', 'fastoba_headdim']:
                            if method in r:
                                result_dict[method] = {
                                    k: float(v) if isinstance(v, (int, float)) else v
                                    for k, v in r[method].items()
                                }
                    else:
                        for method in ['head_prune', 'head_dim_prune']:
                            if method in r:
                                result_dict[method] = {
                                    k: float(v) if isinstance(v, (int, float)) else v
                                    for k, v in r[method].items()
                                }

                    results_serializable.append(result_dict)

                json.dump(results_serializable, f, indent=2)

            print(f"\n结果已保存: {filename}")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
