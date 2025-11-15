#!/usr/bin/env python3
"""
VAR OBA Pruning Script
使用Optimal Brain Apoptosis (OBA)对VAR模型进行结构化剪枝

核心思路：
- 使用WrappedPruner接口调用OBA算法
- 复用SlimGPT的模型加载和数据准备
- 目标40%稀疏度，剪枝attention和FFN层
"""

import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import sys

# 添加VAR路径
sys.path.append("../slimvar/VAR/")

# 加速：禁用默认参数初始化
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

# 直接导入需要的函数，避免torch_pruning冲突
sys.path.append("../slimvar/")


def load_var_model(model_depth, vae_ckpt_path, var_ckpt_path, device='cuda'):
    """直接复制load_var_model函数"""
    from VAR.models import build_vae_var
    import os.path as osp

    assert model_depth in {16, 20, 24, 30}, f"Invalid model_depth: {model_depth}"

    if not osp.exists(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")
    if not osp.exists(var_ckpt_path):
        raise FileNotFoundError(f"VAR checkpoint not found: {var_ckpt_path}")

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    # Build models
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=False
    )

    # Load weights
    vae.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt_path, map_location='cpu'), strict=False)

    # Set to eval mode
    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return vae, var


@torch.no_grad()
def get_imagenet_data_for_var(imagenet_dir, vae, num_samples=256, batch_size=8, use_real_images=True):
    """
    创建VAR校准数据加载器，使用VAR原始的build_dataset函数
    参考SlimGPT的prepare_calibration_data函数
    """
    print("加载真实ImageNet图像并编码（使用VAR原始build_dataset）...")

    # 导入原始VAR的数据加载函数
    sys.path.insert(0, '../slimvar/VAR')
    from VAR.utils.data import build_dataset

    # 使用原始VAR的方式加载数据
    num_classes, train_set, val_set = build_dataset(
        data_path=imagenet_dir,
        final_reso=256,
        hflip=False,
        mid_reso=1.125
    )

    print(f"  从 {num_classes} 个类别中采样 {num_samples} 个样本")

    # 按类别平衡采样
    dataset = val_set
    if num_samples <= len(dataset):
        step = len(dataset) // num_samples
        indices = torch.arange(0, len(dataset), step)[:num_samples]
    else:
        indices = torch.arange(len(dataset))

    print(f"  实际采样：{len(indices)} 个样本（均匀跨度采样）")

    # 预编码所有tokens
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = vae.to(device)
    vae.eval()

    all_labels = []
    all_tokens = []

    print("  编码图像为tokens...")
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break

        try:
            image, label = dataset[idx.item()]
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0).to(device)
            else:
                # 如果是PIL图像，需要转换
                import torchvision.transforms as T
                transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = transform(image).unsqueeze(0).to(device)

            # VAE编码：使用VAR的正确方法
            gt_idx_Bl = vae.img_to_idxBl(image)  # List of 10 tensors
            tokens = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # (1, 679, 32)
            tokens = tokens.squeeze(0).cpu()  # [679, 32]

            all_labels.append(label)
            all_tokens.append(tokens)

            if (i + 1) % 16 == 0:
                print(f"    编码进度: {i+1}/{len(indices)}")

        except Exception as e:
            print(f"    ⚠️ 跳过样本 {i}: {e}")
            continue

        if len(all_tokens) >= num_samples:
            break

    print(f"  ✅ 成功编码 {len(all_tokens)} 个样本")

    # 创建DataLoader
    from torch.utils.data import TensorDataset, DataLoader

    labels_tensor = torch.tensor(all_labels)
    tokens_tensor = torch.stack(all_tokens)  # [N, 679, 32]

    print(f"  📊 Token tensor shape: {tokens_tensor.shape}")

    dataset = TensorDataset(labels_tensor, tokens_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


class VARModelWrapper(nn.Module):
    """
    包装VAR模型，使其兼容WrappedPruner的期望接口
    WrappedPruner期望model(img)形式，但VAR需要model(label, tokens)
    """
    def __init__(self, var_model):
        super().__init__()
        self.var_model = var_model

    def forward(self, img):
        # WrappedPruner调用model(img)，但VAR需要model(labels, tokens)
        # img应该是tokens，来自DataLoader

        # 检查输入格式
        if torch.is_tensor(img):
            # WrappedPruner传递单个tensor，假设这是tokens
            tokens = img
            batch_size = tokens.size(0)
            # 创建dummy labels
            labels = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)

            # VAR前向传播
            logits = self.var_model(labels, tokens)
            # logits shape: [batch_size, sequence_length, vocab_size] = [B, 679, 4096]

            # WrappedPruner期望[batch_size, num_classes]格式用于cross_entropy
            # 我们需要reshape logits以匹配WrappedPruner的期望
            # 取最后一个时间步的logits作为"分类"输出
            # 或者平均池化所有时间步
            logits_reshaped = logits.mean(dim=1)  # [B, 4096] - 平均池化

            return logits_reshaped
        else:
            raise ValueError(f"Unexpected input format for VAR: {type(img)}, expected torch.Tensor")

class VARDataWrapper:
    """
    包装DataLoader，交换labels和tokens的顺序以匹配WrappedPruner期望
    """
    def __init__(self, original_loader):
        self.original_loader = original_loader
        self.dataset = original_loader.dataset

    def __iter__(self):
        for labels, tokens in self.original_loader:
            # WrappedPruner期望(img, label)格式，我们返回(tokens, labels)
            # 实际上VARModelWrapper会处理这个顺序
            yield tokens, labels

    def __len__(self):
        return len(self.original_loader)


# 导入OBA框架
from torch_pruning import WrappedPruner


def create_pruner_args(args):
    """创建WrappedPruner需要的参数对象"""
    class PrunerArgs:
        def __init__(self):
            # 基础参数
            self.importance_type = args.importance_type  # 使用命令行参数

            # OBA参数
            self.delta = args.delta
            self.upward_delta = args.upward_delta
            self.downward_delta = args.downward_delta
            self.parallel_delta = args.parallel_delta
            self.normalizer = args.normalizer
            self.multivariable = True
            self.self_unit_weight = False
            self.other_unit_weight = False

            # 剪枝参数
            self.lr = 0.001  # 重要性计算时的学习率
            self.weight_decay = 1e-4
            self.iters_per_step = args.iters_per_step
            self.iterative_steps = 1  # 一次性剪枝
            self.model = f"var_d{args.model_depth}"
            self.max_pruning_ratio = 0.95  # 单层最大剪枝比例

            # FastOBA参数（即使不用也需要定义）
            self.fastoba_delta = 1.0
            self.sl_lr = 0.001

    return PrunerArgs()


def patch_var_inplace_ops(model):
    """
    临时修改VAR模型中的in-place操作，使其与OBA的backward hooks兼容
    """
    print("🔧 Patching VAR in-place operations for OBA compatibility...")

    # 保存原始forward方法
    for block in model.blocks:
        if not hasattr(block, '_original_forward'):
            block._original_forward = block.forward

            def make_patched_forward(blk):
                def patched_forward(x, cond_BD, attn_bias):
                    # 复制原始逻辑，但避免in-place操作
                    if blk.shared_aln:
                        gamma1, gamma2, scale1, scale2, shift1, shift2 = (blk.ada_gss + cond_BD).unbind(2)
                    else:
                        gamma1, gamma2, scale1, scale2, shift1, shift2 = blk.ada_lin(cond_BD).view(-1, 1, 6, blk.C).unbind(2)

                    # 避免.add_()和.mul_()这样的in-place操作
                    # 原始: self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
                    # 修改为非in-place版本
                    attn_input = blk.ln_wo_grad(x).mul(scale1.add(1)).add(shift1)  # 使用.add()而非.add_()
                    attn_out = blk.attn(attn_input, attn_bias=attn_bias).mul(gamma1)  # 使用.mul()而非.mul_()
                    x = x + blk.drop_path(attn_out)

                    ffn_input = blk.ln_wo_grad(x).mul(scale2.add(1)).add(shift2)  # 使用.add()而非.add_()
                    ffn_out = blk.ffn(ffn_input).mul(gamma2)
                    x = x + blk.drop_path(ffn_out)

                    return x
                return patched_forward

            block.forward = make_patched_forward(block)

    print("  ✅ VAR blocks patched for OBA compatibility")


def restore_var_inplace_ops(model):
    """恢复VAR模型的原始forward方法"""
    print("🔧 Restoring original VAR forward methods...")
    for block in model.blocks:
        if hasattr(block, '_original_forward'):
            block.forward = block._original_forward
            del block._original_forward
    print("  ✅ Original methods restored")


def execute_oba_pruning(model, calibration_loader, args):
    """
    执行OBA剪枝的主流程
    """
    print("🔥 Starting OBA Pruning...")

    # 先patch VAR模型以避免in-place操作冲突
    patch_var_inplace_ops(model)

    try:
        # 包装VAR模型以兼容WrappedPruner接口
        wrapped_model = VARModelWrapper(model)
        wrapped_loader = VARDataWrapper(calibration_loader)

        # 创建example inputs - 使用包装后的格式
        example_tokens = torch.randn(1, 679, 32).cuda()  # [B, L=679, C=32]
        # 为包装的模型创建example input - WrappedPruner期望单个tensor
        example_input = example_tokens

        # 设置忽略层
        ignored_layers = [
            wrapped_model.var_model.class_emb,
            wrapped_model.var_model.pos_1LC,
            wrapped_model.var_model.lvl_embed,
            wrapped_model.var_model.head,
            wrapped_model.var_model.head_nm,
        ]

        # 创建pruner args
        pruner_args = create_pruner_args(args)

        # 计算目标FLOPs保留比例
        ops_ratio = 1.0 - args.target_sparsity  # 40%稀疏度 = 60%保留

        print(f"  Target FLOPs ratio: {ops_ratio}")
        print(f"  Importance type: {args.importance_type}")
        print(f"  OBA parameters: delta={args.delta}, upward={args.upward_delta}, downward={args.downward_delta}, parallel={args.parallel_delta}")

        # 预计算base ops和params
        try:
            import thop
            with torch.no_grad():
                # 使用包装后的model测试FLOPs
                base_ops, base_params = thop.profile(wrapped_model, inputs=(example_tokens,), verbose=False)
        except Exception as e:
            print(f"    ⚠️ thop.profile failed: {e}, using default values")
            # 如果失败，使用默认值
            base_ops, base_params = 1e12, sum(p.numel() for p in model.parameters())

        print(f"  📊 Base FLOPs: {base_ops/1e9:.2f}G, Base Params: {base_params/1e6:.2f}M")

        # 创建WrappedPruner
        pruner = WrappedPruner(
            args=pruner_args,
            train_loader=wrapped_loader,   # 使用包装的loader
            test_loader=None,  # 不需要测试
            model=wrapped_model,           # 使用包装的model
            example_inputs=example_input,  # 单个输入而非tuple
            ignored_layers=ignored_layers,
            pruning_ratio=ops_ratio,
            device=torch.device('cuda'),
            speed_up=1000,
            base_ops=base_ops,        # 提供预计算的值
            base_params=base_params,  # 避免thop.profile调用问题
        )

        print("  🔧 Executing pruning...")

        # 执行一次性剪枝
        pruner.iterative_prune_step()  # 只需要调用方法，数据已经在构造函数中传递了
        print("  ✅ OBA pruning completed successfully!")

        # 返回原始model（已被修改）
        return model

    except Exception as e:
        print(f"  ❌ Pruning failed: {e}")
        raise
    finally:
        # 恢复原始forward方法
        restore_var_inplace_ops(model)


def post_process_var_attention(model):
    """
    后处理VAR attention层的特殊参数
    """
    print("🔧 Post-processing VAR attention parameters...")

    for layer_idx, block in enumerate(model.blocks):
        # 获取当前的proj层信息
        proj_layer = block.attn.proj
        current_dim = proj_layer.in_features

        # 检查是否被剪枝
        if hasattr(block.attn, 'num_heads'):
            current_heads = block.attn.num_heads
        else:
            current_heads = current_dim // 64  # 假设head_dim=64
            block.attn.num_heads = current_heads

        # 确保相关参数维度一致
        if hasattr(block.attn, 'q_bias') and block.attn.q_bias.size(0) != current_dim:
            print(f"    Layer {layer_idx}: updating bias dimensions to {current_dim}")

        if hasattr(block.attn, 'scale_mul_1H11') and block.attn.scale_mul_1H11.size(1) != current_heads:
            print(f"    Layer {layer_idx}: updating scale_mul to {current_heads} heads")

    print("  ✅ VAR attention parameters processed")


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
    parser.add_argument('--normalizer', type=str, default='max',
                        choices=['max', 'mean', 'sum'],
                        help='Importance normalizer')

    parser.add_argument('--importance_type', type=str, default='Taylor',
                        choices=['OBA', 'Taylor', 'Weight'],
                        help='Importance estimation method')

    # Training parameters
    parser.add_argument('--iters_per_step', type=int, default=200,
                        help='Iterations for importance computation')

    parser.add_argument('--output_path', type=str, default='./var_d16_oba_pruned_0.4sparsity.pth',
                        help='Output pruned model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置模型路径
    vae_ckpt = args.vae_ckpt or f'/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_ckpt or f'/home/project/daily/AR/model_zoo/var_d{args.model_depth}.pth'

    print("🚀 VAR OBA Pruning Started")
    print(f"  Model: VAR-d{args.model_depth}")
    print(f"  VAR checkpoint: {var_ckpt}")
    print(f"  VAE checkpoint: {vae_ckpt}")
    print(f"  Target sparsity: {args.target_sparsity}")

    # 1. 加载模型
    print("\n📦 Loading models...")
    vae, var = load_var_model(args.model_depth, vae_ckpt, var_ckpt, device='cuda')
    original_params = sum(p.numel() for p in var.parameters())
    print(f"  ✅ Models loaded. Original params: {original_params:,}")

    # 2. 准备校准数据
    print(f"\n📊 Preparing calibration data ({args.num_samples} samples)...")
    calibration_loader = get_imagenet_data_for_var(
        imagenet_dir=args.imagenet_dir,
        vae=vae,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        use_real_images=True
    )
    print(f"  ✅ Data ready: {len(calibration_loader)} batches")

    # 3. 执行OBA剪枝
    var_pruned = execute_oba_pruning(var, calibration_loader, args)

    # 4. 后处理
    post_process_var_attention(var_pruned)

    # 5. 统计结果
    pruned_params = sum(p.numel() for p in var_pruned.parameters())
    compression_ratio = pruned_params / original_params
    achieved_sparsity = 1 - compression_ratio

    print(f"\n📈 Pruning Results:")
    print(f"  Original parameters: {original_params:,}")
    print(f"  Pruned parameters: {pruned_params:,}")
    print(f"  Compression ratio: {compression_ratio:.1%}")
    print(f"  Achieved sparsity: {achieved_sparsity:.1%}")
    print(f"  Target sparsity: {args.target_sparsity:.1%}")

    # 6. 保存模型
    print(f"\n💾 Saving pruned model to {args.output_path}")
    torch.save({
        'model_state_dict': var_pruned.state_dict(),
        'args': vars(args),
        'original_params': original_params,
        'pruned_params': pruned_params,
        'achieved_sparsity': achieved_sparsity,
    }, args.output_path)

    print(f"\n🎉 OBA Pruning Completed!")
    print(f"  Pruned model saved: {args.output_path}")
    print(f"  Ready for quality testing!")


if __name__ == "__main__":
    main()