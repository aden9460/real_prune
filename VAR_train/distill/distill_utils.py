"""
蒸馏辅助工具函数

提供蒸馏训练所需的工具函数，包括：
- 教师模型加载
- 参数计算
- 配置解析和验证
"""

import torch
import os
from typing import List, Optional
import dist


def load_teacher_model(args):
    """加载教师模型

    Args:
        args: 包含教师模型配置的参数对象

    Returns:
        加载好的教师模型（已冻结）

    Raises:
        FileNotFoundError: 教师模型文件不存在
        KeyError: checkpoint格式不正确
    """
    from models import build_vae_var

    print(f"[加载教师模型] 路径: {args.teacher_model_path}")
    print(f"[加载教师模型] 深度: {args.teacher_depth}")

    # 验证文件存在
    if not os.path.exists(args.teacher_model_path):
        raise FileNotFoundError(f"教师模型文件不存在: {args.teacher_model_path}")

    # 构建教师模型架构（使用完整模型配置）
    print("[加载教师模型] 构建模型架构...")

    # 临时保存学生模型的稀疏度
    student_sparsity = args.sparsity

    # 设置为0，确保教师模型为完整架构（避免注意力头数被剪枝）
    args.sparsity = 0.0
    print(f"[加载教师模型] 临时设置 sparsity=0 (原值={student_sparsity})")

    try:
        teacher_vae, teacher_var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,        # VQVAE超参数（硬编码）
            device=dist.get_device(), patch_nums=args.patch_nums,
            num_classes=1000,  # ImageNet类别数
            depth=args.teacher_depth,  # 教师模型深度
            shared_aln=args.saln, attn_l2_norm=args.anorm,
            flash_if_available=args.fuse, fused_if_available=args.fuse,
            init_adaln=args.aln, init_adaln_gamma=args.alng,
            init_head=args.hd, init_std=args.ini, args=args
        )
    finally:
        # 恢复学生模型的稀疏度
        args.sparsity = student_sparsity
        print(f"[加载教师模型] 恢复 sparsity={student_sparsity}")

    # 加载权重
    print("[加载教师模型] 加载权重...")
    checkpoint = torch.load(args.teacher_model_path, map_location='cpu')

    # 处理不同的checkpoint格式
    if 'trainer' in checkpoint:
        # 训练checkpoint格式
        if 'var_wo_ddp' in checkpoint['trainer']:
            teacher_var.load_state_dict(checkpoint['trainer']['var_wo_ddp'], strict=True)
            print("[加载教师模型] 从训练checkpoint提取模型权重")
        else:
            raise KeyError("在训练checkpoint中未找到var_wo_ddp键")
    else:
        # 直接模型权重格式
        teacher_var.load_state_dict(checkpoint, strict=True)
        print("[加载教师模型] 直接加载模型权重")

    # 设置为评估模式并冻结参数
    teacher_var.eval()
    frozen_params = 0
    for param in teacher_var.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    param_count = count_parameters(teacher_var)
    print(f"[加载教师模型] 成功加载，参数量: {param_count:.2f}M")
    print(f"[加载教师模型] 已冻结参数: {frozen_params/1e6:.2f}M")

    return teacher_var


def count_parameters(model) -> float:
    """计算模型参数数量（百万）

    Args:
        model: PyTorch模型

    Returns:
        参数数量（百万）
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def parse_scale_weights(scale_weights_str: str) -> List[float]:
    """解析尺度权重字符串为列表

    Args:
        scale_weights_str: 逗号分隔的权重字符串，如 "2.0,1.8,1.6,..."

    Returns:
        权重列表 [2.0, 1.8, 1.6, ...]

    Raises:
        ValueError: 权重数量不是10个或格式错误
    """
    try:
        weights = [float(x.strip()) for x in scale_weights_str.split(',')]
        if len(weights) != 10:
            raise ValueError(f"尺度权重必须有10个值，当前有{len(weights)}个")
        if any(w < 0 for w in weights):
            raise ValueError("尺度权重必须非负")
        return weights
    except ValueError as e:
        raise ValueError(f"尺度权重格式错误: {e}")


def parse_feature_layers(feature_layers_str: str) -> List[int]:
    """解析特征层字符串为列表

    Args:
        feature_layers_str: 逗号分隔的层索引字符串，如 "4,8,12,15"

    Returns:
        层索引列表 [4, 8, 12, 15]

    Raises:
        ValueError: 格式错误或层索引无效
    """
    try:
        layers = [int(x.strip()) for x in feature_layers_str.split(',')]
        if any(layer < 0 for layer in layers):
            raise ValueError("层索引必须非负")
        return layers
    except ValueError as e:
        raise ValueError(f"特征层格式错误: {e}")


def validate_teacher_student_compatibility(teacher_model, student_model, args):
    """验证教师模型和学生模型的兼容性

    Args:
        teacher_model: 教师模型
        student_model: 学生模型
        args: 参数配置

    Raises:
        AssertionError: 模型不兼容
    """
    # 检查参数数量
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)

    print(f"[兼容性检查] 教师模型参数: {teacher_params:.2f}M")
    print(f"[兼容性检查] 学生模型参数: {student_params:.2f}M")
    print(f"[兼容性检查] 压缩比: {student_params/teacher_params:.2%}")

    # 检查输出维度
    if hasattr(teacher_model, 'head') and hasattr(student_model, 'head'):
        teacher_vocab = teacher_model.head.out_features
        student_vocab = student_model.head.out_features
        assert teacher_vocab == student_vocab, f"词汇表大小不匹配: 教师{teacher_vocab} vs 学生{student_vocab}"
        print(f"[兼容性检查] 词汇表大小: {teacher_vocab}")

    # 检查序列长度
    if hasattr(teacher_model, 'L') and hasattr(student_model, 'L'):
        assert teacher_model.L == student_model.L, f"序列长度不匹配: 教师{teacher_model.L} vs 学生{student_model.L}"
        print(f"[兼容性检查] 序列长度: {teacher_model.L}")

    # 检查patch配置
    if hasattr(teacher_model, 'patch_nums') and hasattr(student_model, 'patch_nums'):
        assert teacher_model.patch_nums == student_model.patch_nums, "patch配置不匹配"
        print(f"[兼容性检查] patch配置: {teacher_model.patch_nums}")

    print("[兼容性检查] 通过")


def get_scale_boundaries():
    """获取VAR的10个尺度边界

    Returns:
        尺度边界列表，形如 [0, 1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    """
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # VAR的标准patch配置
    boundaries = [0]
    current_pos = 0

    for pn in patch_nums:
        current_pos += pn * pn
        boundaries.append(current_pos)

    return boundaries


def compute_scale_weights(strategy: str = 'linear_decay', **kwargs) -> List[float]:
    """生成尺度权重

    Args:
        strategy: 权重策略 ('linear_decay', 'exponential_decay', 'uniform', 'early_focus')
        **kwargs: 策略相关参数

    Returns:
        10个尺度的权重列表
    """
    if strategy == 'linear_decay':
        # 线性衰减：从高到低
        start_weight = kwargs.get('start_weight', 2.0)
        end_weight = kwargs.get('end_weight', 0.2)
        step = (start_weight - end_weight) / 9
        return [start_weight - i * step for i in range(10)]

    elif strategy == 'exponential_decay':
        # 指数衰减
        base = kwargs.get('base', 0.5)
        scale = kwargs.get('scale', 4.0)
        return [scale * (base ** i) for i in range(10)]

    elif strategy == 'uniform':
        # 均匀权重
        weight = kwargs.get('weight', 1.0)
        return [weight] * 10

    elif strategy == 'early_focus':
        # 早期集中：前几个尺度高权重，后面低权重
        high_weight = kwargs.get('high_weight', 3.0)
        low_weight = kwargs.get('low_weight', 0.1)
        focus_scales = kwargs.get('focus_scales', 4)  # 前4个尺度高权重
        weights = []
        for i in range(10):
            if i < focus_scales:
                weights.append(high_weight)
            else:
                weights.append(low_weight)
        return weights

    else:
        raise ValueError(f"不支持的权重策略: {strategy}")


def create_distill_config_summary(args) -> str:
    """创建蒸馏配置摘要

    Args:
        args: 参数配置

    Returns:
        配置摘要字符串
    """
    if not args.enable_distillation:
        return "蒸馏未启用"

    summary = []
    summary.append("======== 蒸馏配置摘要 ========")
    summary.append(f"蒸馏类型: {args.distill_type}")
    summary.append(f"教师模型: {os.path.basename(args.teacher_model_path)} (深度={args.teacher_depth})")
    summary.append(f"损失权重: 任务={args.distill_alpha:.2f}, 蒸馏={args.distill_beta:.2f}")
    summary.append(f"蒸馏温度: {args.distill_temperature}")

    if args.distill_type in ['scale_aware', 'both']:
        weights = args.scale_weights
        summary.append(f"尺度权重: [{weights[0]:.1f}, {weights[1]:.1f}, ..., {weights[-1]:.1f}]")

    if args.use_feature_distill:
        summary.append(f"特征蒸馏: 启用，层={args.feature_layers}")

    if args.use_attention_distill:
        summary.append("注意力蒸馏: 启用")

    summary.append("=" * 30)

    return "\n".join(summary)


def log_distillation_step(step: int, task_loss: float, distill_loss: float,
                         total_loss: float, logits_similarity: Optional[float] = None):
    """记录蒸馏训练步骤

    Args:
        step: 训练步骤
        task_loss: 任务损失
        distill_loss: 蒸馏损失
        total_loss: 总损失
        logits_similarity: logits相似度（可选）
    """
    log_msg = f"[Step {step}] 任务损失: {task_loss:.4f}, 蒸馏损失: {distill_loss:.4f}, 总损失: {total_loss:.4f}"

    if logits_similarity is not None:
        log_msg += f", 相似度: {logits_similarity:.4f}"

    print(log_msg)