import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np


import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import set_seed
import os.path as osp
import torch_pruning as tp
import sys
sys.path.append("VAR/")
# Disable default parameter initialization for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)



@torch.no_grad()
def prepare_calibration_data(vae, num_samples, use_images=False, image_dir=None, final_reso=256):
    """
    准备校准数据：一次性获取所有tokens（使用原始VAR的build_dataset）

    Args:
        vae: VQVAE model
        num_samples: number of calibration samples
        use_images: if True, load real ImageNet images; if False, use class labels
        image_dir: path to ImageNet root directory (should contain 'train' subdirectory)
        final_reso: final image resolution (default: 256)

    Returns:
        calibration_labels: (num_samples,) class labels
        calibration_tokens: (num_samples, 679, 32) pre-encoded tokens, or None if use_images=False
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_images:
        # 使用原始VAR的build_dataset函数
        print("加载真实ImageNet图像并编码（使用VAR原始build_dataset）...")

        # 导入原始VAR的数据加载函数
        sys.path.insert(0, 'VAR')
        from VAR.utils.data import build_dataset

        # 使用原始VAR的方式加载数据
        num_classes, train_set, val_set = build_dataset(
            data_path=image_dir,
            final_reso=final_reso,
            hflip=False,
            mid_reso=1.125
        )

        # 使用validation set进行calibration（与训练一致的数据增强策略）
        dataset = val_set

        # 按类别平衡采样：从1000类中平均取样
        print(f"  按类别平衡采样：从 {num_classes} 个类别中采样 {num_samples} 个样本")

        # 简化采样：直接从前num_samples个样本中采样（ImageNet val set通常按类别顺序排列）
        # 如果num_samples >= num_classes，则每类至少取1张
        if num_samples <= len(dataset):
            # 计算步长，确保覆盖所有类别
            step = len(dataset) // num_samples
            indices = torch.arange(0, len(dataset), step)[:num_samples]
        else:
            # 如果请求的样本数超过数据集大小，使用全部
            indices = torch.arange(len(dataset))

        print(f"  实际采样：{len(indices)} 个样本（均匀跨度采样）")

        calibration_labels = []
        calibration_tokens = []

        batch_size = 8
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            images = []
            labels = []

            for idx in batch_indices:
                img, label = dataset[int(idx)]
                images.append(img)
                labels.append(label)

            images = torch.stack(images).to(device)  # (B, 3, 256, 256), range [-1, 1]
            labels = torch.tensor(labels).to(device)

            # VQVAE编码：img -> tokens (模仿trainer.py)
            gt_idx_Bl = vae.img_to_idxBl(images)  # List of 10 tensors
            x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # (B, 679, 32)

            calibration_labels.append(labels)
            calibration_tokens.append(x_BLCv.cpu())

            if (i + batch_size) % 64 == 0:
                print(f"  已处理 {i + batch_size}/{num_samples} 张图像")

        calibration_labels = torch.cat(calibration_labels, dim=0)
        calibration_tokens = torch.cat(calibration_tokens, dim=0)

        print(f"✓ 完成！获得 {num_samples} 个样本的tokens")
        return calibration_labels, calibration_tokens

    else:
        # 方案2：简化方案 - 使用类别标签（VAR内部会处理）
        print("使用类别标签作为校准数据（简化方案）")
        calibration_labels = torch.arange(0, num_samples).cuda()
        return calibration_labels, None


model = ct.models.MLModel("/home/project/real_prune/newtransfer/coreml_models/d16_0.4_distill.mlpackage")

# 使用dtype直接指定量化位数：
# 8位量化（推荐）- 最常用，质量损失小
config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4"  # 或者使用字符串: "int8"
    )
)

activation_config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")
)
compressed_model_a8 = cto.linear_quantize_activations(
    model, activation_config, sample_data
)

# (Optional) It's recommended to use with linear_quantize_weights.
weight_config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")
)
compressed_model_w8a8 = cto.linear_quantize_weights(compressed_model_a8, weight_config)
# 其他选项：
# dtype=np.uint8 或 "uint8"  - 8位无符号
# dtype="int4"  - 4位有符号（最小模型，可能有明显质量损失）
# dtype="uint4" - 4位无符号（最小模型）

compressed_model = cto.linear_quantize_weights(model, config)
compressed_model.save("/home/project/real_prune/newtransfer/coreml_models/distill_0.4_int4.mlpackage")



