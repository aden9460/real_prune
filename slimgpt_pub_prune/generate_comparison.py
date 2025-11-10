#!/usr/bin/env python3
"""
批量测试脚本 - 生成所有模型的对比图片
每个模型生成一张图片，然后拼接成大图对比
"""

import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from models import VQVAE, build_vae_var
import os.path as osp

def load_model(model_path, device='cuda'):
    """加载指定路径的模型"""
    MODEL_DEPTH = 16
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    # 构建模型架构
    vae_ckpt = '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = f'/home/project/daily/AR/model_zoo/var_d{MODEL_DEPTH}.pth'

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False
    )

    # 加载VAE权重
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    # 加载剪枝后的VAR权重
    var.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    vae.eval()
    var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)

    return vae, var

def generate_image(var_model, class_label=980, device='cuda'):
    """使用模型生成一张图片"""
    label_tensor = torch.tensor([class_label], device=device)

    # 启用KV缓存
    for b in var_model.blocks:
        b.attn.kv_caching(True)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_image = var_model(label_tensor)

    # 禁用KV缓存
    for b in var_model.blocks:
        b.attn.kv_caching(False)

    return recon_image[0]  # 返回第一张图

def add_text_to_image(image_tensor, text, font_size=20):
    """在图片上添加文字标签"""
    # 转换为PIL图像
    image = torch.clamp(image_tensor, 0, 1)
    image_pil = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    # 添加文字
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 在图片顶部添加白色背景的文字
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 绘制白色背景
    draw.rectangle([0, 0, text_width + 10, text_height + 10], fill="white")
    draw.text((5, 5), text, fill="black", font=font)

    # 转换回tensor
    return torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs("./comparison_images", exist_ok=True)

    sparsities = [0.1, 0.2, 0.3, 0.4]
    strategies = ["natural", "equal", "sqrt"]

    generated_images = []
    image_labels = []

    print("开始生成对比图片...")

    # 1. 生成Chain KFAC图片
    print("生成Chain KFAC图片...")
    for sparsity in sparsities:
        for strategy in strategies:
            model_name = f"var_d16_chain_kfac_s{sparsity}_{strategy}_n150.pth"
            model_path = f"./batch_models/chain_kfac/{model_name}"

            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在 {model_path}")
                continue

            print(f"  处理: {model_name}")
            try:
                vae, var = load_model(model_path, device)
                image = generate_image(var, class_label=980, device=device)

                # 添加标签
                label = f"Chain-KFAC\ns={sparsity} {strategy}"
                labeled_image = add_text_to_image(image, label)

                generated_images.append(labeled_image)
                image_labels.append(f"chain_kfac_s{sparsity}_{strategy}")

                # 保存单张图片
                save_image(labeled_image, f"./comparison_images/{image_labels[-1]}.png")

            except Exception as e:
                print(f"    错误: {e}")

    # 2. 生成SlimGPT图片
    print("生成SlimGPT图片...")
    for sparsity in sparsities:
        model_name = f"var_d16_slimgpt_s{sparsity}_n150.pth"
        model_path = f"./batch_models/slimgpt/{model_name}"

        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在 {model_path}")
            continue

        print(f"  处理: {model_name}")
        try:
            vae, var = load_model(model_path, device)
            image = generate_image(var, class_label=980, device=device)

            # 添加标签
            label = f"SlimGPT\ns={sparsity}"
            labeled_image = add_text_to_image(image, label)

            generated_images.append(labeled_image)
            image_labels.append(f"slimgpt_s{sparsity}")

            # 保存单张图片
            save_image(labeled_image, f"./comparison_images/{image_labels[-1]}.png")

        except Exception as e:
            print(f"    错误: {e}")

    # 3. 创建拼接大图
    print(f"拼接所有图片... 总计: {len(generated_images)} 张")

    if generated_images:
        # 按4列排列
        grid = make_grid(generated_images, nrow=4, padding=10, pad_value=1.0)
        save_image(grid, "./comparison_images/all_models_comparison.png")
        print("✅ 拼接完成: ./comparison_images/all_models_comparison.png")

        # 创建详细的布局图
        create_layout_grid(generated_images, image_labels)
    else:
        print("❌ 没有生成任何图片")

def create_layout_grid(images, labels):
    """创建带有更好布局的对比图"""
    if not images:
        return

    # 重新组织：Chain KFAC按strategy分组，SlimGPT单独一行
    sparsities = [0.1, 0.2, 0.3, 0.4]
    strategies = ["natural", "equal", "sqrt"]

    # 按策略和方法重新组织图片
    organized_images = []
    organized_labels = []

    # Chain KFAC部分 (3行，每行4个sparsity)
    for strategy in strategies:
        row_images = []
        row_labels = []
        for sparsity in sparsities:
            label_key = f"chain_kfac_s{sparsity}_{strategy}"
            if label_key in labels:
                idx = labels.index(label_key)
                row_images.append(images[idx])
                row_labels.append(f"Chain-KFAC {strategy}\nSparsity={sparsity}")

        if row_images:
            organized_images.extend(row_images)
            organized_labels.extend(row_labels)

    # SlimGPT部分 (1行，4个sparsity)
    row_images = []
    row_labels = []
    for sparsity in sparsities:
        label_key = f"slimgpt_s{sparsity}"
        if label_key in labels:
            idx = labels.index(label_key)
            row_images.append(images[idx])
            row_labels.append(f"SlimGPT\nSparsity={sparsity}")

    if row_images:
        organized_images.extend(row_images)
        organized_labels.extend(row_labels)

    # 创建4列的网格布局
    if organized_images:
        grid = make_grid(organized_images, nrow=4, padding=15, pad_value=1.0)
        save_image(grid, "./comparison_images/organized_comparison.png")
        print("✅ 组织化布局完成: ./comparison_images/organized_comparison.png")

if __name__ == "__main__":
    main()