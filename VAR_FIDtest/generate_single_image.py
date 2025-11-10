#!/usr/bin/env python3
"""
单张图片生成脚本 - 模仿 FID_test.py 结构
每个模型生成一张对比图片
"""

import os
import os.path as osp
import torch
import numpy as np
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import build_vae_var
from torchvision.utils import save_image
from PIL import Image, ImageFont, ImageDraw
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--sparsity', type=float, default=0.4)
    parser.add_argument('--var_model', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--method_label', type=str, required=True)
    parser.add_argument('--class_label', type=int, default=980)  # volcano
    return parser.parse_args()

def add_text_to_image(image_tensor, text, font_size=30):
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

    # 在图片顶部添加半透明白色背景的文字
    lines = text.split('\n')
    y_offset = 10

    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 绘制半透明白色背景
        overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, y_offset, text_width + 20, y_offset + text_height + 10],
                              fill=(255, 255, 255, 180))

        # 合并背景
        image_pil = Image.alpha_composite(image_pil.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image_pil)

        # 绘制文字
        draw.text((10, y_offset + 5), line, fill="black", font=font)
        y_offset += text_height + 15

    # 转换回tensor
    return torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0

def main():
    args = parse_args()

    MODEL_DEPTH = args.depth
    assert MODEL_DEPTH in {12, 16, 20, 24, 30}

    # 检查模型文件
    vae_ckpt = '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_model

    print(f"VAE checkpoint: {vae_ckpt}")
    print(f"VAR checkpoint: {var_ckpt}")

    if not osp.exists(vae_ckpt):
        print("❌ VAE checkpoint 不存在")
        return
    if not osp.exists(var_ckpt):
        print("❌ VAR checkpoint 不存在")
        return

    # 构建模型
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("构建 VAE 和 VAR 模型...")

    # 创建一个简单的args对象来替代arg_util.Args
    class SimpleArgs:
        def __init__(self):
            self.depth = MODEL_DEPTH
            self.sparsity = args.sparsity

    simple_args = SimpleArgs()

    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        args=simple_args
    )

    # 加载权重
    print("加载权重...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    checkpoint = torch.load(var_ckpt, map_location='cpu')
    var.load_state_dict(checkpoint, strict=False)

    vae.eval()
    var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)

    print("模型加载完成")

    # 生成图片
    print(f"生成图片 - 类别: {args.class_label}")

    label_tensor = torch.tensor([args.class_label], device=device)

    # 启用KV缓存
    for b in var.blocks:
        b.attn.kv_caching(True)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_image = var(label_tensor)

    # 禁用KV缓存
    for b in var.blocks:
        b.attn.kv_caching(False)

    # 添加标签
    label_text = f"{args.method_label}\nSparsity: {args.sparsity}"
    labeled_image = add_text_to_image(recon_image[0], label_text)

    # 保存图片
    output_dir = "./output/comparison_images"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{args.output_name}.png"
    save_image(labeled_image, output_path)

    print(f"✅ 图片已保存: {output_path}")

if __name__ == "__main__":
    main()