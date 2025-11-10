#!/usr/bin/env python3
"""
对比图片拼接脚本
将所有生成的单张图片拼接成大图进行对比
"""

import os
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
from datetime import datetime

def load_image_as_tensor(image_path):
    """加载图片并转换为tensor"""
    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image) / 255.0
    return torch.from_numpy(image_array).permute(2, 0, 1).float()

def main():
    print("开始拼接对比图片...")

    # 参数设置
    sparsities = [0.1, 0.2, 0.3, 0.4]
    strategies = ["natural", "equal", "sqrt"]
    depth = 16
    num_samples = 50

    # 收集所有图片
    all_images = []
    image_labels = []

    base_dir = "./output/comparison_images"

    # 1. 按组织化方式收集Chain KFAC图片
    print("收集 Chain KFAC 图片...")
    for strategy in strategies:
        strategy_images = []
        for sparsity in sparsities:
            image_name = f"chain_kfac_d{depth}_s{sparsity}_{strategy}_n{num_samples}.png"
            image_path = os.path.join(base_dir, image_name)
            image_tensor = load_image_as_tensor(image_path)

            if image_tensor is not None:
                strategy_images.append(image_tensor)
                image_labels.append(f"Chain-KFAC {strategy} s={sparsity}")
                print(f"  ✅ {image_name}")
            else:
                print(f"  ❌ 缺失: {image_name}")

        all_images.extend(strategy_images)

    # 2. 收集SlimGPT图片
    print("收集 SlimGPT 图片...")
    slimgpt_images = []
    for sparsity in sparsities:
        image_name = f"slimgpt_d{depth}_s{sparsity}_n{num_samples}.png"
        image_path = os.path.join(base_dir, image_name)
        image_tensor = load_image_as_tensor(image_path)

        if image_tensor is not None:
            slimgpt_images.append(image_tensor)
            image_labels.append(f"SlimGPT s={sparsity}")
            print(f"  ✅ {image_name}")
        else:
            print(f"  ❌ 缺失: {image_name}")

    all_images.extend(slimgpt_images)

    if not all_images:
        print("❌ 没有找到任何图片文件")
        return

    print(f"总计收集到 {len(all_images)} 张图片")

    # 3. 创建拼接图 - 4列布局
    print("创建拼接图...")
    grid = make_grid(all_images, nrow=4, padding=10, pad_value=1.0)
    save_image(grid, f"{base_dir}/all_methods_comparison.png")
    print(f"✅ 完整对比图: {base_dir}/all_methods_comparison.png")

    # 4. 分别创建Chain KFAC和SlimGPT的拼接图
    if len(all_images) >= 12:  # Chain KFAC: 3×4=12
        chain_images = all_images[:12]
        chain_grid = make_grid(chain_images, nrow=4, padding=10, pad_value=1.0)
        save_image(chain_grid, f"{base_dir}/chain_kfac_comparison.png")
        print(f"✅ Chain KFAC对比图: {base_dir}/chain_kfac_comparison.png")

    if len(all_images) >= 16:  # SlimGPT: 4
        slimgpt_images = all_images[12:16]
        slimgpt_grid = make_grid(slimgpt_images, nrow=4, padding=10, pad_value=1.0)
        save_image(slimgpt_grid, f"{base_dir}/slimgpt_comparison.png")
        print(f"✅ SlimGPT对比图: {base_dir}/slimgpt_comparison.png")

    # 5. 创建详细布局说明
    create_layout_description(base_dir, len(all_images))

def create_layout_description(output_dir, total_images):
    """创建布局说明文件"""
    layout_info = f"""
# VAR 剪枝方法对比实验结果

## 生成的对比图片

### 1. 完整对比图
- **文件**: all_methods_comparison.png
- **布局**: 4列 × {(total_images + 3) // 4}行
- **内容**: 所有方法的完整对比

### 2. Chain KFAC 对比图
- **文件**: chain_kfac_comparison.png
- **布局**: 4列 × 3行 (sparsity: 0.1, 0.2, 0.3, 0.4)
- **行1**: Chain-KFAC natural strategy
- **行2**: Chain-KFAC equal strategy
- **行3**: Chain-KFAC sqrt strategy

### 3. SlimGPT 对比图
- **文件**: slimgpt_comparison.png
- **布局**: 4列 × 1行 (sparsity: 0.1, 0.2, 0.3, 0.4)

## 分析要点

1. **横向对比**: 同一行内不同sparsity的效果
2. **纵向对比**: 不同方法在相同sparsity下的表现
3. **策略对比**: Chain KFAC的三种scale策略的差异

## 实验设置
- **模型**: VAR-16
- **测试类别**: 980 (volcano)
- **校准样本**: 150
- **Sparsity**: 0.1, 0.2, 0.3, 0.4
- **Scale Strategies**: natural, equal, sqrt

生成时间: {datetime.now()}
"""

    with open(f"{output_dir}/layout_description.md", 'w', encoding='utf-8') as f:
        f.write(layout_info)

    print(f"✅ 布局说明: {output_dir}/layout_description.md")

if __name__ == "__main__":
    main()