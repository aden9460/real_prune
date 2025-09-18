# from torch_fidelity import calculate_metrics

# metrics = calculate_metrics(
#     input1='/home/wangzefang/Projects/EdgeVAR/VAR_FIDtest/output/FID_test/multiscaletestrand9_d24_0.2sparsity_200i_eva_scale_slimgpt.pth',
#     input2='/home/wangzefang/Projects/EdgeVAR/LlamaGen/autoregressive/virtual_images/images',
#     cuda=True,  # 如果要用GPU，改为True
#     fid=True,
#     isc=True,
#     prc=True
# )

# print(metrics)
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    # 获取所有符合格式的文件
    png_files = sorted([
        f for f in os.listdir(sample_dir)
        if f.endswith('.png') and '_' in f and 'img_' in f
    ])
    if num is not None:
        png_files = png_files[:num]
    for fname in tqdm(png_files, desc="Building .npz file from samples"):
        sample_pil = Image.open(os.path.join(sample_dir, fname))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape[3] == 3  # 检查通道数
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

if __name__ == '__main__':
    create_npz_from_sample_folder('/home/suanba/EdgeVAR/real_prune/VAR_FIDtest/output/real_d16_0.2sparsity_150i_slimgpt_1epoch_method')