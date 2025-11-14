################## 1. 生成1000个类别图像用于FID评估
import os
import os.path as osp
import torch, torchvision
import torch.nn.functional as F
import random
import numpy as np
import PIL.Image as PImage
import argparse
import sys
import time
from tqdm import tqdm

# 添加VAR根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加命令行参数
parser = argparse.ArgumentParser(description='VAR图像生成与性能测试')
parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30, 36], help='模型深度')
parser.add_argument('--seed', type=int, default=0, help='随机种子')
parser.add_argument('--cfg', type=float, default=1.5, help='分类器引导强度')
parser.add_argument('--pn', type=str, default='256', choices=['256', '512', '1024'], help='patch数量')
parser.add_argument('--more_smooth', action='store_true', help='更平滑的输出')
parser.add_argument('--output_dir', type=str, default='/wanghuan/data/wangzefang/VAR/FID_test/image/baseline_d24_0.4_5epoch', help='输出目录')
parser.add_argument('--samples_per_class', type=int, default=50, help='每个类别生成的样本数量')
parser.add_argument('--total_classes', type=int, default=1000, help='要生成的类别总数')
parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
parser.add_argument('--sparsity', type=float, default=0.6, help='稀疏度')
args = parser.parse_args()

# 禁用默认参数初始化以加快速度
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import build_vae_var

MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

# 指定权重存放目录
vae_ckpt = '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
# var_ckpt = f'/home/project/daily/AR/model_zoo/var_d{MODEL_DEPTH}.pth'
var_ckpt = "/home/project/real_prune/slimvar/pruned_models/qscaluefix_var_d16_0.2_1000sample_entro.pth"
# var_ckpt = "/home/project/real_prune/slimvar/pruned_models/qscaluefix_var_d16_0.4_1000sample_mag.pth"

# 构建模型
if args.pn == '256':
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,  # VQVAE超参数
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,args=args
)

# 加载权重
print("加载模型权重...")
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

# checkpoint = torch.load(var_ckpt, map_location='cpu')
# if 'trainer' in checkpoint:
#     print("检测到训练检查点文件，正在提取模型权重...")
#     if 'var_wo_ddp' in checkpoint['trainer']:
#         model_weights = checkpoint['trainer']['var_wo_ddp']
#         var.load_state_dict(model_weights, strict=True)
#         print("成功从训练检查点提取模型权重")
#     else:
#         print("警告：在检查点中未找到var_wo_ddp，尝试直接加载...")
#         var.load_state_dict(checkpoint, strict=True)
# else:
#     print("加载原始模型权重...")
#     var.load_state_dict(checkpoint, strict=True)

# vae.eval(), var.eval()
# for p in vae.parameters(): p.requires_grad_(False)
# for p in var.parameters(): p.requires_grad_(False)
# print(f'模型准备完成')

# 从models.helpers导入必要的函数
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_

# 设置参数
seed = args.seed
cfg = args.cfg
more_smooth = args.more_smooth

print(f"使用参数: 模型深度={MODEL_DEPTH}, 随机种子={seed}, CFG强度={cfg}")
print(f"平滑模式: {'开启' if more_smooth else '关闭'}")
print(f"生成样本: {args.total_classes} 个类别，每类 {args.samples_per_class} 张图像")

# 设置随机种子
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 加速设置
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

save_dir = "/wanghuan/data/wangzefang/VAR/FID_test/image/baseline_d24_8_sparsity_0.4_3epoch_wrong_pruning"
os.makedirs(save_dir,exist_ok=True)
# sample

with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        # for class_num in range(5):
            # class_labels  = torch.full((10,), class_num, dtype=torch.long).cuda()
        class_num=0
        class_labels = torch.full((64,), class_num, dtype=torch.long).cuda()
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        for i in range(10):
            t1=time.time()
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
            print(f"total is {time.time()-t1}")
        for image in range(recon_B3HW.shape[0]):
            img = recon_B3HW[image].permute(1, 2, 0).mul(255).cpu().numpy()
            img = PImage.fromarray(img.astype(np.uint8))
            img.save(f'{save_dir}/class_{0:04d}_seed_{args.seed + image:06d}.png')

# print(f"\n已完成 {completed_images} 张FID评估样本的生成")
# print(f"样本已保存至: {args.output_dir}")
# print("完成!")