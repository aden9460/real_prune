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
parser.add_argument('--model_depth', type=int, default=24, choices=[16, 20, 24, 30, 36], help='模型深度')
parser.add_argument('--seed', type=int, default=0, help='随机种子')
parser.add_argument('--cfg', type=float, default=1.5, help='分类器引导强度')
parser.add_argument('--pn', type=str, default='256', choices=['256', '512', '1024'], help='patch数量')
parser.add_argument('--more_smooth', action='store_true', help='更平滑的输出')
parser.add_argument('--output_dir', type=str, default='/wanghuan/data/wangzefang/VAR/FID_test/image/baseline_d24_0.4_5epoch', help='输出目录')
parser.add_argument('--samples_per_class', type=int, default=50, help='每个类别生成的样本数量')
parser.add_argument('--total_classes', type=int, default=1000, help='要生成的类别总数')
parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
parser.add_argument('--sparsity', type=float, default=0.4, help='稀疏度')
args = parser.parse_args()

# 禁用默认参数初始化以加快速度
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import build_vae_var

MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

# 指定权重存放目录
vae_ckpt = '/wanghuan/data/wangzefang/slim_VAR_copy/VAR/model_zoo/vae_ch160v4096z32.pth'
var_ckpt = '/wanghuan/data/wangzefang/slim_VAR_copy/VAR/local_output_1/ar-ckpt-last.pth'

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

checkpoint = torch.load(var_ckpt, map_location='cpu')
if 'trainer' in checkpoint:
    print("检测到训练检查点文件，正在提取模型权重...")
    if 'var_wo_ddp' in checkpoint['trainer']:
        model_weights = checkpoint['trainer']['var_wo_ddp']
        var.load_state_dict(model_weights, strict=True)
        print("成功从训练检查点提取模型权重")
    else:
        print("警告：在检查点中未找到var_wo_ddp，尝试直接加载...")
        var.load_state_dict(checkpoint, strict=True)
else:
    print("加载原始模型权重...")
    var.load_state_dict(checkpoint, strict=True)

vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'模型准备完成')

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

def generate_images_batch(class_ids, batch_indices, base_seed):
    """生成一批图像"""

    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 设置随机种子
    cur_seed = base_seed + batch_indices[0]
    torch.manual_seed(cur_seed)
    random.seed(cur_seed)
    np.random.seed(cur_seed)

    # 准备标签
    B = len(class_ids)
    label_B = torch.tensor(class_ids, device=device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            # 设置随机数生成器
            var.rng.manual_seed(cur_seed)
            rng = var.rng

            # 处理标签输入
            sos = cond_BD = var.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0))

            # 计算层级位置编码
            lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC

            # 构建初始token_map
            next_token_map = sos.unsqueeze(1).expand(2 * B, var.first_l, -1) + var.pos_start.expand(2 * B, var.first_l, -1) + lvl_pos[:, :var.first_l]

            # 初始化当前处理的token数量和特征图
            cur_L = 0
            f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])

            # 为所有Transformer块启用KV缓存，加速自回归生成
            for b in var.blocks: 
                b.attn.kv_caching(True)

            # 获取量化器相关参数
            quant_resi = var.vae_quant_proxy[0].quant_resi

            # 循环处理每个阶段
            for si, pn in enumerate(var.patch_nums):
                # 计算当前阶段的进度比例
                ratio = si / var.num_stages_minus_1
                # 更新当前处理的token数量
                cur_L += pn*pn

                # 处理条件嵌入
                cond_BD_or_gss = var.shared_ada_lin(cond_BD)

                # 设置当前输入
                x = next_token_map

                # 通过所有Transformer块处理输入
                for b in var.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

                # 获取logits
                logits_BlV = var.get_logits(x, cond_BD)

                # 计算CFG强度，随着生成进度增加
                t = cfg * ratio
                # 应用分类器无关引导
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                # 使用top-k/top-p采样生成token索引
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=900, top_p=0.95, num_samples=1)[:, :, 0]

                if not more_smooth:
                    # 将token索引转换为嵌入向量
                    h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
                else:
                    # 使用gumbel softmax平滑预测
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

                # 重塑嵌入向量为特征图
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, var.Cvae, pn, pn)

                # 更新特征图和准备下一阶段的输入
                HW = var.patch_nums[-1]  # 最终尺寸（16x16）
                if si != len(var.patch_nums)-1:
                    # 除了最后一步，将特征图上采样到最大分辨率
                    h = quant_resi[si/(len(var.patch_nums)-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
                    f_hat.add_(h)  # 将上采样后的特征图添加到f_hat
                    # 下采样到下一步的特征图大小
                    next_token_map = F.interpolate(f_hat, size=(var.patch_nums[si+1], var.patch_nums[si+1]), mode='area')
                else:
                    # 最后一步，直接使用不需要上采样
                    h = quant_resi[si/(len(var.patch_nums)-1)](h_BChw)
                    f_hat.add_(h)
                    next_token_map = f_hat

                # 如果不是最后一个阶段，准备下一阶段的输入
                if si != var.num_stages_minus_1:
                    next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
                    next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)

            # 禁用KV缓存
            for b in var.blocks: 
                b.attn.kv_caching(False)

            # 生成最终图像
            recon_B3HW = var.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    # 保存图像
    for i, (class_id, img_idx) in enumerate(zip(class_ids, batch_indices)):
        img = recon_B3HW[i].permute(1, 2, 0).mul(255).cpu().numpy()
        img = PImage.fromarray(img.astype(np.uint8))
        img.save(f'{args.output_dir}/class_{class_id:04d}_seed_{args.seed + img_idx:06d}.png')

# 计算需要的批次数量
batch_size = args.batch_size
total_images = args.total_classes * args.samples_per_class
total_batches = (total_images + batch_size - 1) // batch_size

save_dir = "/wanghuan/data/wangzefang/VAR/FID_test/image/baseline_d24_8_sparsity_0.4_3epoch_wrong_pruning"
os.makedirs(save_dir,exist_ok=True)
# sample

with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        # for class_num in range(5):
            # class_labels  = torch.full((10,), class_num, dtype=torch.long).cuda()
        class_num=0
        class_labels = (980, 980, 437, 437, 22, 22, 562, 562)
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
        for image in range(recon_B3HW.shape[0]):
            img = recon_B3HW[image].permute(1, 2, 0).mul(255).cpu().numpy()
            img = PImage.fromarray(img.astype(np.uint8))
            img.save(f'{save_dir}/class_{0:04d}_seed_{args.seed + image:06d}.png')

# print(f"\n已完成 {completed_images} 张FID评估样本的生成")
# print(f"样本已保存至: {args.output_dir}")
# print("完成!")