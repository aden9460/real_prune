################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from torchvision.utils import save_image
# MODEL_DEPTH = 16    # TODO: =====> please specify MODEL_DEPTH <=====
# assert MODEL_DEPTH in {16, 20, 24, 30}
from utils import arg_util, misc
from tqdm import tqdm
from PIL import Image
args: arg_util.Args = arg_util.init_dist_and_get_args()

MODEL_DEPTH =  args.depth   # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {12,16, 20, 24, 30}
# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
# vae_ckpt, var_ckpt = '/wanghuan/data/wangzefang/slim_VAR_copy/VAR/model_zoo/vae_ch160v4096z32.pth', f'/wanghuan/data/wangzefang/slim_VAR_copy/VAR/model_zoo/var_d{MODEL_DEPTH}.pth'
vae_ckpt = '/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth'
var_ckpt = args.var_model
print(var_ckpt)
#    /home/wangzefang/Projects/project/slim_VAR/slimgpt_pub/sparsity_model/d24_0.4var_1i_256input.pth
# if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
# if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')
if not osp.exists(vae_ckpt): print("var not exist")
if not osp.exists(var_ckpt): print("var not exist")
# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,args=args
    )


# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

checkpoint = torch.load(var_ckpt, map_location='cpu')

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


def dynamic_register_pruned_indices(var, model_weights):
    i = 0
    while True:
        key = f'blocks.{i}.attn.pruned_indices'
        if key not in model_weights:
            break
        attn = var.blocks[i].attn
        pruned_indices_ckpt = model_weights[key]
        if hasattr(attn, 'pruned_indices'):
            del attn._buffers['pruned_indices']
        attn.register_buffer('pruned_indices', torch.zeros_like(pruned_indices_ckpt))
        attn.pruned_indices.copy_(pruned_indices_ckpt)
        i += 1

def adapt_weight_shape(param, ckpt_param):
    # param: 当前模型的参数 (nn.Parameter)
    # ckpt_param: checkpoint里的参数 (Tensor)
    if param.shape == ckpt_param.shape:
        return ckpt_param
    min_shape = tuple(min(a, b) for a, b in zip(param.shape, ckpt_param.shape))
    # 只支持2D和1D
    if len(param.shape) == 2:
        new_param = torch.zeros_like(param)
        # 裁剪/填充
        new_param[:min_shape[0], :min_shape[1]] = ckpt_param[:min_shape[0], :min_shape[1]]
    elif len(param.shape) == 1:
        new_param = torch.zeros_like(param)
        new_param[:min_shape[0]] = ckpt_param[:min_shape[0]]
    else:
        raise ValueError("Only support 1D/2D param for fc1/fc2")
    return new_param

def adapt_fc_weights(var, model_weights):
    i = 0
    while True:
        prefix = f'blocks.{i}.ffn.'
        fc1_w = prefix + 'fc1.weight'
        fc1_b = prefix + 'fc1.bias'
        fc2_w = prefix + 'fc2.weight'
        fc2_b = prefix + 'fc2.bias'
        if fc1_w not in model_weights or fc2_w not in model_weights:
            break
        block = var.blocks[i]
        # fc1
        model_weights[fc1_w] = adapt_weight_shape(block.ffn.fc1.weight, model_weights[fc1_w])
        model_weights[fc1_b] = adapt_weight_shape(block.ffn.fc1.bias, model_weights[fc1_b])
        # fc2
        model_weights[fc2_w] = adapt_weight_shape(block.ffn.fc2.weight, model_weights[fc2_w])
        model_weights[fc2_b] = adapt_weight_shape(block.ffn.fc2.bias, model_weights[fc2_b])
        i += 1

if 'trainer' in checkpoint:
    print("检测到训练检查点文件，正在提取模型权重...")
    if 'var_wo_ddp' in checkpoint['trainer']:
        model_weights = checkpoint['trainer']['var_wo_ddp']
        dynamic_register_pruned_indices(var, model_weights)
        adapt_fc_weights(var, model_weights)
        var.load_state_dict(model_weights, strict=True)
        print("成功从训练检查点提取模型权重")
    else:
        print("警告：在检查点中未找到var_wo_ddp，尝试直接加载...")
        dynamic_register_pruned_indices(var, checkpoint)
        adapt_fc_weights(var, checkpoint)
        var.load_state_dict(checkpoint, strict=True)
else:
    print("加载原始模型权重...")
    dynamic_register_pruned_indices(var, checkpoint)
    adapt_fc_weights(var, checkpoint)
    var.load_state_dict(checkpoint, strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 1.5 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
class_num = 0
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')
output_name = args.output_name
save_dir = f"/home/suanba/EdgeVAR/real_prune/VAR_FIDtest/output/{output_name}"
os.makedirs(save_dir,exist_ok=True)
# sample
progress_bar = tqdm(total=1000, desc="生成FID样本")

with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        for class_num in range(1000):
            progress_bar.update(1)
            class_labels  = torch.full((50,), class_num, dtype=torch.long)
            B = len(class_labels)
            label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
            for i in range(recon_B3HW.shape[0]):
                img = recon_B3HW[i].permute(1, 2, 0).mul(255).cpu().numpy()
                img = PImage.fromarray(img.astype(np.uint8))
                img.save(f'{save_dir}/{class_num:03d}_img_{i:03d}.png')
progress_bar.close()

create_npz_from_sample_folder(save_dir)

# chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
# chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
# chw = PImage.fromarray(chw.astype(np.uint8))
# chw.show()
