################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

from utils import arg_util
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from coremltools.models.neural_network.quantization_utils import quantize_weights
MODEL_DEPTH = 24    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {12,16, 20, 24, 30}
# from pytorch_model_summary import summary
# from ptflops import get_model_complexity_info
# from thop import profile
# from thop import clever_format
# from torchstat import stat
import time
from utils import arg_util
# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
# vae_ckpt, var_ckpt = '/home/wangzefang/Projects/project/VAR/model_zoo/vae_ch160v4096z32.pth', f'/home/wangzefang/Projects/project/VAR/model_zoo/var_d{MODEL_DEPTH}.pth'
vae_ckpt, var_ckpt = '/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/vae_ch160v4096z32.pth', f'/home/suanba/EdgeVAR/slimgpt_pub/model_zoo/model_zoo/var_d{MODEL_DEPTH}.pth'
# var_ckpt = '/home/suanba/EdgeVAR/real_prune/VAR_train/0.4_d16_real_20epoch_slim/ar-ckpt-best.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')


args: arg_util.Args = arg_util.init_dist_and_get_args()
# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        args=args  # 添加args参数
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

checkpoint = torch.load(var_ckpt, map_location='cpu')


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
        # dynamic_register_pruned_indices(var, model_weights)
        adapt_fc_weights(var, model_weights)
        var.load_state_dict(model_weights, strict=True)
        print("成功从训练检查点提取模型权重")
    else:
        print("警告：在检查点中未找到var_wo_ddp，尝试直接加载...")
        # dynamic_register_pruned_indices(var, checkpoint)
        adapt_fc_weights(var, checkpoint)
        var.load_state_dict(checkpoint, strict=True)
else:
    print("加载原始模型权重...")
    dynamic_register_pruned_indices(var, checkpoint)
    # adapt_fc_weights(var, checkpoint)
    var.load_state_dict(checkpoint, strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

# print(var)
# print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
# print(vae)
# load checkpoints

# vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

# vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu', weights_only=True), strict=False)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu', weights_only=True), strict=False)

# vae.eval(), var.eval()
# for p in vae.parameters(): p.requires_grad_(False)
# for p in var.parameters(): p.requires_grad_(False)
# print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
seed = 1 #@param {type:"number"}
torch.manual_seed(seed)
# num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
# cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
# class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
class_labels = (980) #@param {type:"raw"}
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # run faster
# tf32 = True
# torch.backends.cudnn.allow_tf32 = bool(tf32)
# torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
# torch.set_float32_matmul_precision('high' if tf32 else 'highest')


# # sample
# B = len(class_labels)
# label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
# label_B = torch.tensor(class_labels, device=device)
label_B = torch.tensor(class_labels).unsqueeze(0) 
# # with torch.inference_mode():
# #     with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
# #         recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
NAME = "d16_try6"

import coremltools as ct

example_input = label_B.to(device)
# input = torch.rand(1,32,16,16).cuda()

# model_summary = summary(var, example_input)
torch.cuda.synchronize()
zero = time.time()
result = var(example_input)
print(result.shape)
torch.cuda.synchronize()
end = time.time()
img = result[0].permute(1, 2, 0).mul(255).cpu().numpy()
img = PImage.fromarray(img.astype(np.uint8))
img.save(f'current.png')
print("all:{}",end-zero)

# model_summary = summary(vae.fhat_to_img, input)
# dummy_input = torch.randn(1, 3, 256, 256)   # 按你的输入shape修改
# macs, params = profile(var, inputs=(example_input,))

# input  = torch.tensor([1,2,3],dtype=torch.int32)
# result = stat(var,example_input)
# macs,params = get_model_complexity_info(var, example_input, as_strings=True, print_per_layer_stat=True)
# MACs, params = profile(vae.fhat_to_img, inputs=(torch.rand(256,16,16),))
# macs, params = clever_format([macs, params], '%.3f')
# 
# print(f"运算量：{macs}, 参数量：{params}")
# print(result)
# print(model_summary)


traced_model = torch.jit.trace(var, example_input)

traced_model.save(f"var_traced_d{MODEL_DEPTH}_ios15_test{NAME}.pt")

mlmodel = ct.convert(f"var_traced_d{MODEL_DEPTH}_ios15_test{NAME}.pt",
                    inputs=[ct.TensorType(shape=example_input.shape)],debug=True,compute_precision=ct.precision.FLOAT16)
# # 保存模型
mlmodel.save(f"mlmodel/model_d{MODEL_DEPTH}_ios15_test{NAME}_quantized.mlpackage")




# mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=example_input.shape)])#succeed
# # 量化权重到 16 位
# model = ct.models.MLModel("/home/wangzefang/project/VAR/model_d16_ios15_test2_quantized.mlpackage")
# quantized_model = quantize_weights(model, nbits=16)

# # 保存量化后的模型
# quantized_model.save("model_quantized_test2.mlpackage")

# chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
# chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
# chw = PImage.fromarray(chw.astype(np.uint8))
# chw.show()
