import sys
sys.path.append("..")

import os
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import time
import json
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
import PIL.ImageFont as PImageFont

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var, VAR
import torch.nn as nn

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

# CoreML support (optional)
try:
    import coremltools as ct
    from coremltools.models import MLModel
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: coremltools not available. CoreML conversion disabled.")


def setup_deterministic(seed: int = 0):
    """Setup deterministic behavior for reproducible testing"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32 settings
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')


class VarInferWrapper(nn.Module):
    """Wrapper for VAR model inference to enable CoreML conversion"""
    def __init__(self, var_model):
        super().__init__()
        self.var = var_model

    def forward(self, B: torch.Tensor, label_B: torch.Tensor, cfg: torch.Tensor,
                top_k: torch.Tensor, top_p: torch.Tensor, g_seed: torch.Tensor,
                more_smooth: torch.Tensor):
        # Use fixed values for CoreML conversion
        return self.var.autoregressive_infer_cfg(
            B=1,
            label_B=label_B,
            cfg=5.0,
            top_k=900,
            top_p=0.96,
            g_seed=0,
            more_smooth=False
        )



def convert_model_to_coreml(
                            model: VAR,
                            save_dir: str = './coreml_models') -> Optional[str]:
    """
    Convert VAR model to CoreML format

    Args:
        model: VAR model to convert
        config: Model configuration
        save_dir: Directory to save CoreML model

    Returns:
        Path to saved CoreML model, or None if conversion failed
    """
    if not COREML_AVAILABLE:
        print("CoreML not available, skipping conversion")
        return None

    print(f"\n{'='*60}")
    print(f"Converting d16_0.4_distill to CoreML...")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    try:
        # Create wrapper model
        wrapper = VarInferWrapper(model)
        wrapper.eval()

        # Define input specifications for CoreML
        wrapper_inputs = {
            "B": (1,),
            "label_B": (1,),
            "cfg": (1,),
            "top_k": (1,),
            "top_p": (1,),
            "g_seed": (1,),
            "more_smooth": (1,)
        }

        # Create example inputs
        default_values = {
            "B": torch.tensor([1], dtype=torch.float32),
            "label_B": torch.tensor([1], dtype=torch.long),
            "cfg": torch.tensor([5.0], dtype=torch.float32),
            "top_k": torch.tensor([900], dtype=torch.float32),
            "top_p": torch.tensor([0.96], dtype=torch.float32),
            "g_seed": torch.tensor([0], dtype=torch.float32),
            "more_smooth": torch.tensor([0], dtype=torch.float32)
        }

        example_inputs = []
        ml_inputs = []
        for k, shp in wrapper_inputs.items():
            ex = default_values[k]
            ml_inputs.append(ct.TensorType(name=k, shape=shp))
            example_inputs.append(ex.to("cuda"))

        # Setup deterministic mode
        setup_deterministic(seed=0)

        # Move wrapper to device
        wrapper = wrapper.to("cuda")

        print("Tracing model...")
        # Trace the model
        traced = torch.jit.trace(wrapper, tuple(example_inputs))

        print("Converting to CoreML...")
        # Convert to CoreML
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=ml_inputs,
            minimum_deployment_target=ct.target.iOS17
        )

        # Save model
        model_name = "d16_0.4_distill.mlpackage"
        out_path = osp.join(save_dir, model_name)
        mlmodel.save(out_path)

        # Get model size
        import subprocess
        try:
            result = subprocess.run(['du', '-sh', out_path], capture_output=True, text=True)
            size_str = result.stdout.split()[0] if result.returncode == 0 else "Unknown"
        except:
            size_str = "Unknown"

        print(f"✓ CoreML model saved: {out_path}")
        print(f"  Model size: {size_str}")
        print(f"{'='*60}\n")

        return out_path

    except Exception as e:
        print(f"✗ CoreML conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__=="__main__":

    MODEL_DEPTH =  args.depth   # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {12,16, 20, 24, 30}
    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    # vae_ckpt, var_ckpt = '/wanghuan/data/wangzefang/slim_VAR_copy/VAR/model_zoo/vae_ch160v4096z32.pth', f'/wanghuan/data/wangzefang/slim_VAR_copy/VAR/model_zoo/var_d{MODEL_DEPTH}.pth'
    vae_ckpt = '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
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

    if 'trainer' in checkpoint:
        print("检测到训练检查点文件，正在提取模型权重...")
        if 'var_wo_ddp' in checkpoint['trainer']:
            model_weights = checkpoint['trainer']['var_wo_ddp']
            # dynamic_register_pruned_indices(var, model_weights)
            # adapt_fc_weights(var, model_weights)
            var.load_state_dict(model_weights, strict=True)
            print("成功从训练检查点提取模型权重")
        else:
            print("警告：在检查点中未找到var_wo_ddp，尝试直接加载...")
            # dynamic_register_pruned_indices(var, checkpoint)
            # adapt_fc_weights(var, checkpoint)
            var.load_state_dict(checkpoint, strict=True)
    else:
        print("加载原始模型权重...")
        # dynamic_register_pruned_indices(var, checkpoint)
        # adapt_fc_weights(var, checkpoint)
        var.load_state_dict(checkpoint, strict=True)
    # var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    convert_model_to_coreml(var)