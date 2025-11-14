"""
Comprehensive VAR Model Speed Testing Framework
Supports depth pruning, width pruning, and structured pruning on Linux and CoreML
"""

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

# CoreML support (optional)
try:
    import coremltools as ct
    from coremltools.models import MLModel
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: coremltools not available. CoreML conversion disabled.")


@dataclass
class PruningConfig:
    """Configuration for a pruned model variant"""
    name: str
    depth: int  # Number of transformer layers
    width: int  # Embedding dimension
    num_heads: int  # Number of attention heads
    description: str

    @property
    def embed_dim(self):
        return self.width

    def __str__(self):
        return f"{self.name}: depth={self.depth}, width={self.width}, heads={self.num_heads}"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    config_name: str
    platform: str
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    throughput_imgs_per_sec: float
    memory_used_mb: float
    peak_memory_mb: float
    num_params: int
    model_size_mb: float
    warmup_runs: int
    test_runs: int
    batch_size: int

    def to_dict(self):
        return asdict(self)


class MemoryProfiler:
    """Profile memory usage during model execution"""

    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.initial_memory = 0
        self.peak_memory = 0

    def start(self):
        """Start memory profiling"""
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            # For CPU, we'll just track basic metrics
            self.initial_memory = 0

    def get_current_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.use_cuda:
            return torch.cuda.memory_allocated() / 1024**2
        else:
            return 0.0  # CPU memory tracking not available without psutil

    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB"""
        if self.use_cuda:
            return torch.cuda.max_memory_allocated() / 1024**2
        else:
            return 0.0

    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        current = self.get_current_usage()
        peak = self.get_peak_usage()
        return {
            'current_mb': current,
            'peak_mb': peak,
            'allocated_mb': current - self.initial_memory,
        }


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


class VARSpeedTester:
    """Comprehensive speed testing framework for VAR models"""

    def __init__(self,
                 vae_ckpt: str,
                 device: str = 'cuda',
                 patch_nums: Tuple = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)):
        self.vae_ckpt = vae_ckpt
        self.device = device
        self.patch_nums = patch_nums
        self.vae = None
        self.results: List[BenchmarkResult] = []

        # Load VAE once (shared across all VAR variants)
        self._load_vae()

    def _load_vae(self):
        """Load the VQVAE model"""
        print(f"Loading VQVAE from {self.vae_ckpt}")
        vae, _ = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=self.device, patch_nums=self.patch_nums,
            num_classes=1000, depth=16, shared_aln=False,
        )
        vae.load_state_dict(torch.load(self.vae_ckpt, map_location='cpu'), strict=True)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae = vae
        print("VQVAE loaded successfully")

    def load_pruned_weights_from_full_model(self,
                                           pruned_model: VAR,
                                           full_state_dict: dict,
                                           config: PruningConfig,
                                           base_depth: int = 16) -> bool:
        """
        Load pruned weights from full pretrained model

        Inherits weights from base checkpoint:
        - Depth pruning: Load first N layers
        - Width pruning: Slice embedding dimensions (preserves head structure)
        - Structured: Apply both

        Args:
            pruned_model: Pruned VAR model
            full_state_dict: Full base model pretrained weights
            config: Pruning configuration
            base_depth: Base model depth (for inferring base configuration)

        Returns:
            True if successful
        """
        try:
            print(f"  Inheriting weights from base checkpoint...")
            print(f"  Target: depth={config.depth}, width={config.width}, heads={config.num_heads}")

            # Infer base model configuration (following demo.py convention)
            base_config = {
                'depth': base_depth,
                'embed_dim': base_depth * 64,
                'num_heads': base_depth
            }
            print(f"  Base: depth={base_config['depth']}, width={base_config['embed_dim']}, heads={base_config['num_heads']}")

            pruned_state = pruned_model.state_dict()

            # Track attention layers for debugging
            attention_layers_pruned = 0

            # Load weights with pruning
            for key in pruned_state.keys():
                if key in full_state_dict:
                    full_weight = full_state_dict[key]
                    target_shape = pruned_state[key].shape

                    # Prune tensor to match target shape
                    if full_weight.shape == target_shape:
                        pruned_state[key] = full_weight.clone()
                    else:
                        pruned_state[key] = self._prune_tensor(
                            full_weight, target_shape,
                            layer_name=key,
                            config=config,
                            base_config=base_config
                        )

                        # Track attention layer pruning
                        if 'attn' in key:
                            attention_layers_pruned += 1

            # Load the pruned weights
            missing, unexpected = pruned_model.load_state_dict(pruned_state, strict=False)

            print(f"  ✓ Weights inherited successfully")
            print(f"    Attention layers pruned: {attention_layers_pruned}")
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")

            return True

        except Exception as e:
            print(f"  ✗ Weight inheritance failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prune_attention_weight(self,
                                full_tensor: torch.Tensor,
                                target_shape: tuple,
                                layer_name: str,
                                old_heads: int,
                                new_heads: int,
                                old_dim: int,
                                new_dim: int) -> torch.Tensor:
        """
        Prune attention layer weights by preserving head structure
        Handles mat_qkv (combined QKV) and proj layers in VAR

        Args:
            full_tensor: Original weight tensor
            target_shape: Target shape after pruning
            layer_name: Name of the layer
            old_heads: Number of heads in original model
            new_heads: Number of heads in pruned model
            old_dim: Original embedding dimension
            new_dim: Target embedding dimension
        """
        try:
            head_dim = old_dim // old_heads

            # Handle mat_qkv weight: [3*old_dim, old_dim]
            if 'mat_qkv.weight' in layer_name:
                # Reshape to [3, old_heads, head_dim, old_dim]
                reshaped = full_tensor.reshape(3, old_heads, head_dim, old_dim)
                # Select first new_heads and first new_dim
                pruned = reshaped[:, :new_heads, :, :new_dim]
                # Reshape back to [3*new_heads*head_dim, new_dim]
                return pruned.reshape(3 * new_heads * head_dim, new_dim).clone()

            # Handle q_bias, v_bias: [old_dim]
            elif any(x in layer_name for x in ['q_bias', 'v_bias', 'zero_k_bias']):
                # Reshape to [old_heads, head_dim]
                reshaped = full_tensor.reshape(old_heads, head_dim)
                # Select first new_heads
                pruned = reshaped[:new_heads, :]
                # Reshape back to [new_heads * head_dim]
                return pruned.reshape(new_heads * head_dim).clone()

            # Handle projection weight: [old_dim, old_dim]
            elif 'proj.weight' in layer_name:
                # Input dim: slice heads, Output dim: slice embedding
                # Reshape to [old_dim, old_heads, head_dim]
                reshaped = full_tensor.reshape(old_dim, old_heads, head_dim)
                # Select first new_dim output and first new_heads input
                pruned = reshaped[:new_dim, :new_heads, :]
                # Reshape back to [new_dim, new_heads * head_dim]
                return pruned.reshape(new_dim, new_heads * head_dim).clone()

            # Handle projection bias: [old_dim]
            elif 'proj.bias' in layer_name:
                return full_tensor[:new_dim].clone()

            else:
                # Fallback to simple slicing
                slices = tuple(slice(0, dim) for dim in target_shape)
                return full_tensor[slices].clone()

        except Exception as e:
            print(f"    Warning: Attention pruning failed for {layer_name}: {e}")
            print(f"    Falling back to simple slicing")
            slices = tuple(slice(0, dim) for dim in target_shape)
            return full_tensor[slices].clone()

    def _prune_ffn_weight(self,
                         full_tensor: torch.Tensor,
                         target_shape: tuple,
                         layer_name: str,
                         old_dim: int,
                         new_dim: int,
                         mlp_ratio: float = 4.0) -> torch.Tensor:
        """
        Prune FFN layer weights maintaining mlp_ratio

        Args:
            full_tensor: Original weight tensor
            target_shape: Target shape after pruning
            layer_name: Name of the layer
            old_dim: Original embedding dimension
            new_dim: Target embedding dimension
            mlp_ratio: MLP expansion ratio (default: 4.0)
        """
        try:
            old_mlp_dim = int(old_dim * mlp_ratio)
            new_mlp_dim = int(new_dim * mlp_ratio)

            # Handle fc1 weight: [old_mlp_dim, old_dim] -> [new_mlp_dim, new_dim]
            if 'fc1.weight' in layer_name:
                return full_tensor[:new_mlp_dim, :new_dim].clone()

            # Handle fc1 bias: [old_mlp_dim] -> [new_mlp_dim]
            elif 'fc1.bias' in layer_name:
                return full_tensor[:new_mlp_dim].clone()

            # Handle fc2 weight: [old_dim, old_mlp_dim] -> [new_dim, new_mlp_dim]
            elif 'fc2.weight' in layer_name:
                return full_tensor[:new_dim, :new_mlp_dim].clone()

            # Handle fc2 bias: [old_dim] -> [new_dim]
            elif 'fc2.bias' in layer_name:
                return full_tensor[:new_dim].clone()

            else:
                # Fallback
                slices = tuple(slice(0, dim) for dim in target_shape)
                return full_tensor[slices].clone()

        except Exception as e:
            print(f"    Warning: FFN pruning failed for {layer_name}: {e}")
            print(f"    Falling back to simple slicing")
            slices = tuple(slice(0, dim) for dim in target_shape)
            return full_tensor[slices].clone()

    def _prune_tensor(self,
                      full_tensor: torch.Tensor,
                      target_shape: tuple,
                      layer_name: str = "",
                      config: Optional[PruningConfig] = None,
                      base_config: Optional[Dict] = None) -> torch.Tensor:
        """
        Intelligently prune tensor based on layer type

        Width pruning strategy:
        - Transformer blocks (attn + ffn + ln): Apply width pruning
        - Embedding layers: Keep original (no pruning)
        - Input/output projection: Prune only relevant dimensions

        Args:
            full_tensor: Original weight tensor
            target_shape: Target shape after pruning
            layer_name: Name of the layer (for routing)
            config: Pruning configuration
            base_config: Base model configuration
        """
        try:
            # Skip width pruning for embeddings and certain special layers
            skip_pruning_patterns = [
                'class_emb',      # Class embedding
                'pos_start',      # Position start
                'pos_1LC',        # Position embedding
                'lvl_embed',      # Level embedding
                'lvl_1L',         # Level tensor
                'attn_bias',      # Attention bias mask
            ]

            if any(pattern in layer_name for pattern in skip_pruning_patterns):
                # These layers should not be pruned for width
                # Return original if shapes match, otherwise error
                if full_tensor.shape == target_shape:
                    return full_tensor.clone()
                else:
                    print(f"    Warning: Shape mismatch for {layer_name}, expected no pruning")
                    return full_tensor.clone()

            # Check if this is a width pruning scenario
            is_width_pruning = (config and base_config and
                              config.embed_dim != base_config['embed_dim'])

            if not is_width_pruning:
                # Depth pruning only - simple slicing is fine
                slices = tuple(slice(0, dim) for dim in target_shape)
                return full_tensor[slices].clone()

            # Width pruning: dispatch to appropriate handler
            old_dim = base_config['embed_dim']
            new_dim = config.embed_dim

            # 1. Attention layers in transformer blocks
            if 'blocks.' in layer_name and 'attn.' in layer_name:
                old_heads = base_config['num_heads']
                new_heads = config.num_heads
                return self._prune_attention_weight(
                    full_tensor, target_shape, layer_name,
                    old_heads, new_heads, old_dim, new_dim
                )

            # 2. FFN layers in transformer blocks
            elif 'blocks.' in layer_name and 'ffn.' in layer_name:
                return self._prune_ffn_weight(
                    full_tensor, target_shape, layer_name,
                    old_dim, new_dim, mlp_ratio=4.0
                )

            # 3. LayerNorm in transformer blocks
            elif 'blocks.' in layer_name and ('ln' in layer_name or 'norm' in layer_name):
                # LayerNorm params are 1D: [old_dim] -> [new_dim]
                return full_tensor[:new_dim].clone()

            # 4. word_embed: Linear(Cvae, C) - only prune output dimension
            elif 'word_embed' in layer_name:
                if 'weight' in layer_name:
                    # [old_dim, Cvae] -> [new_dim, Cvae]
                    return full_tensor[:new_dim, :].clone()
                elif 'bias' in layer_name:
                    # [old_dim] -> [new_dim]
                    return full_tensor[:new_dim].clone()

            # 5. head_nm (AdaLNBeforeHead) - prune C dimension
            elif 'head_nm' in layer_name:
                if 'weight' in layer_name or 'bias' in layer_name:
                    # These involve C dimension
                    slices = tuple(slice(0, dim) for dim in target_shape)
                    return full_tensor[slices].clone()

            # 6. head: Linear(C, V) - prune input dimension only
            elif layer_name == 'head.weight':
                # [V, old_dim] -> [V, new_dim], V stays the same
                return full_tensor[:, :new_dim].clone()
            elif layer_name == 'head.bias':
                # [V] -> [V], no pruning
                return full_tensor.clone()

            # 7. shared_ada_lin - if exists, handle specially
            elif 'shared_ada_lin' in layer_name:
                # This outputs 6*C, needs to be pruned
                # Input is D (cond_dim), output is 6*C
                if 'weight' in layer_name:
                    # [6*old_dim, D] -> [6*new_dim, D]
                    return full_tensor[:6*new_dim, :].clone()
                elif 'bias' in layer_name:
                    # [6*old_dim] -> [6*new_dim]
                    return full_tensor[:6*new_dim].clone()

            # Fallback: simple slicing for unrecognized layers
            else:
                print(f"    Info: Using simple slicing for {layer_name}")
                slices = tuple(slice(0, dim) for dim in target_shape)
                return full_tensor[slices].clone()

        except Exception as e:
            print(f"    Warning: Tensor pruning failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return correctly shaped tensor
            slices = tuple(slice(0, dim) for dim in target_shape)
            try:
                return full_tensor[slices].clone()
            except:
                return torch.randn(target_shape, dtype=full_tensor.dtype, device=full_tensor.device)

    def create_pruned_model(self, config: PruningConfig, var_ckpt: Optional[str] = None) -> VAR:
        """
        Create a pruned VAR model based on the configuration

        Args:
            config: Pruning configuration
            var_ckpt: Optional checkpoint to load weights from (for finetuned pruned models)

        Returns:
            VAR model instance
        """
        print(f"\nCreating model: {config}")

        # Build VAR model with specified configuration
        var_model = VAR(
            vae_local=self.vae,
            num_classes=1000,
            depth=config.depth,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1 * config.depth / 24,
            norm_eps=1e-6,
            shared_aln=False,
            cond_drop_rate=0.1,
            attn_l2_norm=True,
            patch_nums=self.patch_nums,
            flash_if_available=True,
            fused_if_available=True,
        ).to(self.device)

        # Load checkpoint if provided
        if var_ckpt and osp.exists(var_ckpt):
            print(f"Loading weights from {var_ckpt}")
            state_dict = torch.load(var_ckpt, map_location='cpu')

            # Check if this is a pruned model or full model
            # Full model: config name contains "original" or config matches standard architecture
            is_full_model = (
                "original" in config.name.lower() or
                (config.width == config.depth * 64 and config.num_heads == config.depth)
            )

            if is_full_model:
                # Direct load for original full model (d16/d20/d24/d30)
                try:
                    var_model.load_state_dict(state_dict, strict=True)
                    print(f"  ✓ Full checkpoint loaded")
                except RuntimeError as e:
                    print(f"  Warning: Could not load with strict=True: {e}")
                    var_model.load_state_dict(state_dict, strict=False)
            else:
                # Pruned model: inherit weights from base checkpoint
                print(f"  Pruned model detected, inheriting weights from base checkpoint...")

                # Infer base_depth from checkpoint path or config name
                base_depth = 16  # default
                if var_ckpt:
                    # Try to extract from path like "var_d16.pth" or "d16" in config name
                    import re
                    match = re.search(r'd(\d+)', var_ckpt)
                    if match:
                        base_depth = int(match.group(1))
                    elif 'from-d' in config.name:
                        match = re.search(r'from-d(\d+)', config.name)
                        if match:
                            base_depth = int(match.group(1))

                self.load_pruned_weights_from_full_model(var_model, state_dict, config, base_depth)
        else:
            print(f"No checkpoint provided, initializing random weights...")
            # Initialize weights for models without checkpoints
            var_model.init_weights(
                init_adaln=0.5,
                init_adaln_gamma=1e-5,
                init_head=0.02,
                init_std=-1
            )

        var_model.eval()
        for p in var_model.parameters():
            p.requires_grad_(False)

        # Print model statistics
        num_params = sum(p.numel() for p in var_model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        return var_model

    def setup_deterministic(self, seed: int = 0):
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

    def count_parameters(self, model: nn.Module) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in model.parameters())

    def estimate_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def convert_model_to_coreml(self,
                               model: VAR,
                               config: PruningConfig,
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
        print(f"Converting {config.name} to CoreML...")
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
                example_inputs.append(ex.to(self.device))

            # Setup deterministic mode
            self.setup_deterministic(seed=0)

            # Move wrapper to device
            wrapper = wrapper.to(self.device)

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
            model_name = f"{config.name}.mlpackage"
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

    def generate_and_save_image(self,
                               model: VAR,
                               config: PruningConfig,
                               image_label: int = 1,
                               save_dir: str = './generated_images',
                               cfg: float = 5.0,
                               top_k: int = 900,
                               top_p: float = 0.96,
                               seed: int = 0) -> Optional[str]:
        """
        Generate and save an image using the model

        Args:
            model: VAR model
            config: Model configuration
            image_label: ImageNet class label (0-999)
            save_dir: Directory to save image
            cfg: Classifier-free guidance scale
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            seed: Random seed

        Returns:
            Path to saved image, or None if generation failed
        """
        os.makedirs(save_dir, exist_ok=True)

        try:
            print(f"Generating image for {config.name} (label={image_label})...")

            # Setup deterministic mode
            self.setup_deterministic(seed)

            # Prepare input
            label_B = torch.tensor([image_label], device=self.device)

            # Generate image
            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                    recon_B3HW = model.autoregressive_infer_cfg(
                        B=1,
                        label_B=label_B,
                        cfg=cfg,
                        top_k=top_k,
                        top_p=top_p,
                        g_seed=seed,
                        more_smooth=False
                    )

            # Convert to PIL image
            img_tensor = recon_B3HW[0]  # Get first (and only) image
            img_tensor = img_tensor.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            img = PImage.fromarray(img_tensor)

            # Save image
            img_filename = f"{config.name}_label{image_label}.png"
            img_path = osp.join(save_dir, img_filename)
            img.save(img_path)

            print(f"✓ Image saved: {img_path}")
            return img_path

        except Exception as e:
            print(f"✗ Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_comparison_grid(self,
                                image_paths: List[Dict],
                                save_path: str,
                                grid_cols: int = 4) -> Optional[str]:
        """
        Generate a comparison grid from multiple images

        Args:
            image_paths: List of dicts with 'config', 'path', 'label'
            save_path: Path to save the grid image
            grid_cols: Number of columns in the grid

        Returns:
            Path to saved grid image, or None if generation failed
        """
        if not image_paths:
            return None

        try:
            print(f"\nGenerating comparison grid with {len(image_paths)} images...")

            # Load all images
            images = []
            labels = []
            for img_info in image_paths:
                try:
                    img = PImage.open(img_info['path'])
                    images.append(img)
                    labels.append(img_info['config'])
                except Exception as e:
                    print(f"Warning: Could not load {img_info['path']}: {e}")

            if not images:
                print("No valid images to create grid")
                return None

            # Calculate grid dimensions
            n_images = len(images)
            grid_rows = (n_images + grid_cols - 1) // grid_cols

            # Get image size (assume all same size)
            img_width, img_height = images[0].size
            label_height = 30  # Height for label text

            # Create grid canvas
            grid_width = img_width * grid_cols
            grid_height = (img_height + label_height) * grid_rows
            grid_img = PImage.new('RGB', (grid_width, grid_height), color='white')
            draw = PImageDraw.Draw(grid_img)

            # Try to load a font
            try:
                font = PImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = PImageFont.load_default()

            # Place images and labels in grid
            for idx, (img, label) in enumerate(zip(images, labels)):
                row = idx // grid_cols
                col = idx % grid_cols

                x = col * img_width
                y = row * (img_height + label_height) + label_height

                # Paste image
                grid_img.paste(img, (x, y))

                # Draw label
                label_x = x + 5
                label_y = row * (img_height + label_height) + 5
                draw.text((label_x, label_y), label, fill='black', font=font)

            # Save grid
            grid_img.save(save_path)
            print(f"✓ Comparison grid saved: {save_path}")
            return save_path

        except Exception as e:
            print(f"✗ Grid generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_inference(self,
                          model: VAR,
                          config: PruningConfig,
                          warmup_runs: int = 5,
                          test_runs: int = 20,
                          batch_size: int = 1,
                          cfg: float = 5.0,
                          top_k: int = 900,
                          top_p: float = 0.96,
                          seed: int = 0) -> BenchmarkResult:
        """
        Benchmark inference speed and memory usage

        Args:
            model: VAR model to benchmark
            config: Model configuration
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
            batch_size: Batch size for inference
            cfg: Classifier-free guidance scale
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            seed: Random seed

        Returns:
            BenchmarkResult with timing and memory statistics
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config.name}")
        print(f"{'='*60}")

        self.setup_deterministic(seed)

        # Setup
        class_labels = tuple([1] * batch_size)  # Use class 1 (goldfish)
        label_B = torch.tensor(class_labels, device=self.device)

        # Memory profiler
        profiler = MemoryProfiler(use_cuda=(self.device == 'cuda'))
        profiler.start()

        # Warmup
        print(f"Warmup ({warmup_runs} runs)...")
        with torch.inference_mode():
            for i in range(warmup_runs):
                _ = model.autoregressive_infer_cfg(
                    B=batch_size,
                    label_B=label_B,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    g_seed=seed,
                    more_smooth=False
                )
                if self.device == 'cuda':
                    torch.cuda.synchronize()

        # Actual benchmark
        print(f"Testing ({test_runs} runs)...")
        latencies = []

        with torch.inference_mode():
            for i in range(test_runs):
                start_time = time.perf_counter()

                _ = model.autoregressive_infer_cfg(
                    B=batch_size,
                    label_B=label_B,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    g_seed=seed,
                    more_smooth=False
                )

                if self.device == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0
                latencies.append(latency_ms)

                if (i + 1) % 5 == 0:
                    print(f"  Run {i+1}/{test_runs}: {latency_ms:.2f} ms")

        # Calculate statistics
        latencies_arr = np.array(latencies)
        mean_latency = np.mean(latencies_arr)
        min_latency = np.min(latencies_arr)
        max_latency = np.max(latencies_arr)
        std_latency = np.std(latencies_arr)
        throughput = (batch_size * 1000.0) / mean_latency  # images per second

        # Memory statistics
        memory_stats = profiler.get_memory_stats()

        # Model statistics
        num_params = self.count_parameters(model)
        model_size_mb = self.estimate_model_size(model)

        # Print summary
        print(f"\n{'─'*60}")
        print(f"Results Summary:")
        print(f"  Mean Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Min Latency:  {min_latency:.2f} ms")
        print(f"  Max Latency:  {max_latency:.2f} ms")
        print(f"  Throughput:   {throughput:.3f} images/sec")
        print(f"  Memory Used:  {memory_stats['allocated_mb']:.2f} MB")
        print(f"  Peak Memory:  {memory_stats['peak_mb']:.2f} MB")
        print(f"  Parameters:   {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"  Model Size:   {model_size_mb:.2f} MB")
        print(f"{'─'*60}\n")

        # Create result object
        result = BenchmarkResult(
            config_name=config.name,
            platform='CUDA' if self.device == 'cuda' else 'CPU',
            mean_latency_ms=float(mean_latency),
            min_latency_ms=float(min_latency),
            max_latency_ms=float(max_latency),
            std_latency_ms=float(std_latency),
            throughput_imgs_per_sec=float(throughput),
            memory_used_mb=float(memory_stats['allocated_mb']),
            peak_memory_mb=float(memory_stats['peak_mb']),
            num_params=num_params,
            model_size_mb=float(model_size_mb),
            warmup_runs=warmup_runs,
            test_runs=test_runs,
            batch_size=batch_size,
        )

        return result

    def run_comprehensive_benchmark(self,
                                   configs: List[PruningConfig],
                                   var_ckpt_base: str,
                                   output_dir: str = './speed_test_results',
                                   enable_coreml: bool = False,
                                   coreml_save_dir: str = './coreml_models',
                                   save_images: bool = False,
                                   image_label: int = 1,
                                   image_save_dir: str = './generated_images',
                                   **benchmark_kwargs) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmarks across multiple configurations

        Args:
            configs: List of pruning configurations to test
            var_ckpt_base: Base path for VAR checkpoints (depth will be appended)
            output_dir: Directory to save results
            enable_coreml: Whether to convert models to CoreML
            coreml_save_dir: Directory to save CoreML models
            save_images: Whether to generate and save sample images
            image_label: ImageNet class label for image generation (0-999)
            image_save_dir: Directory to save generated images
            **benchmark_kwargs: Additional arguments for benchmark_inference

        Returns:
            List of benchmark results
        """
        os.makedirs(output_dir, exist_ok=True)
        if enable_coreml:
            os.makedirs(coreml_save_dir, exist_ok=True)
        if save_images:
            os.makedirs(image_save_dir, exist_ok=True)

        results = []
        coreml_models = []  # Track converted CoreML models
        generated_images = []  # Track generated images

        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE VAR SPEED BENCHMARK")
        print(f"{'='*70}")
        print(f"Total configurations to test: {len(configs)}")
        print(f"Results will be saved to: {output_dir}")
        if enable_coreml:
            print(f"CoreML models will be saved to: {coreml_save_dir}")
        if save_images:
            print(f"Images will be saved to: {image_save_dir} (label={image_label})")
        print(f"{'='*70}\n")

        for idx, config in enumerate(configs, 1):
            print(f"\n[{idx}/{len(configs)}] Testing configuration: {config.name}")

            # All models use d16 checkpoint (pruned models will inherit weights)
            var_ckpt = var_ckpt_base

            try:
                # Create model
                model = self.create_pruned_model(config, var_ckpt)

                # Run benchmark
                result = self.benchmark_inference(
                    model=model,
                    config=config,
                    **benchmark_kwargs
                )

                results.append(result)

                # Save individual result
                result_file = osp.join(output_dir, f"{config.name}_result.json")
                with open(result_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"Saved result to {result_file}")

                # Convert to CoreML if enabled
                if enable_coreml:
                    coreml_path = self.convert_model_to_coreml(
                        model=model,
                        config=config,
                        save_dir=coreml_save_dir
                    )
                    if coreml_path:
                        coreml_models.append({
                            'config': config.name,
                            'path': coreml_path
                        })

                # Generate and save image if enabled
                if save_images:
                    img_path = self.generate_and_save_image(
                        model=model,
                        config=config,
                        image_label=image_label,
                        save_dir=image_save_dir,
                        cfg=benchmark_kwargs.get('cfg', 5.0),
                        top_k=benchmark_kwargs.get('top_k', 900),
                        top_p=benchmark_kwargs.get('top_p', 0.96),
                        seed=benchmark_kwargs.get('seed', 0)
                    )
                    if img_path:
                        generated_images.append({
                            'config': config.name,
                            'path': img_path,
                            'label': image_label
                        })

                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"ERROR testing {config.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save comprehensive results
        self._save_comprehensive_results(results, output_dir)

        # Print CoreML models summary if any were converted
        if enable_coreml and coreml_models:
            print(f"\n{'='*70}")
            print(f"COREML MODELS CONVERTED ({len(coreml_models)} total)")
            print(f"{'='*70}")
            for i, model_info in enumerate(coreml_models, 1):
                print(f"{i}. {model_info['config']}")
                print(f"   Path: {model_info['path']}")
            print(f"{'='*70}\n")

            # Save CoreML models list
            coreml_list_file = osp.join(coreml_save_dir, 'coreml_models_list.json')
            with open(coreml_list_file, 'w') as f:
                json.dump(coreml_models, f, indent=2)
            print(f"CoreML models list saved to: {coreml_list_file}\n")

        # Print generated images summary and create comparison grid
        if save_images and generated_images:
            print(f"\n{'='*70}")
            print(f"IMAGES GENERATED ({len(generated_images)} total)")
            print(f"{'='*70}")
            for i, img_info in enumerate(generated_images, 1):
                print(f"{i}. {img_info['config']}")
                print(f"   Path: {img_info['path']}")
            print(f"{'='*70}\n")

            # Save images list
            images_list_file = osp.join(image_save_dir, 'generated_images_list.json')
            with open(images_list_file, 'w') as f:
                json.dump(generated_images, f, indent=2)
            print(f"Images list saved to: {images_list_file}\n")

            # Generate comparison grid
            grid_path = osp.join(image_save_dir, f'comparison_grid_label{image_label}.png')
            self.generate_comparison_grid(generated_images, grid_path, grid_cols=4)

        return results

    def _save_comprehensive_results(self, results: List[BenchmarkResult], output_dir: str):
        """Save comprehensive results and generate comparison report"""
        # Save all results as JSON
        all_results_file = osp.join(output_dir, 'all_results.json')
        with open(all_results_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nSaved all results to {all_results_file}")

        # Generate comparison report
        report_file = osp.join(output_dir, 'comparison_report.txt')
        with open(report_file, 'w') as f:
            f.write("VAR MODEL SPEED COMPARISON REPORT\n")
            f.write("=" * 100 + "\n\n")

            # Table header
            f.write(f"{'Configuration':<30} {'Params':<12} {'Latency(ms)':<15} {'Throughput':<15} {'Memory(MB)':<12}\n")
            f.write("-" * 100 + "\n")

            # Sort by latency
            sorted_results = sorted(results, key=lambda r: r.mean_latency_ms)

            for result in sorted_results:
                params_str = f"{result.num_params/1e6:.1f}M"
                latency_str = f"{result.mean_latency_ms:.2f} ± {result.std_latency_ms:.2f}"
                throughput_str = f"{result.throughput_imgs_per_sec:.3f} img/s"
                memory_str = f"{result.peak_memory_mb:.1f}"

                f.write(f"{result.config_name:<30} {params_str:<12} {latency_str:<15} {throughput_str:<15} {memory_str:<12}\n")

            f.write("\n" + "=" * 100 + "\n\n")

            # Speedup analysis (compared to baseline)
            if len(results) > 0:
                # Assume first result is baseline (original VAR-d16)
                baseline = max(results, key=lambda r: r.num_params)  # Largest model as baseline
                f.write("SPEEDUP ANALYSIS (vs baseline)\n")
                f.write("-" * 100 + "\n")
                f.write(f"Baseline: {baseline.config_name} - {baseline.mean_latency_ms:.2f} ms\n\n")

                for result in sorted_results:
                    speedup = baseline.mean_latency_ms / result.mean_latency_ms
                    param_reduction = (1 - result.num_params / baseline.num_params) * 100
                    memory_reduction = (1 - result.peak_memory_mb / baseline.peak_memory_mb) * 100

                    f.write(f"{result.config_name}:\n")
                    f.write(f"  Speedup: {speedup:.2f}x\n")
                    f.write(f"  Parameter Reduction: {param_reduction:.1f}%\n")
                    f.write(f"  Memory Reduction: {memory_reduction:.1f}%\n\n")

        print(f"Saved comparison report to {report_file}")

        # Print summary to console
        print(f"\n{'='*100}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*100}")
        with open(report_file, 'r') as f:
            print(f.read())


def get_pruning_configs(base_depth: int = 16) -> List[PruningConfig]:
    """
    Define pruning configurations for VAR models

    Args:
        base_depth: Base model depth (16, 20, 24, or 30)
                   Following demo.py convention: width = depth * 64, heads = depth

    Returns:
        List of PruningConfig objects
    """
    assert base_depth in {16, 20, 24, 30}, f"base_depth must be one of {{16, 20, 24, 30}}, got {base_depth}"

    # Calculate base width and heads following demo.py convention
    base_width = base_depth * 64
    base_heads = base_depth

    configs = []

    # 1. Baseline: Original model
    configs.append(PruningConfig(
        name=f"var-d{base_depth}-original",
        depth=base_depth,
        width=base_width,
        num_heads=base_heads,
        description=f"Original VAR-d{base_depth} model"
    ))

    # 2. Depth Pruning: Reduce number of transformer layers
    # Prune to 75%, 62.5%, 50%, 37.5%, 25% of original depth
    depth_ratios = [0.75, 0.625, 0.50, 0.375, 0.25]

    for ratio in depth_ratios:
        pruned_depth = int(base_depth * ratio)
        if pruned_depth >= 4:  # Minimum depth of 4
            reduction_pct = (1 - ratio) * 100
            configs.append(PruningConfig(
                name=f"var-d{pruned_depth}-depth-pruned-from-d{base_depth}",
                depth=pruned_depth,
                width=base_width,  # Keep base width
                num_heads=base_heads,  # Keep base heads
                description=f"Depth pruned: {reduction_pct:.1f}% reduction from d{base_depth}"
            ))

    # 3. Width Pruning: Reduce embedding dimensions
    # Prune to 75%, 50%, 37.5%, 25% of original width
    width_ratios = [0.75, 0.50, 0.375, 0.25]

    for ratio in width_ratios:
        pruned_width = int(base_width * ratio)
        pruned_heads = max(1, int(base_heads * ratio))  # Proportional head reduction

        # Ensure width is divisible by heads for proper attention
        # If not divisible, adjust heads to make it work
        if pruned_width % pruned_heads != 0:
            # Try standard head size (width / 64)
            if pruned_width % 64 == 0:
                pruned_heads = pruned_width // 64
            else:
                # Find largest divisor that's reasonable for attention heads
                for h in range(pruned_heads, 0, -1):
                    if pruned_width % h == 0 and pruned_width // h <= 128:
                        pruned_heads = h
                        break

        # Final check - should always pass now
        if pruned_width % pruned_heads == 0:
            reduction_pct = (1 - ratio) * 100
            configs.append(PruningConfig(
                name=f"var-d{base_depth}-w{pruned_width}-width-pruned",
                depth=base_depth,  # Keep base depth
                width=pruned_width,
                num_heads=pruned_heads,
                description=f"Width pruned: {reduction_pct:.1f}% reduction"
            ))
        else:
            print(f"  ⚠️  Skipping width pruning ratio {ratio:.1%}: w={pruned_width}, h={pruned_heads} (not divisible)")

    # # 4. Structured Pruning: Combined depth + width reduction
    # # Apply same ratio to both depth and width
    # structured_ratios = [0.8, 0.6, 0.4, 0.2]

    # for ratio in structured_ratios:
    #     pruned_depth = int(base_depth * ratio)
    #     pruned_width = int(base_width * ratio)
    #     pruned_heads = max(1, int(base_heads * ratio))

    #     # Ensure width is divisible by heads for proper attention
    #     # If not divisible, adjust heads to make it work
    #     if pruned_width % pruned_heads != 0:
    #         # Try standard head size (width / 64)
    #         if pruned_width % 64 == 0:
    #             pruned_heads = pruned_width // 64
    #         else:
    #             # Find largest divisor that's reasonable for attention heads
    #             for h in range(pruned_heads, 0, -1):
    #                 if pruned_width % h == 0 and pruned_width // h <= 128:
    #                     pruned_heads = h
    #                     break

    #     # Check both depth and width constraints
    #     if pruned_depth >= 4 and pruned_width % pruned_heads == 0:
    #         reduction_pct = (1 - ratio) * 100
    #         configs.append(PruningConfig(
    #             name=f"var-d{pruned_depth}-w{pruned_width}-structured-pruned-from-d{base_depth}",
    #             depth=pruned_depth,
    #             width=pruned_width,
    #             num_heads=pruned_heads,
    #             description=f"Structured pruned: {reduction_pct:.1f}% depth & width from d{base_depth}"
    #         ))
    #     else:
    #         if pruned_depth < 4:
    #             print(f"  ⚠️  Skipping structured pruning ratio {ratio:.1%}: depth {pruned_depth} < 4")
    #         else:
    #             print(f"  ⚠️  Skipping structured pruning ratio {ratio:.1%}: w={pruned_width}, h={pruned_heads} (not divisible)")

    return configs


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='VAR Speed Testing with CoreML Support')
    parser.add_argument('--base-depth', type=int, default=16, choices=[16, 20, 24, 30],
                       help='Base model depth (16, 20, 24, or 30, default: 16)')
    parser.add_argument('--enable-coreml', action='store_true',
                       help='Enable CoreML model conversion')
    parser.add_argument('--coreml-save-dir', type=str, default='./coreml_models',
                       help='Directory to save CoreML models (default: ./coreml_models)')
    parser.add_argument('--save-images', action='store_true',
                       help='Generate and save sample images for quality comparison')
    parser.add_argument('--image-label', type=int, default=1,
                       help='ImageNet class label for image generation (0-999, default: 1=goldfish)')
    parser.add_argument('--image-save-dir', type=str, default='./generated_images',
                       help='Directory to save generated images (default: ./generated_images)')
    parser.add_argument('--output-dir', type=str, default='./speed_test_results',
                       help='Directory to save benchmark results (default: ./speed_test_results)')
    parser.add_argument('--vae-ckpt', type=str,
                       default='../../model_zoo/vae_ch160v4096z32.pth',
                       help='Path to VAE checkpoint')
    parser.add_argument('--var-ckpt', type=str, default=None,
                       help='Path to VAR checkpoint (default: auto-detect based on --base-depth)')
    args = parser.parse_args()

    # Auto-detect VAR checkpoint if not specified
    if args.var_ckpt is None:
        args.var_ckpt = f'../../model_zoo/var_d{args.base_depth}.pth'

    # Configuration
    VAE_CKPT = args.vae_ckpt
    VAR_CKPT = args.var_ckpt
    BASE_DEPTH = args.base_depth
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = args.output_dir

    print(f"Base model depth: d{BASE_DEPTH}")
    print(f"Base model width: {BASE_DEPTH * 64}")
    print(f"Base model heads: {BASE_DEPTH}")
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Check if VAR checkpoint exists
    if not osp.exists(VAR_CKPT):
        print(f"\n⚠️  WARNING: VAR checkpoint not found: {VAR_CKPT}")
        print(f"Please ensure you have the var_d{BASE_DEPTH}.pth checkpoint")
        print(f"Download from: https://huggingface.co/FoundationVision/var")
        return

    if args.enable_coreml:
        if not COREML_AVAILABLE:
            print("ERROR: CoreML requested but coremltools is not available!")
            print("Install with: pip install coremltools")
            return
        print(f"CoreML conversion: ENABLED")
        print(f"CoreML models will be saved to: {args.coreml_save_dir}")
    else:
        print(f"CoreML conversion: DISABLED (use --enable-coreml to enable)")

    if args.save_images:
        print(f"Image generation: ENABLED (label={args.image_label})")
        print(f"Images will be saved to: {args.image_save_dir}")
    else:
        print(f"Image generation: DISABLED (use --save-images to enable)")

    # Initialize tester
    tester = VARSpeedTester(
        vae_ckpt=VAE_CKPT,
        device=DEVICE,
    )

    # Get all pruning configurations for the specified base depth
    configs = get_pruning_configs(base_depth=BASE_DEPTH)

    print(f"\nTotal configurations to benchmark: {len(configs)}")
    for i, cfg in enumerate(configs, 1):
        print(f"{i}. {cfg}")

    # Run comprehensive benchmark
    results = tester.run_comprehensive_benchmark(
        configs=configs,
        var_ckpt_base=VAR_CKPT,
        output_dir=OUTPUT_DIR,
        enable_coreml=args.enable_coreml,
        coreml_save_dir=args.coreml_save_dir,
        save_images=args.save_images,
        image_label=args.image_label,
        image_save_dir=args.image_save_dir,
        warmup_runs=5,
        test_runs=20,
        batch_size=1,
        cfg=5.0,
        top_k=900,
        top_p=0.96,
        seed=0,
    )

    print(f"\n{'='*100}")
    print(f"BENCHMARK COMPLETE!")
    print(f"Total configurations tested: {len(results)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    if args.enable_coreml:
        print(f"CoreML models saved to: {args.coreml_save_dir}")
    if args.save_images:
        print(f"Generated images saved to: {args.image_save_dir}")
        print(f"Comparison grid: {args.image_save_dir}/comparison_grid_label{args.image_label}.png")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
