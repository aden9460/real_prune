"""
GPTQ Quantization for VAR Model (PyTorch Stage)

This script applies GPTQ quantization to the PyTorch VAR model BEFORE converting to CoreML.
The quantization information will be automatically preserved during CoreML conversion.

Workflow:
1. Load original PyTorch VAR model
2. Configure GPTQ with calibration data
3. Apply LayerwiseCompressor or LinearQuantizer
4. Save quantized PyTorch model
5. (Separate step) Convert quantized model to CoreML using transfer.py

Requirements:
- Calibration dataset (~128 samples)
- coremltools.optimize.torch APIs
"""

import sys
sys.path.append("..")

import os
import os.path as osp
import torch
import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var, VAR, VQVAE
from utils import arg_util

# CoreML optimization imports
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError as e:
    COREML_AVAILABLE = False
    print(f"Warning: coremltools not available: {e}")

# Try to import quantization tools separately
GPTQ_AVAILABLE = False
LINEAR_AVAILABLE = False

if COREML_AVAILABLE:
    try:
        from coremltools.optimize.torch.quantization import (
            LayerwiseCompressor,
            LayerwiseCompressorConfig,
        )
        GPTQ_AVAILABLE = True
    except ImportError:
        pass  # GPTQ not available, will use linear

    try:
        from coremltools.optimize.torch.quantization import (
            LinearQuantizer,
            LinearQuantizerConfig
        )
        LINEAR_AVAILABLE = True
    except ImportError:
        pass  # Linear quantizer not available


class CalibrationDataLoader:
    """
    Create calibration data for GPTQ quantization.

    GPTQ needs ~128 representative input samples to compute optimal quantization parameters.

    This loader prepares VAR model inputs correctly:
    - Loads ImageNet images
    - Encodes them to tokens using VQVAE (following VAR training pipeline)
    - Returns (labels, tokens) pairs for calibration

    Based on slimvar/model_slimming_basic_v1.py:prepare_calibration_data
    """
    def __init__(
        self,
        vae,
        data_path: Optional[str] = None,
        batch_size: int = 1,
        num_samples: int = 128,
        use_real_images: bool = True,
        final_reso: int = 256
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.use_real_images = use_real_images
        self.current_idx = 0
        self.vae = vae
        self.final_reso = final_reso

        # Store calibration data
        self.calibration_labels = None
        self.calibration_tokens = None

        if use_real_images and data_path and osp.exists(data_path):
            print(f"Loading calibration images from: {data_path}")
            self._load_and_encode_images(data_path)
        else:
            print("Using synthetic labels for calibration (less accurate)")
            self._generate_synthetic_labels()

    def _load_and_encode_images(self, data_path: str):
        """
        Load ImageNet images and encode them to VAE tokens.

        This follows the same pipeline as VAR training:
        Image -> VQVAE encoder -> tokens (B, 679, 32)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Import VAR's original data loading function
        import sys
        sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
        from utils.data import build_dataset

        try:
            # Use VAR's original build_dataset
            num_classes, train_set, val_set = build_dataset(
                data_path=data_path,
                final_reso=self.final_reso,
                hflip=False,
                mid_reso=1.125
            )

            # Use validation set for calibration
            dataset = val_set

            # Uniform sampling across dataset
            if self.num_samples <= len(dataset):
                step = len(dataset) // self.num_samples
                indices = torch.arange(0, len(dataset), step)[:self.num_samples]
            else:
                indices = torch.arange(len(dataset))

            print(f"  Sampling {len(indices)} images uniformly from validation set")

            labels_list = []
            tokens_list = []

            # Process in batches for efficiency
            process_batch_size = 8
            for i in range(0, len(indices), process_batch_size):
                batch_indices = indices[i:min(i+process_batch_size, len(indices))]
                images = []
                labels = []

                for idx in batch_indices:
                    img, label = dataset[int(idx)]
                    images.append(img)
                    labels.append(label)

                images = torch.stack(images).to(device)  # (B, 3, 256, 256), range [-1, 1]
                labels = torch.tensor(labels, dtype=torch.long)

                # Encode images to tokens using VQVAE (same as VAR training)
                with torch.no_grad():
                    gt_idx_Bl = self.vae.img_to_idxBl(images)  # List of 10 tensors
                    x_BLCv = self.vae.quantize.idxBl_to_var_input(gt_idx_Bl)  # (B, 679, 32)

                labels_list.append(labels)
                tokens_list.append(x_BLCv.cpu())

                if (i + process_batch_size) % 64 == 0:
                    print(f"  Processed {i + process_batch_size}/{len(indices)} images")

            self.calibration_labels = torch.cat(labels_list, dim=0)  # (num_samples,)
            self.calibration_tokens = torch.cat(tokens_list, dim=0)  # (num_samples, 679, 32)

            print(f"✓ Loaded and encoded {len(self.calibration_labels)} calibration samples")
            print(f"  Labels shape: {self.calibration_labels.shape}")
            print(f"  Tokens shape: {self.calibration_tokens.shape}")

        except Exception as e:
            print(f"Warning: Failed to load ImageNet images: {e}")
            print("Falling back to synthetic labels")
            self._generate_synthetic_labels()

    def _generate_synthetic_labels(self):
        """Generate synthetic labels when real data is not available."""
        import random
        random.seed(0)

        labels = [random.randint(0, 999) for _ in range(self.num_samples)]
        self.calibration_labels = torch.tensor(labels, dtype=torch.long)
        self.calibration_tokens = None  # Will use label-only mode

        print(f"✓ Generated {len(self.calibration_labels)} synthetic label samples")

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.calibration_labels):
            raise StopIteration

        # Get batch
        batch_end = min(self.current_idx + self.batch_size, len(self.calibration_labels))

        labels = self.calibration_labels[self.current_idx:batch_end]

        if self.calibration_tokens is not None:
            tokens = self.calibration_tokens[self.current_idx:batch_end]
            batch_data = (labels, tokens)
        else:
            batch_data = (labels,)

        self.current_idx = batch_end
        return batch_data

    def __len__(self):
        return (len(self.calibration_labels) + self.batch_size - 1) // self.batch_size


def create_gptq_config(
    algorithm: str = "gptq",
    weight_dtype: str = "int4",  # or "int8"
    granularity: str = "per_channel",  # or "per_block"
    calibration_nsamples: int = 128
):
    """
    Create GPTQ configuration for LayerwiseCompressor.

    Args:
        algorithm: "gptq" for GPTQ quantization
        weight_dtype: "int4" or "int8"
        granularity: "per_channel" or "per_block"
        calibration_nsamples: Number of calibration samples

    Returns:
        LayerwiseCompressorConfig object
    """
    if not GPTQ_AVAILABLE:
        raise ImportError("LayerwiseCompressor (GPTQ) is not available")

    config_dict = {
        "global_config": {
            "algorithm": algorithm,
            "weight_dtype": weight_dtype,
            "granularity": granularity,
        },
        "input_cacher": "default",
        "calibration_nsamples": calibration_nsamples,
    }

    # Add block_size if using per_block granularity
    if granularity == "per_block":
        config_dict["global_config"]["block_size"] = 128

    return LayerwiseCompressorConfig.from_dict(config_dict)


def create_linear_quantizer_config(
    quantization_scheme: str = "symmetric",
    milestones: List[int] = None
):
    """
    Create LinearQuantizer config as fallback if LayerwiseCompressor doesn't work.

    This is a simpler quantization approach that doesn't use calibration data.
    """
    if not LINEAR_AVAILABLE:
        raise ImportError("LinearQuantizer is not available")

    if milestones is None:
        milestones = [0, 0, 10, 10]

    config_dict = {
        "global_config": {
            "quantization_scheme": quantization_scheme,
            "milestones": milestones,
        }
    }

    return LinearQuantizerConfig.from_dict(config_dict)


def apply_gptq_quantization(
    var_model: VAR,
    config,
    calibration_loader: CalibrationDataLoader,
    device: str = "cuda"
) -> Optional[VAR]:
    """
    Apply GPTQ quantization using LayerwiseCompressor.

    Args:
        var_model: Original VAR model
        config: GPTQ configuration
        calibration_loader: Calibration data loader with (labels, tokens) pairs
        device: Device to run on

    Returns:
        Quantized VAR model, or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Applying GPTQ Quantization with LayerwiseCompressor")
    print(f"{'='*60}")

    try:
        var_model = var_model.to(device)
        var_model.eval()

        # Create compressor
        print("Creating LayerwiseCompressor...")
        compressor = LayerwiseCompressor(var_model, config)

        # Create calibration data iterator
        print(f"Preparing calibration data...")

        def calibration_data_iterator():
            """Iterator that yields input tensors for the VAR model"""
            for batch_data in calibration_loader:
                if len(batch_data) == 2:  # (labels, tokens)
                    labels, tokens = batch_data
                    labels = labels.to(device)
                    tokens = tokens.to(device)

                    # VAR model expects: model(label_B, x_BLCv) for teacher forcing mode
                    # But for GPTQ calibration, we need to prepare the actual inputs
                    # that will be passed through the model during inference

                    # For VAR, this could be:
                    # 1. tokens for teacher forcing mode: yield tokens
                    # 2. labels for autoregressive mode: yield labels

                    # Let's try tokens first (more representative)
                    yield (labels, tokens)
                else:  # (labels,)
                    labels = batch_data[0].to(device)
                    # For label-only mode, we need to simulate VAR's autoregressive inference
                    # This is more complex, but let's try passing labels
                    yield (labels,)

        # Apply compression with calibration data
        print(f"Compressing model with {len(calibration_loader)} batches of calibration data...")
        print("Note: This may take several minutes...")

        # The LayerwiseCompressor expects a dataloader that yields inputs
        # compatible with the model's forward method
        class CalibrationDataIterator:
            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                for batch_data in self.loader:
                    if len(batch_data) == 2:  # (labels, tokens)
                        labels, tokens = batch_data
                        # For VAR model teacher forcing mode
                        yield {
                            'label_B': labels.to(device),
                            'x_BLCv': tokens.to(device)
                        }
                    else:  # (labels,)
                        labels = batch_data[0]
                        # For VAR model autoregressive mode
                        yield {
                            'label_B': labels.to(device)
                        }

            def __len__(self):
                return len(self.loader)

        calibration_iterator = CalibrationDataIterator(calibration_loader)

        # Apply GPTQ compression
        compressed_model = compressor.compress(calibration_iterator)

        print("✓ GPTQ quantization completed successfully")
        return compressed_model

    except Exception as e:
        print(f"✗ GPTQ quantization failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: VAR model may not be compatible with LayerwiseCompressor.")
        print("LayerwiseCompressor requires sequential model architecture (nn.Sequential).")
        print("Consider using LinearQuantizer as fallback (see apply_linear_quantization).")
        return None


def apply_linear_quantization(
    var_model: VAR,
    config
) -> Optional[VAR]:
    """
    Apply linear quantization as fallback (doesn't use calibration data).

    This is simpler than GPTQ but may have lower accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Applying Linear Quantization (Fallback)")
    print(f"{'='*60}")

    try:
        # Create quantizer
        print("Creating LinearQuantizer...")
        quantizer = LinearQuantizer(var_model, config)

        # Prepare example inputs for VAR model
        # VAR expects label_B as input during autoregressive generation
        print("Preparing quantization with example inputs...")
        # Create fixed tensor values (not conditional expressions)
        label_B = torch.tensor([1], dtype=torch.long).cuda()
        example_inputs = (label_B,)

        prepared_model = quantizer.prepare(example_inputs=example_inputs)

        print("Finalizing quantization...")
        quantized_model = quantizer.finalize()

        print("✓ Linear quantization completed successfully")
        return quantized_model

    except Exception as e:
        print(f"✗ Linear quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_quantized_model(
    model: VAR,
    save_path: str,
    config_info: dict
):
    """
    Save quantized model with configuration information.

    Args:
        model: Quantized VAR model
        save_path: Path to save the quantized model
        config_info: Quantization configuration dict
    """
    print(f"\nSaving quantized model to: {save_path}")

    # Save model state dict with config
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantization_config': config_info,
    }
    torch.save(checkpoint, save_path)

    # Get file size
    size_mb = osp.getsize(save_path) / (1024 * 1024)
    print(f"✓ Model saved: {save_path}")
    print(f"  File size: {size_mb:.2f} MB")
    print(f"\nUsage:")
    print(f"  • For CoreML conversion: Use with transfer.py")
    print(f"  • For Linux testing:")
    print(f"      checkpoint = torch.load('{osp.basename(save_path)}')")
    print(f"      model.load_state_dict(checkpoint['model_state_dict'])")


def main():
    """Main execution function."""
    # Check availability and display status
    print(f"\n{'='*60}")
    print("Quantization Tools Availability")
    print(f"{'='*60}")
    print(f"CoreML:            {'✓ Available' if COREML_AVAILABLE else '✗ Not Available'}")
    print(f"GPTQ (Layerwise):  {'✓ Available' if GPTQ_AVAILABLE else '✗ Not Available (will use Linear)'}")
    print(f"Linear Quantizer:  {'✓ Available' if LINEAR_AVAILABLE else '✗ Not Available'}")
    print(f"{'='*60}\n")

    if not COREML_AVAILABLE:
        print("ERROR: coremltools is not available.")
        print("Please install: pip install coremltools")
        return

    if not LINEAR_AVAILABLE and not GPTQ_AVAILABLE:
        print("ERROR: No quantization tools available.")
        print("coremltools.optimize.torch.quantization is not properly installed.")
        return

    # Parse arguments
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    MODEL_DEPTH = args.depth
    assert MODEL_DEPTH in {12, 16, 20, 24, 30}

    # Configuration
    USE_GPTQ = getattr(args, 'use_gptq', False) and GPTQ_AVAILABLE  # Only use GPTQ if available
    WEIGHT_DTYPE = "int4"  # "int4" or "int8"
    CALIBRATION_SAMPLES = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Display configuration
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    print(f"Model depth: {MODEL_DEPTH}")
    print(f"Sparsity/Pruning ratio: {args.sparsity}")
    print(f"Quantization method: {'GPTQ (with fallback)' if USE_GPTQ else 'Linear'}")
    print(f"Quantization dtype: {WEIGHT_DTYPE}")
    print(f"Calibration samples: {CALIBRATION_SAMPLES if USE_GPTQ else 'N/A (not needed)'}")
    print(f"Model path: {args.var_model}")
    print(f"{'='*60}")

    # Load VAE and VAR model
    print(f"\n{'='*60}")
    print(f"Loading VAR Model (depth={MODEL_DEPTH}, sparsity={args.sparsity})")
    print(f"{'='*60}")

    vae_ckpt = '/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth'
    var_ckpt = args.var_model

    if not osp.exists(vae_ckpt):
        print(f"ERROR: VAE checkpoint not found: {vae_ckpt}")
        return
    if not osp.exists(var_ckpt):
        print(f"ERROR: VAR checkpoint not found: {var_ckpt}")
        return

    # Build model
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False, args=args
    )

    # Load checkpoints
    print("Loading checkpoints...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

    checkpoint = torch.load(var_ckpt, map_location='cpu')
    if 'trainer' in checkpoint:
        print("Detected training checkpoint, extracting model weights...")
        if 'var_wo_ddp' in checkpoint['trainer']:
            model_weights = checkpoint['trainer']['var_wo_ddp']
            var.load_state_dict(model_weights, strict=True)
        else:
            var.load_state_dict(checkpoint, strict=True)
    else:
        var.load_state_dict(checkpoint, strict=True)

    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print("✓ Model loaded successfully")

    # Apply quantization
    quantized_model = None
    config_info = {}

    if USE_GPTQ:
        # Try GPTQ with LayerwiseCompressor
        print(f"\nAttempting GPTQ quantization with {WEIGHT_DTYPE}...")

        config = create_gptq_config(
            algorithm="gptq",
            weight_dtype=WEIGHT_DTYPE,
            granularity="per_channel",
            calibration_nsamples=CALIBRATION_SAMPLES
        )

        # Get data path from args if available
        data_path = getattr(args, 'data_path', None)
        if not data_path:
            # Try common ImageNet paths
            possible_paths = [
                '/home/project/daily/AR/imagenet',
                '/imagenet',
                './imagenet'
            ]
            for path in possible_paths:
                if osp.exists(path):
                    data_path = path
                    break

        calibration_loader = CalibrationDataLoader(
            vae=vae,  # Pass VAE for encoding images to tokens
            data_path=data_path,
            batch_size=1,
            num_samples=CALIBRATION_SAMPLES,
            use_real_images=True,  # Set to False to use synthetic labels
            final_reso=256
        )

        quantized_model = apply_gptq_quantization(var, config, calibration_loader, device)
        config_info = {
            'method': 'gptq',
            'weight_dtype': WEIGHT_DTYPE,
            'granularity': 'per_channel',
            'calibration_samples': CALIBRATION_SAMPLES
        }

    # Fallback to LinearQuantizer if GPTQ failed
    if quantized_model is None:
        print("\nFalling back to LinearQuantizer...")
        config = create_linear_quantizer_config(
            quantization_scheme="symmetric",
            milestones=[0, 0, 10, 10]
        )
        quantized_model = apply_linear_quantization(var, config)
        config_info = {
            'method': 'linear',
            'quantization_scheme': 'symmetric'
        }

    # Save quantized model
    if quantized_model is not None:
        save_dir = "./pytorch_quantized_models"
        os.makedirs(save_dir, exist_ok=True)

        quantization_method = config_info.get('method', 'unknown')

        # Include sparsity in filename if model is pruned
        if args.sparsity > 0:
            sparsity_str = f"_s{args.sparsity}"
        else:
            sparsity_str = ""

        save_path = osp.join(
            save_dir,
            f"var_d{MODEL_DEPTH}{sparsity_str}_{quantization_method}_{WEIGHT_DTYPE}.pth"
        )

        # Add sparsity to config info
        config_info['sparsity'] = args.sparsity
        config_info['model_depth'] = MODEL_DEPTH

        save_quantized_model(quantized_model, save_path, config_info)

        print(f"\n{'='*60}")
        print("Quantization completed successfully!")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("1. Test quantized model on Linux:")
        print(f"   checkpoint = torch.load('{osp.basename(save_path)}')")
        print(f"   model.load_state_dict(checkpoint['model_state_dict'])")
        print("")
        print("2. Convert to CoreML:")
        print(f"   python transfer.py --depth {MODEL_DEPTH} --sparsity {args.sparsity} --var_model {save_path}")
    else:
        print("\n✗ Quantization failed. Please check error messages above.")


if __name__ == "__main__":
    main()
