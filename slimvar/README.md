# VAR Basic Pruning Implementation

## Overview

This directory contains the implementation of VAR model pruning using SlimGPT + Torch-Pruning.

## File Structure

```
/home/project/real_prune/slimvar/
├── model_slimming_basic.py          # Basic pruning script (✓ completed)
├── run_basic_pruning.bash           # Test script for basic pruning (✓ completed)
├── pruning_innovations.py           # Advanced features (TODO)
├── model_slimming_var.py            # Integrated main script (TODO)
├── VAR_PRUNING_CALIBRATION_GUIDE.md # Complete documentation
├── INNOVATIONS_README.md            # Quick reference for innovations
└── VAR/                             # VAR model code
    └── models/
        ├── var.py
        ├── basic_var.py
        └── vqvae.py
```

## Quick Start

### 1. Basic Pruning (20% sparsity)

```bash
cd /home/project/real_prune/slimvar
bash run_basic_pruning.bash
```

Or run manually:

```bash
python model_slimming_basic.py \
    --model_depth 16 \
    --num_samples 256 \
    --sparsity 0.2 \
    --minlayer 0 \
    --maxlayer 16 \
    --save_dir ./pruned_models \
    --model_name var_d16_s20.pth
```

### 2. Non-uniform Pruning

Prune deeper layers more aggressively:

```bash
python model_slimming_basic.py \
    --model_depth 16 \
    --num_samples 256 \
    --non_uniform \
    --non_uniform_strategy log_increase \
    --min_sparsity 0.06 \
    --max_sparsity 0.3 \
    --minlayer 0 \
    --maxlayer 16 \
    --save_dir ./pruned_models \
    --model_name var_d16_nonuniform.pth
```

## Basic Pruning Script (`model_slimming_basic.py`)

### What it does:

1. **Loads VAR model** from checkpoint
2. **Collects calibration data** with two modes:
   - **Label-only mode (default)**: Uses class labels 0-255, VAR generates internally
   - **Pre-encoded mode (`--use_images`)**:
     - Loads real ImageNet images
     - **Uses VAR's original data augmentation** (from `VAR/utils/data.py`):
       - Resize shorter edge to 288 (1.125 × 256)
       - CenterCrop to 256×256
       - LANCZOS interpolation
       - Normalize [0,1] → [-1,1] via `(x*2)-1`
     - Pre-encodes all tokens via VQVAE
     - Recommended for better accuracy
3. **Evaluates importance** using SlimGPT's Hessian approximation
   - Computes H = X^T X for each layer
   - Determines which channels/heads to prune
4. **Executes structured pruning** using Torch-Pruning
   - Prunes `attn.proj` input channels →联动 `attn.mat_qkv` output channels
   - Prunes `ffn.fc2` input channels → 联动 `ffn.fc1` output channels
5. **Updates model parameters**
   - Head count, biases, scale parameters
6. **Saves pruned model** and reports statistics

### Key Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_depth` | 16 | VAR model depth (16/20/24/30) |
| `--num_samples` | 256 | Number of calibration samples |
| `--use_images` | False | Use real ImageNet images (pre-encode tokens) |
| `--imagenet_dir` | - | Path to ImageNet train directory |
| `--sparsity` | 0.2 | Target pruning ratio (20%) |
| `--minlayer` | 0 | Start pruning from this layer |
| `--maxlayer` | 16 | Prune up to this layer |
| `--percdamp` | 0.01 | Hessian dampening factor |

### Using Pre-encoded Tokens (Recommended):

For better pruning quality, use real ImageNet images:

```bash
python model_slimming_basic.py \
    --model_depth 16 \
    --num_samples 256 \
    --sparsity 0.2 \
    --use_images \
    --imagenet_dir /path/to/imagenet/train \
    --save_dir ./pruned_models \
    --model_name var_d16_s20_images.pth
```

This will:
1. Load 256 random images from ImageNet
2. Pre-encode them with VQVAE: `img -> tokens (679, 32)`
3. Use pre-encoded tokens for calibration (more accurate activation statistics)
4. Faster forward passes during pruning

### Output:

The script prints:
- Layer-wise sparsity
- Overall model sparsity
- Parameter count before/after
- Inference speed measurement
- MACs and parameter count

Example output:
```
layer 0 sparsity 0.200000
layer 1 sparsity 0.200000
...
Overall sparsity: 0.2000

Parameters after pruning: 0.24B
Parameter reduction: 20.00%

Average inference time: 45.32 ms

Final statistics:
  Parameters: 240.00M
  MACs: 12.50G
```

## Advanced Features (TODO)

See `INNOVATIONS_README.md` for planned advanced features:

1. **Scale-wise Importance Analysis** (`pruning_innovations.py`)
   - Analyze importance across 10 VAR scales
   - Find globally unimportant heads based on cross-scale average

2. **QKV/FC1 Compensation** (`pruning_innovations.py`)
   - Compensate pruned channels before removal
   - Two methods: cosine similarity (fast) and optimal alpha (accurate)

3. **Progressive Pruning** (`pruning_innovations.py`)
   - Multi-stage pruning: 10% → 20% → 30%
   - Lightweight evaluation between stages
   - Only evaluate FID once at the end

4. **Integrated Main Script** (`model_slimming_var.py`)
   - Parameter-controlled feature switches
   - Complete pipeline orchestration

## Implementation Status

- [x] Documentation (Chapter 8 + Quick Reference)
- [x] Basic pruning script
- [x] Test bash script
- [ ] Innovation module (`pruning_innovations.py`)
- [ ] Integrated main script (`model_slimming_var.py`)
- [ ] Experiment scripts

## Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install torch-pruning
```

Make sure you have:
- VQVAE checkpoint: `/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth`
- VAR checkpoint: `/home/project/daily/AR/model_zoo/var_d{depth}.pth`

## References

- **Documentation**: `VAR_PRUNING_CALIBRATION_GUIDE.md` (complete guide)
- **Quick Reference**: `INNOVATIONS_README.md` (innovation features)
- **Reference Implementation**: `tp_prune_reference.py` (Torch-Pruning example)

## Next Steps

1. Test basic pruning script
2. Implement advanced features in `pruning_innovations.py`
3. Create integrated main script
4. Run experiments and evaluate FID

For any questions, refer to the complete documentation in `VAR_PRUNING_CALIBRATION_GUIDE.md`.
