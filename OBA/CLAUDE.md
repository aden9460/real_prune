# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **Optimal Brain Apoptosis (OBA)**, a novel neural network pruning method that directly calculates Hessian-vector products for each parameter. OBA advances beyond traditional approaches like Optimal Brain Damage (OBD) that rely on Fisher matrix approximations. The codebase supports both structured and unstructured pruning for CNNs and Transformers on CIFAR10, CIFAR100, and ImageNet datasets.

## Key Concepts

**Pruning Methods**:
- **OBA**: Calculates Hessian-vector products directly using second-order Taylor expansion
- **FastOBA**: Fast version computing k-th order Taylor terms efficiently
- **Baseline Methods**: C-OBS, C-OBD, Kron-OBS, Kron-OBD, Eigen, Weight (magnitude), Taylor

**Connectivity Analysis**: OBA decomposes the Hessian matrix across layers and identifies:
- Series connectivity (sequential layer dependencies)
- Parallel connectivity (residual/skip connections)
- Direct connectivity importance (upward, downward, parallel)

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate oba
```

**Key Dependencies**: Python 3.11, PyTorch 2.0.1, CUDA 11.7, torch-pruning, thop, wandb

## Dataset Configuration

Edit `dataset.yaml` to specify dataset paths:
- CIFAR10/CIFAR100: Will auto-download to specified directory
- ImageNet: Must be manually downloaded to the specified path

## Common Commands

### Structured Pruning

**One-shot pruning on CIFAR**:
```bash
python train_prune.py --importance_type OBA --dataset cifar10 --model vgg19 --ops_ratios 0.14
```

**Iterative pruning on CIFAR**:
```bash
python iterative_prune.py --importance_type OBA --dataset cifar10 --model resnet32 --ops_ratios 0.6 0.5 0.4 0.3 0.2 0.1 0.05
```

**One-shot pruning on ImageNet**:
```bash
python prune_imagenet.py --importance_type OBA --model vit_b_16 --ops_ratios 0.51
```

**Fine-tune pruned ImageNet model**:
```bash
python train_imagenet.py --save_dir <directory_to_saved_model> --model_name <pruned_model>
```

### Unstructured Pruning

```bash
python train_unstructured_prune.py --importance_type OBA --dataset cifar10 --model resnet20 --pruning_ratio 0.1
```

### FastOBA Usage

Set `--importance_type fastOBA` to use the fast version:
```bash
python train_prune.py --importance_type fastOBA --dataset cifar10 --model vgg19 --ops_ratios 0.14 --order 2
```

The `--order` parameter controls the k-th Taylor term (k=2 corresponds to Hessian-vector product).

## Architecture

### Directory Structure

- `train_prune.py`: Main entry point for one-shot structured pruning on CIFAR
- `iterative_prune.py`: Iterative pruning with multiple target ratios
- `prune_imagenet.py`: ImageNet pruning
- `train_imagenet.py`: ImageNet fine-tuning
- `train_unstructured_prune.py`: Unstructured/unstructured pruning
- `registry.py`: Model and dataset registry
- `dataset.yaml`: Dataset path configuration
- `environment.yaml`: Conda environment specification

### Core Modules

**torch_pruning/**: Custom pruning library (modified from Torch-Pruning)
- `conduct_pruning.py`: `WrappedPruner` class that orchestrates pruning
- `pruner/algorithms/`: Pruner implementations
  - `oba_pruner.py`: Main OBA pruner with hook-based Hessian computation
  - `fastoba_pruner.py`: FastOBA pruner for k-th order Taylor terms
  - `kfac_*.py`: KFAC-based baseline pruners
  - `metapruner.py`: Base pruner class
- `pruner/oba_importance.py`: Hessian importance computation
- `pruner/fastoba_importance.py`: FastOBA importance computation
- `pruner/kfac_importance.py`: KFAC-based importance
- `pruner/kfac_utils/`: KFAC utilities and network builders
- `dependency.py`: Layer dependency graph for structured pruning
- `ops.py`: Operations registry

**engine/**: Model and utility implementations
- `models/cifar/`: CIFAR models (VGG, ResNet, DenseNet, ViT, Swin, etc.)
- `models/imagenet/`: ImageNet models (ResNet50, ViT-B/16, etc.)
- `models/graph/`: Graph models (DGCNN for ModelNet40)
- `utils/`: Dataset loaders, metrics, evaluators

**modules/**: Additional model implementations
- `vgg.py`: VGG variants for ImageNet

### Key Classes and Flow

**WrappedPruner** (`torch_pruning/conduct_pruning.py`):
- Main interface for pruning operations
- Instantiates appropriate pruner based on `importance_type`
- Methods:
  - `iterative_prune_step()`: Single iterative pruning step
  - `onepass_prune_step()`: One-shot pruning to target ratio
  - `onepass_unstructured_prune_step()`: Unstructured pruning

**OBAPruner** (`torch_pruning/pruner/algorithms/oba_pruner.py`):
- Implements OBA algorithm
- Registers forward/backward hooks to capture activations and gradients
- `obtain_importance()`: Computes Hessian-vector products
- Tracks connectivity types (upward, downward, parallel) via dependency graph

**FastOBAPruner** (`torch_pruning/pruner/algorithms/fastoba_pruner.py`):
- Fast k-th order Taylor expansion computation
- `obtain_importance(loss, order)`: Computes k-th order terms
- More efficient than OBA for higher-order terms

### Important Parameters

**Pruning Control**:
- `--ops_ratios`: Target FLOPs ratio (e.g., 0.14 = 14% of original FLOPs)
- `--pruning_ratio`: For unstructured pruning
- `--iterative_steps`: Number of iterative pruning steps
- `--iters_per_step`: Training iterations per pruning step
- `--max_pruning_ratio`: Maximum channel sparsity (default 0.95)

**OBA-Specific**:
- `--delta`: Base delta for OBA importance
- `--upward_delta`: Weight for upward connectivity
- `--downward_delta`: Weight for downward connectivity
- `--parallel_delta`: Weight for parallel connectivity
- `--self_unit_weight`: Use unit weights for self-connections
- `--other_unit_weight`: Use unit weights for cross-layer connections
- `--multivariable`: Use multivariable importance computation

**FastOBA-Specific**:
- `--order`: Order of Taylor expansion (2 = Hessian)
- `--fastoba_delta`: Delta for FastOBA importance

**Training**:
- `--num_epochs`: Dense training epochs (default 200)
- `--sl-num-epochs`: Sparsity learning/fine-tuning epochs (default 150)
- `--lr`: Initial learning rate (default 0.1)
- `--sl-lr`: Sparsity learning rate (default 0.001)
- `--normalizer`: Importance score normalizer ("max", "mean", etc.)

## Testing

The codebase does not include separate test files. Evaluation occurs during training via:
- `eval()` function in training scripts
- Test accuracy computed after each epoch
- Logged to W&B if `--use_wandb` is enabled

## Model Registry

**CIFAR Models** (32x32 input):
- VGG: vgg11, vgg13, vgg16, vgg19
- ResNet: resnet18, resnet20, resnet32, resnet44, resnet56, resnet110
- Others: densenet121, googlenet, mobilenetv2, xception, vit_cifar, swin_t/s/b/l

**ImageNet Models** (224x224 input):
- resnet50, densenet121, mobilenet_v2, vgg16_bn, vgg19_bn
- vit_b_16, inception_v3, googlenet
- regnet_x_1_6gf, resnext50_32x4d

Models defined in `registry.py` with corresponding implementations in `engine/models/`.

## Important Implementation Details

**Hook-Based Computation**: OBA uses forward/backward hooks (`forward_hook`, `backward_hook` in `modules/models.py`) to capture intermediate activations (X, Y) and gradients. These are stored on modules and used to compute Hessian-vector products.

**Dependency Graph**: The pruner builds a dependency graph (`torch_pruning/dependency.py`) to track which channels/parameters depend on each other. This ensures valid pruning that maintains network connectivity.

**Checkpoint Management**:
- Pre-trained models saved to `checkpoints/{dataset}/{model}/best_pretrain.pth`
- Training automatically loads if checkpoint exists, otherwise trains from scratch
- Pruned models saved to `checkpoints/{dataset}/{model}/{log_name}/`

**Multi-Step Pruning**: For iterative pruning, the model is reset to the original pre-trained checkpoint after each pruning ratio target is achieved. This allows comparing different pruning ratios independently.

## Debugging Tips

- If CUDA OOM errors occur, reduce `--batch_size` or `--iters_per_step`
- For OBA, hook registration is critical - ensure `register_hooks()` is called
- Check dependency graph if pruning produces invalid architectures
- Use `--multistep False` for direct one-shot pruning to target ratio
- W&B logging requires `--use_wandb True` and proper wandb setup
