# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Olica is a research implementation for efficient structured pruning of Large Language Models (specifically LLaMA models) without retraining. It achieves model compression using only hundreds of calibration samples, several minutes of runtime, and a single 16GB GPU. The codebase implements the Olica pruning algorithm described in an ICML 2025 paper.

## Environment Setup

```bash
pip install -r requirements.txt
```

**Critical dependency**: `transformers==4.31.0` must be forcibly required (this exact version is mandatory).

## Common Commands

### Pruning a Model

```bash
CUDA_VISIBLE_DEVICES=0 python model_pruning.py \
  --base_model 7b \
  --datasets bookcorpus+alpaca \
  --num_samples 128 \
  --seqlen 128 \
  --percdamp 0.5 \
  --mlp_num 6 \
  --sparsity 0.2 \
  --ratio 0.03 \
  --k 3 \
  --save_dir ./pruned_models
```

### Evaluating a Pruned Model

```bash
CUDA_VISIBLE_DEVICES=0 python model_evaluate.py \
  --base_model 7b \
  --sparsity 0.2 \
  --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
  --save_dir ./pruned_models
```

### Using Scripts

Convenience scripts are available:
- `bash scripts/pruning_and_eval_7b.sh` - Complete pruning and evaluation pipeline for 7B models
- `bash scripts/pruning_and_eval_13b.sh` - Complete pruning and evaluation pipeline for 13B models

## Architecture Overview

### Core Pruning Algorithm

The Olica pruning approach operates in two main phases:

1. **Fast Orthogonal Neuron Decomposition (fast_OND)** - `utils/llama_utils.py`
   - Analyzes model layers to identify importance of MLP neurons
   - Uses forward hooks to capture layer inputs/outputs
   - Computes importance scores based on activation norms
   - Returns ranking of MLP layers for calibration

2. **Structured Pruning** - `utils/llama_utils.py`
   - Prunes attention mechanisms (Q, K, V, O projections)
   - Prunes MLP layers using SVD-based low-rank approximation
   - Uses regression with Cholesky decomposition for weight calibration
   - Preserves model structure through `CustomizedMLP` wrappers

### Key Components

**Model Entry Points:**
- `model_pruning.py` - Main pruning script, orchestrates the entire pruning pipeline
- `model_evaluate.py` - Evaluation script that loads pruned models and runs benchmarks

**Pruning Implementation:**
- `utils/llama_utils.py` - Core pruning algorithms (`fast_OND`, `pruning`, `thinner_mlp`)
- `utils/utils.py` - Helper utilities including:
  - `WrappedGPT` - Layer wrapper for collecting activation statistics
  - `solve()` - Regression solver using Cholesky decomposition for weight calibration
  - `SVDLinearForWidth` - SVD-based low-rank linear layer
  - Dataset loaders (`get_bookcorpus`, `get_alpaca`, `get_c4`)

**Customized Model Architecture:**
- `utils/customized_llama.py` - Modified LLaMA implementation supporting:
  - Variable-rank attention projections per layer
  - SVD-decomposed linear layers
  - Hybrid MLP structure (full + low-rank residual)
  - Configuration with `q_lowranks`, `k_lowranks`, `vo_lowranks`, `mlp_lowranks`, `layer_inter_size`

**Evaluation Framework:**
- `evaluate.py` - Wrapper around lm-evaluation-harness
- `lm_eval/` - Modified lm-evaluation-harness for benchmark tasks
- `ppl_eval/` - Perplexity evaluation on wikitext2 and other datasets

### Important Implementation Details

**Model Paths:**
The code contains hardcoded model paths (e.g., `/userhome/home/hejiujun/ckpts/Llama-{size}`). When working with this codebase, you'll need to:
- Update `model_pruning.py:133` and `model_evaluate.py:66` to point to your LLaMA model location
- Update dataset paths in `utils/utils.py` for bookcorpus, alpaca, and c4 datasets

**Data Caching:**
Calibration data is cached in `./data/` directory with format: `{model_name}_{datasets}_nsample:{num_samples}_seqlen{seqlen}.pkl`

**Pruned Model Storage:**
Pruned models are saved to: `{save_dir}/{model_name}/SR:{sparsity}_{model_name}/`

### Pruning Parameters

Key hyperparameters and their roles:

- `--sparsity`: Target overall sparsity ratio (e.g., 0.2 = 20% of parameters pruned)
- `--mlp_num`: Number of MLP layers to apply low-rank calibration (typically 6)
- `--percdamp`: Dampening factor λ for regression loss (Eq. 7 in paper)
- `--ratio`: Low-rank ratio for SVD decomposition in linear calibration
- `--k`: Ratio controlling QK vs VO parameter distribution
- `--num_samples`: Number of calibration samples per dataset
- `--seqlen`: Sequence length for calibration samples

### Evaluation Metrics

- **Perplexity (PPL)** - Evaluated on wikitext2 dataset
- **Accuracy** - Evaluated on downstream tasks (OpenBookQA, ARC, WinoGrande, HellaSwag, PIQA, BoolQ)
- Reports both `acc` and `acc_norm` (length-normalized accuracy)

## Development Notes

**Critical Version Requirements:**
- Must use `transformers==4.31.0` (later versions may break compatibility)
- The customized LLaMA implementation depends on internal APIs from this specific version

**Working with Forward Hooks:**
The pruning algorithm extensively uses PyTorch forward hooks to capture intermediate activations. When debugging:
- Hooks are registered in `fast_OND()` and must be properly removed with `.remove()`
- Hook functions capture data to lists that must be managed carefully for memory

**Low-Rank Decomposition:**
The pruned model uses a hybrid architecture:
- Attention: SVD-decomposed Q/K projections + reduced-dimension V/O
- MLP: Can be either standard pruned or `CustomizedMLP` (full MLP + low-rank residual)
- Configuration stored in model config with per-layer rank specifications

**Memory Management:**
- Models are moved between CPU and GPU strategically to manage memory
- Uses `torch.cuda.empty_cache()` aggressively after large operations
- Calibration data is stored on CPU to avoid GPU OOM
