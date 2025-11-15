# VAR OBA Pruning Implementation Plan

## Core Implementation Strategy

Based on the analysis of SlimGPT approach, here's the concrete OBA implementation plan:

### 1. Layer Selection (Same as SlimGPT)
```python
target_layers = {
    'ffn.fc2': 'prune_input_channels',   # FFN compression
    'attn.proj': 'prune_input_channels', # Attention head compression
}
```

### 2. OBA Importance Computation
```python
def compute_oba_importance(model, calibration_loader):
    # Initialize OBA importance collector
    importance_scores = {}

    # Register hooks for target layers across all blocks
    for layer_idx in range(model.depth):
        for layer_name in ['ffn.fc2', 'attn.proj']:
            module = getattr(model.blocks[layer_idx], layer_name.split('.')[0])
            target = getattr(module, layer_name.split('.')[1])

            # Store importance for this specific layer
            importance_scores[f'layer_{layer_idx}.{layer_name}'] = []

    # Accumulate importance over calibration data
    model.eval()
    for batch_idx, (labels, tokens) in enumerate(calibration_loader):
        # Forward pass with OBA importance collection
        logits = model(labels, tokens)
        loss = compute_autoregressive_loss(logits, tokens)

        # Compute Hessian-based importance using OBA algorithm
        hessian_importance = compute_hessian_importance(
            model, loss, target_layers,
            delta=1.0, upward_delta=1.0, downward_delta=1.0, parallel_delta=1.0
        )

        # Accumulate
        for key, importance in hessian_importance.items():
            importance_scores[key].append(importance)

        loss.backward()
        model.zero_grad()

    # Average importance across batches
    final_importance = {}
    for key, importance_list in importance_scores.items():
        final_importance[key] = torch.stack(importance_list).mean(dim=0)

    return final_importance

def compute_hessian_importance(model, loss, target_layers, **deltas):
    importance_dict = {}

    for layer_name, layer_module in target_layers.items():
        # 1. First-order importance (self)
        weight = layer_module.weight
        grad = torch.autograd.grad(loss, weight, retain_graph=True)[0]
        first_order = deltas['delta'] * (weight * grad).abs()

        # 2. Upward connectivity (direct output impact)
        # Perturb weight and compute second-order term
        eps = 1e-6
        weight_pert = weight + eps * torch.sign(weight)
        # [Complex Hessian computation using torch.func.jvp]
        upward_term = deltas['upward_delta'] * compute_upward_hessian(weight_pert, loss)

        # 3. Downward connectivity (upstream propagation)
        downward_term = deltas['downward_delta'] * compute_downward_hessian(weight, loss)

        # 4. Parallel connectivity (attention Q-K-V interactions)
        if 'attn' in layer_name:
            parallel_term = deltas['parallel_delta'] * compute_attention_parallel_hessian(weight, loss)
        else:
            parallel_term = 0

        # Total importance
        total_importance = (first_order + upward_term + downward_term + parallel_term).sum(dim=0)
        importance_dict[layer_name] = total_importance

    return importance_dict
```

### 3. Global Pruning Decision
```python
def make_global_pruning_decision(importance_scores, target_sparsity=0.4):
    """OBA's key advantage: global optimization instead of layer-wise greedy"""

    # Flatten all importance scores across layers
    all_importance = []
    layer_info = []

    for layer_key, importance in importance_scores.items():
        layer_idx = int(layer_key.split('_')[1].split('.')[0])
        layer_type = layer_key.split('.')[1]  # 'ffn' or 'attn'

        for channel_idx, imp_score in enumerate(importance):
            all_importance.append(imp_score.item())
            layer_info.append({
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'channel_idx': channel_idx,
                'importance': imp_score.item()
            })

    # Sort by importance (ascending, remove least important)
    sorted_indices = sorted(range(len(all_importance)), key=lambda x: all_importance[x])

    # Determine pruning budget
    total_params = len(all_importance)
    num_to_prune = int(total_params * target_sparsity)

    # Select channels to prune (globally least important)
    prune_indices = sorted_indices[:num_to_prune]

    # Group by layer for execution
    pruning_plan = {}
    for idx in prune_indices:
        info = layer_info[idx]
        layer_key = f"layer_{info['layer_idx']}.{info['layer_type']}"

        if layer_key not in pruning_plan:
            pruning_plan[layer_key] = []
        pruning_plan[layer_key].append(info['channel_idx'])

    return pruning_plan

def execute_pruning_with_slimgpt_method(model, pruning_plan):
    """Reuse SlimGPT's proven pruning execution code"""

    for layer_plan, channel_indices in pruning_plan.items():
        layer_idx = int(layer_plan.split('_')[1].split('.')[0])
        layer_type = layer_plan.split('.')[1]

        if layer_type == 'ffn':
            # Use SlimGPT's FFN pruning logic
            target_layer = model.blocks[layer_idx].ffn.fc2
            target_layer_b = model.blocks[layer_idx].ffn.fc1

            tp.prune_linear_in_channels(target_layer, channel_indices)
            tp.prune_linear_out_channels(target_layer_b, channel_indices)

        elif layer_type == 'attn':
            # Use SlimGPT's attention pruning logic (complex but proven)
            # [Copy exact code from lines 431-486 of model_slimming_basic_v1.py]
            execute_attention_pruning(model.blocks[layer_idx], channel_indices)

    return model

def execute_attention_pruning(block, channel_indices):
    """Exact copy of SlimGPT attention pruning logic"""
    target_layer = block.attn.proj
    sparsity = len(channel_indices) / target_layer.in_features

    # Update num_heads
    block.attn.num_heads = torch.round(torch.tensor(16 * (1 - sparsity))).int()

    # Update biases (exact copy of SlimGPT logic)
    keep_idxs = list(set(range(target_layer.in_features)) - set(channel_indices))
    block.attn.q_bias = nn.Parameter(block.attn.q_bias.data[keep_idxs])
    # ... [rest of SlimGPT attention pruning code]
```

### 4. Main Training Script Template
```python
# var_prune_oba.py
def main():
    # Load models
    var = load_var_d16('/home/project/daily/AR/model_zoo/var_d16.pth')
    vae = load_vae('/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth')

    # Prepare calibration data
    calib_loader = create_calibration_loader(
        imagenet_dir='/home/project/ImageNet-1K',
        vae_model=vae,
        num_samples=256,
        batch_size=8
    )

    # Compute OBA importance
    print("Computing OBA importance scores...")
    importance_scores = compute_oba_importance(var, calib_loader)

    # Make global pruning decision
    print("Making global pruning decisions...")
    pruning_plan = make_global_pruning_decision(importance_scores, target_sparsity=0.4)

    # Execute pruning (reuse SlimGPT execution)
    print("Executing pruning...")
    pruned_var = execute_pruning_with_slimgpt_method(var, pruning_plan)

    # Save pruned model
    torch.save({
        'model_state_dict': pruned_var.state_dict(),
        'pruning_plan': pruning_plan,
        'importance_scores': importance_scores,
    }, './var_d16_oba_pruned_0.4sparsity.pth')

    print("OBA pruning completed!")
    print(f"Pruned model saved to: ./var_d16_oba_pruned_0.4sparsity.pth")

    # Compute final statistics
    original_params = sum(p.numel() for p in var.parameters())
    pruned_params = sum(p.numel() for p in pruned_var.parameters())
    compression_ratio = pruned_params / original_params

    print(f"Compression ratio: {compression_ratio:.2%}")
    print(f"Parameter reduction: {1-compression_ratio:.2%}")

if __name__ == "__main__":
    main()
```

## Key Differences from SlimGPT:

### **Algorithm Level:**
1. **Importance Computation**:
   - SlimGPT: Fisher diagonal approximation (`H = XX^T`, importance = `W²/diag(H)`)
   - OBA: Full Hessian-vector products with connectivity modeling

2. **Optimization Strategy**:
   - SlimGPT: Layer-wise greedy pruning
   - OBA: Global joint optimization across all layers

3. **Connectivity Modeling**:
   - SlimGPT: Independent layer processing
   - OBA: Explicit upward/downward/parallel connectivity terms

### **Engineering Level:**
1. **Pruning Execution**: Reuse SlimGPT's proven torch_pruning calls and manual parameter updates
2. **VAR Specific Handling**: Keep SlimGPT's complex attention parameter management
3. **Data Pipeline**: Same ImageNet → VAE → tokens approach

## Expected Advantages:
1. **Better Global Optimality**: OBA considers cross-layer dependencies
2. **Attention-Aware**: Parallel connectivity explicitly models Q-K-V interactions
3. **More Accurate Importance**: Full Hessian vs diagonal Fisher approximation
4. **Layer Balance**: Global optimization may lead to better layer-wise pruning distribution

## Implementation Priority:
1. ✅ **Phase 1**: Implement importance computation (OBA core)
2. ✅ **Phase 2**: Copy SlimGPT's pruning execution (proven to work)
3. ✅ **Phase 3**: Global optimization (key OBA advantage)
4. ✅ **Phase 4**: Validation and comparison