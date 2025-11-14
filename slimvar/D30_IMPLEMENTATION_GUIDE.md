# D30 Model Pruning Implementation Guide

## Quick Start

For immediate implementation on d30 models, follow this guide.

---

## 1. Pre-Pruning Analysis

### Step 1: Generate Scale_mul Analysis
```bash
cd /home/project/real_prune/slimvar
python analyze_scale_mul.py \
  --model_path /path/to/d30_checkpoint.pth \
  --depth 30 \
  --output_dir ./scale_mul_analysis_d30
```

### Step 2: Verify High-Variance Layers
Expected output:
- Layers 10-23, 26-29 should show variance > 31.60
- Middle layers (10-19) should have highest variance
- Layer 14 should be near peak (~117 variance)

If your results differ significantly, adjust protection strategy accordingly.

---

## 2. Choose Pruning Configuration

### Option A: Conservative 20% Pruning (Recommended for First Trial)
```json
Use: d30_pruning_config_20percent.json

Expected outcomes:
- FID degradation: <10%
- Training speedup: ~1.2x
- Memory savings: ~15%
```

### Option B: Aggressive 40% Pruning
```json
Use: d30_pruning_config_40percent.json

Expected outcomes:
- FID degradation: 10-20%
- Training speedup: ~1.5x
- Memory savings: ~30%
```

---

## 3. Apply Pruning

### Method 1: Using Existing Pruning Script
```bash
python model_slimming_basic.py \
  --model var-d30 \
  --pruning_config d30_pruning_config_20percent.json \
  --checkpoint /path/to/d30_checkpoint.pth \
  --output /path/to/d30_pruned.pth
```

### Method 2: Manual Integration

Add to your training script:

```python
import json

# Load protection config
with open('d30_pruning_config_20percent.json') as f:
    config = json.load(f)

# Apply per-layer pruning
for layer_idx, layer_config in enumerate(config['layer_configs']):
    layer = model.layers[layer_idx]
    target_heads = layer_config['heads_keep']
    
    # Your pruning logic here
    # Example: prune_heads(layer, target_heads)
```

---

## 4. Fine-tuning Strategy

### Recommended Fine-tuning Schedule

**Phase 1: Warmup (10% of original training)**
- Learning rate: 1e-5 (10x smaller than pre-training)
- Batch size: Same as pre-training
- Focus: Let model adapt to reduced capacity

**Phase 2: Main Fine-tuning (20% of original training)**
- Learning rate: 5e-5
- Gradually increase to 1e-4 if stable
- Monitor FID every 5k steps

**Phase 3: Convergence (10% of original training)**
- Learning rate: 1e-6
- Final polish

### Example Fine-tuning Command
```bash
python train_var.py \
  --model_path /path/to/d30_pruned.pth \
  --learning_rate 1e-5 \
  --epochs 50 \
  --finetune \
  --log_dir ./logs/d30_finetune_20p
```

---

## 5. Validation & Monitoring

### Key Metrics to Track

1. **FID Score**
   ```bash
   python FID_test.py \
     --model_path /path/to/d30_pruned_finetuned.pth \
     --dataset imagenet \
     --output fid_results_d30.json
   ```

2. **Scale_mul Post-Pruning**
   ```bash
   python analyze_scale_mul.py \
     --model_path /path/to/d30_pruned_finetuned.pth \
     --depth 30 \
     --output_dir ./scale_mul_analysis_d30_pruned
   ```

3. **Per-Layer Statistics**
   Compare before/after pruning:
   - Protected layers (12-21) should retain >80% of original scale_mul
   - Non-protected layers can drop to 50-60%

### Success Criteria Checklist

- [ ] FID degradation within expected range (<10% for 20%, <20% for 40%)
- [ ] Protected layers maintained high scale_mul (>80% of original)
- [ ] Training converged (loss plateaued)
- [ ] No NaN or divergence during fine-tuning
- [ ] Model generates coherent samples (visual inspection)

---

## 6. Troubleshooting

### Problem: FID Degradation >20% for 20% Pruning

**Possible Causes:**
1. Wrong layers protected
2. Insufficient fine-tuning
3. Learning rate too high

**Solutions:**
- Re-run scale_mul analysis to verify variance distribution
- Extend fine-tuning by 50%
- Reduce learning rate to 1e-6

### Problem: Training Divergence During Fine-tuning

**Possible Causes:**
1. Learning rate too high
2. Batch size too large
3. Over-pruning (>70% in some layers)

**Solutions:**
- Reduce learning rate by 10x
- Use gradient clipping (max_norm=1.0)
- Check pruning config - no layer should be >70% pruned

### Problem: Protected Layers Lose >30% Scale_mul

**Possible Causes:**
1. Protection not applied correctly
2. Global normalization affecting protected layers
3. Fine-tuning hyperparameters wrong

**Solutions:**
- Verify pruning logic - protected layers should have 0% pruning
- Disable any global head pruning mechanisms
- Use lower learning rate (1e-6) for protected layers

---

## 7. Ablation Experiments

To validate the protection strategy, run these experiments:

### Experiment 1: Baseline (No Pruning)
```bash
# Just measure original model
python FID_test.py --model_path /path/to/d30_original.pth
```

### Experiment 2: Smart Pruning (With Protection)
```bash
# Use d30_pruning_config_40percent.json
# Should show <20% FID degradation
```

### Experiment 3: Uniform Pruning (No Protection)
```bash
# Prune 40% uniformly across all layers
# Expected: >30% FID degradation
# This proves protection is necessary
```

### Experiment 4: Wrong Protection
```bash
# Protect early layers (0-9) instead of middle (12-21)
# Expected: Severe degradation
# This proves variance-guided selection is correct
```

---

## 8. Expected Results Summary

### 20% Pruning (Conservative)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| FID Score | X | <1.1X | <10% ↑ |
| Total Heads | 900 | 720 | -20% |
| Training Time/Epoch | T | ~0.85T | -15% |
| GPU Memory | M | ~0.85M | -15% |
| Protected Layer Scale_mul | S | >0.9S | >90% retained |

### 40% Pruning (Aggressive)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| FID Score | X | <1.2X | <20% ↑ |
| Total Heads | 900 | 540 | -40% |
| Training Time/Epoch | T | ~0.65T | -35% |
| GPU Memory | M | ~0.7M | -30% |
| Protected Layer Scale_mul | S | >0.8S | >80% retained |

---

## 9. Production Deployment Checklist

Before deploying pruned d30 model to production:

- [ ] FID score validated on held-out test set
- [ ] Visual quality inspection on diverse samples
- [ ] Latency benchmarks (should be ~1.5x faster for 40% pruning)
- [ ] Memory profiling (should use ~30% less for 40% pruning)
- [ ] Edge case testing (unusual prompts, extreme resolutions)
- [ ] Gradual rollout (A/B test 5% traffic first)
- [ ] Monitoring dashboard (track FID, latency, memory in production)

---

## 10. Advanced Optimizations

### Dynamic Protection During Training
Instead of static protection, adjust protection during fine-tuning:

```python
def dynamic_protection(epoch, total_epochs):
    """Gradually increase protection strength"""
    if epoch < total_epochs * 0.3:
        return 0.40  # Protect layers > 40th percentile
    elif epoch < total_epochs * 0.7:
        return 0.50  # Increase to 50th percentile
    else:
        return 0.60  # Final phase: 60th percentile
```

### Head-Level Pruning Within Protected Layers
For even better results, prune individual heads in protected layers:

```python
# Within layer 14 (highest variance layer)
# Keep top 25 heads (out of 30) by variance
# This allows 17% more pruning without destroying layer function
```

### Iterative Pruning
Instead of one-shot 40% pruning:
1. Prune 20%, fine-tune 50 epochs
2. Analyze new scale_mul distribution
3. Prune another 20%, fine-tune 50 epochs
4. Total: 36% pruned with better preservation

---

## 11. Quick Reference Commands

```bash
# 1. Analyze original model
python analyze_scale_mul.py --model_path d30_orig.pth --depth 30

# 2. Apply 20% pruning
python model_slimming_basic.py --model var-d30 --pruning_config d30_pruning_config_20percent.json

# 3. Fine-tune
python train_var.py --model_path d30_pruned.pth --learning_rate 1e-5 --epochs 50 --finetune

# 4. Validate
python FID_test.py --model_path d30_finetuned.pth

# 5. Analyze pruned model
python analyze_scale_mul.py --model_path d30_finetuned.pth --depth 30
```

---

## 12. Contact & Support

If results deviate significantly from expectations:
1. Check scale_mul analysis matches expected distribution
2. Verify pruning config applied correctly (inspect model architecture)
3. Review fine-tuning logs for anomalies
4. Compare with d16 results as baseline

For questions or issues, refer to:
- Main analysis: `D16_VS_D30_COMPARATIVE_ANALYSIS.md`
- Distribution insights: `SCALE_MUL_DISTRIBUTION_INSIGHTS.md`
- Implementation details: `STRUCT_PRUNE_WITH_HEADS_IMPLEMENTATION.md`

---

**Good luck with your d30 pruning experiments!** 🚀

Remember: Start with 20% pruning to validate the approach, then scale to 40% once confident.
