import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
print("="*80)
print("D16 vs D30 SCALE_MUL DISTRIBUTION ANALYSIS")
print("="*80)

d16_orig = json.load(open('/home/project/real_prune/slimvar/scale_mul_analysis_d16/scale_mul_analysis.json'))
d16_pruned = json.load(open('/home/project/real_prune/VAR_FIDtest/scale_mul_analysis_d16_0.4/scale_mul_analysis.json'))
d30_orig = json.load(open('/home/project/real_prune/slimvar/scale_mul_analysis_d30/scale_mul_analysis.json'))

print("\n### 1. BASIC STATISTICS COMPARISON ###\n")

# Global statistics
print("Global Statistics:")
print(f"  d16 (original): mean={d16_orig['global']['mean']:.4f}, std={d16_orig['global']['std']:.4f}")
print(f"  d16 (40% pruned): mean={d16_pruned['global']['mean']:.4f}, std={d16_pruned['global']['std']:.4f}")
print(f"  d30 (original): mean={d30_orig['global']['mean']:.4f}, std={d30_orig['global']['std']:.4f}")
print()

# Extract per-layer variance
d16_variances = [layer['variance'] for layer in d16_orig['per_layer']]
d30_variances = [layer['variance'] for layer in d30_orig['per_layer']]

d16_means = [layer['mean'] for layer in d16_orig['per_layer']]
d30_means = [layer['mean'] for layer in d30_orig['per_layer']]

print("Per-layer Variance Statistics:")
print(f"  d16: min={min(d16_variances):.2f}, max={max(d16_variances):.2f}, avg={np.mean(d16_variances):.2f}")
print(f"  d30: min={min(d30_variances):.2f}, max={max(d30_variances):.2f}, avg={np.mean(d30_variances):.2f}")
print()

# Percentiles
d16_var_percentiles = np.percentile(d16_variances, [25, 40, 50, 60, 75])
d30_var_percentiles = np.percentile(d30_variances, [25, 40, 50, 60, 75])

print("Variance Percentiles:")
print(f"  d16: Q1={d16_var_percentiles[0]:.2f}, 40%={d16_var_percentiles[1]:.2f}, Q2={d16_var_percentiles[2]:.2f}, 60%={d16_var_percentiles[3]:.2f}, Q3={d16_var_percentiles[4]:.2f}")
print(f"  d30: Q1={d30_var_percentiles[0]:.2f}, 40%={d30_var_percentiles[1]:.2f}, Q2={d30_var_percentiles[2]:.2f}, 60%={d30_var_percentiles[3]:.2f}, Q3={d30_var_percentiles[4]:.2f}")
print()

print("\n### 2. HIGH VARIANCE LAYER IDENTIFICATION ###\n")

# Using 40th percentile as threshold
d16_threshold_40 = d16_var_percentiles[1]
d30_threshold_40 = d30_var_percentiles[1]

d16_high_var_layers = [i for i, v in enumerate(d16_variances) if v > d16_threshold_40]
d30_high_var_layers = [i for i, v in enumerate(d30_variances) if v > d30_threshold_40]

print(f"High Variance Layers (>40th percentile):")
print(f"  d16 threshold: {d16_threshold_40:.2f}")
print(f"  d16 high-var layers: {d16_high_var_layers}")
print(f"  d16 count: {len(d16_high_var_layers)}/16 ({len(d16_high_var_layers)/16*100:.1f}%)")
print()
print(f"  d30 threshold: {d30_threshold_40:.2f}")
print(f"  d30 high-var layers: {d30_high_var_layers}")
print(f"  d30 count: {len(d30_high_var_layers)}/30 ({len(d30_high_var_layers)/30*100:.1f}%)")
print()

print("\n### 3. LAYER STRUCTURE ANALYSIS ###\n")

def classify_layers_by_position(num_layers):
    """Classify layers into early/middle/late thirds"""
    third = num_layers // 3
    early = list(range(0, third))
    middle = list(range(third, 2*third))
    late = list(range(2*third, num_layers))
    return early, middle, late

d16_early, d16_middle, d16_late = classify_layers_by_position(16)
d30_early, d30_middle, d30_late = classify_layers_by_position(30)

print("Layer Structure (by thirds):")
print(f"  d16: early={d16_early}, middle={d16_middle}, late={d16_late}")
print(f"  d30: early={d30_early}, middle={d30_middle}, late={d30_late}")
print()

# Analyze high-variance layer distribution
def count_in_sections(high_var_layers, early, middle, late):
    early_count = len([l for l in high_var_layers if l in early])
    middle_count = len([l for l in high_var_layers if l in middle])
    late_count = len([l for l in high_var_layers if l in late])
    return early_count, middle_count, late_count

d16_e, d16_m, d16_l = count_in_sections(d16_high_var_layers, d16_early, d16_middle, d16_late)
d30_e, d30_m, d30_l = count_in_sections(d30_high_var_layers, d30_early, d30_middle, d30_late)

print("High-Variance Layer Distribution:")
print(f"  d16: early={d16_e}/{len(d16_early)} ({d16_e/len(d16_early)*100:.1f}%), middle={d16_m}/{len(d16_middle)} ({d16_m/len(d16_middle)*100:.1f}%), late={d16_l}/{len(d16_late)} ({d16_l/len(d16_late)*100:.1f}%)")
print(f"  d30: early={d30_e}/{len(d30_early)} ({d30_e/len(d30_early)*100:.1f}%), middle={d30_m}/{len(d30_middle)} ({d30_m/len(d30_middle)*100:.1f}%), late={d30_l}/{len(d30_late)} ({d30_l/len(d30_late)*100:.1f}%)")
print()

print("\n### 4. VARIANCE PEAKS ANALYSIS ###\n")

# Find top variance layers
d16_top5_indices = np.argsort(d16_variances)[-5:][::-1]
d30_top5_indices = np.argsort(d30_variances)[-5:][::-1]

print("Top 5 Variance Layers:")
print("  d16:")
for idx in d16_top5_indices:
    rel_pos = idx / 16 * 100
    print(f"    Layer {idx}: variance={d16_variances[idx]:.2f}, mean={d16_means[idx]:.2f} (position={rel_pos:.1f}%)")

print("\n  d30:")
for idx in d30_top5_indices:
    rel_pos = idx / 30 * 100
    print(f"    Layer {idx}: variance={d30_variances[idx]:.2f}, mean={d30_means[idx]:.2f} (position={rel_pos:.1f}%)")
print()

print("\n### 5. HYPOTHESIS VALIDATION ###\n")

print("Hypothesis 1: High variance layers concentrate in middle section")
d16_middle_ratio = d16_m / len(d16_high_var_layers) if len(d16_high_var_layers) > 0 else 0
d30_middle_ratio = d30_m / len(d30_high_var_layers) if len(d30_high_var_layers) > 0 else 0
print(f"  d16: {d16_m}/{len(d16_high_var_layers)} = {d16_middle_ratio*100:.1f}% in middle")
print(f"  d30: {d30_m}/{len(d30_high_var_layers)} = {d30_middle_ratio*100:.1f}% in middle")
print(f"  Result: {'CONFIRMED' if d16_middle_ratio > 0.5 and d30_middle_ratio > 0.5 else 'REJECTED'}")
print()

print("Hypothesis 2: Early and late layers have lower variance")
d16_early_avg = np.mean([d16_variances[i] for i in d16_early])
d16_middle_avg = np.mean([d16_variances[i] for i in d16_middle])
d16_late_avg = np.mean([d16_variances[i] for i in d16_late])

d30_early_avg = np.mean([d30_variances[i] for i in d30_early])
d30_middle_avg = np.mean([d30_variances[i] for i in d30_middle])
d30_late_avg = np.mean([d30_variances[i] for i in d30_late])

print(f"  d16: early_avg={d16_early_avg:.2f}, middle_avg={d16_middle_avg:.2f}, late_avg={d16_late_avg:.2f}")
print(f"  d30: early_avg={d30_early_avg:.2f}, middle_avg={d30_middle_avg:.2f}, late_avg={d30_late_avg:.2f}")
print(f"  Result: {'CONFIRMED' if (d16_middle_avg > d16_early_avg and d16_middle_avg > d16_late_avg and d30_middle_avg > d30_early_avg and d30_middle_avg > d30_late_avg) else 'PARTIAL/REJECTED'}")
print()

print("Hypothesis 3: Variance correlates with layer depth")
d16_rel_pos = np.arange(16) / 16
d30_rel_pos = np.arange(30) / 30
d16_corr = np.corrcoef(d16_rel_pos, d16_variances)[0, 1]
d30_corr = np.corrcoef(d30_rel_pos, d30_variances)[0, 1]
print(f"  d16 correlation (position vs variance): {d16_corr:.3f}")
print(f"  d30 correlation (position vs variance): {d30_corr:.3f}")
print(f"  Interpretation: {'Weak/No linear correlation' if abs(d16_corr) < 0.3 and abs(d30_corr) < 0.3 else 'Moderate correlation detected'}")
print()

print("\n### 6. PRUNING STRATEGY RECOMMENDATIONS ###\n")

# Strategy 1: Using relative position
print("Strategy 1: Depth-Independent (Relative Position)")
print("  Protect middle 40-70% of layers (high variance concentration)")
d16_protect_start = int(16 * 0.4)
d16_protect_end = int(16 * 0.7)
d30_protect_start = int(30 * 0.4)
d30_protect_end = int(30 * 0.7)
print(f"  d16: protect layers {d16_protect_start}-{d16_protect_end} ({d16_protect_end - d16_protect_start} layers)")
print(f"  d30: protect layers {d30_protect_start}-{d30_protect_end} ({d30_protect_end - d30_protect_start} layers)")
print()

# Strategy 2: Using variance percentile
print("Strategy 2: Variance Percentile-Based")
print("  Protect layers with variance > model's 40th percentile")
print(f"  d16: threshold={d16_threshold_40:.2f}, protect {len(d16_high_var_layers)} layers")
print(f"  d30: threshold={d30_threshold_40:.2f}, protect {len(d30_high_var_layers)} layers")
print()

# Strategy 3: Combined approach
print("Strategy 3: Combined (Position + Variance)")
d16_combined_protect = [i for i in d16_high_var_layers if d16_protect_start <= i <= d16_protect_end]
d30_combined_protect = [i for i in d30_high_var_layers if d30_protect_start <= i <= d30_protect_end]
print(f"  d16: protect {len(d16_combined_protect)} layers: {d16_combined_protect}")
print(f"  d30: protect {len(d30_combined_protect)} layers: {d30_combined_protect}")
print()

print("\n### 7. D30 SPECIFIC PRUNING CONFIGURATIONS ###\n")

def generate_pruning_config(num_layers, prune_ratio, protected_layers, model_name):
    """Generate pruning configuration"""
    total_heads = num_layers * (30 if num_layers == 30 else 16)  # d30 has 30 heads/layer, d16 has 16
    heads_per_layer = 30 if num_layers == 30 else 16
    
    # Calculate target heads to keep
    target_heads_keep = int(total_heads * (1 - prune_ratio))
    
    # Calculate how many heads to prune from non-protected layers
    non_protected_layers = [i for i in range(num_layers) if i not in protected_layers]
    
    if len(non_protected_layers) == 0:
        print(f"  Warning: No non-protected layers! Cannot prune.")
        return None
    
    # Distribute pruning across non-protected layers
    heads_to_prune = int(total_heads * prune_ratio)
    prune_per_layer = heads_to_prune // len(non_protected_layers)
    remainder = heads_to_prune % len(non_protected_layers)
    
    config = {
        "model": model_name,
        "prune_ratio": prune_ratio,
        "strategy": "variance_guided",
        "protected_layers": protected_layers,
        "layer_configs": []
    }
    
    for i in range(num_layers):
        if i in protected_layers:
            layer_prune = 0
        else:
            layer_prune = prune_per_layer + (1 if i < remainder else 0)
        
        layer_keep = heads_per_layer - layer_prune
        config["layer_configs"].append({
            "layer": i,
            "heads_total": heads_per_layer,
            "heads_keep": layer_keep,
            "heads_prune": layer_prune,
            "prune_ratio": layer_prune / heads_per_layer
        })
    
    return config

# Generate d30 40% pruning config
print("D30 Model - 40% Pruning Configuration:")
d30_40_config = generate_pruning_config(30, 0.4, d30_combined_protect, "var-d30")
if d30_40_config:
    print(f"  Total heads: {30*30}")
    print(f"  Target keep: {int(30*30*0.6)}")
    print(f"  Protected layers: {d30_combined_protect}")
    print(f"  Non-protected layers: {[i for i in range(30) if i not in d30_combined_protect]}")
    
    # Show distribution
    keep_counts = [cfg['heads_keep'] for cfg in d30_40_config['layer_configs']]
    print(f"  Heads per layer after pruning: min={min(keep_counts)}, max={max(keep_counts)}, avg={np.mean(keep_counts):.1f}")
print()

# Generate d30 20% pruning config
print("D30 Model - 20% Pruning Configuration:")
d30_20_config = generate_pruning_config(30, 0.2, d30_combined_protect, "var-d30")
if d30_20_config:
    print(f"  Total heads: {30*30}")
    print(f"  Target keep: {int(30*30*0.8)}")
    print(f"  Protected layers: {d30_combined_protect}")
    
    keep_counts = [cfg['heads_keep'] for cfg in d30_20_config['layer_configs']]
    print(f"  Heads per layer after pruning: min={min(keep_counts)}, max={max(keep_counts)}, avg={np.mean(keep_counts):.1f}")
print()

# Save configs
with open('/home/project/real_prune/slimvar/d30_pruning_config_40percent.json', 'w') as f:
    json.dump(d30_40_config, f, indent=2)
with open('/home/project/real_prune/slimvar/d30_pruning_config_20percent.json', 'w') as f:
    json.dump(d30_20_config, f, indent=2)
print("Saved: d30_pruning_config_40percent.json, d30_pruning_config_20percent.json")
print()

print("\n### 8. KEY INSIGHTS ###\n")

print("1. Three-Stage Structure:")
print(f"   d16: early var={d16_early_avg:.2f}, middle var={d16_middle_avg:.2f}, late var={d16_late_avg:.2f}")
print(f"   d30: early var={d30_early_avg:.2f}, middle var={d30_middle_avg:.2f}, late var={d30_late_avg:.2f}")
print(f"   Conclusion: {'Both models show middle-layer variance peaks' if d16_middle_avg > d16_early_avg and d30_middle_avg > d30_early_avg else 'Pattern differs'}")
print()

print("2. Variance Scale Differences:")
d16_max_var = max(d16_variances)
d30_max_var = max(d30_variances)
print(f"   d16 max variance: {d16_max_var:.2f}")
print(f"   d30 max variance: {d30_max_var:.2f}")
print(f"   Ratio: {d30_max_var/d16_max_var:.2f}x")
print(f"   Conclusion: d30 has {'higher' if d30_max_var > d16_max_var else 'lower'} peak variance")
print()

print("3. High-Variance Layer Distribution:")
d16_hv_ratio = len(d16_high_var_layers) / 16
d30_hv_ratio = len(d30_high_var_layers) / 30
print(f"   d16: {len(d16_high_var_layers)}/16 = {d16_hv_ratio*100:.1f}%")
print(f"   d30: {len(d30_high_var_layers)}/30 = {d30_hv_ratio*100:.1f}%")
print(f"   Conclusion: {'Similar ratios' if abs(d16_hv_ratio - d30_hv_ratio) < 0.1 else 'Different distributions'}")
print()

print("4. Protection Strategy Validation:")
print(f"   d16 protected {len(d16_combined_protect)} layers avoided severe degradation in experiments")
print(f"   d30 should protect {len(d30_combined_protect)} layers based on same criteria")
print(f"   Recommendation: Use variance percentile + position filtering for universal protection")
print()

print("\n### 9. VISUALIZATION ###\n")
print("Generating comparison plots...")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Plot 1: Variance curves
ax = axes[0, 0]
ax.plot(range(16), d16_variances, 'o-', label='d16', linewidth=2, markersize=8)
ax.plot(range(30), d30_variances, 's-', label='d30', linewidth=2, markersize=6)
ax.axhline(d16_threshold_40, color='blue', linestyle='--', alpha=0.5, label='d16 40th percentile')
ax.axhline(d30_threshold_40, color='orange', linestyle='--', alpha=0.5, label='d30 40th percentile')
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Variance', fontsize=12)
ax.set_title('Variance Comparison: d16 vs d30', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mean curves
ax = axes[0, 1]
ax.plot(range(16), d16_means, 'o-', label='d16', linewidth=2, markersize=8)
ax.plot(range(30), d30_means, 's-', label='d30', linewidth=2, markersize=6)
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Mean Scale_mul', fontsize=12)
ax.set_title('Mean Comparison: d16 vs d30', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Normalized position variance (relative)
ax = axes[1, 0]
d16_rel_positions = np.linspace(0, 100, 16)
d30_rel_positions = np.linspace(0, 100, 30)
ax.plot(d16_rel_positions, d16_variances, 'o-', label='d16', linewidth=2, markersize=8)
ax.plot(d30_rel_positions, d30_variances, 's-', label='d30', linewidth=2, markersize=6)
ax.axvspan(40, 70, alpha=0.2, color='green', label='Protection zone (40-70%)')
ax.set_xlabel('Relative Layer Position (%)', fontsize=12)
ax.set_ylabel('Variance', fontsize=12)
ax.set_title('Variance vs Relative Position', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: High variance layer positions
ax = axes[1, 1]
d16_hv_rel = [i/16*100 for i in d16_high_var_layers]
d30_hv_rel = [i/30*100 for i in d30_high_var_layers]
ax.scatter(d16_hv_rel, [1]*len(d16_hv_rel), s=200, alpha=0.6, label='d16 high-var layers')
ax.scatter(d30_hv_rel, [2]*len(d30_hv_rel), s=200, alpha=0.6, label='d30 high-var layers')
ax.axvspan(40, 70, alpha=0.2, color='green')
ax.set_xlim(-5, 105)
ax.set_ylim(0.5, 2.5)
ax.set_xlabel('Relative Layer Position (%)', fontsize=12)
ax.set_yticks([1, 2])
ax.set_yticklabels(['d16', 'd30'])
ax.set_title('High-Variance Layer Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Plot 5: Section statistics
ax = axes[2, 0]
sections = ['Early\n(0-33%)', 'Middle\n(33-67%)', 'Late\n(67-100%)']
d16_section_vars = [d16_early_avg, d16_middle_avg, d16_late_avg]
d30_section_vars = [d30_early_avg, d30_middle_avg, d30_late_avg]
x = np.arange(len(sections))
width = 0.35
ax.bar(x - width/2, d16_section_vars, width, label='d16', alpha=0.8)
ax.bar(x + width/2, d30_section_vars, width, label='d30', alpha=0.8)
ax.set_ylabel('Average Variance', fontsize=12)
ax.set_title('Average Variance by Section', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sections)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Pruning impact comparison
ax = axes[2, 1]
d16_orig_means = d16_means
d16_pruned_means = [layer['mean'] for layer in d16_pruned['per_layer']]
ax.plot(range(16), d16_orig_means, 'o-', label='d16 original', linewidth=2)
ax.plot(range(16), d16_pruned_means, 's--', label='d16 after 40% prune', linewidth=2)
# Highlight protected vs non-protected
for i in d16_combined_protect:
    ax.axvspan(i-0.3, i+0.3, alpha=0.2, color='green')
ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Mean Scale_mul', fontsize=12)
ax.set_title('d16: Pruning Impact on Scale_mul', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/project/real_prune/slimvar/d16_vs_d30_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: d16_vs_d30_comparison.png")
print()

print("\n### 10. EXPERIMENTAL RECOMMENDATIONS ###\n")

print("For D30 Model Validation:")
print("  1. Baseline: Train d30 model without pruning")
print("  2. Experiment 1: 20% pruning with protection (use d30_pruning_config_20percent.json)")
print("  3. Experiment 2: 40% pruning with protection (use d30_pruning_config_40percent.json)")
print("  4. Ablation: 40% pruning WITHOUT protection (uniform pruning)")
print()
print("Success Metrics:")
print("  - FID score degradation < 10% for 20% pruning")
print("  - FID score degradation < 20% for 40% pruning")
print("  - Protected layers maintain higher scale_mul values")
print("  - Performance better than uniform pruning baseline")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
