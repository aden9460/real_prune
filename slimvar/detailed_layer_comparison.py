import json
import numpy as np
import pandas as pd

# Load data
d16_orig = json.load(open('/home/project/real_prune/slimvar/scale_mul_analysis_d16/scale_mul_analysis.json'))
d30_orig = json.load(open('/home/project/real_prune/slimvar/scale_mul_analysis_d30/scale_mul_analysis.json'))

print("="*100)
print("DETAILED LAYER-BY-LAYER COMPARISON: D16 vs D30")
print("="*100)
print()

# Create detailed comparison table
print("### D16 LAYER DETAILS ###\n")
print(f"{'Layer':<6} {'Mean':<8} {'Variance':<10} {'Rel.Pos%':<10} {'Section':<12} {'High-Var?':<10}")
print("-" * 70)

d16_threshold = np.percentile([l['variance'] for l in d16_orig['per_layer']], 40)
for i, layer in enumerate(d16_orig['per_layer']):
    rel_pos = (i / 16) * 100
    if i < 5:
        section = "Early"
    elif i < 10:
        section = "Middle"
    else:
        section = "Late"
    high_var = "YES" if layer['variance'] > d16_threshold else "no"
    print(f"{i:<6} {layer['mean']:<8.2f} {layer['variance']:<10.2f} {rel_pos:<10.1f} {section:<12} {high_var:<10}")

print("\n" + "="*100)
print("### D30 LAYER DETAILS ###\n")
print(f"{'Layer':<6} {'Mean':<8} {'Variance':<10} {'Rel.Pos%':<10} {'Section':<12} {'High-Var?':<10}")
print("-" * 70)

d30_threshold = np.percentile([l['variance'] for l in d30_orig['per_layer']], 40)
for i, layer in enumerate(d30_orig['per_layer']):
    rel_pos = (i / 30) * 100
    if i < 10:
        section = "Early"
    elif i < 20:
        section = "Middle"
    else:
        section = "Late"
    high_var = "YES" if layer['variance'] > d30_threshold else "no"
    print(f"{i:<6} {layer['mean']:<8.2f} {layer['variance']:<10.2f} {rel_pos:<10.1f} {section:<12} {high_var:<10}")

print("\n" + "="*100)
print("### MATCHED RELATIVE POSITION COMPARISON ###")
print("="*100)
print()
print("Comparing layers at equivalent relative positions:\n")

# Select representative relative positions
positions = [0, 0.25, 0.5, 0.75, 1.0]
print(f"{'RelPos':<10} {'d16 Layer':<12} {'d16 Var':<10} {'d30 Layer':<12} {'d30 Var':<10} {'Var Ratio':<10}")
print("-" * 75)

for pos in positions:
    d16_idx = int(pos * 15)  # 0-15 for d16
    d30_idx = int(pos * 29)  # 0-29 for d30
    
    d16_var = d16_orig['per_layer'][d16_idx]['variance']
    d30_var = d30_orig['per_layer'][d30_idx]['variance']
    ratio = d30_var / d16_var if d16_var > 0 else 0
    
    print(f"{pos:<10.2f} {d16_idx:<12} {d16_var:<10.2f} {d30_idx:<12} {d30_var:<10.2f} {ratio:<10.2f}x")

print("\n" + "="*100)
print("### VARIANCE DISTRIBUTION COMPARISON ###")
print("="*100)
print()

d16_vars = [l['variance'] for l in d16_orig['per_layer']]
d30_vars = [l['variance'] for l in d30_orig['per_layer']]

percentiles = [0, 10, 25, 40, 50, 60, 75, 90, 100]
print(f"{'Percentile':<12} {'d16':<12} {'d30':<12} {'Ratio (d30/d16)':<15}")
print("-" * 55)

for p in percentiles:
    d16_val = np.percentile(d16_vars, p)
    d30_val = np.percentile(d30_vars, p)
    ratio = d30_val / d16_val if d16_val > 0 else 0
    print(f"{p:<12} {d16_val:<12.2f} {d30_val:<12.2f} {ratio:<15.2f}x")

print("\n" + "="*100)
print("### VARIANCE CONCENTRATION ANALYSIS ###")
print("="*100)
print()

# Calculate what % of total variance is in each section
d16_early_vars = [d16_orig['per_layer'][i]['variance'] for i in range(5)]
d16_middle_vars = [d16_orig['per_layer'][i]['variance'] for i in range(5, 10)]
d16_late_vars = [d16_orig['per_layer'][i]['variance'] for i in range(10, 16)]
d16_total_var = sum(d16_vars)

d30_early_vars = [d30_orig['per_layer'][i]['variance'] for i in range(10)]
d30_middle_vars = [d30_orig['per_layer'][i]['variance'] for i in range(10, 20)]
d30_late_vars = [d30_orig['per_layer'][i]['variance'] for i in range(20, 30)]
d30_total_var = sum(d30_vars)

print(f"{'Section':<12} {'d16 Sum':<12} {'d16 %':<12} {'d30 Sum':<12} {'d30 %':<12}")
print("-" * 65)
print(f"{'Early':<12} {sum(d16_early_vars):<12.2f} {sum(d16_early_vars)/d16_total_var*100:<12.1f} {sum(d30_early_vars):<12.2f} {sum(d30_early_vars)/d30_total_var*100:<12.1f}")
print(f"{'Middle':<12} {sum(d16_middle_vars):<12.2f} {sum(d16_middle_vars)/d16_total_var*100:<12.1f} {sum(d30_middle_vars):<12.2f} {sum(d30_middle_vars)/d30_total_var*100:<12.1f}")
print(f"{'Late':<12} {sum(d16_late_vars):<12.2f} {sum(d16_late_vars)/d16_total_var*100:<12.1f} {sum(d30_late_vars):<12.2f} {sum(d30_late_vars)/d30_total_var*100:<12.1f}")
print(f"{'Total':<12} {d16_total_var:<12.2f} {'100.0':<12} {d30_total_var:<12.2f} {'100.0':<12}")

print("\n**Insight:** Middle section contains", end=" ")
print(f"{sum(d16_middle_vars)/d16_total_var*100:.1f}% (d16) and {sum(d30_middle_vars)/d30_total_var*100:.1f}% (d30) of total variance")
print("despite being only 31% (d16) and 33% (d30) of layers!")

print("\n" + "="*100)
print("### PRUNING IMPACT PREDICTION ###")
print("="*100)
print()

print("If we uniformly prune 40% across all layers:")
print()
print(f"{'Metric':<30} {'d16 Loss':<15} {'d30 Loss (predicted)':<20}")
print("-" * 70)

# Estimated loss proportional to variance destroyed
d16_var_loss = d16_total_var * 0.4
d30_var_loss = d30_total_var * 0.4
print(f"{'Variance destroyed':<30} {d16_var_loss:<15.2f} {d30_var_loss:<15.2f}")
print(f"{'Relative impact':<30} {'1.00x':<15} {d30_var_loss/d16_var_loss:<15.2f}x")
print()

print("If we protect high-variance layers and prune 60% from others:")
print()

# d16: protect 6 layers (6*avg_protected), prune 60% from 10 layers
d16_protected_avg_var = np.mean([d16_orig['per_layer'][i]['variance'] for i in [6,7,8,9,10,11]])
d16_nonprotected_var = sum([d16_orig['per_layer'][i]['variance'] for i in range(16) if i not in [6,7,8,9,10,11]])
d16_smart_loss = d16_nonprotected_var * 0.6

# d30: protect 10 layers, prune 60% from 20 layers
d30_protected_avg_var = np.mean([d30_orig['per_layer'][i]['variance'] for i in range(12, 22)])
d30_nonprotected_var = sum([d30_orig['per_layer'][i]['variance'] for i in range(30) if i not in range(12, 22)])
d30_smart_loss = d30_nonprotected_var * 0.6

print(f"{'Metric':<30} {'d16':<15} {'d30':<15}")
print("-" * 65)
print(f"{'Protected layers avg var':<30} {d16_protected_avg_var:<15.2f} {d30_protected_avg_var:<15.2f}")
print(f"{'Non-protected total var':<30} {d16_nonprotected_var:<15.2f} {d30_nonprotected_var:<15.2f}")
print(f"{'Variance destroyed (60%)':<30} {d16_smart_loss:<15.2f} {d30_smart_loss:<15.2f}")
print(f"{'% of total var destroyed':<30} {d16_smart_loss/d16_total_var*100:<15.1f} {d30_smart_loss/d30_total_var*100:<15.1f}")
print()
print(f"**Key Insight:** Smart pruning destroys only {d16_smart_loss/d16_total_var*100:.1f}% (d16) and {d30_smart_loss/d30_total_var*100:.1f}% (d30)")
print(f"of total variance, vs 40% with uniform pruning!")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
