#!/usr/bin/env python3
"""
Script to create publication-quality dual-axis plot for VAR-D16 sparsity analysis
showing latency and FID vs sparsity rate
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_ipad_data(file_path):
    """Parse the ipad.json file and extract latency and FID data"""

    with open(file_path, 'r') as f:
        content = f.read()

    # Initialize data structure
    data = {
        'sparsity_rates': [],
        'model_names': [],
        'rtx5090_latencies_s': [],
        'ipad_latencies_s': [],
        'fid_scores': []
    }

    lines = content.strip().split('\n')

    # Extract data from different sections
    rtx5090_section = False
    ipad_section = False
    fid_section = False

    for i, line in enumerate(lines):
        line = line.strip()

        if 'rtx5090' in line.lower():
            rtx5090_section = True
            ipad_section = False
            fid_section = False
            continue
        elif 'ipad pro' in line.lower():
            rtx5090_section = False
            ipad_section = True
            fid_section = False
            continue
        elif 'fid' in line.lower():
            rtx5090_section = False
            ipad_section = False
            fid_section = True
            continue
        elif not line:
            continue

        if rtx5090_section:
            # RTX5090 data format: "D16_0.4:" followed by value on next line
            if ':' in line and not line.replace(':', '').replace('_', '').replace('.', '').isdigit():
                model = line.replace(':', '').strip()
                # Look for the value on the next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    try:
                        latency = float(next_line)

                        # Extract sparsity rate from model name
                        if model == 'D16':
                            sparsity = 0.0
                            model_name = 'D16'
                        elif 'D16_0.2' in model:
                            sparsity = 0.2
                            model_name = 'D16_0.2'
                        elif 'D16_0.4' in model:
                            sparsity = 0.4
                            model_name = 'D16_0.4'
                        else:
                            continue

                        # Find existing entry or create new one
                        if model_name not in data['model_names']:
                            data['sparsity_rates'].append(sparsity)
                            data['model_names'].append(model_name)
                            data['rtx5090_latencies_s'].append(latency)
                            data['ipad_latencies_s'].append(None)
                            data['fid_scores'].append(None)
                        else:
                            idx = data['model_names'].index(model_name)
                            data['rtx5090_latencies_s'][idx] = latency

                    except (ValueError, IndexError):
                        continue

        elif ipad_section and (':' in line or '：' in line):
            # Parse iPad latency data - handle both English and Chinese colons
            if '：' in line:
                parts = line.split('：')
            else:
                parts = line.split(':')

            if len(parts) == 2:
                model = parts[0].strip()
                latency_str = parts[1].strip().replace('ms', '')
                try:
                    latency_ms = float(latency_str)
                    latency_s = latency_ms / 1000.0  # Convert to seconds

                    # Extract sparsity rate from model name
                    if model == 'D16':
                        sparsity = 0.0
                        model_name = 'D16'
                    elif 'D16_0.2' in model:
                        sparsity = 0.2
                        model_name = 'D16_0.2'
                    elif 'D16_0.4' in model:
                        sparsity = 0.4
                        model_name = 'D16_0.4'
                    else:
                        continue

                    # Find existing entry or create new one
                    if model_name not in data['model_names']:
                        data['sparsity_rates'].append(sparsity)
                        data['model_names'].append(model_name)
                        data['rtx5090_latencies_s'].append(None)
                        data['ipad_latencies_s'].append(latency_s)
                        data['fid_scores'].append(None)
                    else:
                        idx = data['model_names'].index(model_name)
                        data['ipad_latencies_s'][idx] = latency_s

                except ValueError:
                    continue

        elif fid_section and ':' in line:
            # Parse FID data
            parts = line.split(':')
            if len(parts) == 2:
                model = parts[0].strip()
                try:
                    fid = float(parts[1].strip())

                    # Extract model name
                    if model == 'D16':
                        model_name = 'D16'
                    elif 'D16_0.2' in model:
                        model_name = 'D16_0.2'
                    elif 'D16_0.4' in model:
                        model_name = 'D16_0.4'
                    else:
                        continue

                    # Update existing entry
                    if model_name in data['model_names']:
                        idx = data['model_names'].index(model_name)
                        data['fid_scores'][idx] = fid

                except ValueError:
                    continue

    # Sort by sparsity rate
    sorted_data = sorted(zip(data['sparsity_rates'], data['model_names'],
                           data['rtx5090_latencies_s'], data['ipad_latencies_s'], data['fid_scores']))

    data['sparsity_rates'] = [x[0] for x in sorted_data]
    data['model_names'] = [x[1] for x in sorted_data]
    data['rtx5090_latencies_s'] = [x[2] for x in sorted_data]
    data['ipad_latencies_s'] = [x[3] for x in sorted_data]
    data['fid_scores'] = [x[4] for x in sorted_data]

    return data

def create_dual_axis_plot(data, save_path):
    """Create publication-quality dual-axis plot with RTX5090, iPad Pro, and FID"""

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Convert sparsity to percentage for display
    sparsity_percent = [s * 100 for s in data['sparsity_rates']]

    # Plot RTX5090 latency on left y-axis
    color1 = '#1f77b4'
    ax1.set_xlabel('Sparsity (%)', fontsize=25)
    ax1.set_ylabel('Latency (s)', fontsize=25, color='black')

    line1 = ax1.plot(sparsity_percent, data['rtx5090_latencies_s'], 'o-', color=color1,
                     linewidth=2, markersize=8, markerfacecolor='white',
                     markeredgewidth=2, label='RTX5090')

    # Plot iPad Pro latency on same left y-axis
    color2 = '#ff7f0e'
    line2 = ax1.plot(sparsity_percent, data['ipad_latencies_s'], 's-', color=color2,
                     linewidth=2, markersize=8, markerfacecolor='white',
                     markeredgewidth=2, label='iPad Pro')

    ax1.tick_params(axis='y', labelcolor='black', labelsize=21)
    ax1.tick_params(axis='x', labelsize=21)

    # Create second y-axis for FID
    ax2 = ax1.twinx()
    color3 = '#2ca02c'
    ax2.set_ylabel('FID', fontsize=25, color=color3)
    line3 = ax2.plot(sparsity_percent, data['fid_scores'], '^-', color=color3,
                     linewidth=2, markersize=8, markerfacecolor='white',
                     markeredgewidth=2, label='FID')
    ax2.tick_params(axis='y', labelcolor=color3, labelsize=21)

    # Grid and styling
    ax1.grid(True, alpha=0.3)

    # Set x-axis ticks to show sparsity percentages
    ax1.set_xticks(sparsity_percent)
    ax1.set_xticklabels([f'{int(sp)}%' for sp in sparsity_percent])

    # Set y-axis limits to provide space for legend
    # Expand latency y-axis range with extra space for legend
    rtx_min = min([x for x in data['rtx5090_latencies_s'] if x is not None])
    rtx_max = max([x for x in data['rtx5090_latencies_s'] if x is not None])
    ipad_min = min([x for x in data['ipad_latencies_s'] if x is not None])
    ipad_max = max([x for x in data['ipad_latencies_s'] if x is not None])

    y1_min = min(rtx_min, ipad_min) * 0.8
    y1_max = max(rtx_max, ipad_max) * 1.4  # Extra space for legend
    ax1.set_ylim(y1_min, y1_max)

    # Set FID y-axis range from 1 to 5
    ax2.set_ylim(1, 5)

    # Position legend in upper left corner
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              fontsize=18, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save with high DPI for publication
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    # Also display the plot
    plt.show()

def main():
    """Main function"""
    # File paths
    data_file = '/home/project/real_prune/VAR_FIDtest/image/ipad.json'
    output_file = '/home/project/real_prune/VAR_FIDtest/image/var_sparsity_latency_fid.png'

    # Parse data
    print("Parsing ipad data...")
    data = parse_ipad_data(data_file)

    # Print parsed data for verification
    print("\nParsed data:")
    for i, (sparsity, model, rtx_latency, ipad_latency, fid) in enumerate(zip(
        data['sparsity_rates'], data['model_names'],
        data['rtx5090_latencies_s'], data['ipad_latencies_s'], data['fid_scores'])):
        rtx_str = f"{rtx_latency:.3f}s" if rtx_latency is not None else "N/A"
        ipad_str = f"{ipad_latency:.3f}s" if ipad_latency is not None else "N/A"
        print(f"{model} (Sparsity {sparsity*100:.0f}%): RTX5090: {rtx_str}, iPad Pro: {ipad_str}, FID: {fid}")

    # Create plot
    print("\nCreating publication-quality dual-axis plot...")
    create_dual_axis_plot(data, output_file)

if __name__ == "__main__":
    main()