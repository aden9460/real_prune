#!/usr/bin/env python3
"""
Script to create publication-quality plot for VAR-D16 batch size latency analysis
with different sparsity rates
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_batch_data(file_path):
    """Parse the batch.json file and extract timing data for different sparsity rates"""

    with open(file_path, 'r') as f:
        content = f.read()

    # Initialize data structure
    data = {
        'VAR-d16': {'batch_sizes': [], 'latencies': []},
        'VAR-d16 20%sparsity': {'batch_sizes': [], 'latencies': []},
        'VAR-d16 40%sparsity': {'batch_sizes': [], 'latencies': []},
        'VAR-d16 60%sparsity': {'batch_sizes': [], 'latencies': []}
    }

    lines = content.strip().split('\n')
    current_batch_size = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if this is a batch size line
        # Either "number:" or just "number"
        if ':' not in line and line.isdigit():
            # Just a number like "8"
            current_batch_size = int(line)
            i += 1
            continue
        elif line.endswith(':') and line[:-1].replace(' ', '').isdigit():
            # Number followed by colon like "1:"
            current_batch_size = int(line[:-1])
            i += 1
            continue
        elif ':' in line and current_batch_size is not None:
            # This is a sparsity:latency line
            parts = line.split(':')
            if len(parts) == 2 and parts[1].strip():  # Make sure latency part is not empty
                try:
                    sparsity = float(parts[0])
                    latency = float(parts[1])

                    # Map sparsity to model name
                    if sparsity == 0:
                        model_name = 'VAR-d16'
                    elif sparsity == 0.2:
                        model_name = 'VAR-d16 20%sparsity'
                    elif sparsity == 0.4:
                        model_name = 'VAR-d16 40%sparsity'
                    elif sparsity == 0.6:
                        model_name = 'VAR-d16 60%sparsity'
                    else:
                        i += 1
                        continue

                    data[model_name]['batch_sizes'].append(current_batch_size)
                    data[model_name]['latencies'].append(latency)
                except ValueError:
                    # Skip lines that can't be parsed
                    pass

        i += 1

    return data

def create_batch_plot(data, save_path):
    """Create publication-quality plot for batch size vs latency"""

    plt.figure(figsize=(10, 6))

    # Color palette for different sparsity rates
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    model_names = ['VAR-d16', 'VAR-d16 20%sparsity', 'VAR-d16 40%sparsity', 'VAR-d16 60%sparsity']

    for i, model_name in enumerate(model_names):
        model_data = data[model_name]

        # Sort by batch size
        sorted_data = sorted(zip(model_data['batch_sizes'], model_data['latencies']))
        batch_sizes, latencies = zip(*sorted_data)

        plt.plot(batch_sizes, latencies,
                marker=markers[i], linewidth=2, markersize=6,
                color=colors[i], label=model_name, linestyle=linestyles[i],
                markerfacecolor='white', markeredgewidth=1.5)

    # Styling for publication
    plt.xlabel('Batch Size', fontsize=25)
    plt.ylabel('Latency (s)', fontsize=25)

    # Grid and layout
    plt.grid(True, alpha=0.3)
    # Position legend in upper left corner
    plt.legend(loc='upper left', fontsize=18, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Set x-axis to powers of 2 and use log scale
    batch_sizes_all = []
    for model_data in data.values():
        batch_sizes_all.extend(model_data['batch_sizes'])
    unique_batch_sizes = sorted(list(set(batch_sizes_all)))

    plt.xscale('log', base=2)
    plt.xticks(unique_batch_sizes, [f'$2^{{{int(np.log2(bs))}}}$' for bs in unique_batch_sizes], fontsize=21)
    plt.yticks(fontsize=21)

    # Set axis limits for better visualization with space for legend
    plt.xlim(0.8, max(unique_batch_sizes) * 1.2)

    y_min = min(min(model_data['latencies']) for model_data in data.values()) * 0.9
    y_max = max(max(model_data['latencies']) for model_data in data.values()) * 1.3  # Extra space for legend
    plt.ylim(y_min, y_max)

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
    data_file = '/home/project/real_prune/VAR_FIDtest/image/batch.json'
    output_file = '/home/project/real_prune/VAR_FIDtest/image/var_batch_latency.png'

    # Parse data
    print("Parsing batch data...")
    data = parse_batch_data(data_file)

    # Print parsed data for verification
    print("\nParsed data:")
    for model, model_data in data.items():
        print(f"{model}:")
        sorted_data = sorted(zip(model_data['batch_sizes'], model_data['latencies']))
        for batch_size, latency in sorted_data:
            print(f"  Batch {batch_size}: {latency:.2f} s")
        print()

    # Create plot
    print("Creating publication-quality plot...")
    create_batch_plot(data, output_file)

if __name__ == "__main__":
    main()