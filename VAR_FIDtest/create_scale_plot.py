#!/usr/bin/env python3
"""
Script to create publication-quality plot for VAR model scale latency analysis
"""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_scale_data(file_path):
    """Parse the scale.json file and extract timing data for each model"""

    with open(file_path, 'r') as f:
        content = f.read()

    models = {}

    # Split content by model sections
    sections = content.strip().split('\n\n')

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')
        if not lines:
            continue

        # Extract model name from first line
        model_line = lines[0]
        if ':' in model_line:
            model_name = model_line.replace(':', '').strip()

            scale_times = []

            # Extract scale timing data
            for line in lines[1:]:
                if 'scale cost' in line and 'total' not in line:
                    # Extract scale number and time
                    match = re.search(r'(\d+)scale cost (\d+\.\d+)', line)
                    if match:
                        scale_num = int(match.group(1))
                        time_sec = float(match.group(2))
                        scale_times.append((scale_num, time_sec))

            # Sort by scale number and extract times
            scale_times.sort(key=lambda x: x[0])
            times = [time * 1000 for _, time in scale_times]  # Convert to milliseconds
            scales = [scale for scale, _ in scale_times]

            models[model_name] = {'scales': scales, 'times': times}

    return models

def create_scale_plot(models, save_path):
    """Create publication-quality plot"""

    plt.figure(figsize=(10, 6))

    # Color palette for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    model_names = sorted(models.keys())

    for i, model_name in enumerate(model_names):
        data = models[model_name]
        plt.plot(data['scales'], data['times'],
                marker=markers[i], linewidth=2, markersize=6,
                color=colors[i], label=model_name,
                markerfacecolor='white', markeredgewidth=1.5)

    # Styling for publication
    plt.xlabel('Predicted Scale', fontsize=25)
    plt.ylabel('Latency (ms)', fontsize=25)

    # Grid and layout
    plt.grid(True, alpha=0.3)
    # Position legend in upper left corner
    plt.legend(loc='upper left', fontsize=18, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Set y-axis limits to provide more space for legend
    y_min = min(min(data['times']) for data in models.values()) - 1
    y_max = 23  # Fixed upper limit to provide ample space for legend
    plt.ylim(y_min, y_max)

    # Set x-axis ticks
    plt.xticks(range(10), fontsize=21)
    plt.yticks(fontsize=21)

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
    data_file = '/home/project/real_prune/VAR_FIDtest/image/scale.json'
    output_file = '/home/project/real_prune/VAR_FIDtest/image/var_scale_latency.png'

    # Parse data
    print("Parsing scale data...")
    models = parse_scale_data(data_file)

    # Print parsed data for verification
    print("\nParsed data:")
    for model, data in models.items():
        print(f"{model}:")
        for scale, time in zip(data['scales'], data['times']):
            print(f"  Scale {scale}: {time:.2f} ms")
        print()

    # Create plot
    print("Creating publication-quality plot...")
    create_scale_plot(models, output_file)

if __name__ == "__main__":
    main()