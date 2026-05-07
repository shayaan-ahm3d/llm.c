#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Read the data
cpu_df = pd.read_csv('cpu_times.csv')
gpu_df = pd.read_csv('gpu_times.csv')

# Calculate throughput (tokens per second)
cpu_df['throughput'] = cpu_df['numTokensGenerated'] / cpu_df['totalTimeSeconds']
gpu_df['throughput'] = gpu_df['numTokensGenerated'] / gpu_df['totalTimeSeconds']

# Group by context length and calculate mean throughput
cpu_grouped = cpu_df.groupby('contextLength')['throughput'].mean().reset_index()
gpu_grouped = gpu_df.groupby('contextLength')['throughput'].mean().reset_index()

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(cpu_grouped['contextLength'], cpu_grouped['throughput'],
         marker='o', label='CPU', linewidth=2, markersize=6, color='red')
plt.plot(gpu_grouped['contextLength'], gpu_grouped['throughput'],
         marker='s', label='GPU', linewidth=2, markersize=6, color='green')

plt.xlabel('Context Length', fontsize=12)
plt.ylabel('Token Throughput (tokens/s)', fontsize=12)
plt.title('Token Throughput vs Context Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('throughput_comparison.png', dpi=300)
print("Graph saved as throughput_comparison.png")
plt.show()
