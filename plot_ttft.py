#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Read the data
cpu_df = pd.read_csv('cpu_times.csv')
gpu_df = pd.read_csv('gpu_times.csv')

# Exclude first sample of each run (runs are 100 samples each, so exclude indices 0, 100, 200)
cpu_df = cpu_df[~(cpu_df.index % 100 == 0)].reset_index(drop=True)
gpu_df = gpu_df[~(gpu_df.index % 100 == 0)].reset_index(drop=True)

# Group by context length and calculate mean time to first token
cpu_grouped = cpu_df.groupby('contextLength')['timeToFirstTokenMillis'].mean().reset_index()
gpu_grouped = gpu_df.groupby('contextLength')['timeToFirstTokenMillis'].mean().reset_index()

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(cpu_grouped['contextLength'], cpu_grouped['timeToFirstTokenMillis'],
         marker='o', label='CPU', linewidth=2, markersize=6, color='red')
plt.plot(gpu_grouped['contextLength'], gpu_grouped['timeToFirstTokenMillis'],
         marker='s', label='GPU', linewidth=2, markersize=6, color='green')

plt.xlabel('Context Length', fontsize=12)
plt.ylabel('Time to First Token (ms)', fontsize=12)
plt.title('Time to First Token vs Context Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ttft_comparison.png', dpi=300)
print("Graph saved as ttft_comparison.png")
plt.show()
