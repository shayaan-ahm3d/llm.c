#!/usr/bin/env python3
"""
Averages HellaSwag per-example losses across 3 ordering runs.
Reads cpu_losses_order{0,1,2}.csv and/or gpu_losses_order{0,1,2}.csv.
"""

import pandas as pd
import numpy as np
import os

META = 'hellaswag_sample_metadata.csv'

def average_runs(prefix):
    files = [f'{prefix}_losses_order{i}.csv' for i in range(3)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print(f"Skipping {prefix.upper()}: missing {missing}")
        return

    dfs = [pd.read_csv(f).set_index('exampleIndex') for f in files]
    combined = pd.concat([df['loss'].rename(f'run{i}') for i, df in enumerate(dfs)], axis=1)
    combined['mean_loss'] = combined.mean(axis=1)
    combined['std_loss'] = combined.std(axis=1)

    cumulative_losses = [df['loss'].sum() for df in dfs]
    mean_cumulative = np.mean(cumulative_losses)
    std_cumulative = np.std(cumulative_losses)

    print(f"\n=== {prefix.upper()} ===")
    print(f"Cumulative loss per run: {[f'{v:.4f}' for v in cumulative_losses]}")
    print(f"Mean cumulative loss:    {mean_cumulative:.4f} ± {std_cumulative:.4f}")

    if os.path.exists(META):
        meta = pd.read_csv(META).set_index('seq_index')
        meta.index = dfs[0].index  # align by position (seq_index == exampleIndex after shuffle)
        combined['stratum'] = meta['stratum'].values
        print("\nPer-stratum mean loss:")
        for stratum, group in combined.groupby('stratum'):
            print(f"  {stratum:14s}: {group['mean_loss'].mean():.4f} ± {group['std_loss'].mean():.4f}")

if __name__ == '__main__':
    average_runs('cpu')
    average_runs('gpu')
