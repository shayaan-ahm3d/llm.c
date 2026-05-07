#!/usr/bin/env python3

import csv

Q1, Q2, Q3 = 31, 56, 67

def classify_stratum(ctx_len):
    if ctx_len <= Q1:
        return 'short'
    elif ctx_len <= Q2:
        return 'medium_low'
    elif ctx_len <= Q3:
        return 'medium_high'
    else:
        return 'long'

def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'exampleIndex': int(row['exampleIndex']),
                'contextLength': int(row['contextLength']),
                'loss': float(row['loss']),
            })
    return rows

def main():
    cpu_rows = read_csv('cpu_eval_losses.csv')
    gpu_rows = read_csv('gpu_eval_losses.csv')

    assert len(cpu_rows) == len(gpu_rows), \
        f"Row count mismatch: CPU={len(cpu_rows)}, GPU={len(gpu_rows)}"

    print(f"{'seq':>4} {'ctx':>5} {'cpu_loss':>10} {'gpu_loss':>10} {'diff':>10}")
    print('-' * 44)

    strata_data = {}
    cpu_total = 0.0
    gpu_total = 0.0

    for c, g in zip(cpu_rows, gpu_rows):
        assert c['exampleIndex'] == g['exampleIndex'], \
            f"Index mismatch at position: CPU={c['exampleIndex']}, GPU={g['exampleIndex']}"
        diff = g['loss'] - c['loss']
        ctx = c['contextLength']
        stratum = classify_stratum(ctx)

        print(f"{c['exampleIndex']:>4} {ctx:>5} {c['loss']:>10.4f} {g['loss']:>10.4f} {diff:>+10.4f}")

        cpu_total += c['loss']
        gpu_total += g['loss']

        if stratum not in strata_data:
            strata_data[stratum] = {'cpu': [], 'gpu': [], 'abs_diffs': []}
        strata_data[stratum]['cpu'].append(c['loss'])
        strata_data[stratum]['gpu'].append(g['loss'])
        strata_data[stratum]['abs_diffs'].append(abs(diff))

    print('-' * 44)
    print(f"\nCumulative loss:  CPU={cpu_total:.4f}  GPU={gpu_total:.4f}  diff={gpu_total-cpu_total:+.4f}")

    print(f"\nPer-stratum summary (ctx_len boundaries: Q1={Q1}, Q2={Q2}, Q3={Q3}):")
    print(f"{'stratum':>12} {'n':>4} {'mean_cpu':>10} {'mean_gpu':>10} {'mean|diff|':>12}")
    print('-' * 52)
    for stratum in ['short', 'medium_low', 'medium_high', 'long']:
        if stratum not in strata_data:
            continue
        d = strata_data[stratum]
        n = len(d['cpu'])
        mean_cpu = sum(d['cpu']) / n
        mean_gpu = sum(d['gpu']) / n
        mean_diff = sum(d['abs_diffs']) / n
        print(f"{stratum:>12} {n:>4} {mean_cpu:>10.4f} {mean_gpu:>10.4f} {mean_diff:>12.4f}")

if __name__ == '__main__':
    main()
