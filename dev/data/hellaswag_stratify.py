#!/usr/bin/env python3

import struct
import numpy as np
import random

HEADER_SIZE = 256
HEADER_BYTES = HEADER_SIZE * 4
START_EXAMPLE = 65535

def scan_hellaswag_bin(filename):
    examples = []
    with open(filename, 'rb') as f:
        header = struct.unpack(f'<{HEADER_SIZE}i', f.read(HEADER_BYTES))
        assert header[0] == 20240522, f"Bad magic: {header[0]}"
        assert header[1] == 1, f"Bad version: {header[1]}"
        num_examples = header[2]
        print(f"Scanning {num_examples} examples from {filename}...")

        for i in range(num_examples):
            ex_header = struct.unpack('<3H', f.read(6))
            assert ex_header[0] == START_EXAMPLE, f"Bad START_EXAMPLE at example {i}"
            example_bytes = ex_header[1]
            assert ex_header[2] == i, f"Bad EXAMPLE_INDEX at {i}: got {ex_header[2]}"

            rest = f.read(example_bytes - 6)
            label, num_comp, context_length = struct.unpack('<3H', rest[:6])

            examples.append({
                'original_index': i,
                'context_length': context_length,
                'label': label,
                'raw_rest': rest,
                'stratum': None,
            })

    return examples

def write_hellaswag_bin(filename, examples):
    longest = max(6 + len(ex['raw_rest']) for ex in examples)

    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = 20240522
    header[1] = 1
    header[2] = len(examples)
    header[3] = longest

    with open(filename, 'wb') as f:
        f.write(header.tobytes())
        for new_idx, ex in enumerate(examples):
            total_bytes = 6 + len(ex['raw_rest'])
            ex_header = struct.pack('<3H', START_EXAMPLE, total_bytes, new_idx)
            f.write(ex_header)
            f.write(ex['raw_rest'])

    print(f"Wrote {len(examples)} examples to {filename}")

def main():
    src = 'dev/data/hellaswag/hellaswag_val.bin'
    dst = 'dev/data/hellaswag/hellaswag_sample.bin'
    meta_path = 'hellaswag_sample_metadata.csv'
    per_stratum = 25

    examples = scan_hellaswag_bin(src)

    ctx_lengths = np.array([ex['context_length'] for ex in examples])
    q1 = int(np.percentile(ctx_lengths, 25))
    q2 = int(np.percentile(ctx_lengths, 50))
    q3 = int(np.percentile(ctx_lengths, 75))

    print(f"\nContext length distribution:")
    print(f"  min={ctx_lengths.min()}, Q1={q1}, median={q2}, Q3={q3}, max={ctx_lengths.max()}")
    print(f"\nStrata:")
    print(f"  short:       context_length <= {q1}")
    print(f"  medium_low:  {q1} < context_length <= {q2}")
    print(f"  medium_high: {q2} < context_length <= {q3}")
    print(f"  long:        context_length > {q3}")

    strata = {
        'short':       [ex for ex in examples if ex['context_length'] <= q1],
        'medium_low':  [ex for ex in examples if q1 < ex['context_length'] <= q2],
        'medium_high': [ex for ex in examples if q2 < ex['context_length'] <= q3],
        'long':        [ex for ex in examples if ex['context_length'] > q3],
    }

    for name, stratum in strata.items():
        print(f"  {name}: {len(stratum)} examples available")

    rng = random.Random(42)
    sampled = []
    for name, stratum in strata.items():
        chosen = rng.sample(stratum, per_stratum)
        for ex in chosen:
            ex['stratum'] = name
        sampled.extend(chosen)

    sampled.sort(key=lambda x: x['original_index'])
    print(f"\nSampled {len(sampled)} examples total ({per_stratum} per stratum)")

    write_hellaswag_bin(dst, sampled)

    with open(meta_path, 'w') as f:
        f.write('seq_index,original_index,context_length,stratum\n')
        for seq_idx, ex in enumerate(sampled):
            f.write(f"{seq_idx},{ex['original_index']},{ex['context_length']},{ex['stratum']}\n")
    print(f"Wrote metadata to {meta_path}")

if __name__ == '__main__':
    main()
