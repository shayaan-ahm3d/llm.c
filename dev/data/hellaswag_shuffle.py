#!/usr/bin/env python3
"""
Generates 3 shuffled orderings of hellaswag_sample.bin for multi-order evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from hellaswag_stratify import scan_hellaswag_bin, write_hellaswag_bin
import random

SRC = 'dev/data/hellaswag/hellaswag_sample.bin'
SEEDS = [0, 1, 2]

def main():
    examples = scan_hellaswag_bin(SRC)
    for seed in SEEDS:
        shuffled = examples[:]
        random.Random(seed).shuffle(shuffled)
        dst = f'dev/data/hellaswag/hellaswag_sample_order{seed}.bin'
        write_hellaswag_bin(dst, shuffled)

if __name__ == '__main__':
    main()
