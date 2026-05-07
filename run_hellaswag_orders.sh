#!/usr/bin/env bash

# Runs the HellaSwag eval in 3 different example orderings and saves per-order CSVs.
# Usage:
#   ./run_hellaswag_orders.sh cpu     # CPU only
#   ./run_hellaswag_orders.sh gpu     # GPU only
#   ./run_hellaswag_orders.sh         # both (default)

set -e

MODE=${1:-both}

# Generate the 3 shuffled bin files
python dev/data/hellaswag_shuffle.py

for ORDER in 0 1 2; do
    BIN="dev/data/hellaswag/hellaswag_sample_order${ORDER}.bin"

    if [[ "$MODE" == "cpu" || "$MODE" == "both" ]]; then
        echo "--- CPU order ${ORDER} ---"
        OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores ./train_gpt2 -h 1 -m 100 -x 0 -H "$BIN" -K "cpu_losses_order${ORDER}.csv" -Z 0 > /dev/null 2>&1 <&-
        echo "CPU order ${ORDER} done."
    fi

    if [[ "$MODE" == "gpu" || "$MODE" == "both" ]]; then
        echo "--- GPU order ${ORDER} ---"
        ./train_gpt2cu -h 1 -m 100 -x 1 -Z 0 -H "$BIN" -K "gpu_losses_order${ORDER}.csv" > /dev/null 2>&1 <&-
        echo "GPU order ${ORDER} done."
    fi
done

echo "Done. CSV files written for each order."
