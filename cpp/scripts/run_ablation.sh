#!/usr/bin/env bash
# Quirk ablation: run the C++ MSCKF with each Python-quirk fix on its own, all
# fixes together, and none (= Python behaviour), in parallel.
#
#   scripts/run_ablation.sh /path/to/SEQ OFFSET_SECONDS results/ablation/SEQ
#
# Run from cpp/ on a compute node with >= 9 cores, e.g.
#   srun -p quick -c 10 --mem=16G -t 01:00:00 scripts/run_ablation.sh ...
set -euo pipefail

DATA=$1
OFFSET=$2
OUT=$3
FIXES=(epipolar_full_dot lm_upstream_loops chi2_95 oc_upstream normalize_quaternion
       mask_clamp full_precision_publish all)

mkdir -p "$OUT"
echo "$DATA" > "$OUT/data_path.txt"
echo "$OFFSET" > "$OUT/offset.txt"

build/main_euroc --path "$DATA" --offset "$OFFSET" --threads 1 --out "$OUT/baseline" \
    > "$OUT/baseline.log" 2>&1 &
for fix in "${FIXES[@]}"; do
  build/main_euroc --path "$DATA" --offset "$OFFSET" --threads 1 --fix "$fix" \
      --out "$OUT/$fix" > "$OUT/$fix.log" 2>&1 &
done
wait
echo "done: $OUT"
