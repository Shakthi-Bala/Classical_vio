#!/usr/bin/env bash
# Quirk ablation on one sequence: the C++ MSCKF with no fix (= Python
# behaviour), each Python-quirk fix on its own, all fixes, and all but the
# chi-squared gate, in parallel (one core each).
#
#   scripts/run_ablation.sh /path/to/SEQ OFFSET_SECONDS results/ablation/SEQ [BUILD_DIR]
#
# Run from cpp/ on a node with >= 10 cores. Variant names match VARIANTS in
# scripts/ablation_table.py.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 SEQ_DIR OFFSET OUT_DIR [BUILD_DIR]" >&2; exit 2; }
DATA=$1; OFFSET=$2; OUT=$3; BUILD=${4:-build}
FIXES=(epipolar_full_dot lm_upstream_loops chi2_95 oc_upstream normalize_quaternion
       mask_clamp full_precision_publish all)
mkdir -p "$OUT"

run() {   # run NAME [--fix F]...
  local name=$1; shift
  "$BUILD/main_euroc" --path "$DATA" --offset "$OFFSET" --threads 1 "$@" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1 &
}
run baseline
for fix in "${FIXES[@]}"; do
  run "$fix" --fix "$fix"
done
all_but_chi2=()
for fix in "${FIXES[@]}"; do
  [[ $fix == chi2_95 || $fix == all ]] || all_but_chi2+=(--fix "$fix")
done
run all_but_chi2 "${all_but_chi2[@]}"
wait
echo "done: $OUT"
