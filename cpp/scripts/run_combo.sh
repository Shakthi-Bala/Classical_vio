#!/usr/bin/env bash
# Extra ablation variant "all_but_chi2": every fix except the chi-squared gate.
#   sbatch -c 1 scripts/run_combo.sh SEQ OFFSET PATH
set -euo pipefail
SEQ=$1; OFFSET=$2; DATA=$3
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source /home/sbalamurugan/miniconda3/etc/profile.d/conda.sh
conda activate vio_cpp
build/main_euroc --path "$DATA" --offset "$OFFSET" --threads 1 \
    --fix epipolar_full_dot --fix lm_upstream_loops --fix oc_upstream \
    --fix normalize_quaternion --fix mask_clamp --fix full_precision_publish \
    --out "results/ablation/$SEQ/all_but_chi2" > "results/ablation/$SEQ/all_but_chi2.log" 2>&1
