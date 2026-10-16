#!/usr/bin/env bash
# One EuRoC sequence with one back end and the fixed front end/filter config.
#   sbatch -c 2 scripts/run_backend_job.sh BACKEND SEQ OFFSET PATH [BUILD_DIR]
set -euo pipefail
BACKEND=$1; SEQ=$2; OFFSET=$3; DATA=$4; BUILD=${5:-build}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source /home/sbalamurugan/miniconda3/etc/profile.d/conda.sh
conda activate vio_cpp
OUT="results/backends/$BACKEND/$SEQ"
mkdir -p "$OUT"
"$BUILD/main_euroc" --path "$DATA" --offset "$OFFSET" --threads 1 --backend "$BACKEND" \
    --config config/euroc_fixed.yaml --out "$OUT" > "$OUT/run.log" 2>&1
