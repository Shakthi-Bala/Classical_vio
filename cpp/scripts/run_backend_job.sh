#!/usr/bin/env bash
# One EuRoC sequence with one back end. Output: results/backends/BACKEND/SEQ
# for config/euroc_fixed.yaml (the default), else
# results/backends/BACKEND_VARIANT/SEQ for config/euroc_VARIANT.yaml.
# LOOP=1 adds loop closure (main_euroc --loop) and "_loop" to the name.
#
#   sbatch -c 2 scripts/run_backend_job.sh BACKEND SEQ OFFSET SEQ_DIR [CONFIG] [BUILD_DIR]
set -euo pipefail
[[ $# -ge 4 ]] || { echo "usage: $0 msckf|swo SEQ OFFSET SEQ_DIR [CONFIG] [BUILD_DIR]" >&2; exit 2; }
BACKEND=$1; SEQ=$2; OFFSET=$3; DATA=$4
CONFIG=${5:-config/euroc_fixed.yaml}; BUILD=${6:-build}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate vio_cpp
VARIANT=$(basename "$CONFIG" .yaml); VARIANT=${VARIANT#euroc_}
NAME=$BACKEND; [[ $VARIANT == fixed ]] || NAME=${BACKEND}_$VARIANT
LOOP_ARG=(); [[ ${LOOP:-0} == 1 ]] && { LOOP_ARG=(--loop); NAME=${NAME}_loop; }
OUT="results/backends/$NAME/$SEQ"
mkdir -p "$OUT"
"$BUILD/main_euroc" --path "$DATA" --offset "$OFFSET" --threads 1 --backend "$BACKEND" \
    --config "$CONFIG" "${LOOP_ARG[@]}" --out "$OUT" > "$OUT/run.log" 2>&1
