#!/usr/bin/env bash
# Per-frame timing of every method on MH_01, one after the other on one node
# (no other runs in parallel), so all latency numbers are comparable.
# Accuracy numbers come from the other runs; these runs only write timings.
#
#   sbatch -w NODE -c 2 --mem=8G -t 01:00:00 scripts/run_timing_job.sh SEQ_DIR [BUILD_DIR]
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 SEQ_DIR [BUILD_DIR]" >&2; exit 2; }
DATA=$1; BUILD=${2:-build}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate vio_cpp
PY=${PY:-$HOME/miniconda3/envs/cv/bin/python}
OUT=results/timing
mkdir -p "$OUT"
hostname > "$OUT/node.txt"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" scripts/run_python_reference.py --path "$DATA" --out "$OUT/python" > "$OUT/python.log" 2>&1
"$BUILD/main_euroc" --path "$DATA" --threads 1 --out "$OUT/cpp_python_behaviour" \
    > "$OUT/cpp_python_behaviour.log" 2>&1
"$BUILD/main_euroc" --path "$DATA" --threads 1 --config config/euroc_fixed.yaml \
    --out "$OUT/cpp_msckf_fixed" > "$OUT/cpp_msckf_fixed.log" 2>&1
"$BUILD/main_euroc" --path "$DATA" --threads 1 --config config/euroc_fixed.yaml --backend swo \
    --out "$OUT/cpp_swo" > "$OUT/cpp_swo.log" 2>&1
echo done
