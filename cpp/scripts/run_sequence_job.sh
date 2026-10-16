#!/usr/bin/env bash
# One SLURM job per EuRoC sequence: the 9 C++ ablation variants (baseline =
# Python behaviour) and the deterministic Python reference, all in parallel.
# Accuracy only: these runs share a node, so use the dedicated MH_01 runs for
# latency numbers.
#
#   sbatch -c 10 --mem=24G -t 02:00:00 scripts/run_sequence_job.sh SEQ OFFSET PATH
set -euo pipefail
SEQ=$1; OFFSET=$2; DATA=$3
# sbatch runs a copy of this script from its spool dir, so use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source /home/sbalamurugan/miniconda3/etc/profile.d/conda.sh
conda activate vio_cpp
PY=/home/sbalamurugan/miniconda3/envs/cv/bin/python

mkdir -p "results/sequences/$SEQ"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY scripts/run_python_reference.py --path "$DATA" --offset "$OFFSET" \
      --out "results/sequences/$SEQ/python" > "results/sequences/$SEQ/python.log" 2>&1 &
scripts/run_ablation.sh "$DATA" "$OFFSET" "results/ablation/$SEQ"
wait
echo "finished $SEQ"
