#!/usr/bin/env bash
# One SLURM job per EuRoC sequence: the C++ ablation variants and the
# deterministic Python reference, all in parallel. Accuracy only: these runs
# share a node, so latency numbers come from dedicated runs.
#
#   sbatch -c 11 --mem=24G -t 02:00:00 scripts/run_sequence_job.sh SEQ OFFSET PATH
#
# CONDA_SH and PY (a Python with numpy, scipy, opencv) can be overridden.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 SEQ OFFSET SEQ_DIR" >&2; exit 2; }
SEQ=$1; OFFSET=$2; DATA=$3
# sbatch runs a copy of this script from its spool dir, so use the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate vio_cpp
PY=${PY:-$HOME/miniconda3/envs/cv/bin/python}

mkdir -p "results/sequences/$SEQ"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" scripts/run_python_reference.py --path "$DATA" --offset "$OFFSET" \
      --out "results/sequences/$SEQ/python" > "results/sequences/$SEQ/python.log" 2>&1 &
scripts/run_ablation.sh "$DATA" "$OFFSET" "results/ablation/$SEQ"
wait
echo "finished $SEQ"
