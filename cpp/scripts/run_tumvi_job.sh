#!/usr/bin/env bash
# Both back ends on the six TUM-VI room sequences (512 x 512, EuRoC format),
# then the ATE table results/tumvi/ate.csv. Start offsets and paths come from
# config/tumvi_offsets.csv (scripts/find_start_offsets.py --moved 0.1).
#
#   sbatch -c 12 --mem=24G scripts/run_tumvi_job.sh [BUILD_DIR]
set -euo pipefail
BUILD=${1:-build}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=${PY:-$HOME/miniconda3/envs/cv/bin/python}
OUT=results/tumvi
rm -rf "$OUT"; mkdir -p "$OUT"
while IFS=, read -r name off seq; do
  room=${name#dataset-}; room=${room%_512_16}
  "$BUILD/main_euroc" --path "$seq" --offset "$off" --threads 1 --config config/tumvi_mono_prune.yaml \
      --out "$OUT/msckf/$room" > "$OUT/msckf_$room.log" 2>&1 &
  "$BUILD/main_euroc" --path "$seq" --offset "$off" --threads 1 --config config/tumvi_mono.yaml \
      --backend swo --out "$OUT/swo/$room" > "$OUT/swo_$room.log" 2>&1 &
done < <(tail -n +2 config/tumvi_offsets.csv)
wait
"$PY" - <<'PY'
import csv
import sys
sys.path.insert(0, 'scripts')
import vio_eval as ve
with open('results/tumvi/ate.csv', 'w') as f:
    f.write('sequence,msckf,swo\n')
    for row in csv.DictReader(open('config/tumvi_offsets.csv')):
        room = row['sequence'].replace('dataset-', '').replace('_512_16', '')
        gt = ve.load_gt(row['path'])
        a = [ve.ate(gt, f'results/tumvi/{b}/{room}/traj_cpp.txt') for b in ('msckf', 'swo')]
        f.write(f'{room},' + ','.join('' if x is None else f'{x:.4f}' for x in a) + '\n')
print(open('results/tumvi/ate.csv').read())
PY
