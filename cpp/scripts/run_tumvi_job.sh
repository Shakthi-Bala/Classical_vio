#!/usr/bin/env bash
# Both back ends on the six TUM-VI room sequences (512 x 512, EuRoC format),
# then the ATE table results/tumvi/ate.csv. The start offset of each room is
# the quietest IMU second in its first 8 s (the recordings are handheld).
#
#   sbatch -c 12 --mem=24G scripts/run_tumvi_job.sh TUMVI_DIR [BUILD_DIR]
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 TUMVI_DIR [BUILD_DIR]" >&2; exit 2; }
ROOT=$1; BUILD=${2:-build}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=${PY:-$HOME/miniconda3/envs/cv/bin/python}
OUT=results/tumvi
mkdir -p "$OUT"
for i in 1 2 3 4 5 6; do
  SEQ=$ROOT/dataset-room${i}_512_16
  OFF=$("$PY" -c "
import numpy as np
d = np.loadtxt('$SEQ/mav0/imu0/data.csv', delimiter=',', comments='#')
t = (d[:, 0] - d[0, 0]) * 1e-9
starts = np.arange(0.0, 7.01, 0.5)
score = [d[(t >= s) & (t < s + 1), 1:4].std(0).max() for s in starts]
print(starts[int(np.argmin(score))])")
  echo "room$i offset $OFF" | tee -a "$OUT/offsets.txt"
  "$BUILD/main_euroc" --path "$SEQ" --offset "$OFF" --threads 1 --config config/tumvi_mono_prune.yaml \
      --out "$OUT/msckf/room$i" > "$OUT/msckf_room$i.log" 2>&1 &
  "$BUILD/main_euroc" --path "$SEQ" --offset "$OFF" --threads 1 --config config/tumvi_mono.yaml \
      --backend swo --out "$OUT/swo/room$i" > "$OUT/swo_room$i.log" 2>&1 &
done
wait
"$PY" - "$ROOT" <<'PY'
import sys
sys.path.insert(0, 'scripts')
import vio_eval as ve
root = sys.argv[1]
with open('results/tumvi/ate.csv', 'w') as f:
    f.write('sequence,msckf,swo\n')
    for i in range(1, 7):
        gt = ve.load_gt(f'{root}/dataset-room{i}_512_16')
        a = [ve.ate(gt, f'results/tumvi/{b}/room{i}/traj_cpp.txt') for b in ('msckf', 'swo')]
        f.write(f'room{i},' + ','.join('' if x is None else f'{x:.4f}' for x in a) + '\n')
print(open('results/tumvi/ate.csv').read())
PY
