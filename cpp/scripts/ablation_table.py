"""
Score every run on every EuRoC sequence: Python reference, C++ (= Python
behaviour) and the C++ quirk-fix variants, plus optional external trajectories
(VINS-Fusion, sliding window). ATE uses the same Umeyama SE3 alignment and
ground-truth interpolation as Code/evaluate.py, without the plots.

Usage (from cpp/):
  python scripts/ablation_table.py [--extra NAME=PATTERN ...]
    PATTERN may contain {seq}, e.g. --extra VINS=/scratch/.../vins_{seq}_tum.txt

Writes results/ablation/ate_table.csv and prints a markdown table.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Code'))
from evaluate import interpolate_gt, umeyama_alignment

VARIANTS = ['baseline', 'epipolar_full_dot', 'lm_upstream_loops', 'chi2_95', 'oc_upstream',
            'normalize_quaternion', 'mask_clamp', 'full_precision_publish', 'all_but_chi2', 'all']
DIVERGED_M = 5.0   # ATE above this is reported as a failure


def load_gt(seq_path):
    d = np.loadtxt(os.path.join(seq_path, 'mav0/state_groundtruth_estimate0/data.csv'),
                   delimiter=',', comments='#')
    return d[:, 0] * 1e-9, d[:, 1:4], d[:, 4:8]


def ate(gt, traj_path, t_min=None):
    """ATE RMSE [m] after SE3 Umeyama alignment, or None if missing/empty."""
    if not os.path.exists(traj_path) or os.path.getsize(traj_path) == 0:
        return None
    d = np.atleast_2d(np.loadtxt(traj_path))
    if t_min is not None:
        d = d[d[:, 0] >= t_min]
    gt_t, gt_p, gt_q = gt
    d = d[(d[:, 0] >= gt_t[0]) & (d[:, 0] <= gt_t[-1])]
    if len(d) < 10:
        return None
    p = d[:, 1:4]
    gt_pos, _ = interpolate_gt(gt_t, gt_p, gt_q, d[:, 0])
    R, t, _ = umeyama_alignment(p, gt_pos, with_scale=False)
    err = (R @ p.T).T + t - gt_pos
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra', action='append', default=[],
                        help='NAME=PATTERN with {seq}; scored on the same cropped span')
    args = parser.parse_args()
    extras = [e.split('=', 1) for e in args.extra]

    rows = []
    with open('config/euroc_offsets.csv') as f:
        sequences = list(csv.DictReader(f))
    for s in sequences:
        seq, path, offset = s['sequence'], s['path'], float(s['offset_s'])
        gt = load_gt(path)
        with open(os.path.join(path, 'mav0/imu0/data.csv')) as imu_file:
            next(imu_file)   # header
            imu_t0 = float(next(imu_file).split(',')[0]) * 1e-9
        t_min = imu_t0 + offset
        py_dir = 'results/python' if seq == 'MH_01_easy' else f'results/sequences/{seq}/python'
        # A Python run that crashed leaves a partial trajectory: count it as a failure.
        log = f'{py_dir}/../python.log' if seq != 'MH_01_easy' else f'{py_dir}/run.log'
        crashed = os.path.exists(log) and 'Traceback' in open(log).read()
        row = {'sequence': seq,
               'Python': None if crashed else ate(gt, f'{py_dir}/traj_python.txt')}
        for v in VARIANTS:
            row[v] = ate(gt, f'results/ablation/{seq}/{v}/traj_cpp.txt')
        for name, pattern in extras:
            row[name] = ate(gt, pattern.format(seq=seq), t_min=t_min)
        rows.append(row)

    columns = ['Python'] + VARIANTS + [n for n, _ in extras]
    os.makedirs('results/ablation', exist_ok=True)
    with open('results/ablation/ate_table.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sequence'] + columns)
        w.writeheader()
        for r in rows:
            w.writerow({'sequence': r['sequence'],
                        **{k: '' if r.get(k) is None else f'{r[k]:.4f}' for k in columns}})

    def fmt(x):
        if x is None:
            return 'fail'
        return 'div.' if x > DIVERGED_M else f'{x:.3f}'

    print('| sequence | ' + ' | '.join(columns) + ' |')
    print('|---' * (len(columns) + 1) + '|')
    for r in rows:
        print(f"| {r['sequence']} | " + ' | '.join(fmt(r.get(c)) for c in columns) + ' |')
    def good(r, c):
        return r.get(c) is not None and r[c] <= DIVERGED_M

    # Per column: failures, mean ATE over the sequences it completed, and how
    # often it beats the C++ baseline (= Python behaviour) on the same sequence.
    fails = {c: sum(1 for r in rows if not good(r, c)) for c in columns}
    print('| failures | ' + ' | '.join(str(fails[c]) for c in columns) + ' |')
    print('| mean ATE of completed runs | ' + ' | '.join(
        f"{np.mean([r[c] for r in rows if good(r, c)]):.3f}" if any(good(r, c) for r in rows)
        else '-' for c in columns) + ' |')
    print('| better than baseline | ' + ' | '.join(
        f"{sum(1 for r in rows if good(r, c) and good(r, 'baseline') and r[c] < r['baseline'])}"
        f"/{sum(1 for r in rows if good(r, 'baseline'))}" for c in columns) + ' |')


if __name__ == '__main__':
    main()
