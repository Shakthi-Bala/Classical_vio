"""
ATE of every quirk-ablation run on every EuRoC sequence: the Python reference,
the C++ MSCKF with Python behaviour (baseline) and each fix variant.

Usage (from cpp/): python scripts/ablation_table.py
Writes results/ablation/ate_table.csv (raw ATE; empty = no trajectory) and
prints a markdown table with failures, means and wins over the baseline.
"""
import csv
import os

import numpy as np

import vio_eval as ve

# Same list as FIXES in run_ablation.sh, plus the baseline.
VARIANTS = ['baseline', 'epipolar_full_dot', 'lm_upstream_loops', 'chi2_95', 'oc_upstream',
            'normalize_quaternion', 'mask_clamp', 'full_precision_publish', 'all_but_chi2', 'all']
COLUMNS = ['Python'] + VARIANTS


def main():
    rows = []
    for s in ve.sequences():
        seq, gt = s['sequence'], ve.load_gt(s['path'])
        py = ve.python_trajectory(seq)
        row = {'sequence': seq, 'Python': None if py is None else ve.ate(gt, py)}
        for v in VARIANTS:
            row[v] = ve.ate(gt, f'results/ablation/{seq}/{v}/traj_cpp.txt')
        rows.append(row)

    os.makedirs('results/ablation', exist_ok=True)
    with open('results/ablation/ate_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence'] + COLUMNS)
        for r in rows:
            w.writerow([r['sequence']] + ['' if r[c] is None else f'{r[c]:.4f}' for c in COLUMNS])

    def cell(a):
        return 'fail' if a is None else ('div.' if a > ve.DIVERGED_M else f'{a:.3f}')

    def ok(r, c):
        return not ve.failed(r[c])

    print('| sequence | ' + ' | '.join(COLUMNS) + ' |')
    print('|---' * (len(COLUMNS) + 1) + '|')
    for r in rows:
        print(f"| {r['sequence']} | " + ' | '.join(cell(r[c]) for c in COLUMNS) + ' |')
    print('| failures | ' + ' | '.join(str(sum(not ok(r, c) for r in rows)) for c in COLUMNS) + ' |')
    print('| mean ATE of completed runs | ' + ' | '.join(
        f'{np.mean([r[c] for r in rows if ok(r, c)]):.3f}' for c in COLUMNS) + ' |')
    n_base = sum(ok(r, 'baseline') for r in rows)
    print('| better than baseline | ' + ' | '.join(
        f"{sum(ok(r, c) and ok(r, 'baseline') and r[c] < r['baseline'] for r in rows)}/{n_base}"
        for c in COLUMNS) + ' |')


if __name__ == '__main__':
    main()
