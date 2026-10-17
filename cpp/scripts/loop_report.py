"""
Loop closure on all EuRoC sequences: ATE of the C++ back ends without loop
closure, with it (each pose corrected online, and the final pose graph), and
VINS-Fusion without and with its loop_fusion node (final pose graph).

Usage (from cpp/): python scripts/loop_report.py [--vins-loop PATTERN_WITH_{seq}]
Writes results/final/loop_table.csv and results/final/loop_summary.md.
"""
import argparse
import csv

import numpy as np

import vio_eval as ve

COLUMNS = ['MSCKF', 'MSCKF loop online', 'MSCKF loop final', 'SWO', 'SWO loop online',
           'SWO loop final', 'VINS-Fusion', 'VINS-Fusion loop final']


def paths(seq, vins_loop):
    m, s = 'results/backends/msckf_mono_prune_loop', 'results/backends/swo_mono_loop'
    return {
        'MSCKF': f'{m}/{seq}/traj_cpp.txt',
        'MSCKF loop online': f'{m}/{seq}/traj_loop.txt',
        'MSCKF loop final': f'{m}/{seq}/traj_loop_final.txt',
        'SWO': f'{s}/{seq}/traj_cpp.txt',
        'SWO loop online': f'{s}/{seq}/traj_loop.txt',
        'SWO loop final': f'{s}/{seq}/traj_loop_final.txt',
        'VINS-Fusion': f'results/vins/vins_{seq}_tum.txt',
        'VINS-Fusion loop final': vins_loop.format(seq=seq),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vins-loop', default='results/vins/vins_loop_{seq}_tum.txt')
    args = parser.parse_args()

    table = {}
    for s in ve.sequences():
        gt = ve.load_gt(s['path'])
        table[s['sequence']] = {
            col: ve.ate(gt, path, s['t_min'] if col.startswith('VINS') else None)
            for col, path in paths(s['sequence'], args.vins_loop).items()}

    with open('results/final/loop_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence'] + COLUMNS)
        for seq, row in table.items():
            w.writerow([seq] + ['' if ve.failed(row[c]) else f'{row[c]:.4f}' for c in COLUMNS])

    def cell(a):
        return 'fail' if a is None else ('div.' if a > ve.DIVERGED_M else f'{a:.3f}')

    lines = ['| ATE RMSE [m] | ' + ' | '.join(COLUMNS) + ' |', '|---' * (len(COLUMNS) + 1) + '|']
    for seq, row in table.items():
        lines.append(f'| {seq} | ' + ' | '.join(cell(row[c]) for c in COLUMNS) + ' |')
    lines.append(f'| mean, all {len(table)} | ' + ' | '.join(
        '-' if any(ve.failed(r[c]) for r in table.values())
        else f'{np.mean([r[c] for r in table.values()]):.3f}' for c in COLUMNS) + ' |')
    with open('results/final/loop_summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
