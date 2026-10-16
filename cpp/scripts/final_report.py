"""
Final comparison on all EuRoC sequences: Python MSCKF, C++ MSCKF (Python
behaviour), C++ MSCKF with the quirk fixes, C++ sliding window, VINS-Fusion.

Usage (from cpp/):
  python scripts/final_report.py [--vins PATTERN_WITH_{seq}]
Writes results/final/ate_per_sequence.png and results/final/summary.md.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ablation_table as at

TEXT, GRID, SPINE = '#52514e', '#e6e5e1', '#d4d3cf'
# (column, label, color, marker): validated categorical slots 1-4; marker
# shapes are the secondary encoding the low-contrast slots need.
SERIES = [
    ('Python', 'Python MSCKF', '#2a78d6', 'o'),
    ('MSCKF fixed', 'C++ MSCKF (quirks fixed)', '#eb6834', 's'),
    ('SWO', 'C++ sliding window (Ceres)', '#1baf7a', 'D'),
    ('VINS-Fusion', 'VINS-Fusion (no loop)', '#eda100', '^'),
]
DIVERGED = at.DIVERGED_M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vins', default='/scratch/sbalamurugan/ros2_vio/results/vins_{seq}_tum.txt')
    args = parser.parse_args()

    rows = []
    with open('config/euroc_offsets.csv') as f:
        sequences = list(csv.DictReader(f))
    for s in sequences:
        seq, path, offset = s['sequence'], s['path'], float(s['offset_s'])
        gt = at.load_gt(path)
        with open(os.path.join(path, 'mav0/imu0/data.csv')) as imu_file:
            next(imu_file)
            t_min = float(next(imu_file).split(',')[0]) * 1e-9 + offset
        py_dir = 'results/python' if seq == 'MH_01_easy' else f'results/sequences/{seq}/python'
        log = f'{py_dir}/run.log' if seq == 'MH_01_easy' else f'results/sequences/{seq}/python.log'
        crashed = os.path.exists(log) and 'Traceback' in open(log).read()
        vins_path = args.vins.format(seq=seq)
        if seq == 'MH_01_easy' and not os.path.exists(vins_path):
            vins_path = '/scratch/sbalamurugan/ros2_vio/results/vins_mh01_tum.txt'
        rows.append({
            'sequence': seq,
            'Python': None if crashed else at.ate(gt, f'{py_dir}/traj_python.txt'),
            'MSCKF = Python': at.ate(gt, f'results/ablation/{seq}/baseline/traj_cpp.txt'),
            'MSCKF fixed': at.ate(gt, f'results/backends/msckf/{seq}/traj_cpp.txt'),
            'SWO': at.ate(gt, f'results/backends/swo/{seq}/traj_cpp.txt'),
            # VINS runs the whole bag; score it on the same span as ours.
            'VINS-Fusion': at.ate(gt, vins_path, t_min=t_min),
        })

    # --- figure: ATE per sequence, log scale, failures on top.
    seqs = [r['sequence'] for r in rows]
    x = np.arange(len(seqs))
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    offsets = np.linspace(-0.27, 0.27, len(SERIES))
    top = 3.0
    for (col, label, color, marker), dx in zip(SERIES, offsets):
        ys = [r[col] for r in rows]
        ok = [i for i, y in enumerate(ys) if y is not None and y <= DIVERGED]
        bad = [i for i, y in enumerate(ys) if y is None or y > DIVERGED]
        if not ok and not bad:
            continue
        ax.scatter(x[ok] + dx, [ys[i] for i in ok], s=36, color=color, marker=marker,
                   edgecolor='#fcfcfb', linewidth=0.8, zorder=3, label=label)
        ax.scatter(x[bad] + dx, [top] * len(bad), s=36, color=color, marker='x', zorder=3)
    ax.axhline(top * 0.85, color=SPINE, lw=0.8)
    ax.annotate('failed / diverged', (len(seqs) - 0.5, top), ha='right', va='bottom', fontsize=7,
                color=TEXT, xytext=(0, 3), textcoords='offset points')
    ax.set_yscale('log')
    ax.set_ylim(0.05, top * 1.4)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2], ['0.05', '0.1', '0.2', '0.5', '1', '2'])
    ax.set_xticks(x, [s.replace('_', '\n', 1) for s in seqs], fontsize=7)
    ax.set_ylabel('ATE RMSE [m]  (Umeyama SE3)', color=TEXT)
    ax.set_title('EuRoC: Python vs C++ MSCKF vs C++ sliding window vs VINS-Fusion',
                 color='#0b0b0b', fontsize=10)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=7, ncol=4, loc='upper left')
    fig.tight_layout()
    os.makedirs('results/final', exist_ok=True)
    fig.savefig('results/final/ate_per_sequence.png', facecolor='#fcfcfb')

    # --- markdown table
    cols = ['Python', 'MSCKF = Python', 'MSCKF fixed', 'SWO', 'VINS-Fusion']

    def fmt(v):
        return 'fail' if v is None else ('div.' if v > DIVERGED else f'{v:.3f}')

    lines = ['| ATE RMSE [m] | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 1) + '|']
    for r in rows:
        lines.append(f"| {r['sequence']} | " + ' | '.join(fmt(r[c]) for c in cols) + ' |')
    good = lambda r, c: r[c] is not None and r[c] <= DIVERGED
    lines.append('| failures | ' + ' | '.join(str(sum(not good(r, c) for r in rows)) for c in cols) + ' |')
    # Mean over the sequences every column completed (a fair comparison).
    common = [r for r in rows if all(good(r, c) for c in cols)]
    if common:
        lines.append(f'| mean, {len(common)} seq. all completed | ' +
                     ' | '.join(f"{np.mean([r[c] for r in common]):.3f}" for c in cols) + ' |')
    open('results/final/summary.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
