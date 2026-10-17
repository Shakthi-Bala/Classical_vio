"""
Final comparison on all EuRoC sequences: Python MSCKF, C++ MSCKF (Python
behaviour), C++ MSCKF and C++ sliding window (config/euroc_mono.yaml),
VINS-Fusion; plus the C++ back ends without cam0-only features
(config/euroc_fixed.yaml).

Usage (from cpp/): python scripts/final_report.py [--vins PATTERN_WITH_{seq}]
Writes, in results/final/: ate_table.csv, rpe_table.csv, summary.md (tables),
ate_per_sequence.png and traj_<SEQ>.png (--traj-seq).
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps
import vio_eval as ve

OUT = 'results/final'
COLUMNS = ['Python', 'MSCKF = Python', 'MSCKF', 'SWO', 'VINS-Fusion']
MONO_COLUMNS = ['MSCKF stereo only', 'MSCKF', 'SWO stereo only', 'SWO']
ALL_COLUMNS = COLUMNS + ['MSCKF stereo only', 'SWO stereo only']
# Plotted series: (column, label, colour, marker). Marker shapes are the
# secondary encoding the two low-contrast palette slots need.
SERIES = [
    ('Python', 'Python MSCKF', ps.PYTHON, 'o'),
    ('MSCKF', 'C++ MSCKF', ps.CPP_MSCKF, 's'),
    ('SWO', 'C++ sliding window (Ceres)', ps.CPP_SWO, 'D'),
    ('VINS-Fusion', 'VINS-Fusion (no loop)', ps.VINS, '^'),
]


def trajectory_paths(seq, vins_pattern):
    return {
        'Python': ve.python_trajectory(seq),
        'MSCKF = Python': f'results/ablation/{seq}/baseline/traj_cpp.txt',
        'MSCKF': f'results/backends/msckf_mono/{seq}/traj_cpp.txt',
        'SWO': f'results/backends/swo_mono/{seq}/traj_cpp.txt',
        'VINS-Fusion': vins_pattern.format(seq=seq),
        'MSCKF stereo only': f'results/backends/msckf/{seq}/traj_cpp.txt',
        'SWO stereo only': f'results/backends/swo/{seq}/traj_cpp.txt',
    }


def score(vins_pattern):
    """{seq: {column: (ate, rpe)}}. VINS-Fusion ran on the whole bag, so it is
    cropped to the same start time as our runs."""
    table = {}
    for s in ve.sequences():
        gt = ve.load_gt(s['path'])
        row = {}
        for col, path in trajectory_paths(s['sequence'], vins_pattern).items():
            t_min = s['t_min'] if col == 'VINS-Fusion' else None
            a = None if path is None else ve.ate(gt, path, t_min)
            r = None if ve.failed(a) else ve.rpe(gt, path, t_min)
            row[col] = (a, r)
        table[s['sequence']] = row
    return table


def common_sequences(table, columns):
    """Sequences every column completed: the fair set for means."""
    return [seq for seq, row in table.items() if not any(ve.failed(row[c][0]) for c in columns)]


def ate_lines(table, columns):
    """Markdown ATE table: per sequence, failures, the mean over the sequences
    all columns complete and, for columns without failures, over all."""
    def cell(a):
        return 'fail' if a is None else ('div.' if a > ve.DIVERGED_M else f'{a:.3f}')

    common = common_sequences(table, columns)
    lines = ['| ATE RMSE [m] | ' + ' | '.join(columns) + ' |', '|---' * (len(columns) + 1) + '|']
    for seq, row in table.items():
        lines.append(f'| {seq} | ' + ' | '.join(cell(row[c][0]) for c in columns) + ' |')
    lines.append('| failures | ' + ' | '.join(
        str(sum(ve.failed(row[c][0]) for row in table.values())) for c in columns) + ' |')
    lines.append(f'| mean, {len(common)} seq. all completed | ' + ' | '.join(
        f'{np.mean([table[s][c][0] for s in common]):.3f}' for c in columns) + ' |')
    if len(common) < len(table):
        lines.append(f'| mean, all {len(table)} seq. | ' + ' | '.join(
            '-' if any(ve.failed(row[c][0]) for row in table.values())
            else f'{np.mean([row[c][0] for row in table.values()]):.3f}' for c in columns) + ' |')
    return lines


def rpe_lines(table, columns):
    common = common_sequences(table, columns)
    lines = [f'| RPE @ {ve.RPE_LENGTH_M:.0f} m [% / deg] | ' + ' | '.join(columns) + ' |',
             '|---' * (len(columns) + 1) + '|']
    for seq, row in table.items():
        lines.append(f'| {seq} | ' + ' | '.join(
            'fail' if row[c][1] is None else f'{row[c][1][0]:.2f} / {row[c][1][1]:.2f}'
            for c in columns) + ' |')
    lines.append(f'| mean, {len(common)} seq. all completed | ' + ' | '.join(
        f'{np.mean([table[s][c][1][0] for s in common]):.2f} / '
        f'{np.mean([table[s][c][1][1] for s in common]):.2f}' for c in columns) + ' |')
    return lines


def write_tables(table):
    with open(f'{OUT}/ate_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence'] + ALL_COLUMNS)
        for seq, row in table.items():
            w.writerow([seq] + ['' if ve.failed(row[c][0]) else f'{row[c][0]:.4f}'
                                for c in ALL_COLUMNS])
    with open(f'{OUT}/rpe_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence'] + [f'{c} trans%' for c in ALL_COLUMNS] +
                   [f'{c} rot_deg' for c in ALL_COLUMNS])
        for seq, row in table.items():
            rp = [row[c][1] for c in ALL_COLUMNS]
            w.writerow([seq] + ['' if v is None else f'{v[0]:.3f}' for v in rp] +
                       ['' if v is None else f'{v[1]:.3f}' for v in rp])

    lines = (ate_lines(table, COLUMNS) + [''] + rpe_lines(table, COLUMNS) + ['', 'Cam0-only '
             '(mono) features, config/euroc_mono.yaml vs config/euroc_fixed.yaml:', ''] +
             ate_lines(table, MONO_COLUMNS))
    with open(f'{OUT}/summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return lines


def plot_ate_per_sequence(table):
    seqs = list(table)
    x = np.arange(len(seqs))
    top = 3.0   # failures are drawn at this height
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    for (col, label, color, marker), dx in zip(SERIES, np.linspace(-0.27, 0.27, len(SERIES))):
        ates = [table[s][col][0] for s in seqs]
        ok = [i for i, a in enumerate(ates) if not ve.failed(a)]
        bad = [i for i, a in enumerate(ates) if ve.failed(a)]
        ax.scatter(x[ok] + dx, [ates[i] for i in ok], s=36, color=color, marker=marker,
                   edgecolor=ps.SURFACE, linewidth=0.8, zorder=3, label=label)
        ax.scatter(x[bad] + dx, [top] * len(bad), s=36, color=color, marker='x', zorder=3)
    ax.axhline(top * 0.85, color=ps.SPINE, lw=0.8)
    ax.annotate('failed / diverged', (len(seqs) - 0.5, top), ha='right', va='bottom', fontsize=7,
                color=ps.TEXT, xytext=(0, 3), textcoords='offset points')
    ax.set_yscale('log')
    ax.set_ylim(0.05, top * 1.4)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2], ['0.05', '0.1', '0.2', '0.5', '1', '2'])
    ax.set_xticks(x, [s.replace('_', '\n', 1) for s in seqs], fontsize=7)
    ax.set_ylabel('ATE RMSE [m]  (Umeyama SE3)', color=ps.TEXT)
    ax.set_title('EuRoC: Python vs C++ MSCKF vs C++ sliding window vs VINS-Fusion',
                 color=ps.TITLE, fontsize=10)
    ps.style_axes(ax, grid='y')
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, labelcolor=ps.TEXT, fontsize=7, ncol=4, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{OUT}/ate_per_sequence.png', facecolor=ps.SURFACE)


def plot_trajectory(seq, seq_path):
    """Top view: ground truth, fixed MSCKF and sliding window, SE3-aligned."""
    gt = ve.load_gt(seq_path)
    labels = {c: l for c, l, _, _ in SERIES}
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    for i, (col, color) in enumerate([('MSCKF', ps.CPP_MSCKF), ('SWO', ps.CPP_SWO)]):
        traj = ve.load_tum(trajectory_paths(seq, '')[col], gt)
        p, g = ve.align(traj, gt)
        if i == 0:
            ax.plot(g[:, 0], g[:, 1], color=ps.NEUTRAL, lw=2, ls='--', label='ground truth')
        ate = np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1)))
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.6, label=f'{labels[col]}  (ATE {ate:.2f} m)')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', color=ps.TEXT)
    ax.set_ylabel('y [m]', color=ps.TEXT)
    ax.set_title(f'{seq}: filter vs sliding window (same front end)', color=ps.TITLE, fontsize=10)
    ps.style_axes(ax)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, labelcolor=ps.TEXT, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/traj_{seq}.png', facecolor=ps.SURFACE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vins', default='results/vins/vins_{seq}_tum.txt')
    parser.add_argument('--traj-seq', default='MH_05_difficult')
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    table = score(args.vins)
    print('\n'.join(write_tables(table)))
    plot_ate_per_sequence(table)
    seq_path = next(s['path'] for s in ve.sequences() if s['sequence'] == args.traj_seq)
    plot_trajectory(args.traj_seq, seq_path)


if __name__ == '__main__':
    main()
