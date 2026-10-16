"""
Compare the Python and C++ MSCKF runs: accuracy, speed and front-end parity.

Accuracy uses the existing Code/evaluate.py (Umeyama SE3 ATE, Scaramuzza RPE),
unchanged, so both implementations are scored by exactly the same code.

Usage (from cpp/, in the `cv` env, after run_python_reference.py and main_euroc):
  python scripts/compare.py --gt /path/to/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv

Writes results/summary.md, results/trajectory_xy.png, results/latency.png and
results/<run>/plots/ (all of evaluate.py's plots per run).
"""
import argparse
import contextlib
import io
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Code'))
from evaluate import generate_all_plots, umeyama_alignment

# Validated categorical slots 1-2 (dataviz palette); ground truth is a neutral reference.
COLORS = {'Python': '#2a78d6', 'C++': '#eb6834'}
GT_COLOR = '#8a8984'
TEXT = '#52514e'

RUNS = [   # (label, trajectory, timing csv, plots dir)
    ('Python', 'results/python/traj_python.txt', 'results/python/timing_python.csv',
     'results/python/plots'),
    ('C++', 'results/cpp/traj_cpp.txt', 'results/cpp/timing_cpp.csv', 'results/cpp/plots'),
]


def load_gt(path):
    d = np.loadtxt(path, delimiter=',', comments='#')
    return d[:, 0] * 1e-9, d[:, 1:4], d[:, 4:8]   # t [s], p, q (w x y z)


def load_tum(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1:4], Rotation.from_quat(d[:, 4:8]).as_matrix()


def evaluate(gt, traj_path, out_dir):
    t, p, R = load_tum(traj_path)
    with contextlib.redirect_stdout(io.StringIO()):
        generate_all_plots(gt[0], gt[1], gt[2], t, p, R, output_dir=out_dir)
    report = open(os.path.join(out_dir, 'rmse_report.txt')).read()

    def grab(pattern):
        m = re.search(pattern, report)
        return float(m.group(1)) if m else float('nan')

    return {
        'ate': grab(r'RMSE 3D\s*:\s*([\d.]+)'),
        'rot': grab(r'Overall\s*:\s*([\d.]+)'),
        'rpe10_t': grab(r'L= 10\.0 m :\s*trans ([\d.]+)'),
        'rpe10_r': grab(r'L= 10\.0 m :.*rot ([\d.]+)'),
        'poses': len(t),
        'traj': (t, p),
    }


def timing_stats(path):
    d = np.genfromtxt(path, delimiter=',', names=True)
    fe, be = d['frontend_ms'], d['backend_ms']
    total = fe + be
    data_s = d['timestamp'][-1] - d['timestamp'][0]
    stats = {}
    for name, x in (('frontend', fe), ('backend', be), ('total', total)):
        stats[name] = (x.mean(), np.percentile(x, 50), np.percentile(x, 95), x.max())
    stats['rtf'] = data_s / (total.sum() / 1e3)
    return stats


def read_features(path, max_frames):
    frames = []
    with open(path) as f:
        for line in f:
            if line.startswith('F'):
                if len(frames) == max_frames:
                    break
                frames.append({})
            else:
                p = line.split()
                frames[-1][int(p[0])] = np.array(p[1:], dtype=float)
    return frames


def frontend_parity(py_path, cpp_path, max_frames=3000):
    a, b = read_features(py_path, max_frames), read_features(cpp_path, max_frames)
    n = min(len(a), len(b))
    identical = 0
    for k in range(n):
        same = (a[k].keys() == b[k].keys() and
                all(np.abs(a[k][i] - b[k][i]).max() < 1e-6 for i in a[k]))
        if not same:
            break
        identical += 1
    return {
        'frames': n,
        'identical_prefix': identical,
        'mean_count_py': np.mean([len(x) for x in a[:n]]),
        'mean_count_cpp': np.mean([len(x) for x in b[:n]]),
    }


def plot_trajectories(gt, results, out_path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    t_ref, p_ref = results['Python']['traj']
    gt_xy = gt[1][(gt[0] >= t_ref[0]) & (gt[0] <= t_ref[-1])]
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=GT_COLOR, lw=2, ls='--', label='Ground truth')
    for name in ('Python', 'C++'):
        t, p = results[name]['traj']
        # Align to ground truth (SE3 Umeyama), the same way the ATE is computed.
        gt_at = np.stack([np.interp(t, gt[0], gt[1][:, i]) for i in range(3)], axis=1)
        R, tr, s = umeyama_alignment(p, gt_at)
        pa = (s * R @ p.T).T + tr
        # Python drawn wide underneath, C++ thin on top: the two nearly coincide.
        ax.plot(pa[:, 0], pa[:, 1], color=COLORS[name], lw=4.5 if name == 'Python' else 1.5,
                label=f"{name}  (ATE {results[name]['ate']:.3f} m)")
    ax.set_xlabel('x [m]', color=TEXT)
    ax.set_ylabel('y [m]', color=TEXT)
    ax.set_title('EuRoC MH_01_easy: stereo MSCKF, Python vs C++', color='#0b0b0b')
    ax.set_aspect('equal')
    ax.grid(color='#e6e5e1', lw=0.6)
    for s in ax.spines.values():
        s.set_color('#d4d3cf')
    ax.tick_params(colors=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, facecolor='#fcfcfb')
    plt.close(fig)


def plot_latency(timings, out_path):
    stages = ['frontend', 'backend', 'total']
    x = np.arange(len(stages))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for j, name in enumerate(('Python', 'C++')):
        means = [timings[name][s][0] for s in stages]
        bars = ax.bar(x + (j - 0.5) * (width + 0.02), means, width, color=COLORS[name],
                      label=f"{name}  ({timings[name]['rtf']:.2f}x real time)")
        for b, v in zip(bars, means):
            ax.annotate(f'{v:.1f}', (b.get_x() + b.get_width() / 2, v), ha='center',
                        va='bottom', fontsize=8, color=TEXT, xytext=(0, 2),
                        textcoords='offset points')
    ax.axhline(50, color=GT_COLOR, lw=1, ls='--')
    ax.annotate('50 ms = 20 Hz camera budget', (2.45, 50), ha='right', va='bottom',
                fontsize=8, color=TEXT, xytext=(0, 2), textcoords='offset points')
    ax.set_xticks(x, ['front end', 'back end', 'total'])
    ax.set_ylabel('mean latency per frame [ms]', color=TEXT)
    ax.set_title('Per-frame latency (same compute node, single process)', color='#0b0b0b')
    ax.grid(axis='y', color='#e6e5e1', lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color('#d4d3cf')
    ax.tick_params(colors=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='state_groundtruth_estimate0/data.csv')
    args = parser.parse_args()

    gt = load_gt(args.gt)
    results, timings = {}, {}
    for name, traj, timing, plots in RUNS:
        results[name] = evaluate(gt, traj, plots)
        timings[name] = timing_stats(timing)

    parity = frontend_parity('results/python/features_python.txt', 'results/cpp/features_cpp.txt')

    backend = None
    if os.path.exists('results/python_replay/traj_python.txt') and \
            os.path.exists('results/cpp_replay/traj_cpp.txt'):
        a = np.loadtxt('results/python_replay/traj_python.txt')
        b = np.loadtxt('results/cpp_replay/traj_cpp.txt')
        if len(a) == len(b):
            backend = (len(a), np.linalg.norm(a[:, 1:4] - b[:, 1:4], axis=1).max())

    plot_trajectories(gt, results, 'results/trajectory_xy.png')
    plot_latency(timings, 'results/latency.png')

    py, cp = timings['Python'], timings['C++']
    lines = [
        '# Python vs C++ stereo MSCKF on EuRoC MH_01_easy',
        '',
        'Start offset 40 s, deterministic replay (no threads), same compute node.',
        'Accuracy from the unchanged Code/evaluate.py.',
        '',
        '| | Python | C++ |',
        '|---|---|---|',
        f"| poses | {results['Python']['poses']} | {results['C++']['poses']} |",
        f"| ATE RMSE [m] | {results['Python']['ate']:.4f} | {results['C++']['ate']:.4f} |",
        f"| rotation RMSE [deg] | {results['Python']['rot']:.3f} | {results['C++']['rot']:.3f} |",
        f"| RPE @ 10 m [% / deg] | {results['Python']['rpe10_t']:.2f} / {results['Python']['rpe10_r']:.2f} "
        f"| {results['C++']['rpe10_t']:.2f} / {results['C++']['rpe10_r']:.2f} |",
    ]
    for stage in ('frontend', 'backend', 'total'):
        p, c = py[stage], cp[stage]
        lines.append(f'| {stage} latency mean / p95 [ms] | {p[0]:.2f} / {p[2]:.2f} '
                     f'| {c[0]:.2f} / {c[2]:.2f} |')
    lines += [
        f"| real-time factor | {py['rtf']:.2f}x | {cp['rtf']:.2f}x |",
        f"| speed-up (total mean) | | {py['total'][0] / cp['total'][0]:.1f}x |",
        '',
        '## Parity checks',
        '',
    ]
    if backend:
        lines.append(f'- **Back end** (both filters fed the same Python feature file): '
                     f'max position difference {backend[1]:.1e} m over {backend[0]} poses.')
    lines.append(f"- **Front end**: first {parity['identical_prefix']} of {parity['frames']} "
                 f"frames give identical features (ids and coordinates to 1e-6); mean features "
                 f"per frame Python {parity['mean_count_py']:.1f}, C++ {parity['mean_count_cpp']:.1f}.")
    open('results/summary.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
