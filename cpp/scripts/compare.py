"""
Phase 1 report: the Python and C++ MSCKF on EuRoC MH_01_easy, run with the
same deterministic replay. Accuracy (ATE, RPE), per-frame latency, and the two
parity checks (front-end features, back end on a replayed feature file).

Usage (from cpp/, in the `cv` env, after run_python_reference.py and main_euroc):
  python scripts/compare.py --seq /path/to/MH_01_easy

Writes results/summary.md, results/trajectory_xy.png, results/latency.png and,
per run, results/<run>/plots/ (Code/evaluate.py's full plot set).
"""
import argparse
import contextlib
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import plotstyle as ps
import vio_eval as ve
from evaluate import generate_all_plots

RUNS = {   # label: (trajectory, timing csv, plots dir, colour)
    'Python': ('results/python/traj_python.txt', 'results/python/timing_python.csv',
               'results/python/plots', ps.PYTHON),
    'C++': ('results/cpp/traj_cpp.txt', 'results/cpp/timing_cpp.csv', 'results/cpp/plots',
            ps.CPP_MSCKF),
}
FEATURE_TOLERANCE = 1e-6   # normalized coordinates


def timing_stats(path):
    d = np.genfromtxt(path, delimiter=',', names=True)
    total = d['frontend_ms'] + d['backend_ms']
    stats = {name: (x.mean(), np.percentile(x, 95))
             for name, x in (('frontend', d['frontend_ms']), ('backend', d['backend_ms']),
                             ('total', total))}
    stats['rtf'] = (d['timestamp'][-1] - d['timestamp'][0]) / (total.sum() / 1e3)
    return stats


def frontend_parity(py_path, cpp_path):
    """Number of leading frames whose features agree (same ids, coordinates
    within FEATURE_TOLERANCE), the frame count, and mean features per frame."""
    a, b = ve.read_features(py_path), ve.read_features(cpp_path)
    n = min(len(a), len(b))
    identical = 0
    for (_, fa), (_, fb) in zip(a[:n], b[:n]):
        if fa.keys() != fb.keys() or any(np.abs(fa[i] - fb[i]).max() >= FEATURE_TOLERANCE
                                         for i in fa):
            break
        identical += 1
    return identical, n, np.mean([len(f) for _, f in a[:n]]), np.mean([len(f) for _, f in b[:n]])


def backend_parity():
    """Max position difference when both filters replay the same feature file."""
    a = ve.load_tum('results/python_replay/traj_python.txt')
    b = ve.load_tum('results/cpp_replay/traj_cpp.txt')
    if a is None or b is None or len(a) != len(b):
        print('back-end parity: replay trajectories missing or of different length, skipped')
        return None
    return len(a), np.linalg.norm(a[:, 1:4] - b[:, 1:4], axis=1).max()


def plot_trajectories(gt, out_path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for i, (name, (traj_path, _, _, color)) in enumerate(RUNS.items()):
        traj = ve.load_tum(traj_path, gt)
        p, g = ve.align(traj, gt)
        if i == 0:
            ax.plot(g[:, 0], g[:, 1], color=ps.NEUTRAL, lw=2, ls='--', label='Ground truth')
        ate = np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1)))
        # Python drawn wide underneath, C++ thin on top: the two nearly coincide.
        ax.plot(p[:, 0], p[:, 1], color=color, lw=4.5 if name == 'Python' else 1.5,
                label=f'{name}  (ATE {ate:.3f} m)')
    ax.set_xlabel('x [m]', color=ps.TEXT)
    ax.set_ylabel('y [m]', color=ps.TEXT)
    ax.set_title('EuRoC MH_01_easy: stereo MSCKF, Python vs C++', color=ps.TITLE)
    ax.set_aspect('equal')
    ps.style_axes(ax)
    ax.legend(frameon=False, labelcolor=ps.TEXT)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=ps.SURFACE)
    plt.close(fig)


def plot_latency(timings, out_path):
    stages = ['frontend', 'backend', 'total']
    x = np.arange(len(stages))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for j, (name, (_, _, _, color)) in enumerate(RUNS.items()):
        means = [timings[name][s][0] for s in stages]
        bars = ax.bar(x + (j - 0.5) * (width + 0.02), means, width, color=color,
                      label=f"{name}  ({timings[name]['rtf']:.2f}x real time)")
        for bar, v in zip(bars, means):
            ax.annotate(f'{v:.1f}', (bar.get_x() + bar.get_width() / 2, v), ha='center',
                        va='bottom', fontsize=8, color=ps.TEXT, xytext=(0, 2),
                        textcoords='offset points')
    ax.axhline(50, color=ps.NEUTRAL, lw=1, ls='--')
    ax.annotate('50 ms = 20 Hz camera budget', (x[-1] + width, 50), ha='right', va='bottom',
                fontsize=8, color=ps.TEXT, xytext=(0, 2), textcoords='offset points')
    ax.set_xticks(x, ['front end', 'back end', 'total'])
    ax.set_ylabel('mean latency per frame [ms]', color=ps.TEXT)
    ax.set_title('Per-frame latency (same compute node, single process)', color=ps.TITLE)
    ps.style_axes(ax, grid='y')
    ax.legend(frameon=False, labelcolor=ps.TEXT)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=ps.SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', required=True, help='EuRoC MH_01_easy folder (contains mav0/)')
    parser.add_argument('--offset', type=float, default=40.0, help='start offset of the runs [s]')
    args = parser.parse_args()

    gt = ve.load_gt(args.seq)
    accuracy, timings = {}, {}
    for name, (traj_path, timing_path, plots_dir, _) in RUNS.items():
        accuracy[name] = (ve.ate(gt, traj_path), ve.rpe(gt, traj_path))
        timings[name] = timing_stats(timing_path)
        traj = ve.load_tum(traj_path)
        with contextlib.redirect_stdout(io.StringIO()):
            generate_all_plots(gt[0], gt[1], gt[2], traj[:, 0], traj[:, 1:4],
                               Rotation.from_quat(traj[:, 4:8]).as_matrix(), output_dir=plots_dir)

    plot_trajectories(gt, 'results/trajectory_xy.png')
    plot_latency(timings, 'results/latency.png')

    py, cp = timings['Python'], timings['C++']
    (py_ate, py_rpe), (cp_ate, cp_rpe) = accuracy['Python'], accuracy['C++']
    lines = [
        '# Python vs C++ stereo MSCKF on EuRoC MH_01_easy',
        '',
        f'Start offset {args.offset:g} s, deterministic replay (no threads), same compute node.',
        '',
        '| | Python | C++ |',
        '|---|---|---|',
        f'| ATE RMSE [m] | {py_ate:.4f} | {cp_ate:.4f} |',
        f'| RPE @ {ve.RPE_LENGTH_M:.0f} m [% / deg] | {py_rpe[0]:.2f} / {py_rpe[1]:.2f} '
        f'| {cp_rpe[0]:.2f} / {cp_rpe[1]:.2f} |',
    ]
    for stage in ('frontend', 'backend', 'total'):
        lines.append(f'| {stage} latency mean / p95 [ms] | {py[stage][0]:.2f} / {py[stage][1]:.2f} '
                     f'| {cp[stage][0]:.2f} / {cp[stage][1]:.2f} |')
    lines += [f"| real-time factor | {py['rtf']:.2f}x | {cp['rtf']:.2f}x |",
              f"| speed-up (total mean) | | {py['total'][0] / cp['total'][0]:.1f}x |",
              '', '## Parity checks', '']
    backend = backend_parity()
    if backend:
        lines.append(f'- Back end (both filters fed the same Python feature file): '
                     f'max position difference {backend[1]:.1e} m over {backend[0]} poses.')
    same, n, count_py, count_cpp = frontend_parity('results/python/features_python.txt',
                                                   'results/cpp/features_cpp.txt')
    lines.append(f'- Front end: the first {same} of {n} frames give identical features (ids, '
                 f'and coordinates to {FEATURE_TOLERANCE:g}); mean features per frame '
                 f'Python {count_py:.1f}, C++ {count_cpp:.1f}.')
    with open('results/summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
