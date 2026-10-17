"""
Accuracy vs latency overview: mean ATE over the EuRoC sequences every method
completes, against end-to-end latency per frame on MH_01.

Latency: real-time ROS 2 bag playback on the same compute node (data in ->
pose out), except the Python MSCKF, which has no ROS node (offline per-frame
time). Bars span p50 -> p95, the marker is the mean.

Usage (from cpp/): python scripts/plot_overview.py
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps
import vio_eval as ve


def latency(path, col):
    """(mean, p50, p95) of a per-frame timing column."""
    x = np.genfromtxt(path, delimiter=',', names=True)[col]
    return x.mean(), np.median(x), np.percentile(x, 95)


def mean_ate(methods):
    """Mean ATE per method over the sequences all `methods` completed
    (results/final/ate_table.csv; empty cells are failures)."""
    rows = list(csv.DictReader(open('results/final/ate_table.csv')))
    common = [r for r in rows if all(r[m] for m in methods)]
    return {m: float(np.mean([float(r[m]) for r in common])) for m in methods}, len(common)


def main():
    # Python: offline per-frame time (front end + back end), it has no ROS node.
    py = np.genfromtxt('results/timing/python/timing_python.csv', delimiter=',', names=True)
    py_total = py['frontend_ms'] + py['backend_ms']
    py_lat = (py_total.mean(), np.median(py_total), np.percentile(py_total, 95))
    ates, n_common = mean_ate(['Python', 'MSCKF', 'SWO', 'VINS-Fusion'])
    ros = 'results/ros2/MH_01_easy_{}/timing_ros.csv'   # same node as VINS-Fusion's run
    points = [
        # label, ATE column, latency (mean, p50, p95), color, marker, cores, label side
        ('Python MSCKF (offline, no ROS)', 'Python', py_lat, ps.PYTHON, 'o', '1 core', 'left'),
        ('C++ MSCKF', 'MSCKF', latency(ros.format('msckf'), 'end_to_end_ms'),
         ps.CPP_MSCKF, 's', '1 core', 'right'),
        ('C++ sliding window', 'SWO', latency(ros.format('swo'), 'end_to_end_ms'),
         ps.CPP_SWO, 'D', '1 core', 'right'),
        ('VINS-Fusion', 'VINS-Fusion',
         latency('results/vins/latency_MH_01_easy.csv', 'end_to_end_ms'),
         ps.VINS, '^', '4 cores', 'right'),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for label, col, (mean, p50, p95), color, marker, cores, side in points:
        ate = ates[col]
        ax.plot([p50, p95], [ate, ate], color=color, lw=2.2, alpha=0.45, solid_capstyle='round')
        ax.scatter([mean], [ate], s=90, color=color, marker=marker, edgecolor=ps.SURFACE,
                   linewidth=1.2, zorder=3)
        ax.annotate(f'{label}\n{ate:.3f} m · {mean:.0f} ms ({cores})', (mean, ate),
                    xytext=(8 if side == 'right' else -8, 7), textcoords='offset points',
                    ha='left' if side == 'right' else 'right', fontsize=7.5, color=ps.TEXT)
    y_lo, y_hi = 0.6 * min(ates.values()), 1.15 * max(ates.values())
    ax.set_ylim(y_lo, y_hi)
    ax.axvline(50, color=ps.NEUTRAL, lw=1, ls='--')
    ax.annotate('50 ms = 20 Hz camera budget', (50, y_lo), xytext=(4, 4),
                textcoords='offset points', fontsize=7, color=ps.TEXT)
    ax.set_xscale('log')
    ax.set_xlim(5, 400)
    ax.set_xticks([5, 10, 20, 50, 100, 200], ['5', '10', '20', '50', '100', '200'])
    ax.set_xlabel('latency per frame [ms]  (marker = mean, bar = p50 to p95)', color=ps.TEXT)
    ax.set_ylabel(f'mean ATE, {n_common} EuRoC sequences [m]', color=ps.TEXT)
    ax.set_title('Accuracy vs latency (lower-left is better)', color=ps.TITLE, fontsize=10)
    ps.style_axes(ax)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig('results/final/accuracy_vs_latency.png', facecolor=ps.SURFACE)
    print('wrote results/final/accuracy_vs_latency.png')


if __name__ == '__main__':
    main()
