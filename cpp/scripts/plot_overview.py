"""
Accuracy vs latency overview: mean ATE over the EuRoC sequences every method
completes, against end-to-end latency per frame on MH_01.

Latency: real-time ROS 2 bag playback on the same compute node (data in ->
pose out), except the Python MSCKF, which has no ROS node (offline per-frame
time). Bars span p50 -> p95, the marker is the mean.

Usage (from cpp/): python scripts/plot_overview.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TEXT, GRID, SPINE = '#52514e', '#e6e5e1', '#d4d3cf'


def latency(path, col):
    d = np.genfromtxt(path, delimiter=',', names=True)
    x = d[col]
    return x.mean(), np.median(x), np.percentile(x, 95)


def mean_ate(column):
    """'mean, N seq. all completed' row of the ATE table in results/final/summary.md."""
    lines = open('results/final/summary.md').read().splitlines()
    header = [c.strip() for c in lines[0].strip('|').split('|')]
    for line in lines:
        if line.startswith('| mean,'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            return float(cells[header.index(column)])
    raise ValueError(column)


def main():
    # Python: offline per-frame time (front end + back end), it has no ROS node.
    py = np.genfromtxt('results/python/timing_python.csv', delimiter=',', names=True)
    py_total = py['frontend_ms'] + py['backend_ms']
    py_lat = (py_total.mean(), np.median(py_total), np.percentile(py_total, 95))
    points = [
        # label, ATE column, latency (mean, p50, p95), color, marker, cores, label side
        ('Python MSCKF (offline, no ROS)', 'Python', py_lat, '#2a78d6', 'o', '1 core', 'left'),
        ('C++ MSCKF', 'MSCKF fixed',
         latency('results/ros2/MH_01_easy_msckf_c303/timing_ros.csv', 'end_to_end_ms'),
         '#eb6834', 's', '1 core', 'right'),
        ('C++ sliding window', 'SWO',
         latency('results/ros2/MH_01_easy_swo_c303/timing_ros.csv', 'end_to_end_ms'),
         '#1baf7a', 'D', '1 core', 'right'),
        ('VINS-Fusion', 'VINS-Fusion',
         latency('results/vins/latency_MH_01_easy.csv', 'end_to_end_ms'),
         '#eda100', '^', '4 cores', 'right'),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for label, col, (mean, p50, p95), color, marker, cores, side in points:
        ate = mean_ate(col)
        ax.plot([p50, p95], [ate, ate], color=color, lw=2.2, alpha=0.45, solid_capstyle='round')
        ax.scatter([mean], [ate], s=90, color=color, marker=marker, edgecolor='#fcfcfb',
                   linewidth=1.2, zorder=3)
        ax.annotate(f'{label}\n{ate:.3f} m · {mean:.0f} ms ({cores})', (mean, ate),
                    xytext=(8 if side == 'right' else -8, 7), textcoords='offset points',
                    ha='left' if side == 'right' else 'right', fontsize=7.5, color=TEXT)
    ax.axvline(50, color='#8a8984', lw=1, ls='--')
    ax.annotate('50 ms = 20 Hz camera budget', (50, 0.105), xytext=(4, 0),
                textcoords='offset points', fontsize=7, color=TEXT, va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(5, 400)
    ax.set_xticks([5, 10, 20, 50, 100, 200], ['5', '10', '20', '50', '100', '200'])
    ax.set_ylim(0.1, 0.45)
    ax.set_xlabel('latency per frame [ms]  (marker = mean, bar = p50 to p95)', color=TEXT)
    ax.set_ylabel('mean ATE, 10 EuRoC sequences [m]', color=TEXT)
    ax.set_title('Accuracy vs latency (lower-left is better)', color='#0b0b0b', fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.tick_params(colors=TEXT, labelsize=8)
    fig.tight_layout()
    fig.savefig('results/final/accuracy_vs_latency.png', facecolor='#fcfcfb')
    print('wrote results/final/accuracy_vs_latency.png')


if __name__ == '__main__':
    main()
