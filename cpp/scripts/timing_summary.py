"""
Per-frame latency over repeated timing runs (scripts/run_timing_job.sh into
results/timing and results/timing_repeats/run*): mean and standard deviation
across runs of each run's mean, p95 and real-time factor.

Usage (from cpp/): python scripts/timing_summary.py
Writes results/timing_summary.md.
"""
import glob

import numpy as np

RUNS = ['results/timing'] + sorted(glob.glob('results/timing_repeats/run*'))
METHODS = [
    ('Python MSCKF', 'python/timing_python.csv'),
    ('C++ MSCKF, Python behaviour', 'cpp_python_behaviour/timing_cpp.csv'),
    ('C++ MSCKF', 'cpp_msckf/timing_cpp.csv'),
    ('C++ sliding window', 'cpp_swo/timing_cpp.csv'),
]


def run_stats(path):
    d = np.genfromtxt(path, delimiter=',', names=True)
    total = d['frontend_ms'] + d['backend_ms']
    rtf = (d['timestamp'][-1] - d['timestamp'][0]) / (total.sum() / 1e3)
    return total.mean(), np.percentile(total, 95), rtf


def main():
    node = open(f'{RUNS[0]}/node.txt').read().split('.')[0].strip()
    lines = [f'Per-frame latency on MH_01, {len(RUNS)} runs on {node}, one core '
             '(mean ± standard deviation across runs).', '',
             '| | mean [ms] | p95 [ms] | real-time factor |', '|---|---|---|---|']
    for name, rel in METHODS:
        s = np.array([run_stats(f'{run}/{rel}') for run in RUNS])
        m, sd = s.mean(0), s.std(0, ddof=1)
        lines.append(f'| {name} | {m[0]:.2f} ± {sd[0]:.2f} | {m[1]:.2f} ± {sd[1]:.2f} '
                     f'| {m[2]:.2f} ± {sd[2]:.2f} |')
    with open('results/timing_summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
