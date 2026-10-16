"""
Dot plot of the quirk ablation: ATE of each variant relative to the C++
baseline (= Python behaviour), one dot per EuRoC sequence.

Usage (from cpp/, after ablation_table.py): python scripts/plot_ablation.py
"""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DOT = '#2a78d6'
FAIL = '#d03b3b'
TEXT = '#52514e'
LABELS = {
    'epipolar_full_dot': '1  full epipolar dot product',
    'lm_upstream_loops': '2  LM iterates to convergence',
    'chi2_95': '3  chi-squared gate at 95 %',
    'oc_upstream': '4  upstream OC-EKF constraint',
    'normalize_quaternion': '5  re-normalize quaternion',
    'mask_clamp': '6  mask clamp at border',
    'full_precision_publish': '8  double-precision publish',
    'all_but_chi2': 'all fixes except 3',
    'all': 'all fixes',
}
CAP = 4.0   # ratios above this (and divergence) are drawn at the cap


def main():
    rows = list(csv.DictReader(open('results/ablation/ate_table.csv')))
    variants = list(LABELS)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for i, v in enumerate(variants):
        ratios, fails = [], 0
        for r in rows:
            if not r['baseline']:
                continue
            base = float(r['baseline'])
            if not r[v] or float(r[v]) > 5.0:
                fails += 1
                continue
            ratios.append(float(r[v]) / base)
        jitter = np.linspace(-0.18, 0.18, max(len(ratios), 1))
        ax.scatter(np.minimum(ratios, CAP), i + jitter[:len(ratios)], s=22, color=DOT,
                   edgecolor='#fcfcfb', linewidth=0.8, zorder=3)
        if ratios:
            med = float(np.median(ratios))
            ax.plot([med, med], [i - 0.3, i + 0.3], color='#0b0b0b', lw=2, zorder=4)
        if fails:
            ax.scatter([CAP * 1.15] * fails, i + np.linspace(-0.15, 0.15, fails), marker='x',
                       s=26, color=FAIL, zorder=3)
            ax.annotate(f'{fails} diverged', (CAP * 1.22, i), va='center', fontsize=7, color=TEXT)
    ax.axvline(1.0, color='#8a8984', lw=1, ls='--')
    ax.set_xscale('log')
    ax.set_xlim(0.15, CAP * 2.2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ['0.25x', '0.5x', '1x', '2x', '4x'])
    ax.set_yticks(range(len(variants)), [LABELS[v] for v in variants])
    ax.invert_yaxis()
    ax.set_xlabel('ATE relative to the Python-faithful C++ MSCKF (lower is better)', color=TEXT)
    ax.set_title('Python-quirk fixes on 11 EuRoC sequences (dot = sequence, bar = median)',
                 color='#0b0b0b', fontsize=10)
    ax.grid(axis='x', color='#e6e5e1', lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color('#d4d3cf')
    ax.tick_params(colors=TEXT, labelsize=8)
    fig.tight_layout()
    fig.savefig('results/ablation/ablation.png', facecolor='#fcfcfb')
    print('wrote results/ablation/ablation.png')


if __name__ == '__main__':
    main()
