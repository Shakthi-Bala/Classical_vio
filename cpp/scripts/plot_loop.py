"""
Top view of one sequence: ground truth, the VIO trajectory and the same run
corrected with the final loop-closure pose graph (SE3-aligned to ground truth).

Usage (from cpp/): python scripts/plot_loop.py [--seq MH_05_difficult] [--backend msckf_mono_prune]
Writes results/final/loop_<SEQ>.png.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps
import vio_eval as ve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='MH_05_difficult')
    ap.add_argument('--backend', default='msckf_mono_prune')
    args = ap.parse_args()

    seq_path = next(s['path'] for s in ve.sequences() if s['sequence'] == args.seq)
    gt = ve.load_gt(seq_path)
    run = f'results/backends/{args.backend}_loop/{args.seq}'
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    for i, (name, label, color) in enumerate([
            ('traj_cpp.txt', 'VIO (C++ MSCKF)', ps.CPP_MSCKF),
            ('traj_loop_final.txt', 'with loop closure', ps.CPP_SWO)]):
        p, g = ve.align(ve.load_tum(f'{run}/{name}', gt), gt)
        if i == 0:
            ax.plot(g[:, 0], g[:, 1], color=ps.NEUTRAL, lw=2, ls='--', label='ground truth')
        ate = np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1)))
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.6, label=f'{label}  (ATE {ate:.2f} m)')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', color=ps.TEXT)
    ax.set_ylabel('y [m]', color=ps.TEXT)
    ax.set_title(f'{args.seq}: loop closure (4-DoF pose graph)', color=ps.TITLE, fontsize=10)
    ps.style_axes(ax)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, labelcolor=ps.TEXT, fontsize=8)
    fig.tight_layout()
    out = f'results/final/loop_{args.seq}.png'
    fig.savefig(out, facecolor=ps.SURFACE)
    print('wrote', out)


if __name__ == '__main__':
    main()
