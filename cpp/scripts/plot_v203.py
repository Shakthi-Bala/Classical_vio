"""
Why the stereo-only front end loses V2_03_difficult: features per frame with
and without cam0-only (mono) features, the stretches where cam0 runs at 10 Hz
(long exposure), and one stereo pair from such a stretch.

Usage (from cpp/):
  python scripts/plot_v203.py --seq-path /path/to/V2_03_difficult \
      --stereo RUN_DIR --mono RUN_DIR [--t-pair 65]
RUN_DIR: main_euroc --dump-features output with config/euroc_fixed.yaml
(--stereo) and config/euroc_mono.yaml (--mono). Writes
results/final/v203_tracking.png.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import plotstyle as ps
import vio_eval as ve

CAM0_GAP_S = 0.06   # cam0 frame gap that means a dropped frame (20 Hz -> 10 Hz)


def counts(run_dir):
    """(t, stereo count, cam0-only count) per frame."""
    frames = ve.read_features(os.path.join(run_dir, 'features_cpp.txt'))
    t = np.array([f[0] for f in frames])
    stereo = np.array([sum(np.isfinite(z[2]) for z in f[1].values()) for f in frames])
    total = np.array([len(f[1]) for f in frames])
    return t, stereo, total - stereo


def slow_spans(seq_path, t0, t1):
    """[start, end] intervals in which cam0 skips frames."""
    names = sorted(os.listdir(os.path.join(seq_path, 'mav0/cam0/data')))
    ts = np.array([int(n[:-4]) for n in names]) * 1e-9
    slow = np.diff(ts) > CAM0_GAP_S
    spans, start = [], None
    for i, s in enumerate(slow):
        if s and start is None:
            start = ts[i]
        elif not s and start is not None:
            spans.append((start, ts[i]))
            start = None
    return [(a, b) for a, b in spans if b > t0 and a < t1 and b - a > 0.5], names, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-path', required=True)
    ap.add_argument('--stereo', required=True)
    ap.add_argument('--mono', required=True)
    ap.add_argument('--t-pair', type=float, default=65.0, help='stereo pair time [s from start]')
    args = ap.parse_args()

    t_s, n_s, _ = counts(args.stereo)
    t_m, n_ms, n_mm = counts(args.mono)
    t0 = t_s[0]
    spans, names, ts = slow_spans(args.seq_path, t0, t_s[-1])

    fig = plt.figure(figsize=(9.5, 5.6), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.04)
    ax = fig.add_subplot(grid[0, :])
    for a, b in spans:
        ax.axvspan(a - t0, b - t0, color=ps.GRID, lw=0)
    ax.plot(t_s - t0, n_s, color=ps.NEUTRAL, lw=1.0, label='stereo only (msckf_vio / Python front end)')
    ax.plot(t_m - t0, n_ms + n_mm, color=ps.CPP_SWO, lw=1.0,
            label='with cam0-only features (mono_features)')
    ax.set_xlim(0, t_s[-1] - t0)
    ax.set_ylim(0, None)
    ax.set_xlabel('time [s]', color=ps.TEXT, fontsize=8)
    ax.set_ylabel('features per frame', color=ps.TEXT, fontsize=8)
    ax.set_title('EuRoC V2_03_difficult: shaded = cam0 at 10 Hz (long exposure, motion blur)',
                 color=ps.TITLE, fontsize=9, pad=20)
    ps.style_axes(ax, grid='y')
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, labelcolor=ps.TEXT, fontsize=7, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2)

    i = int(np.searchsorted(ts, t0 + args.t_pair))
    for k, cam in enumerate(['cam0', 'cam1']):
        img = Image.open(os.path.join(args.seq_path, 'mav0', cam, 'data', names[i]))
        a = fig.add_subplot(grid[1, k])
        a.imshow(np.asarray(img), cmap='gray', vmin=0, vmax=255)
        a.set_axis_off()
        a.set_title(f'{cam} at t = {ts[i] - t0:.1f} s' +
                    ('  (blurred)' if cam == 'cam0' else '  (sharp)'), color=ps.TEXT, fontsize=8)
    fig.savefig('results/final/v203_tracking.png', facecolor=ps.SURFACE, bbox_inches='tight')
    print('wrote results/final/v203_tracking.png')


if __name__ == '__main__':
    main()
