"""
Render the README demo GIF: the C++ VIO running on a EuRoC sequence.

  left:  cam0 image with the front end's feature tracks (colour = track age)
  right: top view of the trajectory as it is estimated: ground truth, the
         sliding-window optimizer and the MSCKF (both SE3-aligned to ground truth)

Usage (from cpp/, in the `cv` env; gifsicle on PATH for the final compression).
The README GIF was made with:
  python scripts/make_demo_gif.py --seq-path /path/to/MH_05_difficult \
      --run-dir results/demo/MH_05_difficult \
      --msckf results/backends/msckf/MH_05_difficult/traj_cpp.txt \
      --out results/final/demo_mh05.gif --step 12 --fps 11 --width 800 --lossy 110
(every 12th frame at 11 fps: about 6x real time). `run-dir` holds a
sliding-window run with --dump-features (features_cpp.txt, traj_cpp.txt,
timing_cpp.csv).
"""
import argparse
import bisect
import io
import os
import subprocess

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from PIL import Image

import vio_eval as ve

# Dark theme, validated dark-mode palette steps (dataviz reference palette).
SURFACE = '#1a1a19'
TEXT = '#ffffff'
TEXT2 = '#c3c2b7'
GRID = '#2c2c2a'
GT = '#8a8984'
SWO = '#199e70'      # aqua
MSCKF = '#d95926'    # orange
# Track age: one-hue blue ramp, young -> old (light = long-lived). Starts at the
# mid step so young tracks stay visible on a dark image.
AGE_RAMP = ['#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#cde2fb']


def to_pixels(norm_pts, K, D):
    """Normalized undistorted -> distorted pixel coordinates (radtan)."""
    if len(norm_pts) == 0:
        return np.zeros((0, 2))
    pts = np.hstack([np.asarray(norm_pts), np.ones((len(norm_pts), 1))]).reshape(-1, 1, 3)
    px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    return px.reshape(-1, 2)


def aligned(traj_path, gt):
    """(timestamps, SE3-aligned positions, ATE) of a trajectory."""
    traj = ve.load_tum(traj_path, gt)
    p, g = ve.align(traj, gt)
    return traj[:, 0], p, float(np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-path', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--msckf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--step', type=int, default=12, help='render every Nth frame')
    ap.add_argument('--fps', type=float, default=11.0)
    ap.add_argument('--tail', type=int, default=12, help='track history drawn [frames]')
    ap.add_argument('--width', type=int, default=800)
    ap.add_argument('--lossy', type=int, default=110, help='gifsicle --lossy level')
    ap.add_argument('--colors', type=int, default=128, help='GIF palette size')
    ap.add_argument('--gray-levels', type=int, default=32,
                    help='posterize the camera image (fewer levels = smaller GIF)')
    args = ap.parse_args()

    seq = os.path.basename(os.path.normpath(args.seq_path))
    mav0 = os.path.join(args.seq_path, 'mav0')
    cam = yaml.safe_load(open(os.path.join(mav0, 'cam0/sensor.yaml')))
    fx, fy, cx, cy = cam['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.array(cam['distortion_coefficients'])
    image_ns = sorted(int(f[:-4]) for f in os.listdir(os.path.join(mav0, 'cam0/data'))
                      if f.endswith('.png'))

    gt = ve.load_gt(args.seq_path)
    swo_t, swo_p, swo_ate = aligned(os.path.join(args.run_dir, 'traj_cpp.txt'), gt)
    msckf_t, msckf_p, msckf_ate = aligned(args.msckf, gt)
    gt_span = (gt[0] >= swo_t[0]) & (gt[0] <= swo_t[-1])
    gt_xy = gt[1][gt_span]

    timing = np.genfromtxt(os.path.join(args.run_dir, 'timing_cpp.csv'), delimiter=',', names=True)
    total_ms = dict(zip(np.round(timing['timestamp'], 6), timing['frontend_ms'] + timing['backend_ms']))

    frames = [(t, {i: z[:2] for i, z in feats.items()})   # cam0 (u, v) only
              for t, feats in ve.read_features(os.path.join(args.run_dir, 'features_cpp.txt'))]
    # Pixel history of every feature id: {id: [(frame_index, x, y), ...]}
    history = {}
    for k, (t, feats) in enumerate(frames):
        ids = list(feats)
        px = to_pixels([feats[i] for i in ids], K, D)
        for i, (x, y) in zip(ids, px):
            history.setdefault(i, []).append((k, x, y))

    # Figure: image 752x480 on the left, trajectory on the right.
    dpi = 100
    W = args.width / dpi
    H = W * 0.47
    fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor=SURFACE)
    ax_img = fig.add_axes([0.012, 0.03, 0.56, 0.78])
    ax_tr = fig.add_axes([0.64, 0.17, 0.34, 0.64])
    fig.text(0.012, 0.93, 'Stereo visual-inertial odometry in C++17', color=TEXT, fontsize=13,
             weight='bold', va='center')
    fig.text(0.012, 0.86, f'EuRoC {seq}  ·  live feature tracks (colour = track age)  ·  '
             f'MSCKF filter vs Ceres sliding-window optimizer', color=TEXT2, fontsize=8.5,
             va='center')
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))   # display only

    xmin, ymin = gt_xy[:, :2].min(0) - 1.0
    xmax, ymax = gt_xy[:, :2].max(0) + 1.0

    pil_frames = []
    render_idx = list(range(0, len(frames), args.step))
    for n, k in enumerate(render_idx):
        t, feats = frames[k]
        j = bisect.bisect_left(image_ns, round(t * 1e9))
        ns = min(image_ns[max(j - 1, 0):j + 1], key=lambda s: abs(s * 1e-9 - t))
        image_path = os.path.join(mav0, 'cam0/data', f'{ns}.png')
        img = clahe.apply(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        level = 256 // args.gray_levels
        img = (img // level) * level + level // 2   # posterize: compresses far better

        ax_img.clear()
        ax_img.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax_img.set_xlim(0, img.shape[1])
        ax_img.set_ylim(img.shape[0], 0)
        ax_img.axis('off')
        segs, colors = [], []
        heads = []
        for fid in feats:
            h = [e for e in history[fid] if k - args.tail <= e[0] <= k]
            age = sum(1 for e in history[fid] if e[0] <= k)
            color = AGE_RAMP[min(age // 6, len(AGE_RAMP) - 1)]
            if len(h) >= 2:
                pts = np.array([[x, y] for _, x, y in h])
                segs.append(pts)
                colors.append(color)
            heads.append((h[-1][1], h[-1][2], color))
        ax_img.add_collection(LineCollection(segs, colors=colors, linewidths=1.4))
        if heads:
            hx, hy, hc = zip(*heads)
            ax_img.scatter(hx, hy, s=10, c=hc, edgecolors='none')
        recent = [total_ms.get(round(frames[j][0], 6), np.nan) for j in range(max(0, k - 40), k + 1)]
        ax_img.text(8, 24, f't = {t - frames[0][0]:5.1f} s    {len(feats):3d} features    '
                           f'{np.nanmean(recent):4.1f} ms/frame (one core)', color=TEXT,
                    fontsize=8, bbox=dict(facecolor=SURFACE, alpha=0.8, edgecolor='none', pad=3))

        ax_tr.clear()
        ax_tr.set_facecolor(SURFACE)
        ax_tr.plot(gt_xy[:, 0], gt_xy[:, 1], color=GT, lw=1.0, ls='--', alpha=0.45)
        for tt, pp, color, label in [(msckf_t, msckf_p, MSCKF, f'MSCKF  {msckf_ate:.2f} m'),
                                     (swo_t, swo_p, SWO, f'sliding window  {swo_ate:.2f} m')]:
            m = tt <= t
            if m.sum() > 1:
                ax_tr.plot(pp[m, 0], pp[m, 1], color=color, lw=1.8, label=label)
                ax_tr.scatter(pp[m][-1, 0], pp[m][-1, 1], s=28, color=color,
                              edgecolors=SURFACE, linewidths=1.2, zorder=4)
            else:
                ax_tr.plot([], [], color=color, lw=1.8, label=label)
        ax_tr.plot([], [], color=GT, lw=1.0, ls='--', label='truth')
        ax_tr.text(0.98, 0.03, 'ATE = final RMSE', transform=ax_tr.transAxes, ha='right',
                   color=TEXT2, fontsize=6.5)
        ax_tr.set_xlim(xmin, xmax)
        ax_tr.set_ylim(ymin, ymax)
        ax_tr.set_aspect('equal')
        ax_tr.tick_params(colors=TEXT2, labelsize=7)
        for s in ax_tr.spines.values():
            s.set_color(GRID)
        ax_tr.grid(color=GRID, lw=0.6)
        ax_tr.set_xlabel('x [m]', color=TEXT2, fontsize=8)
        ax_tr.set_ylabel('y [m]', color=TEXT2, fontsize=8)
        ax_tr.set_title('top view, aligned to ground truth', color=TEXT2, fontsize=8, pad=4)
        ax_tr.legend(loc='upper center', bbox_to_anchor=(0.36, -0.15), frameon=False,
                     fontsize=7, labelcolor=TEXT2, ncol=3, handlelength=1.3, columnspacing=0.8)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=SURFACE, dpi=dpi)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert('RGB'))
        if n % 50 == 0:
            print(f'frame {n}/{len(render_idx)}', flush=True)

    # Hold the last frame (final trajectories) for 2.5 s.
    pil_frames += [pil_frames[-1]] * int(2.5 * args.fps)
    # One shared palette (from a frame with all three trajectories) avoids flicker.
    palette_src = pil_frames[-1].quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=palette_src, dither=Image.Dither.NONE) for f in pil_frames]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stem = os.path.splitext(args.out)[0]
    raw = stem + '_raw.gif'
    q[0].save(raw, save_all=True, append_images=q[1:], duration=int(1000 / args.fps), loop=0,
              optimize=True)
    still = stem + '_final.png'
    pil_frames[-1].save(still)
    # Lossy LZW + frame optimization with gifsicle when available.
    try:
        subprocess.run(['gifsicle', '-O3', f'--lossy={args.lossy}', raw, '-o', args.out],
                       check=True)
        os.remove(raw)
    except (OSError, subprocess.CalledProcessError):
        os.replace(raw, args.out)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {len(q)} frames) and {still}')


if __name__ == '__main__':
    main()
