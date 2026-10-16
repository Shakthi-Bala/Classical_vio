"""
Run the Python S-MSCKF (../../Code) deterministically, without threads or the
real-time DataPublisher, so the C++ port has an exact reference to match.

Outputs (in --out):
  traj_python.txt     TUM trajectory: t tx ty tz qx qy qz qw (body in world, Hamilton)
  timing_python.csv   per-frame front-end / back-end wall time [ms]
  features_python.txt every feature message the front end produced:
                        F <timestamp> <n>
                        <id> <u0> <v0> <u1> <v1>   (n lines, undistorted normalized coords)
                      The C++ back end can replay this file (main_euroc --features)
                      to test the filter in isolation from the front end.

--features FILE replays a feature file instead of running the Python front end
(back end only). Use it for the back-end parity test against
`main_euroc --features FILE`: both filters then get bit-identical input. (A live
run is not bit-identical to its own dump: when a frame has no new cam0 features,
image.py's observations are float32 arrays, and numpy 1.x keeps
float32_array * float64_scalar in float32 inside feature.py. The text dump is
float64.)

Event order (same as the C++ main): before each stereo frame, every IMU message
with timestamp <= frame timestamp is fed to both the image processor and the
filter.

Usage (from the repo root, in the `cv` env):
  python cpp/scripts/run_python_reference.py --path /path/to/MH_01_easy --out results/python
"""
import argparse
import os
import sys
import time
from collections import namedtuple

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Code'))

import image as image_module
import msckf as msckf_module
from config import ConfigEuRoC
from dataset import EuRoCDataset
from image import ImageProcessor
from msckf import MSCKF


def _quiet(*args, **kwargs):
    pass


FeatureMeasurement = namedtuple('FeatureMeasurement', ['id', 'u0', 'v0', 'u1', 'v1'])
FeatureMsg = namedtuple('FeatureMsg', ['timestamp', 'features'])


def read_feature_file(path):
    msgs = []
    with open(path) as f:
        lines = f.read().split('\n')
    i = 0
    while i < len(lines) and lines[i]:
        _, t, n = lines[i].split()
        n = int(n)
        feats = []
        for line in lines[i + 1:i + 1 + n]:
            p = line.split()
            feats.append(FeatureMeasurement(int(p[0]), *map(float, p[1:])))
        msgs.append(FeatureMsg(float(t), feats))
        i += 1 + n
    return msgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='EuRoC sequence folder (contains mav0/)')
    parser.add_argument('--out', default='results/python')
    parser.add_argument('--offset', type=float, default=40.0, help='start offset [s], same as vio.py')
    parser.add_argument('--max-frames', type=int, default=-1)
    parser.add_argument('--features', default=None,
                        help='replay this feature file (back end only)')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # The Python modules print on every frame. Silence them so the timing is fair.
    image_module.print = _quiet
    msckf_module.print = _quiet

    config = ConfigEuRoC()
    image_processor = ImageProcessor(config)
    image_processor.draw_features_stereo = lambda: None   # visualization only
    msckf = MSCKF(config)

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=args.offset)

    imu_iter = iter(dataset.imu)
    next_imu = next(imu_iter, None)

    replay = args.features is not None
    frames = read_feature_file(args.features) if replay else dataset.stereo

    traj = open(os.path.join(args.out, 'traj_python.txt'), 'w')
    timing = open(os.path.join(args.out, 'timing_python.csv'), 'w')
    feats = open(os.devnull if replay else os.path.join(args.out, 'features_python.txt'), 'w')
    timing.write('timestamp,frontend_ms,backend_ms\n')

    n_frames = 0
    wall_start = time.perf_counter()
    first_t, last_t = None, None
    for frame in frames:
        t_img = frame.timestamp
        while next_imu is not None and next_imu.timestamp <= t_img:
            image_processor.imu_callback(next_imu)
            msckf.imu_callback(next_imu)
            next_imu = next(imu_iter, None)

        t0 = time.perf_counter()
        if replay:
            feature_msg = frame
        else:
            feature_msg, _ = image_processor.stareo_callback(frame)
        t1 = time.perf_counter()
        result = msckf.feature_callback(feature_msg)
        t2 = time.perf_counter()

        feats.write(f'F {feature_msg.timestamp:.9f} {len(feature_msg.features)}\n')
        for f in feature_msg.features:
            feats.write(f'{f.id} {f.u0:.17g} {f.v0:.17g} {f.u1:.17g} {f.v1:.17g}\n')

        timing.write(f'{t_img:.9f},{(t1 - t0) * 1e3:.4f},{(t2 - t1) * 1e3:.4f}\n')
        if result is not None:
            p = result.pose.t
            q = Rotation.from_matrix(result.pose.R).as_quat()   # x y z w
            traj.write(f'{result.timestamp:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} '
                       f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')

        first_t = t_img if first_t is None else first_t
        last_t = t_img
        n_frames += 1
        if n_frames % 200 == 0:
            print(f'frame {n_frames}  t={t_img:.3f}', flush=True)
        if 0 < args.max_frames <= n_frames:
            break

    wall = time.perf_counter() - wall_start
    for f in (traj, timing, feats):
        f.close()
    data_seconds = last_t - first_t
    print(f'frames: {n_frames}  dataset: {data_seconds:.2f} s  wall: {wall:.2f} s  '
          f'real-time factor: {data_seconds / wall:.3f}')


if __name__ == '__main__':
    main()
