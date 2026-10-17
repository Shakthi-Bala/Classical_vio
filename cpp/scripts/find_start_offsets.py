"""
Pick a start offset for each sequence: both back ends average their first 200
IMU samples (1 s) for gravity and gyro bias, so they must start while the
sensor is still. Rule: the quietest 1 s IMU window (lowest acc + gyro standard
deviation) in the 12 s before ground truth first moves --moved metres, ending
at least 0.5 s before that.

EuRoC (a MAV on the ground, then take-off): --moved 1.0, the default. For MH_01
the rule gives 40.3 s; we keep the 40 s that vio.py used.
TUM-VI (handheld, starts moving slowly): --moved 0.1.

Usage: python scripts/find_start_offsets.py [--moved M] SEQ_DIR... > config/X_offsets.csv
"""
import argparse
import os

import numpy as np

import vio_eval as ve

KEEP = {'MH_01_easy': 40.0}   # offsets fixed by earlier runs
WINDOW = 200                   # IMU samples = 1 s at 200 Hz (the filters' static init)
STEP = 20                      # candidate windows every 0.1 s
# A window's score: accelerometer and gyro standard deviation, each relative to
# a typical value on the ground with motors running (so both count about equally).
ACC_STD_SCALE = 0.1            # [m/s^2]
GYRO_STD_SCALE = 0.02          # [rad/s]


def start_offset(seq_dir, moved_m, lookback=12.0):
    imu = np.loadtxt(os.path.join(seq_dir, 'mav0/imu0/data.csv'), delimiter=',', comments='#')
    t = imu[:, 0] * 1e-9
    gyro, acc = imu[:, 1:4], imu[:, 4:7]
    t_gt, p_gt, _ = ve.load_gt(seq_dir)
    moved = t_gt[np.argmax(np.linalg.norm(p_gt - p_gt[0], axis=1) > moved_m)]

    best = None
    for i in range(0, len(t) - WINDOW, STEP):
        if t[i] < max(moved - lookback, t_gt[0] + 0.5) or t[i] + 1.0 > moved - 0.5:
            continue
        score = (np.linalg.norm(acc[i:i + WINDOW].std(0)) / ACC_STD_SCALE +
                 np.linalg.norm(gyro[i:i + WINDOW].std(0)) / GYRO_STD_SCALE)
        if best is None or score < best[0]:
            best = (score, t[i] - t[0])
    if best is None:
        raise ValueError(f'{seq_dir}: no 1 s IMU window in the {lookback} s before take-off')
    return round(best[1], 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moved', type=float, default=1.0,
                        help='ground-truth displacement that counts as moving [m]')
    parser.add_argument('seq_dirs', nargs='+')
    args = parser.parse_args()
    print('sequence,offset_s,path')
    for seq_dir in args.seq_dirs:
        name = os.path.basename(os.path.normpath(seq_dir))
        offset = KEEP[name] if name in KEEP else start_offset(seq_dir, args.moved)
        print(f'{name},{offset},{os.path.abspath(seq_dir)}')
