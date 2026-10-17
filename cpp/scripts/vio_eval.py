"""
Shared evaluation helpers for the scripts in this folder: ground truth and
trajectory loading, SE3 alignment, ATE / RPE (via Code/evaluate.py, so every
number uses the same code as the original Python report), feature files, and
the list of EuRoC sequences with their start offsets.

Paths are relative to cpp/, where all scripts are run from.
"""
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Code'))
from evaluate import compute_rpe, interpolate_gt, umeyama_alignment  # noqa: E402

DIVERGED_M = 5.0          # an ATE above this counts as a failure
RPE_LENGTH_M = 10.0       # segment length of the reported RPE
OFFSETS_CSV = 'config/euroc_offsets.csv'


def load_gt(seq_path):
    """(t [s], position [N,3], quaternion w x y z [N,4]) of a EuRoC sequence,
    or of a TUM-VI one in EuRoC format (motion capture in mav0/mocap0)."""
    path = os.path.join(seq_path, 'mav0/state_groundtruth_estimate0/data.csv')
    if not os.path.exists(path):
        path = os.path.join(seq_path, 'mav0/mocap0/data.csv')
    d = np.loadtxt(path, delimiter=',', comments='#')
    return d[:, 0] * 1e-9, d[:, 1:4], d[:, 4:8]


def load_tum(path, gt=None, t_min=None):
    """TUM trajectory (t, xyz, quaternion x y z w), optionally cropped to
    t >= t_min and to the ground-truth time span. None if missing or empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    d = np.atleast_2d(np.loadtxt(path))
    if t_min is not None:
        d = d[d[:, 0] >= t_min]
    if gt is not None:
        d = d[(d[:, 0] >= gt[0][0]) & (d[:, 0] <= gt[0][-1])]
    return d if len(d) >= 10 else None


def align(traj, gt):
    """SE3-aligned positions [N,3] and the ground truth at the same times."""
    gt_pos, _ = interpolate_gt(*gt, traj[:, 0])
    R, t, _ = umeyama_alignment(traj[:, 1:4], gt_pos)
    return (R @ traj[:, 1:4].T).T + t, gt_pos


def ate(gt, path, t_min=None):
    """ATE RMSE [m] after SE3 Umeyama alignment, or None."""
    traj = load_tum(path, gt, t_min)
    if traj is None:
        return None
    p, g = align(traj, gt)
    return float(np.sqrt(np.mean(np.sum((p - g) ** 2, axis=1))))


def rpe(gt, path, t_min=None, length=RPE_LENGTH_M):
    """Mean RPE over all segments of `length` m: (translation %, rotation deg)."""
    traj = load_tum(path, gt, t_min)
    if traj is None:
        return None
    gt_pos, gt_rot = interpolate_gt(*gt, traj[:, 0])
    est_rot = Rotation.from_quat(traj[:, 4:8]).as_matrix()
    out = compute_rpe(gt_pos, gt_rot, traj[:, 1:4], est_rot, [length])[length]
    if len(out['trans']) == 0:
        return None
    return float(np.mean(out['trans'])), float(np.mean(out['rot']))


def failed(value):
    return value is None or value > DIVERGED_M


def read_features(path, max_frames=None):
    """Feature dump ("F <t> <n>" + n lines "<id> u0 v0 u1 v1") ->
    [(t, {id: array([u0, v0, u1, v1])}), ...]."""
    frames = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if p[0] == 'F':
                if max_frames is not None and len(frames) == max_frames:
                    break
                frames.append((float(p[1]), {}))
            else:
                frames[-1][1][int(p[0])] = np.array(p[1:], dtype=float)
    return frames


def sequences():
    """[{'sequence', 'path', 'offset', 't_min'}] from config/euroc_offsets.csv;
    t_min = first IMU stamp + offset, the start of every run."""
    out = []
    with open(OFFSETS_CSV) as f:
        for row in csv.DictReader(f):
            with open(os.path.join(row['path'], 'mav0/imu0/data.csv')) as imu:
                next(imu)   # header
                t0 = float(next(imu).split(',')[0]) * 1e-9
            out.append({'sequence': row['sequence'], 'path': row['path'],
                        'offset': float(row['offset_s']), 't_min': t0 + float(row['offset_s'])})
    return out


def python_trajectory(seq):
    """Path of the Python reference trajectory, or None if that run crashed
    (a crashed run leaves a partial trajectory that must not be scored)."""
    run_dir = 'results/python' if seq == 'MH_01_easy' else f'results/sequences/{seq}/python'
    log = 'results/python/run.log' if seq == 'MH_01_easy' else f'results/sequences/{seq}/python.log'
    if os.path.exists(log) and 'Traceback' in open(log).read():
        return None
    return f'{run_dir}/traj_python.txt'
