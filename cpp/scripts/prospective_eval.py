"""Frozen TUM-VI follow-up: explicit partial-GT scoring, no estimator tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

import numpy as np

CPP = Path(__file__).resolve().parents[1]
SEQUENCES = ('corridor2', 'slides1')
CONFIGS = ('tumvi_mono_prune.yaml', 'tumvi_mono.yaml', 'euroc_mono.yaml',
           'euroc_fixed.yaml', 'euroc.yaml')
BACKENDS = {'msckf': 'tumvi_mono_prune.yaml', 'swo': 'tumvi_mono.yaml'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def associate(trajectory, segment, tolerance=.02):
    stamps = trajectory[:, 0]
    indices = np.searchsorted(segment[:, 0], stamps)
    right = np.clip(indices, 0, len(segment) - 1)
    left = np.clip(indices - 1, 0, len(segment) - 1)
    distance = np.minimum(abs(stamps - segment[left, 0]), abs(stamps - segment[right, 0]))
    valid = ((stamps >= segment[0, 0]) & (stamps <= segment[-1, 0])
             & (distance <= tolerance))
    selected = trajectory[valid]
    target = np.column_stack([np.interp(selected[:, 0], segment[:, 0], segment[:, i])
                              for i in (1, 2, 3)])
    return selected, target


def partial_score(gt, trajectory, last_image_time):
    """GT time is seconds; start-only SE3 fit; no interpolation across gaps."""
    if not np.isfinite(trajectory).all() or not np.isfinite(gt).all():
        raise ValueError('Nonfinite trajectory or ground truth')
    if np.any(np.diff(gt[:, 0]) <= 0) or np.any(np.diff(trajectory[:, 0]) <= 0):
        raise ValueError('Timestamps must increase strictly')
    segments = np.split(gt, np.flatnonzero(np.diff(gt[:, 0]) > 1.) + 1)
    if len(segments) < 2:
        raise ValueError('Separate start/end ground-truth segments required')
    initial, initial_gt = associate(trajectory, segments[0])
    final, final_gt = associate(trajectory, segments[-1])
    if min(len(initial), len(final)) < 20:
        raise ValueError('At least 20 associated poses in each segment required')
    if np.linalg.matrix_rank(initial_gt - initial_gt.mean(0), tol=1e-5) < 2:
        raise ValueError('Degenerate start-segment alignment geometry')
    sys.path.insert(0, str(CPP.parent / 'Code'))
    from evaluate import umeyama_alignment
    rotation, translation, scale = umeyama_alignment(initial[:, 1:4], initial_gt,
                                                    with_scale=False)
    assert scale == 1.
    errors = np.linalg.norm((rotation @ final[:, 1:4].T).T + translation - final_gt,
                            axis=1)
    return {
        'end_segment_rmse_m': float(np.sqrt(np.mean(errors**2))),
        'end_segment_mean_error_m': float(errors.mean()),
        'start_alignment_samples': len(initial), 'end_segment_samples': len(final),
        'end_segment_covered_s': float(final[-1, 0] - final[0, 0]),
        'trajectory_rows': len(trajectory), 'gt_segments': len(segments),
        'reaches_end': bool(trajectory[-1, 0] >= last_image_time - .1),
        'metric': 'End-segment translational RMSE after start-only SE3 alignment; '
                  'no scale fit or interpolation across missing GT; not full-trajectory ATE',
    }


def frozen_files():
    files = [CPP / 'build/main_euroc', Path(__file__), CPP.parent / 'Code/evaluate.py',
             CPP / 'docs/followup_protocol.md']
    files += sorted((CPP / 'src').rglob('*.cpp'))
    files += sorted((CPP / 'include').rglob('*.h'))
    files += [CPP / 'config' / name for name in CONFIGS]
    return files


def verify(manifest):
    return all(sha(CPP / name) == digest for name, digest in manifest['hashes'].items())


def freeze(out, commit):
    out.mkdir(parents=True, exist_ok=False)
    hashes = {os.path.relpath(path, CPP): sha(path) for path in frozen_files()}
    prior = json.loads((CPP / 'results/prospective/corridor1_manifest.json').read_text())
    # Neither the previous corridor result nor this follow-up changes the estimator.
    shared = [name for name in prior['hashes'] if name.startswith(('src/', 'include/', 'config/', 'build/'))]
    assert all(hashes[name] == prior['hashes'][name] for name in shared)
    (out / 'config').mkdir()
    for name in CONFIGS:
        shutil.copyfile(CPP / 'config' / name, out / 'config' / name)
    shutil.copyfile(CPP / 'build/main_euroc', out / 'main_euroc')
    (out / 'main_euroc').chmod(0o700)
    manifest = {'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'source_commit': commit, 'sequences': SEQUENCES, 'backends': BACKENDS,
                'offset_s': 0, 'threads': 1, 'loop_closure': False, 'hashes': hashes,
                'same_estimator_as_corridor1': True,
                'known_development_data': 'EuRoC 11 and TUM-VI room1–6; corridor1 MSCKF result already inspected; neither new sequence in inspected prior results'}
    save(out / 'manifest.json', manifest)
    print(json.dumps({'frozen': str(out), 'hashes': len(hashes)}), flush=True)


def fetch(sequence, cache):
    cache.mkdir(parents=True, exist_ok=True)
    name = 'dataset-' + sequence + '_512_16'
    archive = cache / (name + '.tar')
    dataset = cache / name
    if dataset.exists():
        raise RuntimeError('Pre-existing test extraction; inspect provenance before reuse')
    url = 'https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/' + archive.name
    transferred = 0
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=30) as response:
        size = int(response.headers.get('Content-Length') or 0)
        if size > 8 * 1024**3:
            raise RuntimeError('Resource cap: published archive exceeds 8 GiB')
        with archive.open('xb') as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                transferred += len(chunk)
                if transferred > 8 * 1024**3 or time.monotonic() - started > 1800:
                    raise RuntimeError('Bounded download resource limit reached')
                stream.write(chunk)
    record = {'url': url, 'bytes': transferred, 'archive_sha256': sha(archive)}
    save(cache / (sequence + '_download.json'), record)
    print('DOWNLOAD_COMPLETE ' + sequence, flush=True)
    with tarfile.open(archive) as tar:
        tar.extractall(cache, filter='data')
    return dataset, record


def run(frozen, cache, sequence, out):
    manifest = json.loads((frozen / 'manifest.json').read_text())
    assert sequence in manifest['sequences'] and verify(manifest)
    out.mkdir(parents=True, exist_ok=False)
    result = {'sequence': sequence, 'source_commit': manifest['source_commit'],
              'status': 'preparing', 'backends': {}}
    try:
        dataset, download = fetch(sequence, cache)
        result['download'] = download
        image_stamps = sorted(int(p.stem) * 1e-9
                              for p in (dataset / 'mav0/cam0/data').glob('*.png'))
        result['full_image_span_s'] = image_stamps[-1] - image_stamps[0]
        gt_path = dataset / 'mav0/mocap0/data.csv'
        if not gt_path.exists():
            gt_path = dataset / 'mav0/state_groundtruth_estimate0/data.csv'
        # Labels are only used for scoring, never estimator setup or offset selection.
        for backend, config in manifest['backends'].items():
            run_dir = out / backend
            command = [str(frozen / 'main_euroc'), '--path', str(dataset), '--config',
                       str(frozen / 'config' / config), '--threads', '1', '--offset', '0',
                       '--backend', backend, '--out', str(run_dir)]
            row = {'backend': backend, 'configuration': config, 'command': command}
            try:
                assert verify(manifest)
                with (out / (backend + '.log')).open('w') as log:
                    proc = subprocess.run(command, cwd=CPP, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=600)
                row['returncode'] = proc.returncode
                if proc.returncode:
                    raise RuntimeError('Estimator failed; no rerun or retuning')
                gt = np.atleast_2d(np.loadtxt(gt_path, delimiter=',', comments='#'))
                gt[:, 0] *= 1e-9
                trajectory = np.atleast_2d(np.loadtxt(run_dir / 'traj_cpp.txt'))
                row.update(partial_score(gt, trajectory, image_stamps[-1]))
                row['trajectory_sha256'] = sha(run_dir / 'traj_cpp.txt')
                row['ground_truth_sha256'] = sha(gt_path)
                row['status'] = 'completed' if row['reaches_end'] else 'incomplete_trajectory'
            except Exception as error:
                row.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            row['frozen_inputs_unchanged'] = verify(manifest)
            result['backends'][backend] = row
            save(out / 'result.json', result)
        result['status'] = 'completed' if all(r['status'] == 'completed'
                                             for r in result['backends'].values()) else 'completed_with_failures'
    except Exception as error:
        result.update(status='preparation_failed', error=type(error).__name__ + ': ' + str(error))
    result['frozen_inputs_unchanged'] = verify(manifest)
    save(out / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    lock = sub.add_parser('freeze')
    lock.add_argument('--out', type=Path, required=True)
    lock.add_argument('--commit', required=True)
    trial = sub.add_parser('run')
    trial.add_argument('--frozen', type=Path, required=True)
    trial.add_argument('--cache', type=Path, required=True)
    trial.add_argument('--sequence', choices=SEQUENCES, required=True)
    trial.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.out.resolve(), args.commit)
    else:
        run(args.frozen.resolve(), args.cache.resolve(), args.sequence, args.out.resolve())


if __name__ == '__main__':
    main()
