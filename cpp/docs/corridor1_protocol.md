# Prospective TUM-VI corridor1 check

Specified before downloading or evaluating corridor1 on 2026-11-03. Existing
development evidence contains all 11 EuRoC sequences and TUM-VI room1–room6.
No corridor result was found in the inspected local results. This is a prospective
sequence check relative to those records, not an externally audited blind study.

- Estimator: the existing MSCKF binary built from the checkout based on `5a659ca`.
  Freeze its SHA-256, all `src/` and `include/` files, and the transitive configuration
  chain in a manifest before accessing test data. No rebuild or estimator edit.
- Configuration: existing `config/tumvi_mono_prune.yaml` with its inherited
  `tumvi_mono.yaml`, `euroc_mono.yaml`, `euroc_fixed.yaml`, `euroc.yaml`. Retain the
  previously used TUM camera calibration. This does not claim zero adaptation from
  EuRoC: calibration, initialization and SWO camera-scale settings have already
  been investigated on TUM-VI room data.
- Test data: only official `dataset-corridor1_512_16.tar`. Download at most 4 GiB.
  The sequence is selected before seeing outputs. No sequence replacement.
- Execution: MSCKF, one core, offset **0 s**, full sequence, no loop closure.
  Do not use motion-capture labels to choose an initialization offset. Report an
  initialization failure as a failure; no second offset or configuration search.
- Ground truth is available only in start/end motion-capture segments. Split GT
  wherever consecutive timestamps differ by more than 1 s. Associate estimated
  poses only within 20 ms of a GT observation; never interpolate across the gap.
- Fit one rigid SE3 transform on valid estimates in the **first** GT segment.
  Require at least 20 samples and non-collinear positions. Apply this transform
  unchanged to the **last** GT segment and report its translational RMSE, sample
  count, covered duration and trajectory completion. Do not fit scale. Do not
  describe this as full-trajectory ATE or an average over multiple sequences.
- Retain the trajectory, process log, manifest and failures, irrespective of score.
  Compare frozen hashes before and after execution. One successful sequence is
  limited evidence and cannot establish generalization across the benchmark.
- No resume number is added before checking the actual outputs and GT coverage.

Dataset source and GT coverage:
https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset

Resource amendment before data access: the server's Content-Length was
3,518,935,040 bytes, exceeding the original 3 GiB cap. The downloader stopped
before transferring the archive or executing the estimator. Raise the transfer cap
to 4 GiB for that same sequence; retain the original manifest and failure. Estimator,
configuration, offset, alignment and scoring rules remain unchanged. This is a
download-resource amendment, not an additional estimator trial or a tuning round.
