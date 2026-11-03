# Frozen follow-up on corridor2 and slides1

Predefined on 2026-11-03, before acquiring either sequence or inspecting outputs.
The earlier corridor1 MSCKF result is already known and is excluded from this
new batch. EuRoC's 11 sequences and TUM-VI room1–room6 are development data.
No corridor2 or slides1 output was found in the inspected prior local records.
This is a prospective sequence split relative to those records, not an externally
audited blind study. Both TUM-VI sequences use previously supplied calibration.

- Fixed test set: **corridor2 and slides1**, one corridor and one difficult
  low-illumination slide sequence. No substitution, dropping or extra sequences
  based on outcomes.
- Run **both MSCKF and the Ceres sliding-window backend**, once each per sequence.
  MSCKF uses existing `tumvi_mono_prune.yaml`; Ceres uses `tumvi_mono.yaml`.
  Freeze both complete configuration chains, the existing binary, implementation,
  scoring helper and script before data access. Confirm shared estimator inputs
  match the prior corridor1 manifest. No rebuild, tuning or estimator fix.
- Fixed 0-second offset, one OpenCV thread, full sequence, no loop closure.
  No ground-truth-assisted initialization, offset search, checkpoint selection,
  per-sequence setting or reattempt after estimator output.
- Scoring: split ground truth at gaps above 1 second; associate only estimates
  within 20 ms of GT observations in an observed segment. Require at least 20
  poses at each end and non-collinear initial positions. Fit a single SE3 transform
  without scale on the first observed segment; hold it fixed on the last segment.
  Report end-segment translational RMSE, sample counts, duration and completion.
  Never interpolate ground truth through missing intervals. This is **not
  full-trajectory ATE**, and is not comparable to the tuned EuRoC full-alignment
  mean or equivalent to demonstrating generalization over the full benchmark.
- Preserve all four estimator outcomes, including timeouts, poor accuracy and
  initialization/association failures. Publish per-sequence results. Any summary
  mean must state that it is an unweighted mean of two end-segment RMSEs and may
  only be computed if both predefined sequences complete for that backend.
  Report completion/failure counts alongside any summary, not a successes-only
  aggregate. No score threshold changes the stopping rule or triggers retuning.
- Resource bounds: 8 GiB archive and 30 minutes transfer per sequence; 10 minutes
  per estimator. A transport retry before estimator output may resume bytes but
  cannot change the test set, scoring or estimator settings. Resource failures
  remain visible as preparation failures.

Implementation: `scripts/prospective_eval.py`; known-answer scoring tests:
`python -m unittest discover -s test -p test_prospective_eval.py`.
Run `freeze --out FROZEN --commit COMMIT` before `run --frozen FROZEN
--sequence SEQUENCE --cache CACHE --out OUTPUT`. Execute on compute nodes.

Dataset and partial-ground-truth reference:
https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
