# Frozen MSCKF check on TUM-VI corridor1

One prospective check specified in [the protocol](corridor1_protocol.md),
committed in `834def9` before data access. A metadata-only download-size amendment
was committed in `f41f058`. Prior development used EuRoC's 11 sequences and
TUM-VI room1–room6; no earlier corridor result was found in the inspected records.
This is not an externally audited blind study.

| Measurement | Observed result |
|---|---:|
| End-segment translational RMSE, after start-only SE3 alignment | **3.544809 m** |
| End-segment mean positional error | 3.544157 m |
| Start-segment alignment samples | 516 |
| End-segment scored samples | 503 |
| End-segment covered duration | 25.100636 s |
| Full image-sequence duration | 299.459255 s |
| Output poses | 5,970 |
| Reached final image time within 0.1 s | Yes |
| Frozen binary, configuration and checked source hashes unchanged | Yes |

The result measures accumulated error in the final observed segment using a rigid
transform fitted only to the initial segment. It is not full-trajectory ATE:
ground truth is absent during the corridor transit. Do not compare this value
directly with the Ceres EuRoC mean computed after full-trajectory alignment. One
sequence does not establish low-drift generalization across unseen data.

Execution used the existing MSCKF binary, `config/tumvi_mono_prune.yaml`, offset
0 seconds, one OpenCV thread and no loop closure. No parameter, initialization
offset or estimator source was changed after inspecting this result. The estimator
ran once. Network transfer was resumed after a download deadline; a NumPy boolean
JSON-export failure required scoring the saved trajectory again, not rerunning
the estimator.

Scoring split ground truth at gaps above 1 second, associated poses only within
20 ms of an observation, and excluded all poses in the missing-GT interval.
Ground-truth positions were linearly interpolated only inside each observed
segment. A rigid Umeyama alignment without scale was fitted on the first segment,
then held fixed for the last segment. Known-answer checks confirmed gap exclusion,
timestamp rejection, rigid-transform recovery, no scale fit and preservation of
one metre of synthetic end drift.

Archive SHA-256: `074417f29d07bea1bfdc338b8e276d4230c216a2a0c4468fab218997c4a32b25`.
Trajectory SHA-256: `8919e275ab5876aa4e18b799aadf4d56ad17cdd9cc043ad47b2d5a848a0fead6`.
Ground-truth CSV SHA-256: `bde26d7ceb15cb69f3a38cbcad53032025ab3be2c59d43fc06f245af00e3fc06`.

Machine-readable results and frozen hashes are in
`results/prospective/corridor1.json` and `corridor1_manifest.json`.
Dataset attribution, licensing and partial-GT coverage:
[TUM VI Benchmark, Schubert et al.](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset).
