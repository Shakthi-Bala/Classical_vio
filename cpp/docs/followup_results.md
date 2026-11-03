# Frozen two-sequence VIO follow-up

The test set and both backend configurations were committed before new data access. The existing binary/configurations are unchanged from the earlier project; EuRoC11 and TUM-VI room1–room6 remain development data, and corridor1 is a historical test. The new fixed set is corridor2 and slides1; each backend was attempted once per sequence.

| Sequence | Backend | Status | End-segment RMSE (m) | End samples | End duration (s) |
|---|---|---|---:|---:|---:|
| corridor2 | msckf | completed | 0.813747 | 739 | 37.251 |
| corridor2 | swo | completed | 2.634964 | 739 | 37.251 |
| slides1 | msckf | completed | 3.464444 | 946 | 47.252 |
| slides1 | swo | completed | 4.903797 | 946 | 47.252 |

## Aggregation

- msckf: 2/2 completed; unweighted mean of the two end-segment RMSEs 2.139095 m.
- swo: 2/2 completed; unweighted mean of the two end-segment RMSEs 3.769381 m.

## Interpretation and reproducibility

These values score only the final motion-capture segment using a rigid transform fitted to the initial observed segment. The unobserved transit is excluded; no ground truth is interpolated across its gap and no scale is fitted. They are not full-trajectory ATE and must not be compared numerically with the Ceres EuRoC mean after full-trajectory alignment. Completion does not by itself establish accurate tracking. Two sequences do not establish generalization across the benchmark or meet a resume score threshold.

Both the live source inputs and the actual frozen executable/configuration copies were checked against the manifest after execution. No setting, initialization offset, test sequence or scoring rule was selected in response to these outcomes. No estimator rerun or sequence substitution was performed.

Protocol: [followup_protocol.md](followup_protocol.md). Frozen hashes: `results/prospective/followup_manifest.json`; machine-readable outcomes: `results/prospective/followup_results.json`. Scoring and run commands: `scripts/prospective_eval.py`. Eight known-answer scoring tests passed before test-data access.

Dataset attribution and partial-ground-truth reference: [TUM VI Benchmark](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset).
