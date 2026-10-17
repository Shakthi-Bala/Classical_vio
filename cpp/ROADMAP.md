# Roadmap

This is the plan for the port, written before any C++ existed, plus its status. Results,
plots and how to reproduce them are in [README.md](README.md).

| Phase | Scope | Status |
|---|---|---|
| 1 | Faithful C++17/Eigen port of the Python stereo MSCKF in [`../Code`](../Code), verified module by module, offline on EuRoC MH_01_easy | done |
| 2 | Profiling, quirk ablation on all 11 EuRoC sequences, Ceres sliding-window back end, cam0-only features, ROS 2 node, VINS-Fusion baseline, TUM-VI as a second dataset, loop closure, CI on x86-64 and arm64, Docker (all offline on recorded data or bag playback) | done (including loop closure) |
| 3 | Live RealSense, Jetson / onboard the PeAR drone: real-time factor, latency and power on the embedded board | not started |

## Baseline

The Python report (`plots/rmse_report.txt`) gives 0.1025 m ATE on MH_01. That number
came from the threaded `vio.py`, which is not reproducible: thread timing decides which
IMU samples are buffered when an image arrives. The deterministic replay of the same
Python code ([`scripts/run_python_reference.py`](scripts/run_python_reference.py))
gives 0.1289 m, and that is the reference the C++ is compared against.

## Phase 1 milestones

Each milestone ends with a comparison against the Python.

| # | Milestone | Check | Status |
|---|---|---|---|
| M0 | Environment, CMake, gtest | tests build and pass | done |
| M1 | `math_utils` | `Code/test/utils_test.py` ported to gtest | done |
| M2 | Config and EuRoC reader | same start time and frames after the 40 s offset as the Python | done (implied by M5) |
| M3 | `Feature::initialize_position` | `Code/test/feature_initialization_test.py` ported to gtest | done |
| M4 | MSCKF back end fed with the Python features | trajectory equal to the Python's | done: 1e-9 m on all 2862 poses; the back end is also tested on a synthetic trajectory (`test/test_backends.cpp`) |
| M5 | Front end | features equal to the Python's | done: identical on all 2882 frames |
| M6 | End to end | ATE, per-frame latency, real-time factor | done: 0.1284 m vs 0.1289 m, see README |

M4 was the most useful step: a sign error in a Jacobian looks like a front-end problem
in the end-to-end ATE, and feeding both filters the same feature file separates the two.

## Porting traps

These were written down before the port. All of them came up.

1. **Quaternion convention.** `utils.py` uses JPL quaternions stored `[x, y, z, w]`.
   `Eigen::Quaterniond` is Hamilton and its constructor takes `(w, x, y, z)`. Mixing them
   gives the transposed rotation and a filter that still runs. The MSCKF uses
   `Eigen::Vector4d` with its own functions (as msckf_vio does); the sliding window uses
   Hamilton `Eigen::Quaterniond`.
2. **Timestamps.** EuRoC nanoseconds are parsed as `int64_t` and converted to seconds
   once. The IMU window bounds (`<` vs `<=`) follow the Python.
3. **Camera-state order is the covariance block order.** Python dicts iterate in insertion
   order, which is increasing ID; `std::map<StateID, CAMState>` keeps the same order.
4. **Covariance grow / shrink.** Augmentation is `conservativeResize` and fill; removing a
   middle camera state needs the blocks shifted first.
5. **Thin QR.** `np.linalg.qr(H, 'reduced')` becomes `HouseholderQR` and `householderQ()`
   applied to the residual, never a full m x m Q.
6. **Solves instead of inverses.** `np.linalg.solve(S, H P)` becomes `S.ldlt().solve(H P)`.
7. **The chi-squared gate.** `chi2.ppf(0.05, dof)` is the 5th percentile, not the 95 % the
   comment says. Kept for parity (quirk 3 in the README).
8. **numpy broadcasting** hides outer products, and overlapping Eigen block assignments
   need `.eval()`.
9. **float32 vs float64.** The Python front end sometimes hands the filter float32 arrays.
   Reproducing that needed a templated `stereo_match` (quirk 8).

## Phase 2 decisions

- **Profiling.** `perf` is blocked on the compute nodes (`perf_event_paranoid = 4`), so
  profiling uses stage timers and gperftools with a small addr2line converter for
  FlameGraph.
- **Ceres over GTSAM** for the sliding window. GTSAM ships preintegration and iSAM2;
  Ceres needs the factors and the marginalization written by hand, which is the same
  design as VINS-Fusion and keeps that comparison like for like.
- **ROS 2 Humble via RoboStack** (conda, no root on the cluster). Stereo pairing and
  IMU ordering are done in the node rather than with `message_filters`, so live playback
  processes data in the same order as the offline replay.
- **Cam0-only features.** V2_03 showed that dropping every feature without a stereo match
  (the msckf_vio design) leaves the back ends blind for seconds when one camera blurs.
  Both back ends now accept cam0-only observations (`mono_features`), off by default so
  the Python-parity build is unchanged.
- **VINS-Fusion** from the zinuok ROS 2 port, stereo + IMU, loop closure off, scored with
  the same evaluation code on the same time span.

## Phase 3

RealSense D435i live (Kalibr calibration, camera-IMU time sync), then an aarch64 build on
a Jetson or onboard the PeAR drone: real-time factor, latency and power on the board.
