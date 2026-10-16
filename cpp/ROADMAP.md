# Stereo MSCKF → C++17 Port: Roadmap

Goal: replicate the Python S-MSCKF in `../Code` in C++17/Eigen, verified piece by
piece against the Python, then grow it into a profiled ROS 2 VIO with an
optimization backend and a VINS-Fusion comparison.

**Phase 1 (this branch, sim only):** faithful C++ port, EuRoC MH_01_easy, offline.
**Phase 2:** Ceres sliding-window backend, ROS 2 node, profiling, VINS-Fusion baseline.
**Phase 3 (later):** RealSense live / Jetson / PeAR drone.

---

## 0. Baseline to beat / match (Python, `plots/rmse_report.txt`)

| Metric (MH_01_easy, start offset 40 s, 2862 frames, 73.04 m) | Python |
|---|---|
| ATE RMSE 3D (Umeyama SE3) | **0.1025 m** |
| Rotation RMSE | 2.46 deg |
| RPE @ 10 m | 1.41 % / 1.48 deg |
| Per-frame latency | *not measured yet* (Python ran at `ratio = 0.4`, i.e. 2.5× slower than real time) |

Phase 1 is done when the C++ reproduces the ATE within a few mm (same config,
same 40 s offset) and runs faster than real time.

---

## 1. Environment (no sudo needed: conda-forge, verified to solve 2026-10-15)

```bash
conda create -n vio_cpp -c conda-forge --override-channels \
    cxx-compiler "cmake>=3.22" ninja "eigen=3.4" "opencv=4.10" \
    ceres-solver yaml-cpp gtest benchmark libboost-devel
conda activate vio_cpp
```

- **Pin Eigen 3.4 and OpenCV 4.x.** An unpinned solve picks Eigen 5.0 / OpenCV 5.0,
  which ROS 2, VINS-Fusion and most tutorials don't target yet.
- CMake: `find_package(Eigen3 3.4 REQUIRED NO_MODULE)`, `find_package(OpenCV REQUIRED core imgproc video features2d calib3d imgcodecs)`, `find_package(GTest)`, later `find_package(Ceres)`.
- Build with `-DCMAKE_BUILD_TYPE=Release` (`-O3`) for any timing. Debug Eigen is 10–100× slower.
- Timing numbers: the login node (`login-01`) is shared and noisy. Run benchmarks on a
  compute node (`srun`), and report the CPU model next to them.
- Profilers available: `perf` (installed). Add `-fno-omit-frame-pointer` for clean stacks.

Suggested layout (your call):

```
cpp/
  CMakeLists.txt
  include/msckf/   math_utils.h  config.h  dataset.h  feature.h  image_processor.h  msckf.h
  src/             ...           main_euroc.cpp
  test/            test_math_utils.cpp  test_feature_init.cpp  test_msckf_backend.cpp
  config/          euroc.yaml
  scripts/         dump_golden.py  eval.py      (Python helpers for parity + metrics)
```

---

## 2. Module map: Python → C++

| Python | Key functions | C++ notes |
|---|---|---|
| `utils.py` | `skew`, `to_rotation`, `to_quaternion`, `quaternion_multiplication`, `small_angle_quaternion`, `from_two_vectors`, `Isometry3d` | `Eigen::Vector4d` for JPL quats (see trap #1). `Isometry3d`: `Eigen::Isometry3d` works (it's just R,t). |
| `config.py` | `ConfigEuRoC`, `OptimizationConfigEuRoC` | struct + `yaml-cpp` loader. Keep every value identical for parity. Later, read EuRoC's `mav0/cam0/sensor.yaml` directly. |
| `dataset.py` | `GroundTruthReader`, `IMUDataReader`, `ImageReader`, `Stereo`, `DataPublisher` | Phase 1: **skip the threaded publisher**. Merge IMU + stereo by timestamp in one loop (deterministic, clean timing). Keep the 40 s start offset. |
| `feature.py` | `cost` (L33), `jacobian` (L57), `generate_initial_guess` (L99), `check_motion` (L124), `initialize_position` (L167, hand-rolled LM) | Port the hand-rolled LM first (parity). Later swapping it for Ceres is a nice first Ceres exercise. |
| `image.py` | `stareo_callback` (L94), `initialize_first_frame` (L160), `track_features` (L213), `add_new_features` (L344), `prune_features` (L419), `integrate_imu_data` (L467), `predict_feature_tracking` (L728), `stereo_match` (L761), `undistort_points` / `distort_points` (L847/883) | OpenCV C++ calls map 1:1: `cv::buildOpticalFlowPyramid`, `cv::calcOpticalFlowPyrLK` (+ `OPTFLOW_USE_INITIAL_FLOW`), `cv::FastFeatureDetector`, `cv::undistortPoints`, `cv::projectPoints`, `cv::Rodrigues`. `cv::fisheye::*` only for equidistant (EuRoC is radtan). Grid 4×5, 3–5 feats/cell, FAST 15, LK 15×15 / 3 levels / 30 it / 0.01. |
| `msckf.py` | `initialize_gravity_and_bias` (L231), `batch_imu_processing` (L270), `process_model` (L301), `predict_new_state` (L378, RK4), `state_augmentation` (L441), `add_feature_observations` (L490), `measurement_jacobian` (L513), `feature_jacobian` (L583, null-space projection), `measurement_update` (L626), `gating_test` (L695), `remove_lost_features` (L705), `find_redundant_cam_states` / `prune_cam_state_buffer` (L780/819), `online_reset` (L948), `publish` (L977) | The core. Dynamic `Eigen::MatrixXd` covariance (21 + 6N). See traps #3–#7. |
| `evaluate.py` | Umeyama, RPE (Scaramuzza), plots | **Don't port.** Have both C++ and Python write TUM-format trajectories and evaluate with this same script (or `evo`), so the comparison can't be skewed by a metric difference. |
| `vio.py`, `viewer.py` | threads, canvas frames, pangolin | Phase 1: a plain `main_euroc.cpp`. Visualization later (ROS 2 + RViz). |

Reference implementation (same math, original C++): [KumarRobotics/msckf_vio](https://github.com/KumarRobotics/msckf_vio)
(`include/msckf_vio/math_utils.hpp`, `src/msckf_vio.cpp`, `src/image_processor.cpp`).
Use it to unstick yourself, not to copy from. It's ROS 1 and uses SuiteSparse SPQR.

---

## 3. Porting traps (read before writing code)

1. **Quaternion convention.** `utils.py` uses **JPL, stored `[x, y, z, w]`**.
   `Eigen::Quaterniond` is **Hamilton**, and its constructor is `(w, x, y, z)` even
   though `.coeffs()` returns `[x, y, z, w]`. Mixing them gives the transposed
   rotation, and the filter still "runs", just wrong. Use `Eigen::Vector4d` + your own
   `quaternionToRotation` etc. (msckf_vio does the same). Note `from_two_vectors` returns
   a conjugate (Hamilton → JPL) at the end.
2. **Timestamps.** Parse the EuRoC ns field as `int64_t`, then convert to `double`
   seconds once. `1.4e9 s` in a double has about 0.2 µs resolution, which is fine, but
   never use `==` on times. Use the same IMU-window bounds (`<` vs `<=`) as Python.
3. **Ordering of camera states = covariance block order.** Python dict iteration
   order = insertion order = increasing ID. Use `std::map<StateID, CAMState>` (sorted
   by ID, same order). An `unordered_map` here silently scrambles the covariance.
4. **Covariance grow/shrink.** Augmentation: `conservativeResize` then fill blocks.
   Pruning: shift blocks up/left before resizing. `conservativeResize` does **not**
   move data around for you when you delete a middle block.
5. **Thin QR** (`np.linalg.qr(H, 'reduced')`): `Eigen::HouseholderQR<MatrixXd> qr(H);`
   then `R = qr.matrixQR().topRows(n).triangularView<Upper>()` and
   `Qᵀr = (qr.householderQ().transpose() * r).head(n)`. Never form full Q explicitly
   (m×m). Sparse SPQR is a later optimization.
6. **`np.linalg.solve(S, H P).T`** → `S.ldlt().solve(H * P).transpose()` (S is SPD).
   Null-space via `np.linalg.svd` → `Eigen::JacobiSVD(..., ComputeFullU)`, left null
   space = last columns of U.
7. **`chi2.ppf(0.05, dof)`** table → `boost::math::chi_squared` + `quantile(dist, 0.05)`,
   precomputed for dof 1..99 in the constructor (same as Python, `msckf.py:117-119`).
   Note: that's the **5th percentile**, not the 95th the comment claims, so the gate is
   much stricter than a 95 % test. Upstream msckf_vio has the same quirk. Keep 0.05 for
   parity. Trying 0.95 is a good Phase 2 ablation.
8. **numpy broadcasting** hides outer products: e.g. in `process_model`,
   `(A1 @ u - w)[:, None] * u.T / (u @ u)` is `(A1*u - w) * u.transpose() / u.squaredNorm()`.
   Translate each such line on purpose. Aliasing: `A = A * B` is fine, but
   `P.block(..) = Phi * P.block(..)` needs `.eval()` when the blocks overlap.
9. **Eigen alignment.** C++17 handles over-aligned `new`, so fixed-size members in
   structs are fine. Still prefer `std::vector<T>`/`std::map` holding the structs, not
   raw `new`.
10. **Row- vs column-major.** Only matters when mapping `cv::Mat` ↔ Eigen buffers
    (`cv::cv2eigen` / `cv::eigen2cv` from `opencv2/core/eigen.hpp` handle it).
11. **Determinism.** 2-point RANSAC is commented out in `image.py`, so the Python
    front end is deterministic. Expect tiny float drift (LK/BLAS order), not random
    divergence. That's why tight parity tests are realistic here.

---

## 4. Phase 1 milestones (each ends with a test that compares against Python)

| # | Milestone | Done when |
|---|---|---|
| M0 | Env + CMake skeleton + one gtest passes | `ctest` green |
| M1 | `math_utils` | Port `Code/test/utils_test.py` to gtest. Plus round-trips vs Python-dumped values (rotation↔quat, quat mult, small-angle) to 1e-12 |
| M2 | Config + EuRoC reader | Same IMU/image counts and first/last timestamps after the 40 s offset as Python |
| M3 | `Feature::initializePosition` | Port `Code/test/feature_initialization_test.py`. Plus same 3-D point as Python on dumped real tracks (1e-6) |
| M4 | **MSCKF back end, fed by Python features** | Dump Python's `feature_msg` stream (+ IMU) to a file. C++ back end consumes it. Trajectory ≈ Python (ATE diff < 1 mm). Isolates filter bugs from front-end bugs. Also unit-test `process_model` (Φ, P after 1 step) and one `measurement_update` against golden dumps |
| M5 | Image processor (front end) | Per-frame feature counts + tracked pixel coords ≈ Python on the first ~200 frames (a few % count diff and sub-pixel coords are fine) |
| M6 | **End to end** | `main_euroc` writes a TUM trajectory. Same eval script gives ATE ≈ 0.10 m. Report latency per frame (front end / back end split, `std::chrono::steady_clock`, mean/p50/p95/max) and real-time factor = dataset seconds ÷ wall seconds |

The golden-dump trick for M4/M5: a small `scripts/dump_golden.py` that imports the
existing Python classes and saves inputs/outputs to `.npy`/`.csv`. That's Python
tooling, not part of the C++ port. M4 is the single highest-value step, because a wrong
sign in a Jacobian looks exactly like a front-end problem in the end-to-end ATE.

Python side, one small change needed for a fair comparison: make `vio.py` also write
the TUM trajectory and per-frame timings, and run it with the publisher ratio removed
(process as fast as possible), so the Python latency is measured the same way.

---

## 5. Phase 2: the resume part

**5a. Profile the C++ MSCKF first** (`perf record -g`, flame graph). Expect hot spots in
LK tracking, the stacked-H QR, and `P` updates (21+6·20 = 141² dense). Optimizations
worth measuring: exploit the `Φ` block structure, sparse QR, avoid temporaries,
`-march=native`. Report before/after.

**5b. Sliding-window optimizer (Ceres).** The MSCKF is a *filter*. It has no
sliding-window optimizer to "port". This is a **second back end** behind a common
interface, so you can compare MSCKF vs SWO on the same front end:
- keyframe window (~10), states: pose, velocity, biases
- IMU **preintegration** factor (Forster et al. 2017) with bias Jacobians, which you'll derive
- reprojection factors (inverse depth, as in VINS-Mono)
- **marginalization** via Schur complement → prior factor (the hardest part)
- `ceres::Manifold` (Ceres ≥ 2.1, replaces `LocalParameterization`) for the quaternion

Ceres vs GTSAM: GTSAM ships `ImuFactor`/preintegration + iSAM2, so less to write but
less learned. Ceres makes you write the factors and marginalization yourself, the same
design as VINS-Fusion, which keeps that comparison apples-to-apples. **Recommendation:
Ceres.**

**5c. ROS 2 node.** `rclcpp` component: `message_filters` ApproximateTime for stereo,
`sensor_msgs/Imu` callback, publish `nav_msgs/Odometry`, `nav_msgs/Path`, TF. Feed
EuRoC via rosbag2. The provided `MH_01_easy.bag` is ROS 1, so convert with
`pip install rosbags && rosbags-convert MH_01_easy.bag`. No sudo/containers on this
cluster: install ROS 2 via **RoboStack** (conda, `ros-humble-*` / `ros-jazzy-*`).
Measure callback-to-publish latency with `ros2_tracing` or timestamps in the message.

**5d. VINS-Fusion baseline.** Run it yourself on MH_01 (stereo+IMU, loop closure **off**
for fairness) and evaluate with the same script. Upstream is ROS 1; ROS 2 forks exist.
If it won't build, quote published numbers and label them as such in the table.

**Final results table to fill:**

| | ATE RMSE [m] | RPE@10m [%/deg] | Latency/frame mean·p95 [ms] | RTF (x86) | RTF (Jetson) |
|---|---|---|---|---|---|
| Python MSCKF | 0.1025 | 1.41 / 1.48 | | | n/a |
| C++ MSCKF | | | | | |
| C++ SWO (Ceres) | | | | | |
| VINS-Fusion (no loop) | | | | | |

Add MH_02–MH_05 / V1_xx later. One sequence alone is a weak claim.

---

## 6. Phase 3 (later, real world)

RealSense D435i/T265 live (calibrate with Kalibr, time-sync IMU↔camera), then Jetson
(aarch64 build, RTF + latency + power) / PeAR drone onboard.
