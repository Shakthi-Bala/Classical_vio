# Stereo visual-inertial odometry in C++17: MSCKF and sliding-window optimizer

C++17 / Eigen / OpenCV / Ceres port of the Python S-MSCKF in [`../Code`](../Code),
checked module by module against the Python, then extended with a Ceres
sliding-window back end and a ROS 2 node. Plan and status: [ROADMAP.md](ROADMAP.md).

| What | Result |
|---|---|
| Port fidelity | C++ filter = Python filter to **1e-9 m** on identical input; identical features on all 2882 frames |
| Speed (MH_01, one core) | **8.6 ms/frame, 5.8x real time** (Python: 34.5 ms, 1.45x), after profiling-driven optimization |
| Accuracy, 11 EuRoC sequences | Python-faithful MSCKF 0.615 m mean ATE -> **0.249 m** with the quirk fixes (0 failures) |
| Sliding-window optimizer | **0.164 m** mean ATE on the 10 sequences it completes; fails on V2_03 |
| ROS 2 node | real-time bag playback gives the **same trajectory as offline**; 10.9 ms end-to-end (MSCKF) |
| vs VINS-Fusion (10 seq. all complete) | sliding window **0.164 m**, fixed MSCKF 0.207 m, VINS-Fusion 0.207 m; VINS-Fusion most robust (only one besides the MSCKF to complete V2_03) |

Sections: [Phase 1 port](#results-euroc-mh_01_easy) ·
[Profiling](#profiling-and-optimization-124---86-ms-per-frame) ·
[Quirk ablation](#python-quirk-ablation-11-euroc-sequences) ·
[Sliding window](#sliding-window-optimizer-ceres-back-end) · [ROS 2](#ros-2-node) ·
[Final comparison](#final-comparison-11-euroc-sequences) · [Build](#build) ·
[Code tour](#code-tour-read-in-this-order)

![trajectory](results/trajectory_xy.png)

## Results (EuRoC MH_01_easy)

Same start offset (40 s), deterministic replay without threads, same SLURM compute
node, accuracy from the unchanged [`Code/evaluate.py`](../Code/evaluate.py):

| | Python | C++ |
|---|---|---|
| ATE RMSE (Umeyama SE3) | 0.1289 m | 0.1284 m |
| Rotation RMSE | 1.98 deg | 1.98 deg |
| RPE @ 10 m | 1.49 % / 1.41 deg | 1.49 % / 1.39 deg |
| Front end, mean / p95 | 15.95 / 17.48 ms | 6.34 / 8.71 ms |
| Back end, mean / p95 | 18.59 / 44.02 ms | 2.29 / 5.06 ms |
| Total per frame, mean / p95 | 34.54 / 60.22 ms | **8.63 / 12.32 ms** |
| Real-time factor | 1.45x | **5.80x** (one core) |

C++ numbers are after the profiling-driven optimizations below (OpenCV limited to
one thread with `--threads 1`). The straight port, before optimizing, ran at
12.42 ms per frame (4.02x).

![latency](results/latency.png)

**Parity:**
- **Back end:** fed the same feature file, the C++ and Python filters agree to
  **1e-9 m** (the print precision) on all 2862 poses.
- **Front end:** all 2882 frames give **identical features** (ids, and coordinates
  to 1e-6).
- The remaining 0.5 mm ATE difference comes from numpy, not from the port: when a
  frame has no new cam0 features, `image.py` hands the filter float32 arrays, and
  numpy 1.x keeps `float32_array * float64_scalar` in float32 inside `feature.py`.
  That ~1e-7 difference grows to centimetres along the trajectory, because the
  filter's gating and pruning decisions are sensitive to it.

**Why 0.1289 m and not the 0.1025 m in `plots/rmse_report.txt`?** That number came
from `vio.py`, which runs the image, IMU and filter threads against a wall clock
(`ratio = 0.4`). Which IMU samples a thread has seen when an image arrives depends
on timing, so that run is not reproducible, and as shown above tiny differences
grow to cm. Both columns here use the same deterministic replay; use these for
Python-vs-C++ claims. Running several sequences (MH_02–05, V1) would give a fairer
accuracy picture than one run of one sequence.

**Where the speed-up comes from:** the front end is mostly OpenCV (FAST, pyramidal
LK, undistortion), which is already C++ under Python, so the straight port only
gained 1.8x there (from removing per-point Python loops); the filter is pure numpy
on small matrices and gained 5.2x. The first C++ frame takes ~200 ms (OpenCV
start-up); it is included in the statistics and is the "max" value.

## Profiling and optimization (12.4 -> 8.6 ms per frame)

`perf` is blocked on the compute nodes (`perf_event_paranoid = 4`, no sudo), so the
profile comes from two tools that need no privileges:

- **Stage timers**: [`profiler.h`](include/msckf/profiler.h), one `MSCKF_PROFILE("name")`
  line per stage; `main_euroc` prints the table and writes `profile_stages.csv`.
- **Sampling profiler**: gperftools (`LD_PRELOAD=libprofiler.so`, 1 kHz), converted to
  a flame graph by [`scripts/gperf_to_folded.py`](scripts/gperf_to_folded.py) (addr2line)
  and FlameGraph: [before](results/profile/flame_before.svg),
  [after](results/profile/flame_after.svg).

![stages](results/profile/stages_before_after.png)

| Change | Why it was slow | Effect | Output |
|---|---|---|---|
| Build each image pyramid once ([`create_image_pyramids`](src/image_processor.cpp)) | 5 KLT calls per frame each rebuilt both pyramids: 10 builds for 3 images | front end 9.25 -> 7.63 ms | bit-identical features and trajectory |
| `P - K (H P)` instead of `(I - K H) P` in [`measurement_update`](src/msckf.cpp) | an extra 141 x 141 x 141 product, and `H P` computed twice | update 2.06 -> 1.51 ms | trajectory equal to 1e-9 m |
| Gating on the non-zero columns of `H` only ([`gating_test`](src/msckf.cpp)) | `H P H^T` with a 141-wide `H` that is zero outside 2-20 camera blocks | gating 0.86 -> 0.12 ms | trajectory equal to 1e-9 m |
| `--threads 1` for OpenCV | ~100 points per KLT call are too few to pay for threads | front end 7.6 -> 7.3 ms (before the two back-end changes) | identical |

What is left: KLT optical flow (stereo L-R forward + backward, and temporal) is now
about half of a frame. Cutting it further means changing the algorithm (fewer LK
iterations, skip the backward check), which changes the results, so it belongs to
the quirk/ablation step, not to "same output, faster".

## Python-quirk ablation (11 EuRoC sequences)

Each quirk in the table at the end has an opt-in fix (`--fix NAME`, or the `fixes:`
section of the YAML; [`config/euroc_fixed.yaml`](config/euroc_fixed.yaml) has the
recommended set). All 11 sequences were run with each fix alone, all fixes, and all
but the chi-squared one. Start offsets come from
[`scripts/find_start_offsets.py`](scripts/find_start_offsets.py) (quietest IMU second
before take-off; it picks 40.3 s for MH_01, where `vio.py` used 40 s).

![ablation](results/ablation/ablation.png)

| ATE RMSE [m] | Python | C++ = Python | **C++ fixed** (all but 3) |
|---|---|---|---|
| MH_01_easy | 0.129 | 0.128 | **0.088** |
| MH_02_easy | 0.472 | 0.576 | **0.205** |
| MH_03_medium | 0.214 | 0.214 | 0.234 |
| MH_04_difficult | 0.880 | 1.030 | **0.415** |
| MH_05_difficult | 0.819 | 0.869 | **0.420** |
| V1_01_easy | 0.346 | 0.384 | **0.117** |
| V1_02_medium | 0.128 | 0.128 | 0.148 |
| V1_03_difficult | 0.429 | 0.429 | **0.244** |
| V2_01_easy | 0.276 | 0.270 | **0.095** |
| V2_02_medium | 0.204 | 0.204 | **0.100** |
| V2_03_difficult | crashes | 2.535 | **0.677** |
| mean (11 seq.) | - | 0.615 | **0.249** |

Full table (every variant): [`results/ablation/ate_table.csv`](results/ablation/ate_table.csv).

What it shows:
- **Quirk 4 (observability constraint) is the big one**: the Python OC-EKF projects with
  `R_w_i^T g` and leaves `Phi[0:3,0:3]` alone; the upstream form halves the error on
  most sequences.
- **Quirk 3 is load-bearing**: a "correct" 95 % chi-squared gate alone diverges on 8 of
  11 sequences. With the broken epipolar check (quirk 1) and RANSAC disabled, the
  over-strict 5 % gate is what keeps stereo outliers out. Even with every other fix,
  the strict gate is still better (all-but-3 beats all).
- **Run-to-run noise is real**: quirk 5 does not change the math (the propagated
  quaternion keeps its norm), only the rounding, yet it moves single-sequence ATE by
  up to a few percent. Only consistent wins across sequences count.
- **The Python crashes on V2_03**: a frame with zero tracked features makes
  `stereo_match` return `[]`, which `track_features` cannot unpack. The C++ handles it.
- Python vs "C++ = Python" differs on MH_02/04/05: the float32 effect described above,
  amplified over longer, faster sequences. Same algorithm, different rounding.

## Sliding-window optimizer (Ceres) back end

`main_euroc --backend swo` runs a keyframe-based sliding-window optimizer on the
same front end ([`include/msckf/swo/`](include/msckf/swo), [`src/swo/`](src/swo)):

- **States** per keyframe: position, orientation (Hamilton quaternion with a
  right-perturbation manifold, [`quaternion_manifold.h`](include/msckf/swo/quaternion_manifold.h)),
  velocity, accelerometer and gyroscope bias. Window of 10 keyframes.
- **IMU factor**: on-manifold preintegration (Forster et al. 2017) with bias
  Jacobians for first-order bias correction, re-integrated when the bias moves far
  ([`imu_preintegration.cpp`](src/swo/imu_preintegration.cpp)).
- **Visual factors**: landmarks are inverse depths in their anchor keyframe, initialized
  from the stereo pair; every stereo observation is a reprojection residual (Huber
  loss). The reprojection Jacobians are hand-derived
  ([`ReprojectionFactorAnalytic`](include/msckf/swo/factors.h)); autodiff was a third of the
  solve time.
- **Marginalization**: the oldest keyframe and its landmarks become a Gaussian prior via
  the Schur complement ([`marginalization.cpp`](src/swo/marginalization.cpp)).
- **Keyframes**: a new keyframe when parallax > 10 px, < 50 % of features still tracked,
  or 0.25 s passed; other frames report the IMU prediction.
- **Tests** ([`test/test_swo.cpp`](test/test_swo.cpp)): manifold Plus/Minus and Jacobian,
  bias Jacobians vs numerical differentiation, IMU factor zero at the prediction,
  marginalization equal to the joint solution, analytic vs autodiff reprojection Jacobians.

Tried and rejected (both kept as options, off by default): restarting the window after
tracking loss (`reset_min_tracked`) and ignoring vision when fewer than N landmarks are
usable (`min_vision_landmarks`). Both made V2_03 worse (33.6 m and 15.2 m vs 5.0 m).
The per-keyframe `debug_log` showed why it diverges: tracking is lost completely for
several seconds and the few re-acquired landmarks then move the whole window.

## ROS 2 node

[`ros2/msckf_vio_ros`](ros2/msckf_vio_ros) wraps either back end as a ROS 2 (Humble)
node. It builds the same library as a CMake subproject, so it runs exactly the code
above.

- Subscribes `/imu0`, `/cam0/image_raw`, `/cam1/image_raw`; publishes `odometry`,
  `path` and the `world -> imu` TF.
- Stereo images are paired by timestamp, and a pair is processed only after every IMU
  sample up to its timestamp: the same order as the offline replay, independent of
  how DDS interleaves the topics.
- Played back in real time from the MH_01 ROS 2 bag, the node gives the **same 2862
  poses and the same ATE as `main_euroc`** (0.0884 m MSCKF, 0.1013 m sliding window;
  positions within 1e-4 m, the difference being OpenCV 4.13 / Eigen 5 in the ROS env).

| MH_01, ROS 2, real-time playback, 1 core | process mean / p95 | end to end mean / p95 |
|---|---|---|
| MSCKF | 10.3 / 13.1 ms | 10.9 / 14.2 ms |
| Sliding window | 19.2 / 38.0 ms | 22.1 / 38.9 ms |

End to end = from the arrival of the later image of a pair to the published odometry
(includes DDS delivery and queueing). Run it with
[`ros2/run_euroc_bag.sh`](ros2/run_euroc_bag.sh) (environment: see below).

## Final comparison (11 EuRoC sequences)

![final](results/final/ate_per_sequence.png)

| ATE RMSE [m] | Python | MSCKF = Python | MSCKF fixed | SWO | VINS-Fusion |
|---|---|---|---|---|---|
| MH_01_easy | 0.129 | 0.128 | 0.088 | 0.101 | 0.208 |
| MH_02_easy | 0.472 | 0.576 | 0.205 | 0.083 | 0.214 |
| MH_03_medium | 0.214 | 0.214 | 0.234 | 0.215 | 0.313 |
| MH_04_difficult | 0.880 | 1.030 | 0.415 | 0.280 | 0.439 |
| MH_05_difficult | 0.819 | 0.869 | 0.420 | 0.175 | 0.312 |
| V1_01_easy | 0.346 | 0.384 | 0.117 | 0.078 | 0.113 |
| V1_02_medium | 0.128 | 0.128 | 0.148 | 0.082 | 0.107 |
| V1_03_difficult | 0.429 | 0.429 | 0.244 | 0.299 | 0.113 |
| V2_01_easy | 0.276 | 0.270 | 0.095 | 0.117 | 0.131 |
| V2_02_medium | 0.204 | 0.204 | 0.100 | 0.210 | 0.123 |
| V2_03_difficult | fail | 2.535 | 0.677 | div. | 0.313 |
| failures | 1 | 0 | 0 | 1 | 0 |
| mean, 10 seq. all completed | 0.390 | 0.423 | 0.207 | 0.164 | 0.207 |

What it shows:
- **The C++ sliding window is the most accurate** where it works: mean 0.164 m on the
  10 sequences all methods complete, against 0.207 m for both the fixed MSCKF and
  VINS-Fusion. It wins clearly on the Machine Hall sequences (MH_02 0.08 m, MH_05 0.18 m).
- **VINS-Fusion is the most robust**: it completes V2_03 at 0.31 m (our MSCKF 0.68 m,
  our sliding window diverges) and is best on V1_03. It has two things ours lacks: a
  visual-inertial initialization that does not assume a static start, and failure
  detection with re-initialization.
- **The fixed C++ MSCKF matches VINS-Fusion on average** (0.207 m each) at half the
  sliding window's compute, and never fails.
- **Python -> C++ fixed**: 0.390 -> 0.207 m on the same 10 sequences, 3-4x faster.

VINS-Fusion trajectories, the summary CSV and the patch set applied to the ROS 2 port
([zinuok/VINS-Fusion-ROS2](https://github.com/zinuok/VINS-Fusion-ROS2) @ 72023bc) are in
[`results/vins/`](results/vins). Patches: vio.csv timestamps were truncated to whole
seconds; reliable sensor QoS (best effort lost 902 of 7364 images); a single-threaded
executor (the port's multi-threaded executor started ~190 threads on 192-core nodes and
froze); a frame-drop switch (off for these runs). Every run lost 0 messages at rate 1.0.

All runs use the deterministic replay, the start offsets of
[`config/euroc_offsets.csv`](config/euroc_offsets.csv) and the same ATE code.
"MSCKF fixed" and "SWO" use [`config/euroc_fixed.yaml`](config/euroc_fixed.yaml).
VINS-Fusion (ROS 2 port, stereo + IMU, loop closure off, upstream EuRoC config) runs
on the whole bag and is scored on the same time span as ours.

| MH_01, one core, `--threads 1` | mean / p95 per frame | real-time factor |
|---|---|---|
| Python MSCKF | 34.5 / 60.2 ms | 1.45x |
| C++ MSCKF, Python behaviour | 8.6 / 12.3 ms | 5.80x |
| C++ MSCKF, quirks fixed | 11.3 / 14.6 ms | 4.41x |
| C++ sliding window | 19.9 / 43.5 ms | 2.51x |

## Build

```bash
conda create -n vio_cpp -c conda-forge --override-channels \
    cxx-compiler "cmake>=3.22" ninja "eigen=3.4" "opencv=4.10" \
    ceres-solver yaml-cpp gtest benchmark libboost-devel
conda activate vio_cpp

cd ~/Classical_vio
# Use the env's cmake: a pip cmake 3.25 in ~/.local/bin shadows it and cannot
# configure Ceres' SuiteSparse dependency.
$CONDA_PREFIX/bin/cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
$CONDA_PREFIX/bin/cmake --build cpp/build
./cpp/build/msckf_tests                              # 16 unit tests
```

ROS 2 node (RoboStack Humble env `ros2_vio`, no sudo needed):

```bash
source /scratch/sbalamurugan/ros2_vio/activate.sh   # conda activate ros2_vio + PATH/ROS vars
W=/scratch/sbalamurugan/ros2_vio/msckf_ws
cd ~/Classical_vio/cpp
colcon --log-base $W/log build --base-paths ros2 --build-base $W/build \
    --install-base $W/install --cmake-args -DCMAKE_BUILD_TYPE=Release -G Ninja
# node + bag in real time, on a compute node:
sbatch -p short -c 4 --mem=16G -t 01:00:00 ros2/run_euroc_bag.sh \
    /scratch/sbalamurugan/ros2_vio/bags/MH_01_easy_ros2 swo results/ros2/MH_01_easy_swo 40
```

## Run and reproduce the table

```bash
cd ~/Classical_vio/cpp
DATA=/home/sbalamurugan/YourDirectoryID_p4/MH_01_easy

# C++ (front end + back end); --dump-features saves the front-end output
./build/main_euroc --path $DATA --out results/cpp --dump-features

# Python reference, deterministic (in the `cv` env)
python scripts/run_python_reference.py --path $DATA --out results/python

# Back-end parity: both filters on the same Python feature file
./build/main_euroc --path $DATA --out results/cpp_replay --features results/python/features_python.txt
python scripts/run_python_reference.py --path $DATA --out results/python_replay \
    --features results/python/features_python.txt

# Metrics, plots, results/summary.md
python scripts/compare.py --gt $DATA/mav0/state_groundtruth_estimate0/data.csv
```

All sequences (SLURM): `scripts/run_sequence_job.sh` (Python + 9 ablation variants),
`scripts/run_backend_job.sh` (one back end with `config/euroc_fixed.yaml`), then
`scripts/ablation_table.py`, `scripts/plot_ablation.py` and `scripts/final_report.py`.
Sliding window: add `--backend swo`. For timing, run on a compute node
(`srun -p quick -c 4 ...`), not the shared login node.

## Code tour (read in this order)

| File | Python | What to look at |
|---|---|---|
| [math_utils](include/msckf/math_utils.h) | `utils.py` | JPL quaternions as `Eigen::Vector4d`; header vs source; `const&` |
| [config](include/msckf/config.h) + [euroc.yaml](config/euroc.yaml) | `config.py` | structs with default values; yaml-cpp; a small template (`read_matrix`) |
| [messages](include/msckf/messages.h), [state](include/msckf/state.h) | namedtuples, `IMUState`, `CAMState` | `std::map` instead of `dict`, and why the order matters |
| [dataset](src/dataset.cpp) | `dataset.py` | `std::filesystem`, file parsing, int64 ns timestamps |
| [feature](src/feature.cpp) | `feature.py` | `inline static` class members; structured bindings `for (auto& [id, m] : ...)` |
| [msckf](src/msckf.cpp) | `msckf.py` | fixed-size vs dynamic Eigen matrices; `conservativeResize`; thin QR; `ldlt().solve` instead of inverses; **aliasing** (`symmetrize`, `.eval()`) |
| [image_processor](src/image_processor.cpp) | `image.py` | OpenCV C++ API; a member **template** (`stereo_match<PointT>`) used to mirror numpy's float32/float64 choice |
| [main_euroc](src/main_euroc.cpp) | `vio.py` | `std::optional`, `std::chrono::steady_clock` timing |
| [test/](test/) | `Code/test/` | gtest: `TEST`, `EXPECT_NEAR`, seeded random numbers |

## Python quirks reproduced on purpose

The goal of Phase 1 is parity, so these behaviours of the Python code are copied,
each marked `NOTE (Python parity)` in the source. Each fix is a measurable Phase 2
ablation.

| # | Where | Python behaviour | Upstream msckf_vio |
|---|---|---|---|
| 1 | `image.py` `stereo_match` | epipolar error uses `(pt1 * line)[0]`, one term of the dot product | full dot product `pt1 . line` |
| 2 | `feature.py` `initialize_position` | `is_cost_reduced` never reset: LM accepts one step, then stops updating | do-while loops, iterates to convergence |
| 3 | `msckf.py` `__init__` | `chi2.ppf(0.05, dof)`: 5th percentile gate (comment says 95 %) | same quirk |
| 4 | `msckf.py` `process_model` | OC-EKF uses `u = R_w_i^T g` (pre-propagation orientation), leaves `Phi[0:3,0:3]` | `u = R(q_null) g`, sets `Phi[0:3,0:3]` |
| 5 | `msckf.py` `predict_new_state` | quaternion not re-normalized; small-gyro branch would raise (`ndarray @ scalar`) | normalizes; working branch (used here) |
| 6 | `image.py` `add_new_features` | `mask[y-3:y+4, x-3:x+4]` with negative start selects nothing near the top/left border | clamps |
| 7 | `dataset.py` | start time = first IMU sample + offset (`stereo.start_time()` is still `-inf`) | n/a |
| 8 | `image.py` `publish` | cam0 coordinates float64 if any new feature is present, else float32 | n/a |

Not reproduced: the Python `map_server` dict iterates in insertion order and the
C++ `std::map` in ID order. That can only matter when an update hits the 1500-row
cap, and the exact back-end parity above shows it never did on MH_01.
