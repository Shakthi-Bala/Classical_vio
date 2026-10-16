# Stereo MSCKF in C++17

C++17 / Eigen / OpenCV port of the Python S-MSCKF in [`../Code`](../Code), checked
module by module against the Python implementation. Phase 1 of [ROADMAP.md](ROADMAP.md).

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

## Build

```bash
conda create -n vio_cpp -c conda-forge --override-channels \
    cxx-compiler "cmake>=3.22" ninja "eigen=3.4" "opencv=4.10" \
    ceres-solver yaml-cpp gtest benchmark libboost-devel
conda activate vio_cpp

cd ~/Classical_vio
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure     # 11 unit tests
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

For timing, run on a compute node (`srun -p quick -c 4 ...`), not the shared login node.

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
