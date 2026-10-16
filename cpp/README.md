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
| Front end, mean / p95 | 15.95 / 17.48 ms | 8.87 / 9.70 ms |
| Back end, mean / p95 | 18.59 / 44.02 ms | 3.55 / 7.83 ms |
| Total per frame, mean / p95 | 34.54 / 60.22 ms | 12.42 / 16.72 ms |
| Real-time factor | 1.45x | **4.02x** |

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
LK, undistortion), which is already C++ under Python, so it only gains 1.8x (from
removing per-point Python loops). The filter is pure numpy on small matrices and
gains 5.2x. The first C++ frame takes ~260 ms (OpenCV start-up); it is included in
the statistics and is the "max" value.

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
