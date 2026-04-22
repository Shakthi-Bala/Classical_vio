# Classical_vio
# Stereo MSCKF — Visual-Inertial Odometry


MSCKF (Multi-State Constraint Kalman Filter) is an EKF-based **tightly-coupled**
visual-inertial odometry algorithm.
[S-MSCKF](https://arxiv.org/abs/1712.00036) is its stereo version.
This project is a Python re-implementation translated from the official C++
implementation [KumarRobotics/msckf_vio](https://github.com/KumarRobotics/msckf_vio).

---

## Requirements

```
Python 3.6+
numpy
scipy
opencv-python   (cv2)
matplotlib
pangolin        (optional — only needed for --view)
```

Install the required packages:

```bash
pip install numpy scipy opencv-python matplotlib
```

---

## Dataset

Download the [EuRoC MAV dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).
This project has been tested on **MH_01_easy**.

Expected folder structure:

```
MH_01_easy/
└── mav0/
    ├── cam0/data/          # left camera images
    ├── cam1/data/          # right camera images
    ├── imu0/data.csv       # IMU measurements
    └── state_groundtruth_estimate0/data.csv
```

> **Absolute path note:** the default dataset path in `vio.py` is hardcoded to
> `/home/alien/YourDirectoryID_p4/MH_01_easy`.
> Pass `--path` explicitly if your dataset lives elsewhere (see commands below).

---

## Running

### Basic run (no visualisation)

```bash
cd Code
python vio.py
```

### With a custom dataset path

```bash
python vio.py --path /path/to/MH_01_easy
```

### With live 3-D trajectory viewer (requires pangolin)

```bash
python vio.py --view
```

---

## Outputs

| Location | Contents |
|---|---|
| `output_frames/frame_XXXXX.png` | Per-frame canvas: grey GT path, black estimated path, camera frustum, stereo feature inset with GT/EST position text |
| `plots/1_rotation_error.png` | Roll (blue), Pitch (green), Yaw (red) error over time |
| `plots/2_translation_error.png` | X (red), Y (green), Z (blue) translation error over time |
| `plots/3a_trajectory_xy.png` | GT (pink) vs estimated (blue) — top-down X–Y view |
| `plots/3b_trajectory_xz.png` | GT (pink) vs estimated (blue) — X–Z view |
| `plots/4_scale_drift.png` | Scale ratio vs cumulative distance travelled |
| `plots/5a_rpe_scaramuzza.png` | RPE mean ± std at multiple segment lengths (Scaramuzza IROS 2018) |
| `plots/5b_relative_errors_vs_dist.png` | Relative translation (%) and yaw error (deg) vs distance |
| `plots/6_ate_over_time.png` | Absolute Trajectory Error (Umeyama SE3 aligned) |
| `plots/rmse_report.txt` | RMSE summary: ATE, rotation, scale drift |

All plots and the RMSE report are generated automatically at the end of the run.

---

## License

Follows the [license of msckf_vio](https://github.com/KumarRobotics/msckf_vio/blob/master/LICENSE.txt).
Code adapted from [uoip/stereo_msckf](https://github.com/uoip/stereo_msckf).



 ## Dataset link 
 ```bash
https://drive.google.com/drive/folders/10sMVL7Df9jisSE0bZe9T_aaj9QrB7xNw
```
