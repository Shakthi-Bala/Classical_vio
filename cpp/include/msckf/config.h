#pragma once

#include <string>

#include <Eigen/Dense>

// All tunable parameters. C++ port of Code/config.py.
// Defaults equal the Python values; a YAML file (config/euroc.yaml) can
// override any of them.

namespace msckf {

// Parameters of the feature triangulation (Levenberg-Marquardt).
struct OptimizationConfig {
  double translation_threshold = -1.0;   // < 0 disables Feature::check_motion
  double huber_epsilon = 0.01;
  double estimation_precision = 5e-7;
  double initial_damping = 1e-3;
  int outer_loop_max_iteration = 5;
  int inner_loop_max_iteration = 5;
  // Fix for quirk 2 (README "Python quirks"): at least one LM step per outer
  // iteration, like upstream's do-while loops.
  bool lm_upstream_loops = false;
};

// Opt-in fixes for behaviours of the Python code that the port reproduces on
// purpose (README "Python quirks"). All false = identical to the Python.
struct Fixes {
  bool epipolar_full_dot = false;       // 1: full pt1 . line in the stereo check
  bool chi2_95 = false;                 // 3: gate at the 95th percentile, not the 5th
  bool oc_upstream = false;             // 4: OC-EKF with R(q_null) g and Phi[0:3,0:3]
  bool normalize_quaternion = false;    // 5: re-normalize q after propagation
  bool mask_clamp = false;              // 6: mask features near the top/left border too
  bool full_precision_publish = false;  // 8: undistort published points in double
};

// Sliding-window optimizer (Ceres) back end.
struct SwoConfig {
  int window_size = 10;                // keyframes kept in the optimization
  double kf_parallax_px = 10.0;        // new keyframe if mean parallax exceeds this
  double kf_min_tracked_ratio = 0.5;   // ... or if fewer of the last keyframe's features remain
  double kf_max_interval = 0.25;       // ... or if this many seconds passed
  int max_iterations = 8;              // Levenberg-Marquardt iterations per keyframe
  double pixel_sigma = 1.5;            // reprojection noise [px]
  double huber_px = 1.5;               // robust loss threshold [px]
  double min_depth = 0.1;              // [m]
  double max_depth = 40.0;             // [m]
  double outlier_px = 3.0;             // drop a landmark above this mean reprojection error
  // Start-up prior on the first keyframe (standard deviations).
  double init_pos_sigma = 1e-3;        // [m]
  double init_rot_sigma = 1e-3;        // [rad]
  double init_vel_sigma = 0.5;         // [m/s]
  double init_acc_bias_sigma = 0.1;    // [m/s^2]
  double init_gyro_bias_sigma = 0.01;  // [rad/s]
  std::string debug_log;               // if set: one CSV line per keyframe solve
};

struct CameraConfig {
  std::string distortion_model = "radtan";   // "radtan" or "equidistant"
  Eigen::Vector4d intrinsics = Eigen::Vector4d::Zero();          // fx, fy, cx, cy
  Eigen::Vector4d distortion_coeffs = Eigen::Vector4d::Zero();   // k1, k2, p1, p2
  Eigen::Vector2i resolution = Eigen::Vector2i::Zero();          // width, height
  // Takes a vector from the IMU frame to this camera's frame (Kalibr T_cam_imu;
  // the name is config.py's).
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
};

// Camera -> IMU transform, derived from CameraConfig::T_imu_cam.
struct CameraExtrinsics {
  Eigen::Matrix3d R_cam_imu;   // takes a vector from the camera frame to the IMU frame
  Eigen::Vector3d t_cam_imu;   // camera origin in the IMU frame
};
CameraExtrinsics camera_to_imu(const CameraConfig& cam);

struct Config {
  OptimizationConfig optimization_config;
  Fixes fixes;
  SwoConfig swo;

  // Image processor.
  int grid_row = 4;
  int grid_col = 5;
  int grid_min_feature_num = 3;
  int grid_max_feature_num = 5;
  int fast_threshold = 15;
  double stereo_threshold = 5;    // epipolar error [px]
  int max_iteration = 30;         // KLT
  double track_precision = 0.01;  // KLT
  int pyramid_levels = 3;
  int patch_size = 15;

  // MSCKF.
  double gravity_acc = 9.81;
  int max_cam_state_size = 20;
  double position_std_threshold = 8.0;   // online reset above this [m]
  double rotation_threshold = 0.2618;    // keyframe pruning [rad] (15 deg)
  double translation_threshold = 0.4;    // keyframe pruning [m]
  double tracking_rate_threshold = 0.5;

  // Noise (variances, not standard deviations).
  double gyro_noise = 0.005 * 0.005;
  double acc_noise = 0.05 * 0.05;
  double gyro_bias_noise = 0.001 * 0.001;
  double acc_bias_noise = 0.01 * 0.01;
  double observation_noise = 0.035 * 0.035;

  // Initial state and covariance.
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  double velocity_cov = 0.25;
  double gyro_bias_cov = 0.01;
  double acc_bias_cov = 0.01;
  double extrinsic_rotation_cov = 3.0462e-4;   // (1 deg)^2
  double extrinsic_translation_cov = 2.5e-5;   // (5 mm)^2

  // Calibration.
  CameraConfig cam0;
  CameraConfig cam1;
  Eigen::Matrix4d T_cn_cnm1 = Eigen::Matrix4d::Identity();   // cam0 -> cam1
  Eigen::Matrix4d T_imu_body = Eigen::Matrix4d::Identity();

  int grid_num() const { return grid_row * grid_col; }
};

// Loads a YAML file. Keys that are missing keep their default value above.
// A top-level `base: other.yaml` (relative to this file) is loaded first, so a
// variant only needs the keys it changes.
Config load_config(const std::string& yaml_path);

// Turns on one fix by name ("epipolar_full_dot", "lm_upstream_loops", ...,
// or "all"). Throws on an unknown name.
void enable_fix(Config& config, const std::string& name);

}  // namespace msckf
