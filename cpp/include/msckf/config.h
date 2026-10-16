#pragma once

#include <string>

#include <Eigen/Dense>

// All tunable parameters. C++ port of Code/config.py.
// Defaults equal the Python values; config/euroc.yaml can override any of them.

namespace msckf {

// Parameters of the feature triangulation (Levenberg-Marquardt).
struct OptimizationConfig {
  double translation_threshold = -1.0;   // < 0 disables Feature::check_motion
  double huber_epsilon = 0.01;
  double estimation_precision = 5e-7;
  double initial_damping = 1e-3;
  int outer_loop_max_iteration = 5;
  int inner_loop_max_iteration = 5;
};

struct CameraConfig {
  std::string distortion_model = "radtan";   // "radtan" or "equidistant"
  Eigen::Vector4d intrinsics = Eigen::Vector4d::Zero();          // fx, fy, cx, cy
  Eigen::Vector4d distortion_coeffs = Eigen::Vector4d::Zero();   // k1, k2, p1, p2
  Eigen::Vector2i resolution = Eigen::Vector2i::Zero();          // width, height
  // Same matrix and name as config.py's T_imu_cam0 / T_imu_cam1.
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
};

struct Config {
  OptimizationConfig optimization_config;

  // Image processor.
  int grid_row = 4;
  int grid_col = 5;
  int grid_min_feature_num = 3;
  int grid_max_feature_num = 5;
  int fast_threshold = 15;
  double ransac_threshold = 3;   // unused: 2-point RANSAC is disabled in image.py
  double stereo_threshold = 5;
  int max_iteration = 30;
  double track_precision = 0.01;
  int pyramid_levels = 3;
  int patch_size = 15;

  // MSCKF.
  double gravity_acc = 9.81;
  double frame_rate = 20;
  int max_cam_state_size = 20;
  double position_std_threshold = 8.0;
  double rotation_threshold = 0.2618;
  double translation_threshold = 0.4;
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
  double extrinsic_rotation_cov = 3.0462e-4;
  double extrinsic_translation_cov = 2.5e-5;

  // Calibration.
  CameraConfig cam0;
  CameraConfig cam1;
  Eigen::Matrix4d T_cn_cnm1 = Eigen::Matrix4d::Identity();   // cam0 -> cam1
  Eigen::Matrix4d T_imu_body = Eigen::Matrix4d::Identity();

  int grid_num() const { return grid_row * grid_col; }
};

// Loads a YAML file. Keys that are missing keep their default value above.
Config load_config(const std::string& yaml_path);

}  // namespace msckf
