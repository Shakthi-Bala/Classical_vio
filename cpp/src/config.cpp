#include "msckf/config.h"

#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace msckf {

namespace {

// Overwrites `value` only if `key` exists in `node`.
template <typename T>
void read(const YAML::Node& node, const std::string& key, T& value) {
  if (node[key]) {
    value = node[key].as<T>();
  }
}

// Reads a flat list of numbers into any fixed-size Eigen matrix (row-major order).
template <typename Derived>
void read_matrix(const YAML::Node& node, const std::string& key,
                 Eigen::MatrixBase<Derived>& m) {
  if (!node[key]) {
    return;
  }
  using Scalar = typename Derived::Scalar;
  const std::vector<Scalar> values = node[key].as<std::vector<Scalar>>();
  if (static_cast<Eigen::Index>(values.size()) != m.size()) {
    throw std::runtime_error("config: '" + key + "' expects " +
                             std::to_string(m.size()) + " numbers");
  }
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    for (Eigen::Index j = 0; j < m.cols(); ++j) {
      m(i, j) = values[i * m.cols() + j];
    }
  }
}

void read_camera(const YAML::Node& node, CameraConfig& cam) {
  if (!node) {
    return;
  }
  read(node, "distortion_model", cam.distortion_model);
  read_matrix(node, "intrinsics", cam.intrinsics);
  read_matrix(node, "distortion_coeffs", cam.distortion_coeffs);
  read_matrix(node, "resolution", cam.resolution);
  read_matrix(node, "T_imu_cam", cam.T_imu_cam);
}

}  // namespace

Config load_config(const std::string& yaml_path) {
  const YAML::Node n = YAML::LoadFile(yaml_path);
  Config c;

  if (const YAML::Node o = n["optimization"]) {
    OptimizationConfig& oc = c.optimization_config;
    read(o, "translation_threshold", oc.translation_threshold);
    read(o, "huber_epsilon", oc.huber_epsilon);
    read(o, "estimation_precision", oc.estimation_precision);
    read(o, "initial_damping", oc.initial_damping);
    read(o, "outer_loop_max_iteration", oc.outer_loop_max_iteration);
    read(o, "inner_loop_max_iteration", oc.inner_loop_max_iteration);
  }

  read(n, "grid_row", c.grid_row);
  read(n, "grid_col", c.grid_col);
  read(n, "grid_min_feature_num", c.grid_min_feature_num);
  read(n, "grid_max_feature_num", c.grid_max_feature_num);
  read(n, "fast_threshold", c.fast_threshold);
  read(n, "ransac_threshold", c.ransac_threshold);
  read(n, "stereo_threshold", c.stereo_threshold);
  read(n, "max_iteration", c.max_iteration);
  read(n, "track_precision", c.track_precision);
  read(n, "pyramid_levels", c.pyramid_levels);
  read(n, "patch_size", c.patch_size);

  read(n, "gravity_acc", c.gravity_acc);
  read(n, "frame_rate", c.frame_rate);
  read(n, "max_cam_state_size", c.max_cam_state_size);
  read(n, "position_std_threshold", c.position_std_threshold);
  read(n, "rotation_threshold", c.rotation_threshold);
  read(n, "translation_threshold", c.translation_threshold);
  read(n, "tracking_rate_threshold", c.tracking_rate_threshold);

  // The YAML stores standard deviations (easier to read); the filter wants variances.
  auto read_std_as_var = [&n](const std::string& key, double& variance) {
    if (n[key]) {
      const double sigma = n[key].as<double>();
      variance = sigma * sigma;
    }
  };
  read_std_as_var("gyro_noise_std", c.gyro_noise);
  read_std_as_var("acc_noise_std", c.acc_noise);
  read_std_as_var("gyro_bias_noise_std", c.gyro_bias_noise);
  read_std_as_var("acc_bias_noise_std", c.acc_bias_noise);
  read_std_as_var("observation_noise_std", c.observation_noise);

  read_matrix(n, "velocity", c.velocity);
  read(n, "velocity_cov", c.velocity_cov);
  read(n, "gyro_bias_cov", c.gyro_bias_cov);
  read(n, "acc_bias_cov", c.acc_bias_cov);
  read(n, "extrinsic_rotation_cov", c.extrinsic_rotation_cov);
  read(n, "extrinsic_translation_cov", c.extrinsic_translation_cov);

  read_camera(n["cam0"], c.cam0);
  read_camera(n["cam1"], c.cam1);
  read_matrix(n, "T_cn_cnm1", c.T_cn_cnm1);
  read_matrix(n, "T_imu_body", c.T_imu_body);

  return c;
}

}  // namespace msckf
