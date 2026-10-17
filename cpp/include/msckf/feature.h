#pragma once

#include <map>

#include <Eigen/Dense>

#include "msckf/config.h"
#include "msckf/math_utils.h"
#include "msckf/messages.h"
#include "msckf/state.h"

// A 3-D landmark and its stereo observations. C++ port of Code/feature.py.

namespace msckf {

class Feature {
 public:
  // Stereo extrinsics, shared by all features (class attributes in Python).
  // Takes a vector from the cam0 frame to the cam1 frame.
  inline static Eigen::Matrix3d R_cam0_cam1 = Eigen::Matrix3d::Identity();
  inline static Eigen::Vector3d t_cam0_cam1 = Eigen::Vector3d::Zero();

  Feature() = default;
  Feature(FeatureID new_id, const OptimizationConfig& config)
      : id(new_id), optimization_config(config) {}

  // Squared reprojection error of one observation.
  // T_c0_ci takes a vector from the c0 frame to the ci frame,
  // x = (alpha, beta, rho) is the inverse-depth position in c0, z the measurement.
  double cost(const Isometry3d& T_c0_ci, const Eigen::Vector3d& x,
              const Eigen::Vector2d& z) const;

  // Jacobian J (2x3), residual r and Huber weight w of one observation.
  void jacobian(const Isometry3d& T_c0_ci, const Eigen::Vector3d& x,
                const Eigen::Vector2d& z, Eigen::Matrix<double, 2, 3>& J,
                Eigen::Vector2d& r, double& w) const;

  // Two-view depth initial guess; returns the point in the c1 frame.
  // T_c1_c2 takes a vector from the c2 frame to the c1 frame.
  Eigen::Vector3d generate_initial_guess(const Isometry3d& T_c1_c2,
                                         const Eigen::Vector2d& z1,
                                         const Eigen::Vector2d& z2) const;

  // True if the camera moved enough (perpendicular to the ray) to triangulate:
  // more than optimization_config.translation_threshold, or `threshold`.
  bool check_motion(const CamStateServer& cam_states) const;
  bool check_motion(const CamStateServer& cam_states, double threshold) const;

  // True if at least one observation has its cam1 part (Config::mono_features).
  bool has_stereo_observation() const;

  // Levenberg-Marquardt triangulation; sets `position` (world frame) and
  // `is_initialized`. Returns true if the point is in front of every camera.
  bool initialize_position(const CamStateServer& cam_states);

  FeatureID id = 0;
  // <StateID, (u0, v0, u1, v1)>, sorted by state ID like the Python dict.
  std::map<StateID, Eigen::Vector4d> observations;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  bool is_initialized = false;
  OptimizationConfig optimization_config;
};

}  // namespace msckf
