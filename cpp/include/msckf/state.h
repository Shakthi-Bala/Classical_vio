#pragma once

#include <map>

#include <Eigen/Dense>

#include "msckf/messages.h"

// Filter state. C++ port of IMUState / CAMState in Code/msckf.py.
// Quaternions are JPL [x, y, z, w] (see math_utils.h).

namespace msckf {

struct IMUState {
  StateID id = 0;
  double timestamp = 0.0;

  // Takes a vector from the world frame to the IMU (body) frame.
  Eigen::Vector4d orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  // Position / velocity of the IMU frame in the world frame.
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();

  // Copies of the state at the last propagation, used to keep the
  // observability-constrained null space (OC-EKF).
  Eigen::Vector4d orientation_null = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  Eigen::Vector3d position_null = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity_null = Eigen::Vector3d::Zero();

  // IMU-camera extrinsics (estimated online).
  Eigen::Matrix3d R_imu_cam0 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_cam0_imu = Eigen::Vector3d::Zero();
};

struct CAMState {
  StateID id = 0;
  double timestamp = 0.0;

  // Takes a vector from the world frame to the cam0 frame.
  Eigen::Vector4d orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  // Position of the cam0 frame in the world frame.
  Eigen::Vector3d position = Eigen::Vector3d::Zero();

  Eigen::Vector4d orientation_null = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  Eigen::Vector3d position_null = Eigen::Vector3d::Zero();
};

// Sorted by ID. IDs only grow, so this is also insertion order, which is what
// the Python dict gives. The order defines where each camera's 6x6 block sits
// in the covariance matrix, so it must never be an unordered container.
using CamStateServer = std::map<StateID, CAMState>;

}  // namespace msckf
