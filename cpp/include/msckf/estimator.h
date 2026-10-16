#pragma once

#include <cstddef>
#include <optional>

#include <Eigen/Dense>

#include "msckf/math_utils.h"
#include "msckf/messages.h"

// Common interface of the back ends (MSCKF filter, sliding-window optimizer),
// so main_euroc can run either with the same front end.

namespace msckf {

// Both back ends start by averaging this many IMU samples (1 s at 200 Hz),
// assuming the sensor is static, for gravity direction and gyro bias.
inline constexpr size_t kStaticInitImuSamples = 200;

struct VioResult {
  double timestamp = 0.0;
  Isometry3d pose;        // body (IMU) frame in the world frame
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Isometry3d cam0_pose;   // cam0 frame in the world frame
};

class Estimator {
 public:
  virtual ~Estimator() = default;

  virtual void imu_callback(const ImuMsg& imu_msg) = 0;

  // Returns the pose at the feature message's time, or nothing while the
  // estimator is still initializing.
  virtual std::optional<VioResult> feature_callback(const FeatureMsg& feature_msg) = 0;
};

}  // namespace msckf
