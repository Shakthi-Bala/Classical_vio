#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

// Messages passed between the data source, the image processor and the filter.
// These replace the namedtuples in Code/dataset.py and Code/image.py.

namespace msckf {

using FeatureID = std::int64_t;
using StateID = std::int64_t;

struct ImuMsg {
  double timestamp = 0.0;   // seconds
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();      // rad/s
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();   // m/s^2
};

struct StereoMsg {
  double timestamp = 0.0;
  cv::Mat cam0_image;   // 8-bit grayscale
  cv::Mat cam1_image;
};

// One observation of a feature, in undistorted normalized coordinates. With
// Config::mono_features a feature without a stereo match has NaN cam1
// coordinates.
struct FeatureMeasurement {
  FeatureID id = 0;
  double u0 = 0.0, v0 = 0.0;   // cam0
  double u1 = 0.0, v1 = 0.0;   // cam1
};

// True if an observation (u0, v0, u1, v1) has its cam1 part.
inline bool is_stereo(const Eigen::Vector4d& z) { return std::isfinite(z(2)); }

struct FeatureMsg {
  double timestamp = 0.0;
  std::vector<FeatureMeasurement> features;
};

}  // namespace msckf
