#pragma once

#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>

#include "msckf/config.h"
#include "msckf/estimator.h"
#include "msckf/swo/imu_preintegration.h"
#include "msckf/swo/marginalization.h"
#include "msckf/swo/quaternion_manifold.h"

// Keyframe-based sliding-window visual-inertial optimizer (Ceres), the second
// back end next to the MSCKF. It consumes the same feature messages.
//
//  - Each keyframe has a full state (p, q, v, ba, bg); consecutive keyframes
//    are tied by an IMU preintegration factor.
//  - Landmarks are inverse depths in their first (anchor) keyframe; every
//    stereo observation in a later keyframe is a reprojection factor.
//  - Frames that add little parallax are not keyframes: their pose is the IMU
//    prediction from the latest optimized keyframe.
//  - When the window is full, the oldest keyframe and the landmarks anchored
//    in it are marginalized into a Gaussian prior (Schur complement).

namespace msckf::swo {

class SlidingWindowEstimator final : public Estimator {
 public:
  explicit SlidingWindowEstimator(const Config& config);
  ~SlidingWindowEstimator() override;

  void imu_callback(const ImuMsg& imu_msg) override;
  std::optional<VioResult> feature_callback(const FeatureMsg& feature_msg) override;

  struct Stats {
    long frames = 0;
    long keyframes = 0;
    long marginalizations = 0;
    long landmarks_optimized = 0;   // summed over solves
    long outliers_removed = 0;
  };
  const Stats& stats() const { return stats_; }

 private:
  struct KeyFrame {
    long id = 0;
    double timestamp = 0.0;
    double p[3] = {0.0, 0.0, 0.0};
    double q[4] = {0.0, 0.0, 0.0, 1.0};   // Eigen order (x, y, z, w)
    double sb[9] = {0.0};                  // v | ba | bg
    std::unique_ptr<ImuPreintegration> preint;   // from the previous keyframe
    std::map<FeatureID, Eigen::Vector4d> obs;    // (u0, v0, u1, v1), normalized

    Eigen::Map<Eigen::Vector3d> position() { return Eigen::Map<Eigen::Vector3d>(p); }
    Eigen::Map<Eigen::Quaterniond> orientation() { return Eigen::Map<Eigen::Quaterniond>(q); }
    Eigen::Map<Eigen::Vector3d> velocity() { return Eigen::Map<Eigen::Vector3d>(sb); }
    Eigen::Map<Eigen::Vector3d> acc_bias() { return Eigen::Map<Eigen::Vector3d>(sb + 3); }
    Eigen::Map<Eigen::Vector3d> gyro_bias() { return Eigen::Map<Eigen::Vector3d>(sb + 6); }
  };

  struct Landmark {
    long anchor_id = 0;
    Eigen::Vector3d bearing = Eigen::Vector3d::UnitZ();   // (x, y, 1) in anchor cam0
    double inv_depth[1] = {1.0};
    std::map<long, Eigen::Vector4d> obs;                  // keyframe id -> observation
  };

  void initialize_gravity_and_bias();
  void initialize_first_keyframe(const FeatureMsg& msg);
  void integrate_imu_until(double t);
  bool is_keyframe(const FeatureMsg& msg) const;
  void add_keyframe(const FeatureMsg& msg, const Eigen::Vector3d& p, const Eigen::Quaterniond& q,
                    const Eigen::Vector3d& v);
  void add_observations(KeyFrame& kf, const FeatureMsg& msg);
  std::optional<double> stereo_inverse_depth(const Eigen::Vector4d& z) const;

  bool usable(const Landmark& lm) const;
  KeyFrame* keyframe(long id) const;
  // Visual factors of one landmark: (cost, blocks) pairs; caller owns the costs.
  void landmark_factors(Landmark& lm, std::vector<ResidualBlockInfo>& out) const;
  void repropagate_if_needed(KeyFrame& prev, KeyFrame& kf) const;

  void optimize();
  void remove_outliers();
  void marginalize_oldest();
  void erase_landmark(FeatureID id);
  VioResult make_result(double t, const Eigen::Vector3d& p, const Eigen::Quaterniond& q,
                        const Eigen::Vector3d& v) const;

  Config config_;
  SwoConfig sc_;
  ImuNoise noise_;

  // Extrinsics.
  Eigen::Matrix3d R_bc0_, R_bc1_, R_c1c0_;
  Eigen::Vector3d t_bc0_, t_bc1_, t_c1c0_;
  double focal_ = 1.0;   // cam0 fx, converts normalized errors to pixels

  // IMU.
  std::vector<ImuMsg> imu_buffer_;
  bool is_gravity_set_ = false;
  Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, -9.81);
  Eigen::Quaterniond q_init_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d gyro_bias_init_ = Eigen::Vector3d::Zero();
  double last_imu_t_ = 0.0;
  Eigen::Vector3d last_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero();
  std::unique_ptr<ImuPreintegration> cur_preint_;   // last keyframe -> now

  // Window and map.
  std::deque<std::unique_ptr<KeyFrame>> window_;
  long next_kf_id_ = 0;
  std::map<FeatureID, Landmark> landmarks_;

  // Priors and shared Ceres objects (the problems do not own them).
  std::unique_ptr<ceres::CostFunction> init_prior_;   // on the first keyframe until marginalized
  std::unique_ptr<MarginalizationFactor> marg_prior_;
  RightQuaternionManifold quaternion_manifold_;
  std::unique_ptr<ceres::LossFunction> huber_;

  Stats stats_;
};

}  // namespace msckf::swo
