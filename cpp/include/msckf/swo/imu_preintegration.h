#pragma once

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "msckf/config.h"

// On-manifold IMU preintegration (Forster et al., "On-Manifold Preintegration
// for Real-Time Visual-Inertial Odometry", T-RO 2017), Euler integration.
//
// Between two keyframes i and j it accumulates, in the frame of body i:
//   delta_q  (Delta R_ij),  delta_v,  delta_p
// for a fixed bias guess (ba_lin, bg_lin), plus
//   - the 9x9 covariance of [dtheta, dv, dp], and
//   - the Jacobians of delta_* w.r.t. the biases, used to correct the
//     preintegration to first order when the bias estimate changes.
// The raw samples are kept so it can be re-integrated with a new bias.

namespace msckf::swo {

struct ImuNoise {
  // Continuous-time densities, as variances (same meaning as config.py).
  double gyro_noise = 0.0;
  double acc_noise = 0.0;
  double gyro_bias_noise = 0.0;
  double acc_bias_noise = 0.0;

  static ImuNoise from_config(const Config& c) {
    return {c.gyro_noise, c.acc_noise, c.gyro_bias_noise, c.acc_bias_noise};
  }
};

class ImuPreintegration {
 public:
  ImuPreintegration(const Eigen::Vector3d& ba_lin, const Eigen::Vector3d& bg_lin,
                    const ImuNoise& noise);

  // Integrates one measurement held constant for dt seconds.
  void integrate(double dt, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro);

  // Re-integrates every stored sample with a new bias linearization point.
  void repropagate(const Eigen::Vector3d& ba_lin, const Eigen::Vector3d& bg_lin);

  // Predicts the state at j from the state at i (gravity g in the world frame).
  void predict(const Eigen::Vector3d& p_i, const Eigen::Quaterniond& q_i,
               const Eigen::Vector3d& v_i, const Eigen::Vector3d& g, Eigen::Vector3d& p_j,
               Eigen::Quaterniond& q_j, Eigen::Vector3d& v_j) const;

  // Full 15x15 covariance of the IMU residual [dtheta, dv, dp, dba, dbg].
  Eigen::Matrix<double, 15, 15> residual_covariance() const;

  double sum_dt = 0.0;
  Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
  Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();

  Eigen::Vector3d ba_lin = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg_lin = Eigen::Vector3d::Zero();

  Eigen::Matrix3d dR_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_dbg = Eigen::Matrix3d::Zero();

  Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();

 private:
  struct Sample {
    double dt;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
  };
  void reset();
  void integrate_sample(const Sample& s);

  ImuNoise noise_;
  std::vector<Sample> samples_;
};

}  // namespace msckf::swo
