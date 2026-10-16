#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>

#include "msckf/swo/imu_preintegration.h"
#include "msckf/swo/quaternion_manifold.h"

// Cost functions of the sliding-window optimizer. All use ceres automatic
// differentiation, so their Jacobians are exact by construction.
//
// Parameter blocks per keyframe (Hamilton convention, body = IMU frame):
//   p  [3]  position of the body in the world frame
//   q  [4]  orientation R_wb as Eigen quaternion coefficients (x, y, z, w)
//   sb [9]  velocity (world) | accelerometer bias | gyroscope bias
// Per landmark:
//   inv_depth [1]  inverse depth along its bearing in the anchor's cam0 frame

namespace msckf::swo {

// IMU preintegration factor between keyframes i and j (15 residuals:
// rotation, velocity, position, acc-bias walk, gyro-bias walk).
class ImuFactor {
 public:
  ImuFactor(const ImuPreintegration& pre, const Eigen::Vector3d& gravity)
      : delta_q_(pre.delta_q), delta_v_(pre.delta_v), delta_p_(pre.delta_p),
        ba_lin_(pre.ba_lin), bg_lin_(pre.bg_lin), dR_dbg_(pre.dR_dbg), dv_dba_(pre.dv_dba),
        dv_dbg_(pre.dv_dbg), dp_dba_(pre.dp_dba), dp_dbg_(pre.dp_dbg), dt_(pre.sum_dt),
        g_(gravity) {
    // sqrt_info^T sqrt_info = covariance^-1, so |sqrt_info r|^2 is the Mahalanobis norm.
    Eigen::Matrix<double, 15, 15> cov = pre.residual_covariance();
    cov.diagonal().array() += 1e-12;   // keep it invertible for very short intervals
    const Eigen::Matrix<double, 15, 15> info = cov.inverse();
    sqrt_info_ = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(0.5 * (info + info.transpose()))
                     .matrixL()
                     .transpose();
  }

  template <typename T>
  bool operator()(const T* p_i_, const T* q_i_, const T* sb_i, const T* p_j_, const T* q_j_,
                  const T* sb_j, T* residual) const {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    const Eigen::Map<const Vec3> p_i(p_i_), p_j(p_j_);
    const Eigen::Map<const Eigen::Quaternion<T>> q_i(q_i_), q_j(q_j_);
    const Eigen::Map<const Vec3> v_i(sb_i), ba_i(sb_i + 3), bg_i(sb_i + 6);
    const Eigen::Map<const Vec3> v_j(sb_j), ba_j(sb_j + 3), bg_j(sb_j + 6);

    // First-order bias correction of the preintegrated measurements.
    const Vec3 dba = ba_i - ba_lin_.cast<T>();
    const Vec3 dbg = bg_i - bg_lin_.cast<T>();
    const Eigen::Quaternion<T> dq =
        delta_q_.cast<T>() * quaternion_exp<T>(Vec3(dR_dbg_.cast<T>() * dbg));
    const Vec3 dv = delta_v_.cast<T>() + dv_dba_.cast<T>() * dba + dv_dbg_.cast<T>() * dbg;
    const Vec3 dp = delta_p_.cast<T>() + dp_dba_.cast<T>() * dba + dp_dbg_.cast<T>() * dbg;

    const T dt = T(dt_);
    const Vec3 g = g_.cast<T>();
    const Eigen::Quaternion<T> q_i_inv = q_i.conjugate();

    Eigen::Matrix<T, 15, 1> r;
    r.template segment<3>(0) = quaternion_log<T>(dq.conjugate() * (q_i_inv * q_j));
    r.template segment<3>(3) = q_i_inv * (v_j - v_i - g * dt) - dv;
    r.template segment<3>(6) = q_i_inv * (p_j - p_i - v_i * dt - T(0.5) * g * dt * dt) - dp;
    r.template segment<3>(9) = ba_j - ba_i;
    r.template segment<3>(12) = bg_j - bg_i;

    Eigen::Map<Eigen::Matrix<T, 15, 1>> out(residual);
    out = sqrt_info_.cast<T>() * r;
    return true;
  }

  static ceres::CostFunction* create(const ImuPreintegration& pre,
                                     const Eigen::Vector3d& gravity) {
    return new ceres::AutoDiffCostFunction<ImuFactor, 15, 3, 4, 9, 3, 4, 9>(
        new ImuFactor(pre, gravity));
  }

 private:
  Eigen::Quaterniond delta_q_;
  Eigen::Vector3d delta_v_, delta_p_, ba_lin_, bg_lin_;
  Eigen::Matrix3d dR_dbg_, dv_dba_, dv_dbg_, dp_dba_, dp_dbg_;
  double dt_;
  Eigen::Vector3d g_;
  Eigen::Matrix<double, 15, 15> sqrt_info_;
};

// Observation of a landmark (anchored in keyframe a) by camera c of keyframe j.
class ReprojectionFactor {
 public:
  // bearing: (x, y, 1) of the landmark in the anchor's cam0 frame.
  // R_bc0/t_bc0: cam0 -> body. R_bc/t_bc: camera c -> body. z: normalized coords.
  ReprojectionFactor(const Eigen::Vector3d& bearing, const Eigen::Vector2d& z,
                     const Eigen::Matrix3d& R_bc0, const Eigen::Vector3d& t_bc0,
                     const Eigen::Matrix3d& R_bc, const Eigen::Vector3d& t_bc, double sqrt_info)
      : bearing_(bearing), z_(z), R_bc0_(R_bc0), t_bc0_(t_bc0), R_bc_(R_bc), t_bc_(t_bc),
        sqrt_info_(sqrt_info) {}

  template <typename T>
  bool operator()(const T* p_a_, const T* q_a_, const T* p_j_, const T* q_j_,
                  const T* inv_depth, T* residual) const {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    const Eigen::Map<const Vec3> p_a(p_a_), p_j(p_j_);
    const Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_), q_j(q_j_);

    const Vec3 p_c0_a = bearing_.cast<T>() / inv_depth[0];
    const Vec3 p_b_a = R_bc0_.cast<T>() * p_c0_a + t_bc0_.cast<T>();
    const Vec3 p_w = q_a * p_b_a + p_a;
    const Vec3 p_b_j = q_j.conjugate() * (p_w - p_j);
    const Vec3 p_c_j = R_bc_.transpose().cast<T>() * (p_b_j - t_bc_.cast<T>());

    residual[0] = T(sqrt_info_) * (p_c_j.x() / p_c_j.z() - T(z_.x()));
    residual[1] = T(sqrt_info_) * (p_c_j.y() / p_c_j.z() - T(z_.y()));
    return true;
  }

  static ceres::CostFunction* create(const Eigen::Vector3d& bearing, const Eigen::Vector2d& z,
                                     const Eigen::Matrix3d& R_bc0, const Eigen::Vector3d& t_bc0,
                                     const Eigen::Matrix3d& R_bc, const Eigen::Vector3d& t_bc,
                                     double sqrt_info) {
    return new ceres::AutoDiffCostFunction<ReprojectionFactor, 2, 3, 4, 3, 4, 1>(
        new ReprojectionFactor(bearing, z, R_bc0, t_bc0, R_bc, t_bc, sqrt_info));
  }

 private:
  Eigen::Vector3d bearing_;
  Eigen::Vector2d z_;
  Eigen::Matrix3d R_bc0_;
  Eigen::Vector3d t_bc0_;
  Eigen::Matrix3d R_bc_;
  Eigen::Vector3d t_bc_;
  double sqrt_info_;
};

// cam1 observation in the anchor keyframe itself: constrains only the depth.
class StereoAnchorFactor {
 public:
  // R_c1c0/t_c1c0: cam0 -> cam1.
  StereoAnchorFactor(const Eigen::Vector3d& bearing, const Eigen::Vector2d& z1,
                     const Eigen::Matrix3d& R_c1c0, const Eigen::Vector3d& t_c1c0,
                     double sqrt_info)
      : bearing_(bearing), z1_(z1), R_c1c0_(R_c1c0), t_c1c0_(t_c1c0), sqrt_info_(sqrt_info) {}

  template <typename T>
  bool operator()(const T* inv_depth, T* residual) const {
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    const Vec3 p_c0 = bearing_.cast<T>() / inv_depth[0];
    const Vec3 p_c1 = R_c1c0_.cast<T>() * p_c0 + t_c1c0_.cast<T>();
    residual[0] = T(sqrt_info_) * (p_c1.x() / p_c1.z() - T(z1_.x()));
    residual[1] = T(sqrt_info_) * (p_c1.y() / p_c1.z() - T(z1_.y()));
    return true;
  }

  static ceres::CostFunction* create(const Eigen::Vector3d& bearing, const Eigen::Vector2d& z1,
                                     const Eigen::Matrix3d& R_c1c0,
                                     const Eigen::Vector3d& t_c1c0, double sqrt_info) {
    return new ceres::AutoDiffCostFunction<StereoAnchorFactor, 2, 1>(
        new StereoAnchorFactor(bearing, z1, R_c1c0, t_c1c0, sqrt_info));
  }

 private:
  Eigen::Vector3d bearing_;
  Eigen::Vector2d z1_;
  Eigen::Matrix3d R_c1c0_;
  Eigen::Vector3d t_c1c0_;
  double sqrt_info_;
};

// Gaussian prior on one keyframe's full state (fixes the gauge at start-up).
class StatePriorFactor {
 public:
  StatePriorFactor(const Eigen::Vector3d& p0, const Eigen::Quaterniond& q0,
                   const Eigen::Matrix<double, 9, 1>& sb0,
                   const Eigen::Matrix<double, 15, 1>& sigma)
      : p0_(p0), q0_(q0), sb0_(sb0), inv_sigma_(sigma.cwiseInverse()) {}

  template <typename T>
  bool operator()(const T* p_, const T* q_, const T* sb_, T* residual) const {
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(p_);
    const Eigen::Map<const Eigen::Quaternion<T>> q(q_);
    const Eigen::Map<const Eigen::Matrix<T, 9, 1>> sb(sb_);
    Eigen::Matrix<T, 15, 1> r;
    r.template segment<3>(0) = p - p0_.cast<T>();
    r.template segment<3>(3) = quaternion_log<T>(q0_.cast<T>().conjugate() * q);
    r.template segment<9>(6) = sb - sb0_.cast<T>();
    Eigen::Map<Eigen::Matrix<T, 15, 1>> out(residual);
    out = inv_sigma_.cast<T>().cwiseProduct(r);
    return true;
  }

  static ceres::CostFunction* create(const Eigen::Vector3d& p0, const Eigen::Quaterniond& q0,
                                     const Eigen::Matrix<double, 9, 1>& sb0,
                                     const Eigen::Matrix<double, 15, 1>& sigma) {
    return new ceres::AutoDiffCostFunction<StatePriorFactor, 15, 3, 4, 9>(
        new StatePriorFactor(p0, q0, sb0, sigma));
  }

 private:
  Eigen::Vector3d p0_;
  Eigen::Quaterniond q0_;
  Eigen::Matrix<double, 9, 1> sb0_;
  Eigen::Matrix<double, 15, 1> inv_sigma_;
};

}  // namespace msckf::swo
