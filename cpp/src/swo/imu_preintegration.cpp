#include "msckf/swo/imu_preintegration.h"

#include "msckf/math_utils.h"
#include "msckf/swo/quaternion_manifold.h"

namespace msckf::swo {

namespace {

// Right Jacobian of SO(3).
Eigen::Matrix3d right_jacobian(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  const Eigen::Matrix3d K = skew(phi);
  if (theta < 1e-8) {
    return Eigen::Matrix3d::Identity() - 0.5 * K;
  }
  const double t2 = theta * theta;
  return Eigen::Matrix3d::Identity() - (1.0 - std::cos(theta)) / t2 * K +
         (theta - std::sin(theta)) / (t2 * theta) * K * K;
}

}  // namespace

ImuPreintegration::ImuPreintegration(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
                                     const ImuNoise& noise)
    : ba_lin(ba), bg_lin(bg), noise_(noise) {}

void ImuPreintegration::reset() {
  sum_dt = 0.0;
  delta_q.setIdentity();
  delta_v.setZero();
  delta_p.setZero();
  dR_dbg.setZero();
  dv_dba.setZero();
  dv_dbg.setZero();
  dp_dba.setZero();
  dp_dbg.setZero();
  covariance.setZero();
}

void ImuPreintegration::integrate(double dt, const Eigen::Vector3d& acc,
                                  const Eigen::Vector3d& gyro) {
  if (dt <= 0.0) {
    return;
  }
  samples_.push_back({dt, acc, gyro});
  integrate_sample(samples_.back());
}

void ImuPreintegration::repropagate(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg) {
  ba_lin = ba;
  bg_lin = bg;
  reset();
  for (const Sample& s : samples_) {
    integrate_sample(s);
  }
}

void ImuPreintegration::integrate_sample(const Sample& s) {
  const double dt = s.dt;
  const Eigen::Vector3d acc = s.acc - ba_lin;
  const Eigen::Vector3d omega_dt = (s.gyro - bg_lin) * dt;

  const Eigen::Matrix3d R = delta_q.toRotationMatrix();   // Delta R_ik
  const Eigen::Matrix3d dR = quaternion_exp(omega_dt).toRotationMatrix();
  const Eigen::Matrix3d Jr = right_jacobian(omega_dt);
  const Eigen::Matrix3d acc_hat = skew(acc);

  // Noise propagation of [dtheta, dv, dp] (Forster Eq. 62).
  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
  A.block<3, 3>(0, 0) = dR.transpose();
  A.block<3, 3>(3, 0) = -R * acc_hat * dt;
  A.block<3, 3>(6, 0) = -0.5 * R * acc_hat * dt * dt;
  A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

  Eigen::Matrix<double, 9, 3> B_g = Eigen::Matrix<double, 9, 3>::Zero();
  B_g.block<3, 3>(0, 0) = Jr * dt;
  Eigen::Matrix<double, 9, 3> B_a = Eigen::Matrix<double, 9, 3>::Zero();
  B_a.block<3, 3>(3, 0) = R * dt;
  B_a.block<3, 3>(6, 0) = 0.5 * R * dt * dt;

  // Discrete-time noise of a continuous density sigma^2: sigma^2 / dt.
  covariance = A * covariance * A.transpose() +
               B_g * (noise_.gyro_noise / dt) * B_g.transpose() +
               B_a * (noise_.acc_noise / dt) * B_a.transpose();

  // Bias Jacobians (Forster Eq. 69-70, incremental form). Order matters:
  // each line uses the values from before this sample.
  dp_dba += dv_dba * dt - 0.5 * R * dt * dt;
  dp_dbg += dv_dbg * dt - 0.5 * R * acc_hat * dR_dbg * dt * dt;
  dv_dba += -R * dt;
  dv_dbg += -R * acc_hat * dR_dbg * dt;
  dR_dbg = dR.transpose() * dR_dbg - Jr * dt;

  // Mean.
  delta_p += delta_v * dt + 0.5 * R * acc * dt * dt;
  delta_v += R * acc * dt;
  delta_q = (delta_q * quaternion_exp(omega_dt)).normalized();
  sum_dt += dt;
}

void ImuPreintegration::predict(const Eigen::Vector3d& p_i, const Eigen::Quaterniond& q_i,
                                const Eigen::Vector3d& v_i, const Eigen::Vector3d& g,
                                Eigen::Vector3d& p_j, Eigen::Quaterniond& q_j,
                                Eigen::Vector3d& v_j) const {
  const double dt = sum_dt;
  q_j = (q_i * delta_q).normalized();
  v_j = v_i + g * dt + q_i * delta_v;
  p_j = p_i + v_i * dt + 0.5 * g * dt * dt + q_i * delta_p;
}

Eigen::Matrix<double, 15, 15> ImuPreintegration::residual_covariance() const {
  Eigen::Matrix<double, 15, 15> C = Eigen::Matrix<double, 15, 15>::Zero();
  C.topLeftCorner<9, 9>() = covariance;
  // Bias random walk over the interval.
  C.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * noise_.acc_bias_noise * sum_dt;
  C.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * noise_.gyro_bias_noise * sum_dt;
  return C;
}

}  // namespace msckf::swo
