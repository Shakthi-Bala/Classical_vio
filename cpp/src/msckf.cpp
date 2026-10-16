#include "msckf/msckf.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include <boost/math/distributions/chi_squared.hpp>

namespace msckf {

MSCKF::MSCKF(const Config& config) : config_(config) {
  // Chi-squared table. NOTE (Python parity): chi2.ppf(0.05, dof) is the 5th
  // percentile, not the 95 % confidence the comment in msckf.py claims.
  // Upstream msckf_vio does the same. Kept on purpose.
  chi_squared_test_table_.assign(100, 0.0);
  for (int i = 1; i < 100; ++i) {
    const boost::math::chi_squared_distribution<double> dist(i);
    chi_squared_test_table_[i] = boost::math::quantile(dist, 0.05);
  }

  state_server_.imu_state.velocity = config_.velocity;
  reset_state_cov();

  Eigen::Matrix<double, 12, 12>& Q = state_server_.continuous_noise_cov;
  Q.setIdentity();
  Q.block<3, 3>(0, 0) *= config_.gyro_noise;
  Q.block<3, 3>(3, 3) *= config_.gyro_bias_noise;
  Q.block<3, 3>(6, 6) *= config_.acc_noise;
  Q.block<3, 3>(9, 9) *= config_.acc_bias_noise;

  gravity_ = Eigen::Vector3d(0.0, 0.0, -config_.gravity_acc);

  // IMU-camera extrinsics.
  const Eigen::Matrix4d T_cam0_imu = config_.cam0.T_imu_cam.inverse();
  state_server_.imu_state.R_imu_cam0 = T_cam0_imu.topLeftCorner<3, 3>().transpose();
  state_server_.imu_state.t_cam0_imu = T_cam0_imu.topRightCorner<3, 1>();

  // Stereo extrinsics, shared with the features.
  Feature::R_cam0_cam1 = config_.T_cn_cnm1.topLeftCorner<3, 3>();
  Feature::t_cam0_cam1 = config_.T_cn_cnm1.topRightCorner<3, 1>();
  T_imu_body_ = Isometry3d(config_.T_imu_body.topLeftCorner<3, 3>(),
                           config_.T_imu_body.topRightCorner<3, 1>());
}

void MSCKF::imu_callback(const ImuMsg& imu_msg) {
  // Buffered; processed when the next image arrives.
  imu_msg_buffer_.push_back(imu_msg);
  if (!is_gravity_set_ && imu_msg_buffer_.size() >= 200) {
    initialize_gravity_and_bias();
    is_gravity_set_ = true;
  }
}

std::optional<VioResult> MSCKF::feature_callback(const FeatureMsg& feature_msg) {
  if (!is_gravity_set_) {
    return std::nullopt;
  }
  // The frame of the first image is the world origin.
  if (is_first_img_) {
    is_first_img_ = false;
    state_server_.imu_state.timestamp = feature_msg.timestamp;
  }

  batch_imu_processing(feature_msg.timestamp);
  state_augmentation(feature_msg.timestamp);
  add_feature_observations(feature_msg);
  remove_lost_features();
  prune_cam_state_buffer();

  const VioResult result = publish(feature_msg.timestamp);
  online_reset();
  return result;
}

void MSCKF::initialize_gravity_and_bias() {
  // Assumes the IMU is static during the first 200 samples.
  Eigen::Vector3d sum_angular_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_linear_acc = Eigen::Vector3d::Zero();
  for (const ImuMsg& m : imu_msg_buffer_) {
    sum_angular_vel += m.angular_velocity;
    sum_linear_acc += m.linear_acceleration;
  }
  const double n = static_cast<double>(imu_msg_buffer_.size());
  state_server_.imu_state.gyro_bias = sum_angular_vel / n;
  const Eigen::Vector3d gravity_imu = sum_linear_acc / n;

  const double gravity_norm = gravity_imu.norm();
  gravity_ = Eigen::Vector3d(0.0, 0.0, -gravity_norm);

  // Orientation such that to_rotation(q) * [0, 0, 1] = measured gravity direction.
  const Eigen::Vector3d gravity_imu_normalized = gravity_imu / gravity_norm;
  state_server_.imu_state.orientation =
      from_two_vectors(Eigen::Vector3d(0.0, 0.0, 1.0), gravity_imu_normalized);
}

void MSCKF::batch_imu_processing(double time_bound) {
  size_t used_imu_msg_count = 0;
  for (const ImuMsg& m : imu_msg_buffer_) {
    if (m.timestamp < state_server_.imu_state.timestamp) {
      ++used_imu_msg_count;
      continue;
    }
    if (m.timestamp > time_bound) {
      break;
    }
    process_model(m.timestamp, m.angular_velocity, m.linear_acceleration);
    ++used_imu_msg_count;
  }

  state_server_.imu_state.id = next_imu_state_id_++;
  imu_msg_buffer_.erase(imu_msg_buffer_.begin(), imu_msg_buffer_.begin() + used_imu_msg_count);
}

void MSCKF::process_model(double time, const Eigen::Vector3d& m_gyro,
                          const Eigen::Vector3d& m_acc) {
  IMUState& imu_state = state_server_.imu_state;
  const Eigen::Vector3d gyro = m_gyro - imu_state.gyro_bias;
  const Eigen::Vector3d acc = m_acc - imu_state.acc_bias;
  const double dt = time - imu_state.timestamp;

  // Continuous-time error dynamics (MSCKF paper, Appendix A).
  using Matrix21 = Eigen::Matrix<double, 21, 21>;
  Matrix21 F = Matrix21::Zero();
  Eigen::Matrix<double, 21, 12> G = Eigen::Matrix<double, 21, 12>::Zero();

  const Eigen::Matrix3d R_w_i = to_rotation(imu_state.orientation);
  F.block<3, 3>(0, 0) = -skew(gyro);
  F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -R_w_i.transpose() * skew(acc);
  F.block<3, 3>(6, 9) = -R_w_i.transpose();
  F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -R_w_i.transpose();
  G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

  // Matrix exponential to 3rd order (dt ~ 5 ms).
  const Matrix21 Fdt = F * dt;
  const Matrix21 Fdt_square = Fdt * Fdt;
  const Matrix21 Fdt_cube = Fdt_square * Fdt;
  Matrix21 Phi = Matrix21::Identity() + Fdt + 0.5 * Fdt_square + (1.0 / 6.0) * Fdt_cube;

  // Propagate the nominal state (RK4).
  predict_new_state(dt, gyro, acc);

  // Observability constraint on Phi (OC-EKF).
  // NOTE (Python parity): msckf.py uses u = R_w_i^T g with the pre-propagation
  // orientation and does not modify Phi[0:3, 0:3]. Upstream msckf_vio uses
  // u = R(orientation_null) g and also sets Phi[0:3, 0:3]. Kept as in Python.
  const Eigen::Vector3d u = R_w_i.transpose() * gravity_;
  const double u_squared = u.dot(u);

  const Eigen::Matrix3d A1 = Phi.block<3, 3>(6, 0);
  const Eigen::Vector3d w1 = skew(imu_state.velocity_null - imu_state.velocity) * gravity_;
  Phi.block<3, 3>(6, 0) = A1 - (A1 * u - w1) * u.transpose() / u_squared;

  const Eigen::Matrix3d A2 = Phi.block<3, 3>(12, 0);
  const Eigen::Vector3d w2 =
      skew(dt * imu_state.velocity_null + imu_state.position_null - imu_state.position) *
      gravity_;
  Phi.block<3, 3>(12, 0) = A2 - (A2 * u - w2) * u.transpose() / u_squared;

  // Propagate the covariance.
  const Matrix21 Q =
      Phi * G * state_server_.continuous_noise_cov * G.transpose() * Phi.transpose() * dt;

  Eigen::MatrixXd& P = state_server_.state_cov;
  P.topLeftCorner<21, 21>() = Phi * P.topLeftCorner<21, 21>() * Phi.transpose() + Q;

  const Eigen::Index n_cam = P.cols() - 21;
  if (n_cam > 0) {
    P.topRightCorner(21, n_cam) = Phi * P.topRightCorner(21, n_cam);
    P.bottomLeftCorner(n_cam, 21) = P.topRightCorner(21, n_cam).transpose();
  }
  symmetrize(P);

  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;
  imu_state.timestamp = time;
}

void MSCKF::predict_new_state(double dt, const Eigen::Vector3d& gyro,
                              const Eigen::Vector3d& acc) {
  const double gyro_norm = gyro.norm();
  Eigen::Matrix4d Omega = Eigen::Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skew(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro.transpose();

  IMUState& s = state_server_.imu_state;
  const Eigen::Vector4d q = s.orientation;
  const Eigen::Vector3d v = s.velocity;
  const Eigen::Vector3d p = s.position;

  Eigen::Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (std::cos(gyro_norm * dt * 0.5) * Eigen::Matrix4d::Identity() +
             std::sin(gyro_norm * dt * 0.5) / gyro_norm * Omega) * q;
    dq_dt2 = (std::cos(gyro_norm * dt * 0.25) * Eigen::Matrix4d::Identity() +
              std::sin(gyro_norm * dt * 0.25) / gyro_norm * Omega) * q;
  } else {
    // Python's version of this branch raises (ndarray @ scalar); this is the
    // intended small-angle formula from upstream msckf_vio.
    dq_dt = (Eigen::Matrix4d::Identity() + 0.5 * dt * Omega) * std::cos(gyro_norm * dt * 0.5) * q;
    dq_dt2 = (Eigen::Matrix4d::Identity() + 0.25 * dt * Omega) * std::cos(gyro_norm * dt * 0.25) * q;
  }
  const Eigen::Matrix3d dR_dt_transpose = to_rotation(dq_dt).transpose();
  const Eigen::Matrix3d dR_dt2_transpose = to_rotation(dq_dt2).transpose();

  // RK4 for velocity and position.
  const Eigen::Vector3d k1_v_dot = to_rotation(q).transpose() * acc + gravity_;
  const Eigen::Vector3d k1_p_dot = v;

  const Eigen::Vector3d k1_v = v + k1_v_dot * dt / 2;
  const Eigen::Vector3d k2_v_dot = dR_dt2_transpose * acc + gravity_;
  const Eigen::Vector3d k2_p_dot = k1_v;

  const Eigen::Vector3d k2_v = v + k2_v_dot * dt / 2;
  const Eigen::Vector3d k3_v_dot = dR_dt2_transpose * acc + gravity_;
  const Eigen::Vector3d k3_p_dot = k2_v;

  const Eigen::Vector3d k3_v = v + k3_v_dot * dt;
  const Eigen::Vector3d k4_v_dot = dR_dt_transpose * acc + gravity_;
  const Eigen::Vector3d k4_p_dot = k3_v;

  // NOTE (Python parity): q is not re-normalized here (upstream does).
  s.orientation = dq_dt;
  s.velocity = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
  s.position = p + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);
}

void MSCKF::state_augmentation(double time) {
  const IMUState& imu_state = state_server_.imu_state;
  const Eigen::Matrix3d& R_i_c = imu_state.R_imu_cam0;
  const Eigen::Vector3d& t_c_i = imu_state.t_cam0_imu;

  // New camera pose from the IMU pose and the extrinsics.
  const Eigen::Matrix3d R_w_i = to_rotation(imu_state.orientation);
  const Eigen::Matrix3d R_w_c = R_i_c * R_w_i;
  const Eigen::Vector3d t_c_w = imu_state.position + R_w_i.transpose() * t_c_i;

  CAMState& cam_state = state_server_.cam_states[imu_state.id];
  cam_state.id = imu_state.id;
  cam_state.timestamp = time;
  cam_state.orientation = to_quaternion(R_w_c);
  cam_state.position = t_c_w;
  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Jacobian of the new camera pose w.r.t. the IMU state (MSCKF paper, App. B).
  Eigen::Matrix<double, 6, 21> J = Eigen::Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skew(R_w_i.transpose() * t_c_i);
  J.block<3, 3>(3, 12) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(3, 18) = R_w_i.transpose();

  // Grow the covariance by 6 rows and columns.
  Eigen::MatrixXd& P = state_server_.state_cov;
  const Eigen::Index old_size = P.rows();
  P.conservativeResize(old_size + 6, old_size + 6);   // keeps the old block

  P.block(old_size, 0, 6, old_size) = J * P.block(0, 0, 21, old_size);
  P.block(0, old_size, old_size, 6) = P.block(old_size, 0, 6, old_size).transpose();
  P.block<6, 6>(old_size, old_size) = J * P.topLeftCorner<21, 21>() * J.transpose();

  symmetrize(P);
}

void MSCKF::add_feature_observations(const FeatureMsg& feature_msg) {
  const StateID state_id = state_server_.imu_state.id;
  const size_t curr_feature_num = map_server_.size();
  size_t tracked_feature_num = 0;

  for (const FeatureMeasurement& f : feature_msg.features) {
    const Eigen::Vector4d z(f.u0, f.v0, f.u1, f.v1);
    auto it = map_server_.find(f.id);
    if (it == map_server_.end()) {
      Feature feature(f.id, config_.optimization_config);
      feature.observations[state_id] = z;
      map_server_.emplace(f.id, feature);
    } else {
      it->second.observations[state_id] = z;
      ++tracked_feature_num;
    }
  }

  tracking_rate_ = curr_feature_num > 0
                       ? static_cast<double>(tracked_feature_num) / curr_feature_num
                       : 1.0;
}

void MSCKF::measurement_jacobian(StateID cam_state_id, FeatureID feature_id,
                                 Eigen::Matrix<double, 4, 6>& H_x,
                                 Eigen::Matrix<double, 4, 3>& H_f,
                                 Eigen::Vector4d& r) const {
  const CAMState& cam_state = state_server_.cam_states.at(cam_state_id);
  const Feature& feature = map_server_.at(feature_id);

  // Cam0 and cam1 poses.
  const Eigen::Matrix3d R_w_c0 = to_rotation(cam_state.orientation);
  const Eigen::Vector3d& t_c0_w = cam_state.position;
  const Eigen::Matrix3d R_w_c1 = Feature::R_cam0_cam1 * R_w_c0;
  const Eigen::Vector3d t_c1_w = t_c0_w - R_w_c1.transpose() * Feature::t_cam0_cam1;

  const Eigen::Vector3d& p_w = feature.position;
  const Eigen::Vector4d& z = feature.observations.at(cam_state_id);

  // Feature in the cam0 and cam1 frames.
  const Eigen::Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
  const Eigen::Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);

  Eigen::Matrix<double, 4, 3> dz_dpc0 = Eigen::Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

  Eigen::Matrix<double, 4, 3> dz_dpc1 = Eigen::Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2) * p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2) * p_c1(2));

  Eigen::Matrix<double, 3, 6> dpc0_dxc;
  dpc0_dxc.leftCols<3>() = skew(p_c0);
  dpc0_dxc.rightCols<3>() = -R_w_c0;

  Eigen::Matrix<double, 3, 6> dpc1_dxc;
  dpc1_dxc.leftCols<3>() = Feature::R_cam0_cam1 * skew(p_c0);
  dpc1_dxc.rightCols<3>() = -R_w_c1;

  const Eigen::Matrix3d& dpc0_dpg = R_w_c0;
  const Eigen::Matrix3d& dpc1_dpg = R_w_c1;

  H_x = dz_dpc0 * dpc0_dxc + dz_dpc1 * dpc1_dxc;
  H_f = dz_dpc0 * dpc0_dpg + dz_dpc1 * dpc1_dpg;

  // Observability constraint on H_x.
  const Eigen::Matrix<double, 4, 6> A = H_x;
  Eigen::Matrix<double, 6, 1> u;
  u.head<3>() = to_rotation(cam_state.orientation_null) * gravity_;
  u.tail<3>() = skew(p_w - cam_state.position_null) * gravity_;
  H_x = A - (A * u) * u.transpose() / u.dot(u);

  r = z - Eigen::Vector4d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2),
                          p_c1(0) / p_c1(2), p_c1(1) / p_c1(2));
}

void MSCKF::feature_jacobian(FeatureID feature_id, const std::vector<StateID>& cam_state_ids,
                             Eigen::MatrixXd& H_x, Eigen::VectorXd& r) const {
  const Feature& feature = map_server_.at(feature_id);

  std::vector<StateID> valid_cam_state_ids;
  for (const StateID cam_id : cam_state_ids) {
    if (feature.observations.count(cam_id)) {
      valid_cam_state_ids.push_back(cam_id);
    }
  }

  const Eigen::Index rows = 4 * static_cast<Eigen::Index>(valid_cam_state_ids.size());
  const Eigen::Index cols = 21 + 6 * static_cast<Eigen::Index>(state_server_.cam_states.size());
  Eigen::MatrixXd H_xj = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::MatrixXd H_fj = Eigen::MatrixXd::Zero(rows, 3);
  Eigen::VectorXd r_j = Eigen::VectorXd::Zero(rows);

  Eigen::Index stack_count = 0;
  for (const StateID cam_id : valid_cam_state_ids) {
    Eigen::Matrix<double, 4, 6> H_xi;
    Eigen::Matrix<double, 4, 3> H_fi;
    Eigen::Vector4d r_i;
    measurement_jacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    const int idx = cam_state_index(cam_id);
    H_xj.block<4, 6>(stack_count, 21 + 6 * idx) = H_xi;
    H_fj.block<4, 3>(stack_count, 0) = H_fi;
    r_j.segment<4>(stack_count) = r_i;
    stack_count += 4;
  }

  // Left null space of H_fj: the last (rows - 3) columns of U.
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_fj, Eigen::ComputeFullU);
  const Eigen::MatrixXd A = svd.matrixU().rightCols(rows - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;
}

void MSCKF::measurement_update(const Eigen::MatrixXd& H, const Eigen::VectorXd& r) {
  if (H.rows() == 0 || r.rows() == 0) {
    return;
  }

  // Thin QR of a tall H: only R (cols x cols) and Q^T r are needed, so the
  // m x m matrix Q is never formed.
  Eigen::MatrixXd H_thin;
  Eigen::VectorXd r_thin;
  if (H.rows() > H.cols()) {
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(H);
    H_thin = qr.matrixQR().topRows(H.cols()).triangularView<Eigen::Upper>();
    r_thin = (qr.householderQ().transpose() * r).head(H.cols());
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Kalman gain K = P H^T S^-1, via a solve instead of an inverse.
  const Eigen::MatrixXd& P = state_server_.state_cov;
  const Eigen::MatrixXd S =
      H_thin * P * H_thin.transpose() +
      config_.observation_noise *
          Eigen::MatrixXd::Identity(H_thin.rows(), H_thin.rows());
  const Eigen::MatrixXd K = S.ldlt().solve(H_thin * P).transpose();

  const Eigen::VectorXd delta_x = K * r_thin;

  // IMU state.
  const Eigen::Matrix<double, 21, 1> delta_x_imu = delta_x.head<21>();
  IMUState& imu_state = state_server_.imu_state;
  const Eigen::Vector4d dq = small_angle_quaternion(delta_x_imu.segment<3>(0));
  imu_state.orientation = quaternion_multiplication(dq, imu_state.orientation);
  imu_state.orientation = quaternion_normalize(imu_state.orientation);
  imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  imu_state.velocity += delta_x_imu.segment<3>(6);
  imu_state.acc_bias += delta_x_imu.segment<3>(9);
  imu_state.position += delta_x_imu.segment<3>(12);

  const Eigen::Vector4d dq_extrinsic = small_angle_quaternion(delta_x_imu.segment<3>(15));
  imu_state.R_imu_cam0 = to_rotation(dq_extrinsic) * imu_state.R_imu_cam0;
  imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  // Camera states, in the same order as their covariance blocks.
  int i = 0;
  for (auto& [cam_id, cam_state] : state_server_.cam_states) {
    const Eigen::Matrix<double, 6, 1> delta_x_cam = delta_x.segment<6>(21 + 6 * i);
    const Eigen::Vector4d dq_cam = small_angle_quaternion(delta_x_cam.head<3>());
    cam_state.orientation = quaternion_multiplication(dq_cam, cam_state.orientation);
    cam_state.orientation = quaternion_normalize(cam_state.orientation);
    cam_state.position += delta_x_cam.tail<3>();
    ++i;
  }

  // Covariance.
  const Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(K.rows(), K.rows()) - K * H_thin;
  state_server_.state_cov = I_KH * P;   // product -> evaluated into a temporary first
  symmetrize(state_server_.state_cov);
}

bool MSCKF::gating_test(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, int dof) const {
  const Eigen::MatrixXd P1 = H * state_server_.state_cov * H.transpose();
  const Eigen::MatrixXd P2 =
      config_.observation_noise * Eigen::MatrixXd::Identity(H.rows(), H.rows());
  const double gamma = r.dot((P1 + P2).ldlt().solve(r));
  return gamma < chi_squared_test_table_.at(dof);
}

void MSCKF::remove_lost_features() {
  Eigen::Index jacobian_row_size = 0;
  std::vector<FeatureID> invalid_feature_ids;
  std::vector<FeatureID> processed_feature_ids;

  for (auto& [feature_id, feature] : map_server_) {
    // Still tracked: keep it.
    if (feature.observations.count(state_server_.imu_state.id)) {
      continue;
    }
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature_id);
      continue;
    }
    if (!feature.is_initialized) {
      if (!feature.check_motion(state_server_.cam_states)) {
        invalid_feature_ids.push_back(feature_id);
        continue;
      }
      if (!feature.initialize_position(state_server_.cam_states)) {
        invalid_feature_ids.push_back(feature_id);
        continue;
      }
    }
    jacobian_row_size += 4 * static_cast<Eigen::Index>(feature.observations.size()) - 3;
    processed_feature_ids.push_back(feature_id);
  }

  for (const FeatureID id : invalid_feature_ids) {
    map_server_.erase(id);
  }
  if (processed_feature_ids.empty()) {
    return;
  }

  Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(
      jacobian_row_size, 21 + 6 * static_cast<Eigen::Index>(state_server_.cam_states.size()));
  Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
  Eigen::Index stack_count = 0;

  for (const FeatureID feature_id : processed_feature_ids) {
    const Feature& feature = map_server_.at(feature_id);
    std::vector<StateID> cam_state_ids;
    for (const auto& [cam_id, measurement] : feature.observations) {
      cam_state_ids.push_back(cam_id);
    }

    Eigen::MatrixXd H_xj;
    Eigen::VectorXd r_j;
    feature_jacobian(feature_id, cam_state_ids, H_xj, r_j);

    if (gating_test(H_xj, r_j, static_cast<int>(cam_state_ids.size()) - 1)) {
      H_x.block(stack_count, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_count, r_j.rows()) = r_j;
      stack_count += H_xj.rows();
    }
    // Upper bound on the update size, to bound the run time.
    if (stack_count > 1500) {
      break;
    }
  }

  measurement_update(H_x.topRows(stack_count), r.head(stack_count));

  for (const FeatureID id : processed_feature_ids) {
    map_server_.erase(id);
  }
}

std::vector<StateID> MSCKF::find_redundant_cam_states() const {
  // Random access into the (sorted) camera states.
  std::vector<std::pair<StateID, const CAMState*>> cam_state_pairs;
  for (const auto& [id, cam_state] : state_server_.cam_states) {
    cam_state_pairs.emplace_back(id, &cam_state);
  }

  const int key_cam_state_idx = static_cast<int>(cam_state_pairs.size()) - 4;
  int cam_state_idx = key_cam_state_idx + 1;
  int first_cam_state_idx = 0;

  const Eigen::Vector3d& key_position = cam_state_pairs[key_cam_state_idx].second->position;
  const Eigen::Matrix3d key_rotation =
      to_rotation(cam_state_pairs[key_cam_state_idx].second->orientation);

  // Drop a recent state if it barely moved from the key frame, else the oldest.
  std::vector<StateID> rm_cam_state_ids;
  for (int i = 0; i < 2; ++i) {
    const CAMState& cam_state = *cam_state_pairs[cam_state_idx].second;
    const Eigen::Matrix3d rotation = to_rotation(cam_state.orientation);

    const double distance = (cam_state.position - key_position).norm();
    const double angle =
        2.0 * std::acos(to_quaternion(rotation * key_rotation.transpose())(3));

    if (angle < config_.rotation_threshold && distance < config_.translation_threshold &&
        tracking_rate_ > config_.tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_pairs[cam_state_idx].first);
      ++cam_state_idx;
    } else {
      rm_cam_state_ids.push_back(cam_state_pairs[first_cam_state_idx].first);
      ++first_cam_state_idx;
      ++cam_state_idx;
    }
  }

  std::sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());
  return rm_cam_state_ids;
}

void MSCKF::prune_cam_state_buffer() {
  if (static_cast<int>(state_server_.cam_states.size()) < config_.max_cam_state_size) {
    return;
  }

  const std::vector<StateID> rm_cam_state_ids = find_redundant_cam_states();

  // Pass 1: size the Jacobian; drop observations that cannot be used.
  Eigen::Index jacobian_row_size = 0;
  for (auto& [feature_id, feature] : map_server_) {
    std::vector<StateID> involved_cam_state_ids;
    for (const StateID cam_id : rm_cam_state_ids) {
      if (feature.observations.count(cam_id)) {
        involved_cam_state_ids.push_back(cam_id);
      }
    }
    if (involved_cam_state_ids.empty()) {
      continue;
    }
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }
    if (!feature.is_initialized) {
      if (!feature.check_motion(state_server_.cam_states) ||
          !feature.initialize_position(state_server_.cam_states)) {
        for (const StateID cam_id : involved_cam_state_ids) {
          feature.observations.erase(cam_id);
        }
        continue;
      }
    }
    jacobian_row_size += 4 * static_cast<Eigen::Index>(involved_cam_state_ids.size()) - 3;
  }

  // Pass 2: stack the Jacobians.
  Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(
      jacobian_row_size, 21 + 6 * static_cast<Eigen::Index>(state_server_.cam_states.size()));
  Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
  Eigen::Index stack_count = 0;

  for (auto& [feature_id, feature] : map_server_) {
    std::vector<StateID> involved_cam_state_ids;
    for (const StateID cam_id : rm_cam_state_ids) {
      if (feature.observations.count(cam_id)) {
        involved_cam_state_ids.push_back(cam_id);
      }
    }
    if (involved_cam_state_ids.empty()) {
      continue;
    }

    Eigen::MatrixXd H_xj;
    Eigen::VectorXd r_j;
    feature_jacobian(feature_id, involved_cam_state_ids, H_xj, r_j);

    if (gating_test(H_xj, r_j, static_cast<int>(involved_cam_state_ids.size()))) {
      H_x.block(stack_count, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_count, r_j.rows()) = r_j;
      stack_count += H_xj.rows();
    }
    for (const StateID cam_id : involved_cam_state_ids) {
      feature.observations.erase(cam_id);
    }
  }

  measurement_update(H_x.topRows(stack_count), r.head(stack_count));

  // Remove the camera states and their rows/columns of the covariance.
  for (const StateID cam_id : rm_cam_state_ids) {
    const Eigen::Index start = 21 + 6 * cam_state_index(cam_id);
    const Eigen::Index end = start + 6;
    Eigen::MatrixXd& P = state_server_.state_cov;
    const Eigen::Index size = P.rows();
    if (end < size) {
      // Shift the trailing blocks up and left. The source and destination
      // overlap, so evaluate the source into a temporary first.
      P.block(start, 0, size - end, size) = P.block(end, 0, size - end, size).eval();
      P.block(0, start, size, size - end) = P.block(0, end, size, size - end).eval();
    }
    P.conservativeResize(size - 6, size - 6);

    state_server_.cam_states.erase(cam_id);
  }
}

void MSCKF::reset_state_cov() {
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(21, 21);
  P.block<3, 3>(3, 3) = config_.gyro_bias_cov * Eigen::Matrix3d::Identity();
  P.block<3, 3>(6, 6) = config_.velocity_cov * Eigen::Matrix3d::Identity();
  P.block<3, 3>(9, 9) = config_.acc_bias_cov * Eigen::Matrix3d::Identity();
  P.block<3, 3>(15, 15) = config_.extrinsic_rotation_cov * Eigen::Matrix3d::Identity();
  P.block<3, 3>(18, 18) = config_.extrinsic_translation_cov * Eigen::Matrix3d::Identity();
  state_server_.state_cov = P;
}

void MSCKF::online_reset() {
  if (config_.position_std_threshold <= 0) {
    return;
  }
  const Eigen::MatrixXd& P = state_server_.state_cov;
  const double max_position_std =
      std::max({std::sqrt(P(12, 12)), std::sqrt(P(13, 13)), std::sqrt(P(14, 14))});
  if (max_position_std < config_.position_std_threshold) {
    return;
  }
  // Uncertainty too large: drop the window and the map (dead-reckoning reset).
  state_server_.cam_states.clear();
  map_server_.clear();
  reset_state_cov();
}

VioResult MSCKF::publish(double time) const {
  const IMUState& imu_state = state_server_.imu_state;
  const Isometry3d T_i_w(to_rotation(imu_state.orientation).transpose(), imu_state.position);

  VioResult result;
  result.timestamp = time;
  result.pose = T_imu_body_ * T_i_w * T_imu_body_.inverse();
  result.velocity = T_imu_body_.R * imu_state.velocity;

  const Eigen::Matrix3d R_w_c = imu_state.R_imu_cam0 * T_i_w.R.transpose();
  const Eigen::Vector3d t_c_w = imu_state.position + T_i_w.R * imu_state.t_cam0_imu;
  result.cam0_pose = Isometry3d(R_w_c.transpose(), t_c_w);
  return result;
}

int MSCKF::cam_state_index(StateID cam_id) const {
  const auto it = state_server_.cam_states.find(cam_id);
  return static_cast<int>(std::distance(state_server_.cam_states.begin(), it));
}

void MSCKF::symmetrize(Eigen::MatrixXd& P) {
  // `P = (P + P.transpose()) / 2` would read P while overwriting it (aliasing),
  // so build the result in a temporary first.
  const Eigen::MatrixXd P_sym = (P + P.transpose()) / 2.0;
  P = P_sym;
}

}  // namespace msckf
