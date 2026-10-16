#pragma once

#include <map>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "msckf/config.h"
#include "msckf/estimator.h"
#include "msckf/feature.h"
#include "msckf/math_utils.h"
#include "msckf/messages.h"
#include "msckf/state.h"

// Multi-State Constraint Kalman Filter (stereo). C++ port of MSCKF in Code/msckf.py.
//
// Error-state layout (21 + 6N):
//   [0:3) orientation  [3:6) gyro bias  [6:9) velocity  [9:12) acc bias
//   [12:15) position   [15:18) R_imu_cam0  [18:21) t_cam0_imu
//   then 6 per camera state (orientation, position), in camera-ID order.

namespace msckf {

class MSCKF : public Estimator {
 public:
  explicit MSCKF(const Config& config);

  void imu_callback(const ImuMsg& imu_msg) override;

  // Returns the new pose, or nothing while the filter is still initializing.
  std::optional<VioResult> feature_callback(const FeatureMsg& feature_msg) override;

  const IMUState& imu_state() const { return state_server_.imu_state; }
  const Eigen::MatrixXd& state_cov() const { return state_server_.state_cov; }
  size_t num_cam_states() const { return state_server_.cam_states.size(); }

 private:
  struct StateServer {
    IMUState imu_state;
    CamStateServer cam_states;
    Eigen::MatrixXd state_cov;   // (21 + 6N) x (21 + 6N)
    Eigen::Matrix<double, 12, 12> continuous_noise_cov;
  };

  void initialize_gravity_and_bias();
  void batch_imu_processing(double time_bound);
  void process_model(double time, const Eigen::Vector3d& m_gyro, const Eigen::Vector3d& m_acc);
  void predict_new_state(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc);
  void state_augmentation(double time);
  void add_feature_observations(const FeatureMsg& feature_msg);

  // Jacobians and residual of one feature seen in one camera state.
  void measurement_jacobian(StateID cam_state_id, FeatureID feature_id,
                            Eigen::Matrix<double, 4, 6>& H_x,
                            Eigen::Matrix<double, 4, 3>& H_f,
                            Eigen::Vector4d& r) const;
  // Stacked Jacobian of one feature over several camera states, projected
  // onto the left null space of the feature Jacobian.
  void feature_jacobian(FeatureID feature_id, const std::vector<StateID>& cam_state_ids,
                        Eigen::MatrixXd& H_x, Eigen::VectorXd& r) const;
  void measurement_update(const Eigen::MatrixXd& H, const Eigen::VectorXd& r);
  bool gating_test(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, int dof) const;

  void remove_lost_features();
  std::vector<StateID> find_redundant_cam_states() const;
  void prune_cam_state_buffer();

  void reset_state_cov();
  void online_reset();
  VioResult publish(double time) const;

  // Position of a camera state's 6x6 block, counted in camera states.
  int cam_state_index(StateID cam_id) const;
  static void symmetrize(Eigen::MatrixXd& P);

  Config config_;
  std::vector<ImuMsg> imu_msg_buffer_;
  StateServer state_server_;
  std::map<FeatureID, Feature> map_server_;
  std::vector<double> chi_squared_test_table_;   // index = degrees of freedom

  // Class attributes of IMUState in Python.
  Eigen::Vector3d gravity_ = Eigen::Vector3d(0.0, 0.0, -9.81);
  Isometry3d T_imu_body_;
  StateID next_imu_state_id_ = 0;

  double tracking_rate_ = 0.0;
  bool is_gravity_set_ = false;
  bool is_first_img_ = true;
};

}  // namespace msckf
