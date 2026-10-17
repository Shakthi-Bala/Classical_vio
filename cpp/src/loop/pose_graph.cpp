#include "msckf/loop/pose_graph.h"

#include <cmath>

#include <ceres/ceres.h>

namespace msckf::loop {

namespace {

// Loop edges get a robust loss (meters / radians), sequential edges none,
// as in VINS-Fusion.
constexpr double kLoopHuber = 0.1;

template <typename T>
T normalize_angle(const T& a) {
  // atan2(sin, cos) keeps autodiff derivatives intact.
  return ceres::atan2(ceres::sin(a), ceres::cos(a));
}

template <typename T>
Eigen::Matrix<T, 3, 3> ypr_to_rotation_t(const T& yaw, const T& pitch, const T& roll) {
  const T cy = ceres::cos(yaw), sy = ceres::sin(yaw);
  const T cp = ceres::cos(pitch), sp = ceres::sin(pitch);
  const T cr = ceres::cos(roll), sr = ceres::sin(roll);
  Eigen::Matrix<T, 3, 3> R;
  R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
       sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
       -sp, cp * sr, cp * cr;
  return R;
}

// Relative position of j in i's frame and relative yaw, against a measurement.
struct FourDofError {
  FourDofError(const Eigen::Vector3d& t_ij, double yaw_ij, double pitch_i, double roll_i)
      : t_ij(t_ij), yaw_ij(yaw_ij), pitch_i(pitch_i), roll_i(roll_i) {}

  template <typename T>
  bool operator()(const T* yaw_i, const T* p_i, const T* yaw_j, const T* p_j, T* r) const {
    const Eigen::Matrix<T, 3, 3> R_w_i = ypr_to_rotation_t(yaw_i[0], T(pitch_i), T(roll_i));
    const Eigen::Matrix<T, 3, 1> dp(p_j[0] - p_i[0], p_j[1] - p_i[1], p_j[2] - p_i[2]);
    const Eigen::Matrix<T, 3, 1> t_est = R_w_i.transpose() * dp;
    for (int k = 0; k < 3; ++k) {
      r[k] = t_est(k) - T(t_ij(k));
    }
    r[3] = normalize_angle(yaw_j[0] - yaw_i[0] - T(yaw_ij));
    return true;
  }

  Eigen::Vector3d t_ij;
  double yaw_ij, pitch_i, roll_i;
};

}  // namespace

Eigen::Matrix3d ypr_to_rotation(double yaw, double pitch, double roll) {
  return ypr_to_rotation_t(yaw, pitch, roll);
}

Eigen::Vector3d rotation_to_ypr(const Eigen::Matrix3d& R) {
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  const double pitch = std::atan2(-R(2, 0), std::hypot(R(0, 0), R(1, 0)));
  const double roll = std::atan2(R(2, 1), R(2, 2));
  return {yaw, pitch, roll};
}

int PoseGraph::add_keyframe(const Isometry3d& T_w_b_vio, const Isometry3d& T_w_b_init) {
  Node n;
  n.vio = T_w_b_vio;
  const Eigen::Vector3d ypr_vio = rotation_to_ypr(T_w_b_vio.R);
  n.pitch = ypr_vio(1);
  n.roll = ypr_vio(2);
  n.yaw[0] = rotation_to_ypr(T_w_b_init.R)(0);
  for (int k = 0; k < 3; ++k) {
    n.p[k] = T_w_b_init.t(k);
  }
  nodes_.push_back(n);

  const int j = size() - 1;
  for (int i = std::max(0, j - kSequentialEdges); i < j; ++i) {
    const Isometry3d& a = nodes_[i].vio;
    const Isometry3d& b = nodes_[j].vio;
    const double yaw_ij = rotation_to_ypr(b.R)(0) - rotation_to_ypr(a.R)(0);
    edges_.push_back({i, j, a.R.transpose() * (b.t - a.t), yaw_ij, false});
  }
  return j;
}

void PoseGraph::add_loop(int i, int j, const Eigen::Vector3d& t_ij, double yaw_ij) {
  edges_.push_back({i, j, t_ij, yaw_ij, true});
  ++num_loops_;
}

void PoseGraph::optimize(int max_iterations) {
  if (nodes_.size() < 2) {
    return;
  }
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  ceres::HuberLoss huber(kLoopHuber);
  for (Node& n : nodes_) {
    problem.AddParameterBlock(n.yaw, 1);
    problem.AddParameterBlock(n.p, 3);
  }
  for (const Edge& e : edges_) {
    Node& a = nodes_[e.i];
    Node& b = nodes_[e.j];
    auto* cost = new ceres::AutoDiffCostFunction<FourDofError, 4, 1, 3, 1, 3>(
        new FourDofError(e.t_ij, e.yaw_ij, a.pitch, a.roll));
    problem.AddResidualBlock(cost, e.loop ? &huber : nullptr, a.yaw, a.p, b.yaw, b.p);
  }
  problem.SetParameterBlockConstant(nodes_[0].yaw);
  problem.SetParameterBlockConstant(nodes_[0].p);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = max_iterations;
  options.num_threads = 1;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
}

Isometry3d PoseGraph::pose(int i) const {
  const Node& n = nodes_[i];
  return Isometry3d(ypr_to_rotation(n.yaw[0], n.pitch, n.roll),
                    Eigen::Vector3d(n.p[0], n.p[1], n.p[2]));
}

}  // namespace msckf::loop
