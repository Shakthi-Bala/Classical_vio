#include "msckf/feature.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "msckf/profiler.h"

namespace msckf {

namespace {
// Bounds of the LM damping factor (feature.py / upstream msckf_vio).
constexpr double kMinDamping = 1e-10;
constexpr double kMaxDamping = 1e12;
}  // namespace

double Feature::cost(const Isometry3d& T_c0_ci, const Eigen::Vector3d& x,
                     const Eigen::Vector2d& z) const {
  // h as in Equation (37).
  const double alpha = x(0), beta = x(1), rho = x(2);
  const Eigen::Vector3d h = T_c0_ci.R * Eigen::Vector3d(alpha, beta, 1.0) + rho * T_c0_ci.t;

  const Eigen::Vector2d z_hat = h.head<2>() / h(2);
  return (z_hat - z).squaredNorm();
}

void Feature::jacobian(const Isometry3d& T_c0_ci, const Eigen::Vector3d& x,
                       const Eigen::Vector2d& z, Eigen::Matrix<double, 2, 3>& J,
                       Eigen::Vector2d& r, double& w) const {
  const double alpha = x(0), beta = x(1), rho = x(2);
  const Eigen::Vector3d h = T_c0_ci.R * Eigen::Vector3d(alpha, beta, 1.0) + rho * T_c0_ci.t;
  const double h1 = h(0), h2 = h(1), h3 = h(2);

  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.R.leftCols<2>();
  W.col(2) = T_c0_ci.t;

  J.row(0) = W.row(0) / h3 - W.row(2) * h1 / (h3 * h3);
  J.row(1) = W.row(1) / h3 - W.row(2) * h2 / (h3 * h3);

  const Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
  r = z_hat - z;

  // Huber weight. NOTE (Python parity): feature.py uses eps / (2 e); upstream
  // msckf_vio uses sqrt(2 eps / e) (quirk 9 in README).
  const double e = r.norm();
  if (e <= optimization_config.huber_epsilon) {
    w = 1.0;
  } else {
    w = optimization_config.huber_epsilon / (2.0 * e);
  }
}

Eigen::Vector3d Feature::generate_initial_guess(const Isometry3d& T_c1_c2,
                                                const Eigen::Vector2d& z1,
                                                const Eigen::Vector2d& z2) const {
  const double depth = two_view_depth(T_c1_c2.R, T_c1_c2.t, z1, z2);
  return Eigen::Vector3d(z1(0), z1(1), 1.0) * depth;
}

bool Feature::has_stereo_observation() const {
  for (const auto& [cam_id, z] : observations) {
    if (is_stereo(z)) {
      return true;
    }
  }
  return false;
}

bool Feature::check_motion(const CamStateServer& cam_states) const {
  return check_motion(cam_states, optimization_config.translation_threshold);
}

bool Feature::check_motion(const CamStateServer& cam_states, double threshold) const {
  if (threshold < 0) {
    return true;
  }

  const StateID first_id = observations.begin()->first;
  const StateID last_id = observations.rbegin()->first;

  const CAMState& first_cam = cam_states.at(first_id);
  const CAMState& last_cam = cam_states.at(last_id);
  const Isometry3d first_cam_pose(to_rotation(first_cam.orientation).transpose(),
                                  first_cam.position);
  const Isometry3d last_cam_pose(to_rotation(last_cam.orientation).transpose(),
                                 last_cam.position);

  // Direction of the feature when it was first observed, in the world frame.
  const Eigen::Vector4d& z_first = observations.at(first_id);
  Eigen::Vector3d feature_direction(z_first(0), z_first(1), 1.0);
  feature_direction.normalize();
  feature_direction = first_cam_pose.R * feature_direction;

  // Translation between the first and last frame, perpendicular to the ray.
  const Eigen::Vector3d translation = last_cam_pose.t - first_cam_pose.t;
  const double parallel = translation.dot(feature_direction);
  const Eigen::Vector3d orthogonal_translation = translation - parallel * feature_direction;

  return orthogonal_translation.norm() > threshold;
}

bool Feature::initialize_position(const CamStateServer& cam_states) {
  MSCKF_PROFILE("backend/*/triangulation");
  std::vector<Isometry3d> cam_poses;
  std::vector<Eigen::Vector2d> measurements;

  const Isometry3d T_cam1_cam0 = Isometry3d(R_cam0_cam1, t_cam0_cam1).inverse();

  size_t last_cam0 = 0;   // index of the last cam0 observation
  for (const auto& [cam_id, m] : observations) {
    const auto it = cam_states.find(cam_id);
    if (it == cam_states.end()) {
      continue;
    }
    const CAMState& cam_state = it->second;

    // Takes a vector from this camera frame to the world frame.
    const Isometry3d cam0_pose(to_rotation(cam_state.orientation).transpose(),
                               cam_state.position);
    last_cam0 = cam_poses.size();
    measurements.push_back(m.head<2>());
    cam_poses.push_back(cam0_pose);
    if (is_stereo(m)) {   // cam0-only observations (Config::mono_features) have no cam1 part
      measurements.push_back(m.tail<2>());
      cam_poses.push_back(cam0_pose * T_cam1_cam0);
    }
  }
  if (cam_poses.empty()) {
    return false;
  }

  // Re-express every pose so it takes a vector from the first camera frame
  // to that camera frame.
  const Isometry3d T_c0_w = cam_poses[0];
  for (Isometry3d& pose : cam_poses) {
    pose = pose.inverse() * T_c0_w;
  }

  // Initial guess from the first and the last cam0 observation.
  const Eigen::Vector3d initial_position =
      generate_initial_guess(cam_poses[last_cam0], measurements[0], measurements[last_cam0]);
  Eigen::Vector3d solution(initial_position(0) / initial_position(2),
                           initial_position(1) / initial_position(2),
                           1.0 / initial_position(2));

  // Levenberg-Marquardt.
  const OptimizationConfig& oc = optimization_config;
  double lambda = oc.initial_damping;
  int inner_loop_count = 0;
  int outer_loop_count = 0;
  bool is_cost_reduced = false;
  double delta_norm = std::numeric_limits<double>::infinity();

  double total_cost = 0.0;
  for (size_t i = 0; i < cam_poses.size(); ++i) {
    total_cost += cost(cam_poses[i], solution, measurements[i]);
  }

  // NOTE (Python parity): `is_cost_reduced` is never reset inside the outer
  // loop, so after the first accepted step the inner loop does not run again.
  // Upstream msckf_vio uses do-while loops and keeps iterating. Kept as in
  // feature.py by default; oc.lm_upstream_loops forces one step per outer
  // iteration like the do-while (quirk 2 in README).
  while (outer_loop_count < oc.outer_loop_max_iteration &&
         delta_norm > oc.estimation_precision) {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w = 0.0;
      jacobian(cam_poses[i], solution, measurements[i], J, r, w);
      A += w * w * J.transpose() * J;
      b += w * w * J.transpose() * r;
    }

    bool force_step = oc.lm_upstream_loops;
    while (inner_loop_count < oc.inner_loop_max_iteration && (!is_cost_reduced || force_step)) {
      force_step = false;
      const Eigen::Vector3d delta =
          (A + lambda * Eigen::Matrix3d::Identity()).partialPivLu().solve(b);
      const Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();

      double new_cost = 0.0;
      for (size_t i = 0; i < cam_poses.size(); ++i) {
        new_cost += cost(cam_poses[i], new_solution, measurements[i]);
      }

      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = std::max(lambda / 10.0, kMinDamping);
      } else {
        is_cost_reduced = false;
        lambda = std::min(lambda * 10.0, kMaxDamping);
      }
      ++inner_loop_count;
    }
    inner_loop_count = 0;
    ++outer_loop_count;
  }

  // Inverse depth -> 3-D point in the first camera frame.
  const Eigen::Vector3d final_position(solution(0) / solution(2),
                                       solution(1) / solution(2),
                                       1.0 / solution(2));

  // The point must be in front of every camera.
  bool is_valid_solution = true;
  for (const Isometry3d& pose : cam_poses) {
    const Eigen::Vector3d p = pose.R * final_position + pose.t;
    if (p(2) <= 0) {
      is_valid_solution = false;
      break;
    }
  }

  position = T_c0_w.R * final_position + T_c0_w.t;
  is_initialized = is_valid_solution;
  return is_valid_solution;
}

}  // namespace msckf
