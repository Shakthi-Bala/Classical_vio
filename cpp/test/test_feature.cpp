// gtest port of Code/test/feature_initialization_test.py.

#include <random>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "msckf/feature.h"

using namespace msckf;

namespace {

// Six cameras on the unit sphere, on the coordinate axes, all looking at the
// origin. Each pose takes a vector from the camera frame to the world frame.
std::vector<Isometry3d> sphere_cameras() {
  auto pose = [](std::initializer_list<double> r, Eigen::Vector3d t) {
    Eigen::Matrix3d R;
    auto it = r.begin();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R(i, j) = *it++;
      }
    }
    return Isometry3d(R, t);
  };
  return {
      pose({0, 0, -1, 1, 0, 0, 0, -1, 0}, Eigen::Vector3d(1, 0, 0)),
      pose({-1, 0, 0, 0, 0, -1, 0, -1, 0}, Eigen::Vector3d(0, 1, 0)),
      pose({0, 0, 1, -1, 0, 0, 0, -1, 0}, Eigen::Vector3d(-1, 0, 0)),
      pose({1, 0, 0, 0, 0, 1, 0, -1, 0}, Eigen::Vector3d(0, -1, 0)),
      pose({0, -1, 0, -1, 0, 0, 0, 0, -1}, Eigen::Vector3d(0, 0, 1)),
      pose({1, 0, 0, 0, 1, 0, 0, 0, 1}, Eigen::Vector3d(0, 0, -1)),
  };
}

// Triangulates `feature` from the six cameras with the given pixel noise.
Eigen::Vector3d triangulate(const Eigen::Vector3d& feature, double noise_std,
                            std::mt19937& rng, bool* status) {
  // Mono setup, as in the Python test: cam1 == cam0.
  Feature::R_cam0_cam1 = Eigen::Matrix3d::Identity();
  Feature::t_cam0_cam1 = Eigen::Vector3d::Zero();

  const std::vector<Isometry3d> cam_poses = sphere_cameras();
  std::normal_distribution<double> noise(0.0, noise_std);

  CamStateServer cam_states;
  Feature f(0, OptimizationConfig{});
  for (int i = 0; i < 6; ++i) {
    CAMState cs;
    cs.id = i;
    cs.timestamp = i;
    cs.orientation = to_quaternion(cam_poses[i].R.transpose());
    cs.position = cam_poses[i].t;
    cam_states[i] = cs;

    const Isometry3d cam_pose_inv = cam_poses[i].inverse();
    const Eigen::Vector3d p = cam_pose_inv.R * feature + cam_pose_inv.t;
    const double u = p(0) / p(2) + noise(rng);
    const double v = p(1) / p(2) + noise(rng);
    f.observations[i] = Eigen::Vector4d(u, v, u, v);
  }
  *status = f.initialize_position(cam_states);
  return f.position;
}

}  // namespace

TEST(Feature, InitializationWithNoise) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> uniform(0.0, 0.5);
  for (int trial = 0; trial < 20; ++trial) {
    const Eigen::Vector3d feature(uniform(rng), uniform(rng), uniform(rng));
    bool status = false;
    const Eigen::Vector3d estimate = triangulate(feature, 0.01, rng, &status);
    EXPECT_TRUE(status);
    EXPECT_LT((estimate - feature).norm(), 0.05) << "trial " << trial;
  }
}

TEST(Feature, InitializationNoiseFreeIsExact) {
  std::mt19937 rng(11);
  const Eigen::Vector3d feature(0.3, -0.2, 0.1);
  bool status = false;
  const Eigen::Vector3d estimate = triangulate(feature, 0.0, rng, &status);
  EXPECT_TRUE(status);
  EXPECT_NEAR((estimate - feature).norm(), 0.0, 1e-9);
}

TEST(Feature, HuberWeight) {
  Feature f(0, OptimizationConfig{});
  const Isometry3d identity;
  Eigen::Matrix<double, 2, 3> J;
  Eigen::Vector2d r;
  double w = 0.0;
  // Point (0.1, 0.2) at depth 1; a small residual keeps the full weight.
  f.jacobian(identity, Eigen::Vector3d(0.1, 0.2, 1.0), Eigen::Vector2d(0.1, 0.205), J, r, w);
  EXPECT_DOUBLE_EQ(w, 1.0);
  // A large residual is down-weighted.
  f.jacobian(identity, Eigen::Vector3d(0.1, 0.2, 1.0), Eigen::Vector2d(0.1, 0.5), J, r, w);
  EXPECT_LT(w, 1.0);
}
