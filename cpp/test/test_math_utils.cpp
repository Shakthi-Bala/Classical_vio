// gtest port of Code/test/utils_test.py.

#include <random>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "msckf/math_utils.h"

using namespace msckf;

namespace {

constexpr double kTol = 1e-9;

// Each test seeds its own generator, so a failure is reproducible and does not
// depend on which other tests ran (Python's tests used np.random unseeded).
Eigen::Vector4d random_unit_quaternion(std::mt19937& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::Vector4d q(normal(rng), normal(rng), normal(rng), normal(rng));
  return q / q.norm();
}

Eigen::Vector3d random_vector3(std::mt19937& rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  return Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
}

}  // namespace

TEST(MathUtils, Skew) {
  const Eigen::Vector3d w(1.0, 2.0, 3.0);
  const Eigen::Matrix3d w_hat = skew(w);

  EXPECT_EQ(Eigen::FullPivLU<Eigen::Matrix3d>(w_hat).rank(), 2);
  EXPECT_TRUE((w_hat * w).isZero(0.0));   // exactly zero, as in Python
  // skew(a) * b is the cross product.
  const Eigen::Vector3d b(-0.5, 4.0, 0.25);
  EXPECT_NEAR((w_hat * b - w.cross(b)).norm(), 0.0, kTol);
}

TEST(MathUtils, Normalize) {
  const Eigen::Vector4d v(1.0, 2.0, 3.0, 4.0);
  const Eigen::Vector4d q = quaternion_normalize(v);
  EXPECT_NEAR(q.norm(), 1.0, kTol);
  EXPECT_NEAR((q * v.norm() - v).norm(), 0.0, kTol);   // same direction
}

TEST(MathUtils, ToRotation) {
  std::mt19937 rng(1);
  // Known value from utils_test.py (note the transpose there).
  Eigen::Vector4d q(-1.0, 1.0, 3.0, 2.0);
  q /= q.norm();
  Eigen::Matrix3d R_gt;
  R_gt << -1 / 3., -14 / 15., -2 / 15.,
           2 / 3., -1 / 3., 2 / 3.,
          -2 / 3., 2 / 15., 11 / 15.;
  R_gt.transposeInPlace();
  EXPECT_NEAR((to_rotation(q) - R_gt).norm(), 0.0, kTol);

  for (int i = 0; i < 20; ++i) {
    const Eigen::Vector4d qi = random_unit_quaternion(rng);
    const Eigen::Matrix3d R = to_rotation(qi);
    const Eigen::Matrix3d R_inv = to_rotation(quaternion_conjugate(qi));
    EXPECT_NEAR((R * R_inv - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
    EXPECT_NEAR((R * R.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
  }
}

TEST(MathUtils, ToQuaternion) {
  std::mt19937 rng(2);
  const Eigen::Vector4d q = to_quaternion(Eigen::Matrix3d::Identity());
  EXPECT_NEAR((q - Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)).norm(), 0.0, kTol);

  for (int i = 0; i < 20; ++i) {
    const Eigen::Matrix3d R = to_rotation(random_unit_quaternion(rng));
    const Eigen::Matrix3d R2 = to_rotation(to_quaternion(R));
    EXPECT_NEAR((R - R2).norm(), 0.0, kTol);
  }
}

TEST(MathUtils, Multiplication) {
  std::mt19937 rng(3);
  for (int i = 0; i < 20; ++i) {
    const Eigen::Vector4d q1 = random_unit_quaternion(rng);
    const Eigen::Vector4d q2 = random_unit_quaternion(rng);
    const Eigen::Matrix3d R_prod = to_rotation(q1) * to_rotation(q2);
    const Eigen::Matrix3d R_prod_cp = to_rotation(quaternion_multiplication(q1, q2));
    EXPECT_NEAR((R_prod - R_prod_cp).norm(), 0.0, kTol);
  }
}

TEST(MathUtils, FromTwoVectors) {
  std::mt19937 rng(4);
  for (int i = 0; i < 20; ++i) {
    const Eigen::Vector3d v0 = random_vector3(rng).normalized();
    const Eigen::Vector3d v1 = random_vector3(rng).normalized();

    const Eigen::Matrix3d R = to_rotation(from_two_vectors(v0, v1));
    EXPECT_NEAR((R * v0 - v1).norm(), 0.0, kTol);

    const Eigen::Matrix3d R_inv = to_rotation(from_two_vectors(v1, v0));
    EXPECT_NEAR((R * R_inv - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
  }
}

TEST(MathUtils, SmallAngleQuaternion) {
  // Empty in utils_test.py. For a small rotation vector the JPL rotation is
  // R = I - skew(dtheta) + O(|dtheta|^2).
  const Eigen::Vector3d dtheta(1e-4, -2e-4, 3e-4);
  const Eigen::Vector4d q = small_angle_quaternion(dtheta);
  EXPECT_NEAR(q.norm(), 1.0, kTol);
  const Eigen::Matrix3d R_first_order = Eigen::Matrix3d::Identity() - skew(dtheta);
  EXPECT_LT((to_rotation(q) - R_first_order).norm(), 1e-6);
  // Zero rotation vector -> identity quaternion.
  EXPECT_NEAR((small_angle_quaternion(Eigen::Vector3d::Zero()) -
               Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)).norm(), 0.0, kTol);
}

TEST(MathUtils, Isometry3d) {
  std::mt19937 rng(5);
  const Isometry3d T(to_rotation(random_unit_quaternion(rng)), random_vector3(rng));
  const Isometry3d T_identity = T * T.inverse();
  EXPECT_NEAR((T_identity.R - Eigen::Matrix3d::Identity()).norm(), 0.0, kTol);
  EXPECT_NEAR(T_identity.t.norm(), 0.0, kTol);

  const Isometry3d T1(to_rotation(random_unit_quaternion(rng)), random_vector3(rng));
  const Isometry3d T2(to_rotation(random_unit_quaternion(rng)), random_vector3(rng));
  const Isometry3d T3 = T1 * T2;
  EXPECT_NEAR((T3.matrix() - T1.matrix() * T2.matrix()).norm(), 0.0, kTol);
}
