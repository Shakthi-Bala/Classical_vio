#pragma once

#include <Eigen/Dense>

// Math helpers for the MSCKF. C++ port of Code/utils.py.
//
// Quaternion convention: JPL, stored as Eigen::Vector4d [x, y, z, w]
// (scalar LAST). This is NOT Eigen::Quaterniond, which is Hamilton.

namespace msckf {

// Skew-symmetric matrix of v, so that skew(a) * b == a.cross(b).
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

// JPL quaternion -> rotation matrix (Trawny & Roumeliotis, Eq. 78).
Eigen::Matrix3d to_rotation(const Eigen::Vector4d& q);

// Rotation matrix -> JPL quaternion.
Eigen::Vector4d to_quaternion(const Eigen::Matrix3d& R);

Eigen::Vector4d quaternion_normalize(const Eigen::Vector4d& q);

Eigen::Vector4d quaternion_conjugate(const Eigen::Vector4d& q);

// Returns q1 * q2 (JPL), normalized.
Eigen::Vector4d quaternion_multiplication(const Eigen::Vector4d& q1,
                                          const Eigen::Vector4d& q2);

// Small rotation vector dtheta -> quaternion (Trawny & Roumeliotis, Eq. 238-239).
Eigen::Vector4d small_angle_quaternion(const Eigen::Vector3d& dtheta);

// JPL quaternion of the rotation taking v0 to v1.
Eigen::Vector4d from_two_vectors(const Eigen::Vector3d& v0,
                                 const Eigen::Vector3d& v1);

// Least-squares depth along the ray (z1, 1) of view 1 such that the point
// projects to z2 in view 2; (R_21, t_21) take a vector from view 1 to view 2.
// Not finite if the rays are parallel.
double two_view_depth(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
                      const Eigen::Vector2d& z1, const Eigen::Vector2d& z2);

// 3-D rigid transform: x' = R * x + t.
struct Isometry3d {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Isometry3d() = default;
  Isometry3d(const Eigen::Matrix3d& R_in, const Eigen::Vector3d& t_in)
      : R(R_in), t(t_in) {}

  Eigen::Matrix4d matrix() const;
  Isometry3d inverse() const;
  Isometry3d operator*(const Isometry3d& other) const;
};

}  // namespace msckf
