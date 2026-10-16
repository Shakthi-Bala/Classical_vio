#include "msckf/math_utils.h"

#include <cmath>

namespace msckf {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S <<   0.0, -v(2),  v(1),
        v(2),   0.0, -v(0),
       -v(1),  v(0),   0.0;
  return S;
}

Eigen::Matrix3d to_rotation(const Eigen::Vector4d& q_in) {
  const Eigen::Vector4d q = quaternion_normalize(q_in);
  const Eigen::Vector3d vec = q.head<3>();
  const double w = q(3);

  return (2.0 * w * w - 1.0) * Eigen::Matrix3d::Identity()
         - 2.0 * w * skew(vec)
         + 2.0 * vec * vec.transpose();
}

Eigen::Vector4d to_quaternion(const Eigen::Matrix3d& R) {
  Eigen::Vector4d q;
  if (R(2, 2) < 0.0) {
    if (R(0, 0) > R(1, 1)) {
      const double t = 1.0 + R(0, 0) - R(1, 1) - R(2, 2);
      q << t, R(0, 1) + R(1, 0), R(2, 0) + R(0, 2), R(1, 2) - R(2, 1);
    } else {
      const double t = 1.0 - R(0, 0) + R(1, 1) - R(2, 2);
      q << R(0, 1) + R(1, 0), t, R(2, 1) + R(1, 2), R(2, 0) - R(0, 2);
    }
  } else {
    if (R(0, 0) < -R(1, 1)) {
      const double t = 1.0 - R(0, 0) - R(1, 1) + R(2, 2);
      q << R(0, 2) + R(2, 0), R(2, 1) + R(1, 2), t, R(0, 1) - R(1, 0);
    } else {
      const double t = 1.0 + R(0, 0) + R(1, 1) + R(2, 2);
      q << R(1, 2) - R(2, 1), R(2, 0) - R(0, 2), R(0, 1) - R(1, 0), t;
    }
  }
  return quaternion_normalize(q);
}

Eigen::Vector4d quaternion_normalize(const Eigen::Vector4d& q) {
  return q / q.norm();
}

Eigen::Vector4d quaternion_conjugate(const Eigen::Vector4d& q) {
  return Eigen::Vector4d(-q(0), -q(1), -q(2), q(3));
}

Eigen::Vector4d quaternion_multiplication(const Eigen::Vector4d& q1_in,
                                          const Eigen::Vector4d& q2_in) {
  const Eigen::Vector4d q1 = quaternion_normalize(q1_in);
  const Eigen::Vector4d q2 = quaternion_normalize(q2_in);

  Eigen::Matrix4d L;
  L <<  q1(3),  q1(2), -q1(1), q1(0),
       -q1(2),  q1(3),  q1(0), q1(1),
        q1(1), -q1(0),  q1(3), q1(2),
       -q1(0), -q1(1), -q1(2), q1(3);

  return quaternion_normalize(L * q2);
}

Eigen::Vector4d small_angle_quaternion(const Eigen::Vector3d& dtheta) {
  const Eigen::Vector3d dq = dtheta / 2.0;
  const double dq_square_norm = dq.squaredNorm();

  Eigen::Vector4d q;
  if (dq_square_norm <= 1.0) {
    q << dq, std::sqrt(1.0 - dq_square_norm);
  } else {
    q << dq, 1.0;
    q /= std::sqrt(1.0 + dq_square_norm);
  }
  return q;
}

Eigen::Vector4d from_two_vectors(const Eigen::Vector3d& v0_in,
                                 const Eigen::Vector3d& v1_in) {
  const Eigen::Vector3d v0 = v0_in.normalized();
  const Eigen::Vector3d v1 = v1_in.normalized();
  const double d = v0.dot(v1);

  Eigen::Vector4d q;
  if (d < -0.999999) {
    // Vectors are nearly opposite: rotate 180 deg about any perpendicular axis.
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX().cross(v0);
    if (axis.norm() < 0.000001) {
      axis = Eigen::Vector3d::UnitY().cross(v0);
    }
    q << axis, 0.0;
  } else if (d > 0.999999) {
    q << 0.0, 0.0, 0.0, 1.0;
  } else {
    const double s = std::sqrt((1.0 + d) * 2.0);
    const Eigen::Vector3d axis = v0.cross(v1);
    q << axis / s, 0.5 * s;
  }

  q = quaternion_normalize(q);
  return quaternion_conjugate(q);  // Hamilton -> JPL
}

double two_view_depth(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
                      const Eigen::Vector2d& z1, const Eigen::Vector2d& z2) {
  const Eigen::Vector3d m = R_21 * Eigen::Vector3d(z1(0), z1(1), 1.0);
  const Eigen::Vector2d a = m.head<2>() - z2 * m(2);
  const Eigen::Vector2d b = z2 * t_21(2) - t_21.head<2>();
  return a.dot(b) / a.dot(a);
}

Eigen::Matrix4d Isometry3d::matrix() const {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m.block<3, 3>(0, 0) = R;
  m.block<3, 1>(0, 3) = t;
  return m;
}

Isometry3d Isometry3d::inverse() const {
  return Isometry3d(R.transpose(), -R.transpose() * t);
}

Isometry3d Isometry3d::operator*(const Isometry3d& other) const {
  return Isometry3d(R * other.R, R * other.t + t);
}

}  // namespace msckf
