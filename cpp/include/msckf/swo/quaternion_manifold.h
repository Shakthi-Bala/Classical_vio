#pragma once

#include <Eigen/Geometry>
#include <ceres/manifold.h>

// Unit-quaternion manifold with a right (body-frame) perturbation:
//
//   Plus(q, d)  = q * Exp(d)          d: rotation vector [rad], 3-D
//   Minus(y, x) = Log(x^-1 * y)
//
// Storage is Eigen's coefficient order (x, y, z, w), so a parameter block can
// be viewed as Eigen::Map<Eigen::Quaterniond>. The sliding-window optimizer
// uses Hamilton quaternions (Eigen), unlike the MSCKF port, which keeps the
// JPL convention of the Python code.

namespace msckf::swo {

// Small-angle-safe exponential map for any scalar type (double or ceres::Jet).
template <typename T>
Eigen::Quaternion<T> quaternion_exp(const Eigen::Matrix<T, 3, 1>& d) {
  const T theta_sq = d.squaredNorm();
  if (theta_sq > T(1e-12)) {
    using std::cos;
    using std::sin;
    using std::sqrt;
    const T theta = sqrt(theta_sq);
    const T half = theta / T(2);
    const Eigen::Matrix<T, 3, 1> v = d * (sin(half) / theta);
    return Eigen::Quaternion<T>(cos(half), v.x(), v.y(), v.z());
  }
  // First order; normalized so the result stays a unit quaternion.
  Eigen::Quaternion<T> q(T(1), d.x() / T(2), d.y() / T(2), d.z() / T(2));
  q.normalize();
  return q;
}

// Rotation vector of q (inverse of quaternion_exp), shortest rotation.
template <typename T>
Eigen::Matrix<T, 3, 1> quaternion_log(const Eigen::Quaternion<T>& q_in) {
  Eigen::Quaternion<T> q = q_in;
  if (q.w() < T(0)) {
    q.coeffs() = -q.coeffs();
  }
  const Eigen::Matrix<T, 3, 1> v = q.vec();
  const T sin_half_sq = v.squaredNorm();
  if (sin_half_sq > T(1e-12)) {
    using std::atan2;
    using std::sqrt;
    const T sin_half = sqrt(sin_half_sq);
    const T theta = T(2) * atan2(sin_half, q.w());
    return v * (theta / sin_half);
  }
  return T(2) * v;   // small angle
}

class RightQuaternionManifold final : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 3; }

  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
    const Eigen::Map<const Eigen::Quaterniond> q(x);
    const Eigen::Map<const Eigen::Vector3d> d(delta);
    Eigen::Map<Eigen::Quaterniond> out(x_plus_delta);
    out = (q * quaternion_exp(Eigen::Vector3d(d))).normalized();
    return true;
  }

  // d Plus(x, d) / d d at d = 0  (4 x 3, row-major): 0.5 * q * [e_k; 0].
  bool PlusJacobian(const double* x, double* jacobian) const override {
    const Eigen::Map<const Eigen::Quaterniond> q(x);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
    for (int k = 0; k < 3; ++k) {
      Eigen::Quaterniond e(0.0, 0.0, 0.0, 0.0);
      e.vec()(k) = 0.5;
      J.col(k) = (q * e).coeffs();
    }
    return true;
  }

  bool Minus(const double* y, const double* x, double* y_minus_x) const override {
    const Eigen::Map<const Eigen::Quaterniond> qy(y), qx(x);
    // Named Map: "Eigen::Map<...>(y_minus_x) = ..." would declare a new variable.
    Eigen::Map<Eigen::Vector3d> out(y_minus_x);
    out = quaternion_log(Eigen::Quaterniond(qx.inverse() * qy));
    return true;
  }

  // d Minus(y, x) / d y at y = x  (3 x 4, row-major): 2 * vec(x^-1 * dy).
  bool MinusJacobian(const double* x, double* jacobian) const override {
    const Eigen::Map<const Eigen::Quaterniond> q(x);
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobian);
    const Eigen::Quaterniond q_inv = q.inverse();
    for (int k = 0; k < 4; ++k) {
      Eigen::Quaterniond e(0.0, 0.0, 0.0, 0.0);
      e.coeffs()(k) = 1.0;
      J.col(k) = 2.0 * (q_inv * e).vec();
    }
    return true;
  }
};

}  // namespace msckf::swo
