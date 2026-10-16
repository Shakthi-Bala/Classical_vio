// Unit tests of the sliding-window optimizer building blocks.

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include "msckf/config.h"
#include "msckf/swo/factors.h"
#include "msckf/swo/imu_preintegration.h"
#include "msckf/swo/marginalization.h"
#include "msckf/swo/quaternion_manifold.h"

using namespace msckf::swo;

namespace {

constexpr double kDt = 0.005;   // IMU period [s] (200 Hz)

struct Samples {
  std::vector<Eigen::Vector3d> acc, gyro;
};

// Random IMU samples for a moving, rotating body.
Samples random_samples(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  auto randn3 = [&](double s) {
    return Eigen::Vector3d(normal(rng), normal(rng), normal(rng)) * s;
  };
  Samples out;
  for (int i = 0; i < n; ++i) {
    out.acc.push_back(Eigen::Vector3d(0.0, 0.0, 9.81) + randn3(1.0));
    out.gyro.push_back(randn3(0.5));
  }
  return out;
}

ImuPreintegration integrate(const Samples& s, const Eigen::Vector3d& ba,
                            const Eigen::Vector3d& bg) {
  ImuPreintegration pre(ba, bg, ImuNoise::from_config(msckf::Config{}));
  for (size_t i = 0; i < s.acc.size(); ++i) {
    pre.integrate(kDt, s.acc[i], s.gyro[i]);
  }
  return pre;
}

}  // namespace

TEST(Swo, QuaternionManifoldPlusMinus) {
  const RightQuaternionManifold m;
  Eigen::Quaterniond q(Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()));
  const Eigen::Vector3d d(0.1, -0.2, 0.05);
  Eigen::Quaterniond q_plus;
  m.Plus(q.coeffs().data(), d.data(), q_plus.coeffs().data());
  Eigen::Vector3d d_back;
  m.Minus(q_plus.coeffs().data(), q.coeffs().data(), d_back.data());
  EXPECT_NEAR((d_back - d).norm(), 0.0, 1e-12);

  // PlusJacobian against central differences.
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> J;
  m.PlusJacobian(q.coeffs().data(), J.data());
  for (int k = 0; k < 3; ++k) {
    const double h = 1e-6;
    Eigen::Vector3d dp = Eigen::Vector3d::Zero(), dm = Eigen::Vector3d::Zero();
    dp(k) = h;
    dm(k) = -h;
    Eigen::Quaterniond qp, qm;
    m.Plus(q.coeffs().data(), dp.data(), qp.coeffs().data());
    m.Plus(q.coeffs().data(), dm.data(), qm.coeffs().data());
    EXPECT_NEAR((J.col(k) - (qp.coeffs() - qm.coeffs()) / (2 * h)).norm(), 0.0, 1e-8);
  }
}

TEST(Swo, PreintegrationBiasJacobians) {
  const Samples s = random_samples(40, 3);   // 0.2 s
  const Eigen::Vector3d ba(0.05, -0.02, 0.1), bg(0.01, 0.02, -0.01);
  const ImuPreintegration base = integrate(s, ba, bg);
  const double h = 1e-5;
  for (int k = 0; k < 3; ++k) {
    Eigen::Vector3d e = Eigen::Vector3d::Zero();
    e(k) = h;
    // Gyro-bias Jacobians are first-order approximations (Forster Eq. 69-70),
    // so they match central differences only to ~1e-4 over 0.2 s.
    const ImuPreintegration gp = integrate(s, ba, bg + e), gm = integrate(s, ba, bg - e);
    const Eigen::Vector3d dR_num =
        (quaternion_log(Eigen::Quaterniond(base.delta_q.inverse() * gp.delta_q)) -
         quaternion_log(Eigen::Quaterniond(base.delta_q.inverse() * gm.delta_q))) / (2 * h);
    EXPECT_NEAR((base.dR_dbg.col(k) - dR_num).norm(), 0.0, 1e-4) << "dR/dbg col " << k;
    EXPECT_NEAR((base.dv_dbg.col(k) - (gp.delta_v - gm.delta_v) / (2 * h)).norm(), 0.0, 1e-4);
    EXPECT_NEAR((base.dp_dbg.col(k) - (gp.delta_p - gm.delta_p) / (2 * h)).norm(), 0.0, 1e-4);
    // delta_v and delta_p are linear in the accelerometer bias: exact up to rounding.
    const ImuPreintegration ap = integrate(s, ba + e, bg), am = integrate(s, ba - e, bg);
    EXPECT_NEAR((base.dv_dba.col(k) - (ap.delta_v - am.delta_v) / (2 * h)).norm(), 0.0, 1e-6);
    EXPECT_NEAR((base.dp_dba.col(k) - (ap.delta_p - am.delta_p) / (2 * h)).norm(), 0.0, 1e-6);
  }
}

TEST(Swo, ImuFactorZeroAtPrediction) {
  const Samples s = random_samples(30, 4);
  const Eigen::Vector3d ba(0.02, 0.0, -0.03), bg(0.001, -0.002, 0.003);
  const ImuPreintegration pre = integrate(s, ba, bg);
  const Eigen::Vector3d g(0.0, 0.0, -9.81);

  // State i, with the biases the preintegration was linearized at.
  double p_i[3] = {1.0, 2.0, 3.0};
  double sb_i[9];
  Eigen::Map<Eigen::Matrix<double, 9, 1>> sb_i_map(sb_i);
  sb_i_map << 0.3, -0.1, 0.2, ba, bg;
  Eigen::Quaterniond q_i(Eigen::AngleAxisd(0.4, Eigen::Vector3d(0, 1, 1).normalized()));

  // State j = the IMU prediction from state i, same biases.
  Eigen::Vector3d p_j, v_j;
  Eigen::Quaterniond q_j;
  pre.predict(Eigen::Map<Eigen::Vector3d>(p_i), q_i, sb_i_map.head<3>(), g, p_j, q_j, v_j);
  double sb_j[9];
  Eigen::Map<Eigen::Matrix<double, 9, 1>> sb_j_map(sb_j);
  sb_j_map << v_j, ba, bg;

  std::unique_ptr<ceres::CostFunction> f(ImuFactor::create(pre, g));
  const double* params[] = {p_i, q_i.coeffs().data(), sb_i, p_j.data(), q_j.coeffs().data(), sb_j};
  Eigen::Matrix<double, 15, 1> r;
  f->Evaluate(params, r.data(), nullptr);
  // The residual is whitened (in standard deviations): 1e-6 sigma is zero.
  EXPECT_LT(r.norm(), 1e-6);
}

namespace {

// Linear residuals for the marginalization test: r = A x - b, and r = B (y - x) - c.
struct UnaryLinear {
  Eigen::Matrix2d A;
  Eigen::Vector2d b;
  template <typename T>
  bool operator()(const T* x, T* r) const {
    const Eigen::Matrix<T, 2, 1> xv(x[0], x[1]);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> out(r);
    out = A.cast<T>() * xv - b.cast<T>();
    return true;
  }
};
struct BinaryLinear {
  Eigen::Matrix2d B;
  Eigen::Vector2d c;
  template <typename T>
  bool operator()(const T* x, const T* y, T* r) const {
    const Eigen::Matrix<T, 2, 1> d(y[0] - x[0], y[1] - x[1]);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> out(r);
    out = B.cast<T>() * d - c.cast<T>();
    return true;
  }
};

}  // namespace

TEST(Swo, MarginalizationMatchesJointSolution) {
  // Joint problem over x and y; then marginalize x and solve for y alone.
  // For a linear least-squares problem both must give the same y exactly.
  const UnaryLinear f1{(Eigen::Matrix2d() << 2, 0.5, -0.3, 1.5).finished(),
                       Eigen::Vector2d(1, -2)};
  const BinaryLinear f2{(Eigen::Matrix2d() << 1.2, 0.1, 0.0, 0.8).finished(),
                        Eigen::Vector2d(0.5, 0.3)};
  const UnaryLinear f3{(Eigen::Matrix2d() << 0.7, -0.2, 0.4, 1.1).finished(),
                       Eigen::Vector2d(-1, 0.4)};
  auto c1 = std::make_unique<ceres::AutoDiffCostFunction<UnaryLinear, 2, 2>>(new UnaryLinear(f1));
  auto c2 =
      std::make_unique<ceres::AutoDiffCostFunction<BinaryLinear, 2, 2, 2>>(new BinaryLinear(f2));
  auto c3 = std::make_unique<ceres::AutoDiffCostFunction<UnaryLinear, 2, 2>>(new UnaryLinear(f3));

  ceres::Problem::Options po;
  po.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Solver::Options so;
  so.logging_type = ceres::SILENT;
  // Solve to machine precision, so the comparison tests the math, not the
  // solver's default stopping rule.
  so.function_tolerance = so.gradient_tolerance = so.parameter_tolerance = 1e-16;
  so.max_num_iterations = 100;
  ceres::Solver::Summary summary;

  double x_joint[2] = {0, 0}, y_joint[2] = {0, 0};
  {
    ceres::Problem p(po);
    p.AddResidualBlock(c1.get(), nullptr, x_joint);
    p.AddResidualBlock(c2.get(), nullptr, x_joint, y_joint);
    p.AddResidualBlock(c3.get(), nullptr, y_joint);
    ceres::Solve(so, &p, &summary);
  }

  // Linearize f1, f2 at an arbitrary point, marginalize x.
  double x[2] = {0.3, -0.7}, y[2] = {1.1, 0.2};
  Marginalizer marg;
  marg.add({c1.get(), nullptr, {x}});
  marg.add({c2.get(), nullptr, {x, y}});
  std::unique_ptr<MarginalizationFactor> prior = marg.marginalize({x}, {});
  ASSERT_NE(prior, nullptr);
  {
    ceres::Problem p(po);
    p.AddResidualBlock(prior.get(), nullptr, prior->block_pointers());
    p.AddResidualBlock(c3.get(), nullptr, y);
    ceres::Solve(so, &p, &summary);
  }
  EXPECT_NEAR(y[0], y_joint[0], 1e-8);
  EXPECT_NEAR(y[1], y_joint[1], 1e-8);
}

TEST(Swo, AnalyticReprojectionMatchesAutodiff) {
  // Hand-picked, well-conditioned geometry: point ~3 m in front of the anchor.
  const Eigen::Matrix3d R_bc0 = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d t_bc0(0.05, -0.02, 0.01);
  const Eigen::Matrix3d R_bc1 =
      Eigen::AngleAxisd(-0.05, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Vector3d t_bc1(-0.06, -0.02, 0.0);
  const Eigen::Vector3d bearing(0.1, -0.05, 1.0);
  const Eigen::Vector2d z(0.12, -0.03);
  const double sqrt_info = 300.0;   // ~ focal length / pixel sigma

  double p_a[3] = {0.1, 0.2, 0.3}, p_j[3] = {0.4, 0.1, 0.2}, rho[1] = {0.33};
  Eigen::Quaterniond q_a(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 0, 1).normalized()));
  Eigen::Quaterniond q_j(Eigen::AngleAxisd(0.35, Eigen::Vector3d(1, 0.2, 1).normalized()));
  const double* params[] = {p_a, q_a.coeffs().data(), p_j, q_j.coeffs().data(), rho};

  std::unique_ptr<ceres::CostFunction> ad(
      ReprojectionFactor::create(bearing, z, R_bc0, t_bc0, R_bc1, t_bc1, sqrt_info));
  ReprojectionFactorAnalytic an(bearing, z, R_bc0, t_bc0, R_bc1, t_bc1, sqrt_info);

  const int sizes[] = {3, 4, 3, 4, 1};
  double r_ad[2], r_an[2];
  std::vector<std::vector<double>> J_ad(5), J_an(5);
  double* jp_ad[5];
  double* jp_an[5];
  for (int k = 0; k < 5; ++k) {
    J_ad[k].resize(2 * sizes[k]);
    J_an[k].resize(2 * sizes[k]);
    jp_ad[k] = J_ad[k].data();
    jp_an[k] = J_an[k].data();
  }
  ad->Evaluate(params, r_ad, jp_ad);
  an.Evaluate(params, r_an, jp_an);
  EXPECT_NEAR(r_ad[0], r_an[0], 1e-12);
  EXPECT_NEAR(r_ad[1], r_an[1], 1e-12);

  // Compare in tangent space: J_ambient * PlusJacobian (quaternion blocks).
  using RowMajorXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const RightQuaternionManifold m;
  for (int k = 0; k < 5; ++k) {
    Eigen::MatrixXd A = Eigen::Map<RowMajorXd>(J_ad[k].data(), 2, sizes[k]);
    Eigen::MatrixXd B = Eigen::Map<RowMajorXd>(J_an[k].data(), 2, sizes[k]);
    if (sizes[k] == 4) {
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> P;
      m.PlusJacobian(params[k], P.data());
      A = A * P;
      B = B * P;
    }
    EXPECT_NEAR((A - B).norm(), 0.0, 1e-9 * std::max(1.0, A.norm())) << "block " << k;
  }
}
