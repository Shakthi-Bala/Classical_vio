// Unit tests of the sliding-window optimizer building blocks.

#include <random>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include "msckf/swo/factors.h"
#include "msckf/swo/imu_preintegration.h"
#include "msckf/swo/marginalization.h"
#include "msckf/swo/quaternion_manifold.h"

using namespace msckf::swo;

namespace {

std::mt19937 rng(3);
std::normal_distribution<double> normal(0.0, 1.0);

Eigen::Vector3d randn3(double s = 1.0) {
  return Eigen::Vector3d(normal(rng), normal(rng), normal(rng)) * s;
}

// Random IMU samples (5 ms apart) for a moving, rotating body.
struct Samples {
  std::vector<Eigen::Vector3d> acc, gyro;
};
Samples random_samples(int n) {
  Samples s;
  for (int i = 0; i < n; ++i) {
    s.acc.push_back(Eigen::Vector3d(0.0, 0.0, 9.81) + randn3(1.0));
    s.gyro.push_back(randn3(0.5));
  }
  return s;
}

ImuPreintegration integrate(const Samples& s, const Eigen::Vector3d& ba, const Eigen::Vector3d& bg) {
  ImuPreintegration pre(ba, bg, ImuNoise{});
  for (size_t i = 0; i < s.acc.size(); ++i) {
    pre.integrate(0.005, s.acc[i], s.gyro[i]);
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
  const Samples s = random_samples(40);   // 0.2 s
  const Eigen::Vector3d ba(0.05, -0.02, 0.1), bg(0.01, 0.02, -0.01);
  const ImuPreintegration base = integrate(s, ba, bg);
  const double h = 1e-5;
  for (int k = 0; k < 3; ++k) {
    Eigen::Vector3d e = Eigen::Vector3d::Zero();
    e(k) = h;
    // Gyro bias.
    const ImuPreintegration gp = integrate(s, ba, bg + e), gm = integrate(s, ba, bg - e);
    const Eigen::Vector3d dR_num =
        (quaternion_log(Eigen::Quaterniond(base.delta_q.inverse() * gp.delta_q)) -
         quaternion_log(Eigen::Quaterniond(base.delta_q.inverse() * gm.delta_q))) / (2 * h);
    EXPECT_NEAR((base.dR_dbg.col(k) - dR_num).norm(), 0.0, 1e-4) << "dR/dbg col " << k;
    EXPECT_NEAR((base.dv_dbg.col(k) - (gp.delta_v - gm.delta_v) / (2 * h)).norm(), 0.0, 1e-4);
    EXPECT_NEAR((base.dp_dbg.col(k) - (gp.delta_p - gm.delta_p) / (2 * h)).norm(), 0.0, 1e-4);
    // Accelerometer bias.
    const ImuPreintegration ap = integrate(s, ba + e, bg), am = integrate(s, ba - e, bg);
    EXPECT_NEAR((base.dv_dba.col(k) - (ap.delta_v - am.delta_v) / (2 * h)).norm(), 0.0, 1e-6);
    EXPECT_NEAR((base.dp_dba.col(k) - (ap.delta_p - am.delta_p) / (2 * h)).norm(), 0.0, 1e-6);
  }
}

TEST(Swo, ImuFactorZeroAtPrediction) {
  const Samples s = random_samples(30);
  const Eigen::Vector3d ba(0.02, 0.0, -0.03), bg(0.001, -0.002, 0.003);
  const ImuPreintegration pre = integrate(s, ba, bg);
  const Eigen::Vector3d g(0.0, 0.0, -9.81);

  double p_i[3] = {1.0, 2.0, 3.0}, sb_i[9] = {0.3, -0.1, 0.2, 0.02, 0.0, -0.03, 0.001, -0.002, 0.003};
  Eigen::Quaterniond q_i(Eigen::AngleAxisd(0.4, Eigen::Vector3d(0, 1, 1).normalized()));
  Eigen::Vector3d p_j, v_j;
  Eigen::Quaterniond q_j;
  pre.predict(Eigen::Map<Eigen::Vector3d>(p_i), q_i, Eigen::Map<Eigen::Vector3d>(sb_i), g, p_j,
              q_j, v_j);
  double sb_j[9];
  Eigen::Map<Eigen::Vector3d> v_j_out(sb_j);
  v_j_out = v_j;
  for (int k = 3; k < 9; ++k) {
    sb_j[k] = sb_i[k];
  }

  std::unique_ptr<ceres::CostFunction> f(ImuFactor::create(pre, g));
  const double* params[] = {p_i, q_i.coeffs().data(), sb_i, p_j.data(), q_j.coeffs().data(), sb_j};
  Eigen::Matrix<double, 15, 1> r;
  f->Evaluate(params, r.data(), nullptr);
  EXPECT_LT(r.norm(), 1e-6);   // whitened, so this is tiny in physical units
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
  const UnaryLinear f1{(Eigen::Matrix2d() << 2, 0.5, -0.3, 1.5).finished(), Eigen::Vector2d(1, -2)};
  const BinaryLinear f2{(Eigen::Matrix2d() << 1.2, 0.1, 0.0, 0.8).finished(), Eigen::Vector2d(0.5, 0.3)};
  const UnaryLinear f3{(Eigen::Matrix2d() << 0.7, -0.2, 0.4, 1.1).finished(), Eigen::Vector2d(-1, 0.4)};
  auto c1 = std::make_unique<ceres::AutoDiffCostFunction<UnaryLinear, 2, 2>>(new UnaryLinear(f1));
  auto c2 = std::make_unique<ceres::AutoDiffCostFunction<BinaryLinear, 2, 2, 2>>(new BinaryLinear(f2));
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
