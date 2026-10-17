// End-to-end tests of both back ends on synthetic data: a known trajectory in
// a room of landmarks, exact IMU samples and exact feature observations (no
// front end). Checks the propagation, the measurement updates / optimization
// and the cam0-only observations of Config::mono_features.

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "msckf/config.h"
#include "msckf/estimator.h"
#include "msckf/msckf.h"
#include "msckf/swo/sliding_window.h"

using namespace msckf;

namespace {

constexpr double kImuRate = 200.0;     // [Hz]
constexpr double kCameraRate = 20.0;   // [Hz]
constexpr double kStaticTime = 1.5;    // [s] at rest first: gravity / bias initialization
constexpr double kDuration = 11.5;     // [s]
constexpr double kGravity = 9.81;      // [m/s^2]
constexpr double kBaseline = 0.11;     // [m] cam1 is to the right of cam0
// Accelerometer bias the estimators do not know: integrated without vision it
// is metres of drift over the run, so only the visual updates keep it small.
const Eigen::Vector3d kAccBias(0.08, -0.05, 0.06);   // [m/s^2], body frame

// a (1 - cos(w tau)) and its first two derivatives: starts at rest.
struct Wave {
  double a, w;
  double f(double tau) const { return a * (1.0 - std::cos(w * tau)); }
  double df(double tau) const { return a * w * std::sin(w * tau); }
  double ddf(double tau) const { return a * w * w * std::cos(w * tau); }
};

// Body frame = cam0 frame (x right, y down, z forward). The body starts at the
// origin looking along world +x, then moves in x, y, z and turns about world z.
class Trajectory {
 public:
  Eigen::Vector3d position(double t) const {
    const double tau = std::max(t - kStaticTime, 0.0);
    return {x_.f(tau), y_.f(tau), z_.f(tau)};
  }
  Eigen::Matrix3d rotation(double t) const {   // body -> world
    const double tau = std::max(t - kStaticTime, 0.0);
    return Eigen::AngleAxisd(yaw_.f(tau), Eigen::Vector3d::UnitZ()).toRotationMatrix() * R0();
  }
  ImuMsg imu(double t) const {
    const double tau = std::max(t - kStaticTime, 0.0);
    const bool moving = t > kStaticTime;
    const Eigen::Vector3d acc_w =
        moving ? Eigen::Vector3d(x_.ddf(tau), y_.ddf(tau), z_.ddf(tau)) : Eigen::Vector3d::Zero();
    const Eigen::Vector3d omega_w(0.0, 0.0, moving ? yaw_.df(tau) : 0.0);
    const Eigen::Matrix3d R_wb = rotation(t);
    ImuMsg m;
    m.timestamp = t;
    m.linear_acceleration =
        R_wb.transpose() * (acc_w + Eigen::Vector3d(0.0, 0.0, kGravity)) + kAccBias;
    m.angular_velocity = R_wb.transpose() * omega_w;
    return m;
  }

 private:
  static Eigen::Matrix3d R0() {
    Eigen::Matrix3d R;   // columns: body x, y, z in the world frame
    R << 0, 0, 1,
        -1, 0, 0,
         0, -1, 0;
    return R;
  }
  Wave x_{1.0, 0.8}, y_{0.5, 1.2}, z_{0.2, 1.5}, yaw_{0.4, 0.6};
};

// Random points on the walls and floor of a room around the trajectory.
std::vector<Eigen::Vector3d> make_landmarks(unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 400; ++i) {
    points.emplace_back(7.0, -6.0 + 12.0 * u(rng), -1.5 + 4.5 * u(rng));   // front wall
    points.emplace_back(-1.0 + 8.0 * u(rng), 6.0, -1.5 + 4.5 * u(rng));    // left wall
    points.emplace_back(-1.0 + 8.0 * u(rng), -6.0, -1.5 + 4.5 * u(rng));   // right wall
    points.emplace_back(-1.0 + 8.0 * u(rng), -6.0 + 12.0 * u(rng), -1.5);  // floor
  }
  return points;
}

// The recommended configuration (config/euroc_fixed.yaml): every Python-quirk
// fix except the 95 % chi-squared gate.
Config synthetic_config(bool mono_features) {
  Config config;
  for (const char* fix : {"epipolar_full_dot", "lm_upstream_loops", "oc_upstream",
                          "normalize_quaternion", "mask_clamp", "full_precision_publish"}) {
    enable_fix(config, fix);
  }
  config.mono_features = mono_features;
  config.gravity_acc = kGravity;
  config.cam0.T_imu_cam = Eigen::Matrix4d::Identity();
  config.cam0.intrinsics << 450.0, 450.0, 376.0, 240.0;
  config.T_cn_cnm1 = Eigen::Matrix4d::Identity();
  config.T_cn_cnm1(0, 3) = -kBaseline;   // p_c1 = p_c0 - (b, 0, 0)
  config.cam1 = config.cam0;
  config.cam1.T_imu_cam = config.T_cn_cnm1 * config.cam0.T_imu_cam;
  return config;
}

// Observations of the visible landmarks at time t. With `mono_every` > 0,
// every mono_every-th landmark has no cam1 observation (NaN); with
// mono_every < 0 there are no observations at all (IMU only).
FeatureMsg observe(const Trajectory& traj, const std::vector<Eigen::Vector3d>& landmarks,
                   double t, int mono_every) {
  const Eigen::Matrix3d R_wc = traj.rotation(t);
  const Eigen::Vector3d p_wc = traj.position(t);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  FeatureMsg msg;
  msg.timestamp = t;
  if (mono_every < 0) {
    return msg;
  }
  for (size_t id = 0; id < landmarks.size(); ++id) {
    const Eigen::Vector3d p0 = R_wc.transpose() * (landmarks[id] - p_wc);
    const Eigen::Vector3d p1 = p0 - Eigen::Vector3d(kBaseline, 0.0, 0.0);
    // Field of view about 75 x 55 degrees.
    if (p0.z() < 0.3 || std::abs(p0.x() / p0.z()) > 0.75 || std::abs(p0.y() / p0.z()) > 0.5) {
      continue;
    }
    const bool mono = mono_every > 0 && id % mono_every == 0;
    msg.features.push_back({static_cast<FeatureID>(id), p0.x() / p0.z(), p0.y() / p0.z(),
                            mono ? nan : p1.x() / p1.z(), mono ? nan : p1.y() / p1.z()});
  }
  return msg;
}

struct Errors {
  double max_position = 0.0;   // [m]
  double max_rotation = 0.0;   // [rad]
  int poses = 0;
};

// Runs an estimator through the whole trajectory and compares each pose,
// relative to the first one, with the ground truth (this removes the unknown
// yaw and origin of the estimator's world frame).
Errors run(Estimator& estimator, int mono_every) {
  const Trajectory traj;
  const std::vector<Eigen::Vector3d> landmarks = make_landmarks(7);
  const int n_imu = static_cast<int>(kDuration * kImuRate);
  const int imu_per_image = static_cast<int>(kImuRate / kCameraRate);

  Errors errors;
  std::optional<VioResult> first;
  double first_t = 0.0;
  for (int i = 0; i <= n_imu; ++i) {
    const double t = i / kImuRate;
    estimator.imu_callback(traj.imu(t));
    if (i == 0 || i % imu_per_image != 0) {
      continue;
    }
    const std::optional<VioResult> result =
        estimator.feature_callback(observe(traj, landmarks, t, mono_every));
    if (!result) {
      continue;
    }
    if (!first) {
      first = result;
      first_t = t;
    }
    const Isometry3d est = first->pose.inverse() * result->pose;
    const Isometry3d gt = Isometry3d(traj.rotation(first_t), traj.position(first_t)).inverse() *
                          Isometry3d(traj.rotation(t), traj.position(t));
    errors.max_position = std::max(errors.max_position, (est.t - gt.t).norm());
    errors.max_rotation =
        std::max(errors.max_rotation, Eigen::AngleAxisd(est.R.transpose() * gt.R).angle());
    ++errors.poses;
  }
  return errors;
}

// Expected number of poses: one per image after the 1 s static initialization.
constexpr int kMinPoses = static_cast<int>((kDuration - 1.0) * kCameraRate) - 1;

}  // namespace

// The trajectory moves about 2 m and turns 46 degrees; the observations are
// exact and the IMU only has the unknown accelerometer bias. Measured maximum
// errors: MSCKF 4.0 cm / 0.50 deg (the filter needs time to learn the bias),
// sliding window 1.8 mm / 0.02 deg. Tolerances are ~2.5x those.
constexpr double kMsckfMaxPosition = 0.10, kMsckfMaxRotation = 0.02;   // [m], [rad]
constexpr double kSwoMaxPosition = 0.005, kSwoMaxRotation = 0.001;

TEST(Backends, ImuOnlyDrifts) {
  // Control: without features the same run is far outside the tolerances
  // below, so passing them requires working visual updates.
  MSCKF filter(synthetic_config(false));
  const Errors e = run(filter, -1);
  EXPECT_GT(e.max_position, 1.0);
}

TEST(Backends, MsckfStereo) {
  MSCKF filter(synthetic_config(false));
  const Errors e = run(filter, 0);
  EXPECT_GE(e.poses, kMinPoses);
  EXPECT_LT(e.max_position, kMsckfMaxPosition);
  EXPECT_LT(e.max_rotation, kMsckfMaxRotation);
}

TEST(Backends, MsckfHalfMono) {
  MSCKF filter(synthetic_config(true));
  const Errors e = run(filter, 2);
  EXPECT_GE(e.poses, kMinPoses);
  EXPECT_LT(e.max_position, kMsckfMaxPosition);
  EXPECT_LT(e.max_rotation, kMsckfMaxRotation);
}

TEST(Backends, SlidingWindowStereo) {
  swo::SlidingWindowEstimator swo(synthetic_config(false));
  const Errors e = run(swo, 0);
  EXPECT_GE(e.poses, kMinPoses);
  EXPECT_LT(e.max_position, kSwoMaxPosition);
  EXPECT_LT(e.max_rotation, kSwoMaxRotation);
}

TEST(Backends, SlidingWindowHalfMono) {
  swo::SlidingWindowEstimator swo(synthetic_config(true));
  const Errors e = run(swo, 2);
  EXPECT_GE(e.poses, kMinPoses);
  EXPECT_LT(e.max_position, kSwoMaxPosition);
  EXPECT_LT(e.max_rotation, kSwoMaxRotation);
}
