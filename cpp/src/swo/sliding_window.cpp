#include "msckf/swo/sliding_window.h"

#include <cmath>
#include <cstdio>
#include <set>

#include <ceres/problem.h>
#include <ceres/solver.h>

#include "msckf/math_utils.h"
#include "msckf/profiler.h"
#include "msckf/swo/factors.h"

namespace msckf::swo {

namespace {

// Re-integrate a preintegration when the bias estimate moved this far from
// its linearization point (the first-order correction is no longer good).
// Same thresholds as VINS-Fusion.
constexpr double kRepropagateAccBias = 0.10;    // [m/s^2]
constexpr double kRepropagateGyroBias = 0.01;   // [rad/s]

}  // namespace

SlidingWindowEstimator::SlidingWindowEstimator(const Config& config)
    : cfg_(config.swo), noise_(ImuNoise::from_config(config)) {
  const CameraExtrinsics cam0 = camera_to_imu(config.cam0);
  R_bc0_ = cam0.R_cam_imu;
  t_bc0_ = cam0.t_cam_imu;
  R_c1c0_ = config.T_cn_cnm1.topLeftCorner<3, 3>();
  t_c1c0_ = config.T_cn_cnm1.topRightCorner<3, 1>();
  // T_bc1 = T_bc0 * T_c0c1.
  R_bc1_ = R_bc0_ * R_c1c0_.transpose();
  t_bc1_ = t_bc0_ - R_bc1_ * t_c1c0_;
  focal_ = config.cam0.intrinsics(0);

  // Robust loss on whitened residuals (1 unit = pixel_sigma pixels).
  huber_ = std::make_unique<ceres::HuberLoss>(cfg_.huber_px / cfg_.pixel_sigma);
  if (!cfg_.debug_log.empty()) {
    debug_log_.open(cfg_.debug_log);
    debug_log_ << "t,landmarks,initial_cost,final_cost,iterations,dp_vs_imu,dv_vs_imu,"
                  "speed,outliers_removed\n";
  }
}

SlidingWindowEstimator::~SlidingWindowEstimator() = default;

void SlidingWindowEstimator::imu_callback(const ImuMsg& imu_msg) {
  imu_buffer_.push_back(imu_msg);
  if (!is_gravity_set_ && imu_buffer_.size() >= kStaticInitImuSamples) {
    initialize_gravity_and_bias();
    is_gravity_set_ = true;
  }
}

void SlidingWindowEstimator::initialize_gravity_and_bias() {
  Eigen::Vector3d acc_sum = Eigen::Vector3d::Zero(), gyro_sum = Eigen::Vector3d::Zero();
  for (const ImuMsg& m : imu_buffer_) {
    acc_sum += m.linear_acceleration;
    gyro_sum += m.angular_velocity;
  }
  const double n = static_cast<double>(imu_buffer_.size());
  const Eigen::Vector3d acc_mean = acc_sum / n;
  gyro_bias_init_ = gyro_sum / n;
  gravity_ = Eigen::Vector3d(0.0, 0.0, -acc_mean.norm());
  // R_wb maps the measured specific force (pointing up) onto +z.
  q_init_ = Eigen::Quaterniond::FromTwoVectors(acc_mean, Eigen::Vector3d::UnitZ());
}

std::optional<double> SlidingWindowEstimator::stereo_inverse_depth(const Eigen::Vector4d& z) const {
  const double depth = two_view_depth(R_c1c0_, t_c1c0_, z.head<2>(), z.tail<2>());
  if (!std::isfinite(depth) || depth < cfg_.min_depth || depth > cfg_.max_depth) {
    return std::nullopt;
  }
  return 1.0 / depth;
}

void SlidingWindowEstimator::integrate_imu_until(double t) {
  // Each sample is held until the next one (Euler); the last one until t.
  size_t used = 0;
  for (const ImuMsg& m : imu_buffer_) {
    if (m.timestamp > t) {
      break;
    }
    if (m.timestamp > last_imu_t_) {
      cur_preint_->integrate(m.timestamp - last_imu_t_, last_acc_, last_gyro_);
      last_imu_t_ = m.timestamp;
    }
    last_acc_ = m.linear_acceleration;
    last_gyro_ = m.angular_velocity;
    ++used;
  }
  imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + used);
  if (t > last_imu_t_) {
    cur_preint_->integrate(t - last_imu_t_, last_acc_, last_gyro_);
    last_imu_t_ = t;
  }
}

void SlidingWindowEstimator::initialize_first_keyframe(const FeatureMsg& msg) {
  // Discard IMU samples up to the first frame; remember the latest one.
  size_t used = 0;
  for (const ImuMsg& m : imu_buffer_) {
    if (m.timestamp > msg.timestamp) {
      break;
    }
    last_acc_ = m.linear_acceleration;
    last_gyro_ = m.angular_velocity;
    ++used;
  }
  imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + used);
  last_imu_t_ = msg.timestamp;

  push_keyframe(msg, Eigen::Vector3d::Zero(), q_init_, Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), gyro_bias_init_);
  KeyFrame& kf = *window_.back();
  Eigen::Matrix<double, 15, 1> sigma;
  sigma << Eigen::Vector3d::Constant(cfg_.init_pos_sigma),
      Eigen::Vector3d::Constant(cfg_.init_rot_sigma), Eigen::Vector3d::Constant(cfg_.init_vel_sigma),
      Eigen::Vector3d::Constant(cfg_.init_acc_bias_sigma),
      Eigen::Vector3d::Constant(cfg_.init_gyro_bias_sigma);
  init_prior_.reset(StatePriorFactor::create(kf.position(), kf.orientation(),
                                             Eigen::Map<Eigen::Matrix<double, 9, 1>>(kf.sb),
                                             sigma));
}

void SlidingWindowEstimator::push_keyframe(const FeatureMsg& msg, const Eigen::Vector3d& p,
                                           const Eigen::Quaterniond& q, const Eigen::Vector3d& v,
                                           const Eigen::Vector3d& ba, const Eigen::Vector3d& bg) {
  auto kf = std::make_unique<KeyFrame>();
  kf->id = next_kf_id_++;
  kf->timestamp = msg.timestamp;
  kf->position() = p;
  kf->orientation() = q.normalized();
  kf->velocity() = v;
  kf->acc_bias() = ba;
  kf->gyro_bias() = bg;
  kf->preint = std::move(cur_preint_);   // null for the first keyframe
  cur_preint_ = std::make_unique<ImuPreintegration>(ba, bg, noise_);
  add_observations(*kf, msg);
  window_.push_back(std::move(kf));
  ++stats_.keyframes;
}

void SlidingWindowEstimator::add_observations(KeyFrame& kf, const FeatureMsg& msg) {
  for (const FeatureMeasurement& f : msg.features) {
    const Eigen::Vector4d z(f.u0, f.v0, f.u1, f.v1);
    auto it = landmarks_.find(f.id);
    if (it == landmarks_.end()) {
      // New landmark, anchored here, depth from the stereo pair.
      const std::optional<double> inv_depth = stereo_inverse_depth(z);
      if (!inv_depth) {
        continue;
      }
      Landmark lm;
      lm.anchor_id = kf.id;
      lm.bearing = Eigen::Vector3d(z(0), z(1), 1.0);
      lm.inv_depth[0] = *inv_depth;
      lm.obs[kf.id] = z;
      landmarks_.emplace(f.id, lm);
    } else {
      it->second.obs[kf.id] = z;
    }
    kf.obs[f.id] = z;
  }
}

bool SlidingWindowEstimator::is_keyframe(const FeatureMsg& msg) const {
  const KeyFrame& last = *window_.back();
  int tracked = 0;
  double parallax = 0.0;
  for (const FeatureMeasurement& f : msg.features) {
    const auto it = last.obs.find(f.id);
    if (it == last.obs.end()) {
      continue;
    }
    ++tracked;
    parallax += std::hypot(f.u0 - it->second(0), f.v0 - it->second(1)) * focal_;
  }
  if (tracked == 0) {
    return true;
  }
  return parallax / tracked >= cfg_.kf_parallax_px ||
         tracked < cfg_.kf_min_tracked_ratio * static_cast<double>(last.obs.size()) ||
         msg.timestamp - last.timestamp >= cfg_.kf_max_interval;
}

SlidingWindowEstimator::KeyFrame* SlidingWindowEstimator::keyframe(KeyFrameID id) {
  for (const auto& kf : window_) {
    if (kf->id == id) {
      return kf.get();
    }
  }
  return nullptr;
}

bool SlidingWindowEstimator::in_window(KeyFrameID id) const {
  for (const auto& kf : window_) {
    if (kf->id == id) {
      return true;
    }
  }
  return false;
}

bool SlidingWindowEstimator::usable(const Landmark& lm) const {
  return lm.obs.size() >= 2 && lm.obs.begin()->first == lm.anchor_id && in_window(lm.anchor_id);
}

void SlidingWindowEstimator::landmark_factors(Landmark& lm, std::vector<ResidualBlockInfo>& out,
                                              CostFunctions& owned) {
  const double sqrt_info = focal_ / cfg_.pixel_sigma;
  KeyFrame* anchor = keyframe(lm.anchor_id);
  auto add = [&](ceres::CostFunction* cost, std::vector<double*> blocks) {
    owned.emplace_back(cost);
    out.push_back({cost, huber_.get(), std::move(blocks)});
  };
  for (const auto& [kf_id, z] : lm.obs) {
    if (kf_id == lm.anchor_id) {
      add(StereoAnchorFactor::create(lm.bearing, z.tail<2>(), R_c1c0_, t_c1c0_, sqrt_info),
          {lm.inv_depth});
      continue;
    }
    KeyFrame* kf = keyframe(kf_id);
    const std::vector<double*> blocks = {anchor->p, anchor->q, kf->p, kf->q, lm.inv_depth};
    add(new ReprojectionFactorAnalytic(lm.bearing, z.head<2>(), R_bc0_, t_bc0_, R_bc0_, t_bc0_,
                                       sqrt_info),
        blocks);
    add(new ReprojectionFactorAnalytic(lm.bearing, z.tail<2>(), R_bc0_, t_bc0_, R_bc1_, t_bc1_,
                                       sqrt_info),
        blocks);
  }
}

void SlidingWindowEstimator::repropagate_if_needed(KeyFrame& prev, KeyFrame& kf) const {
  ImuPreintegration& pre = *kf.preint;
  if ((prev.acc_bias() - pre.ba_lin).norm() > kRepropagateAccBias ||
      (prev.gyro_bias() - pre.bg_lin).norm() > kRepropagateGyroBias) {
    pre.repropagate(prev.acc_bias(), prev.gyro_bias());
  }
}

long SlidingWindowEstimator::optimize() {
  MSCKF_PROFILE("swo/optimize");
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  CostFunctions owned;

  for (const auto& kf : window_) {
    problem.AddParameterBlock(kf->p, 3);
    problem.AddParameterBlock(kf->q, 4, &quaternion_manifold_);
    problem.AddParameterBlock(kf->sb, 9);
  }
  if (init_prior_) {
    KeyFrame& kf0 = *window_.front();
    problem.AddResidualBlock(init_prior_.get(), nullptr, kf0.p, kf0.q, kf0.sb);
  }
  if (marg_prior_) {
    problem.AddResidualBlock(marg_prior_.get(), nullptr, marg_prior_->block_pointers());
  }
  for (size_t k = 1; k < window_.size(); ++k) {
    KeyFrame& prev = *window_[k - 1];
    KeyFrame& kf = *window_[k];
    repropagate_if_needed(prev, kf);
    owned.emplace_back(ImuFactor::create(*kf.preint, gravity_));
    problem.AddResidualBlock(owned.back().get(), nullptr, prev.p, prev.q, prev.sb, kf.p, kf.q,
                             kf.sb);
  }
  long n_landmarks = 0;
  for (auto& [id, lm] : landmarks_) {
    if (!usable(lm)) {
      continue;
    }
    std::vector<ResidualBlockInfo> factors;
    landmark_factors(lm, factors, owned);
    problem.AddParameterBlock(lm.inv_depth, 1);
    problem.SetParameterLowerBound(lm.inv_depth, 0, 1.0 / cfg_.max_depth);
    problem.SetParameterUpperBound(lm.inv_depth, 0, 1.0 / cfg_.min_depth);
    for (const ResidualBlockInfo& f : factors) {
      problem.AddResidualBlock(f.cost, f.loss, f.blocks);
    }
    ++n_landmarks;
  }
  stats_.landmarks_optimized += n_landmarks;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;   // landmarks eliminated first
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = cfg_.max_iterations;
  options.num_threads = 1;
  options.logging_type = ceres::SILENT;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  last_solve_ = {summary.initial_cost, summary.final_cost,
                 static_cast<int>(summary.iterations.size())};
  return n_landmarks;
}

void SlidingWindowEstimator::erase_landmark(FeatureID id) {
  const auto it = landmarks_.find(id);
  if (it == landmarks_.end()) {
    return;
  }
  for (const auto& [kf_id, z] : it->second.obs) {
    if (KeyFrame* kf = keyframe(kf_id)) {
      kf->obs.erase(id);
    }
  }
  landmarks_.erase(it);
}

void SlidingWindowEstimator::remove_outliers() {
  MSCKF_PROFILE("swo/remove_outliers");
  std::vector<FeatureID> bad;
  for (auto& [id, lm] : landmarks_) {
    if (!usable(lm)) {
      continue;
    }
    std::vector<ResidualBlockInfo> factors;
    CostFunctions owned;
    landmark_factors(lm, factors, owned);
    // Whitened residual * pixel_sigma = pixels.
    double sum_px = 0.0;
    for (const ResidualBlockInfo& f : factors) {
      double r[2];
      f.cost->Evaluate(f.blocks.data(), r, nullptr);
      sum_px += std::hypot(r[0], r[1]) * cfg_.pixel_sigma;
    }
    if (sum_px / static_cast<double>(factors.size()) > cfg_.outlier_px) {
      bad.push_back(id);
    }
  }
  for (const FeatureID id : bad) {
    erase_landmark(id);
  }
  stats_.outliers_removed += static_cast<long>(bad.size());
}

void SlidingWindowEstimator::marginalize_oldest() {
  MSCKF_PROFILE("swo/marginalize");
  KeyFrame& kf0 = *window_[0];
  KeyFrame& kf1 = *window_[1];

  Marginalizer marginalizer;
  CostFunctions owned;
  if (marg_prior_) {
    marginalizer.add({marg_prior_.get(), nullptr, marg_prior_->block_pointers()});
  }
  if (init_prior_) {
    marginalizer.add({init_prior_.get(), nullptr, {kf0.p, kf0.q, kf0.sb}});
  }
  owned.emplace_back(ImuFactor::create(*kf1.preint, gravity_));
  marginalizer.add({owned.back().get(), nullptr, {kf0.p, kf0.q, kf0.sb, kf1.p, kf1.q, kf1.sb}});

  std::set<double*> drop = {kf0.p, kf0.q, kf0.sb};
  for (auto& [id, lm] : landmarks_) {
    if (lm.anchor_id != kf0.id || !usable(lm)) {
      continue;
    }
    std::vector<ResidualBlockInfo> factors;
    landmark_factors(lm, factors, owned);
    for (const ResidualBlockInfo& f : factors) {
      marginalizer.add(f);
    }
    drop.insert(lm.inv_depth);
  }
  std::set<double*> quaternion_blocks;
  for (const auto& kf : window_) {
    quaternion_blocks.insert(kf->q);
  }
  // The old prior is one of the inputs; it is replaced only after the call.
  marg_prior_ = marginalizer.marginalize(drop, quaternion_blocks);
  init_prior_.reset();
  ++stats_.marginalizations;

  // Landmarks anchored in kf0: move the anchor to their next keyframe.
  const Eigen::Matrix3d R_w0 = kf0.orientation().toRotationMatrix();
  const Eigen::Vector3d p_w0 = kf0.position();
  std::vector<FeatureID> to_erase;
  for (auto& [id, lm] : landmarks_) {
    lm.obs.erase(kf0.id);
    if (lm.anchor_id != kf0.id) {
      continue;
    }
    if (lm.obs.empty()) {
      to_erase.push_back(id);
      continue;
    }
    KeyFrame* new_anchor = keyframe(lm.obs.begin()->first);
    const Eigen::Vector4d& z = lm.obs.begin()->second;
    const Eigen::Vector3d p_w =
        R_w0 * (R_bc0_ * (lm.bearing / lm.inv_depth[0]) + t_bc0_) + p_w0;
    const Eigen::Vector3d p_b =
        new_anchor->orientation().conjugate() * (p_w - new_anchor->position());
    const Eigen::Vector3d p_c0 = R_bc0_.transpose() * (p_b - t_bc0_);
    lm.anchor_id = new_anchor->id;
    lm.bearing = Eigen::Vector3d(z(0), z(1), 1.0);
    if (p_c0.z() > cfg_.min_depth && p_c0.z() < cfg_.max_depth) {
      lm.inv_depth[0] = 1.0 / p_c0.z();
    } else if (const std::optional<double> inv_depth = stereo_inverse_depth(z)) {
      lm.inv_depth[0] = *inv_depth;
    } else {
      to_erase.push_back(id);
    }
  }
  window_.pop_front();   // kf0 is gone; erase_landmark skips missing keyframes
  for (const FeatureID id : to_erase) {
    erase_landmark(id);
  }
}

VioResult SlidingWindowEstimator::make_result(double t, const Eigen::Vector3d& p,
                                              const Eigen::Quaterniond& q,
                                              const Eigen::Vector3d& v) const {
  VioResult result;
  result.timestamp = t;
  result.pose = Isometry3d(q.toRotationMatrix(), p);
  result.velocity = v;
  result.cam0_pose = result.pose * Isometry3d(R_bc0_, t_bc0_);
  return result;
}

std::optional<VioResult> SlidingWindowEstimator::feature_callback(const FeatureMsg& msg) {
  if (!is_gravity_set_) {
    return std::nullopt;
  }
  ++stats_.frames;
  if (window_.empty()) {
    initialize_first_keyframe(msg);
    KeyFrame& kf = *window_.back();
    return make_result(msg.timestamp, kf.position(), Eigen::Quaterniond(kf.orientation()),
                       kf.velocity());
  }

  integrate_imu_until(msg.timestamp);
  KeyFrame& last = *window_.back();
  Eigen::Vector3d p, v;
  Eigen::Quaterniond q;
  cur_preint_->predict(last.position(), Eigen::Quaterniond(last.orientation()), last.velocity(),
                       gravity_, p, q, v);
  if (!is_keyframe(msg)) {
    // Not enough new information: report the IMU prediction.
    return make_result(msg.timestamp, p, q, v);
  }

  push_keyframe(msg, p, q, v, last.acc_bias(), last.gyro_bias());
  const long n_landmarks = optimize();
  const long outliers_before = stats_.outliers_removed;
  remove_outliers();
  KeyFrame& kf = *window_.back();
  if (debug_log_.is_open()) {
    char line[256];
    std::snprintf(line, sizeof(line), "%.6f,%ld,%.6g,%.6g,%d,%.4f,%.4f,%.4f,%ld\n",
                  msg.timestamp, n_landmarks, last_solve_.initial_cost, last_solve_.final_cost,
                  last_solve_.iterations, (Eigen::Vector3d(kf.position()) - p).norm(),
                  (Eigen::Vector3d(kf.velocity()) - v).norm(), kf.velocity().norm(),
                  stats_.outliers_removed - outliers_before);
    debug_log_ << line;
  }
  const VioResult result = make_result(msg.timestamp, kf.position(),
                                       Eigen::Quaterniond(kf.orientation()), kf.velocity());
  if (static_cast<int>(window_.size()) > cfg_.window_size) {
    marginalize_oldest();
  }
  return result;
}

}  // namespace msckf::swo
