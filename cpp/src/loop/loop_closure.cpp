#include "msckf/loop/loop_closure.h"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace msckf::loop {

namespace {

constexpr int kMaxCandidates = 3;            // PnP attempts per keyframe
constexpr float kRatioTest = 0.75f;          // Lowe's ratio for ORB matches
constexpr double kMaxEpipolarPx = 2.0;       // stereo match check [px]
constexpr double kMinDepth = 0.1, kMaxDepth = 8.0;    // stereo landmarks [m]: 11 cm baseline
constexpr double kMaxLoopYaw = 30.0 * M_PI / 180.0;   // loop vs current estimate

// Pixel -> undistorted normalized coordinates.
std::vector<cv::Point2d> undistort(const std::vector<cv::KeyPoint>& keypoints,
                                   const CameraConfig& cam) {
  std::vector<cv::Point2d> in, out;
  for (const cv::KeyPoint& kp : keypoints) {
    in.emplace_back(kp.pt.x, kp.pt.y);
  }
  if (in.empty()) {
    return out;
  }
  const Eigen::Vector4d& k = cam.intrinsics;
  const cv::Matx33d K(k(0), 0, k(2), 0, k(1), k(3), 0, 0, 1);
  const cv::Vec4d D(cam.distortion_coeffs(0), cam.distortion_coeffs(1),
                    cam.distortion_coeffs(2), cam.distortion_coeffs(3));
  if (cam.distortion_model == "equidistant") {
    cv::fisheye::undistortPoints(in, out, K, D);
  } else {
    cv::undistortPoints(in, out, K, D);
  }
  return out;
}

// Matches a -> b with the ratio test; pairs (index in a, index in b).
std::vector<std::pair<int, int>> match(const cv::Mat& a, const cv::Mat& b) {
  std::vector<std::pair<int, int>> out;
  if (a.empty() || b.rows < 2) {
    return out;
  }
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(a, b, knn, 2);
  for (const auto& m : knn) {
    if (m.size() == 2 && m[0].distance < kRatioTest * m[1].distance) {
      out.emplace_back(m[0].queryIdx, m[0].trainIdx);
    }
  }
  return out;
}

double yaw_of(const Eigen::Matrix3d& R) { return rotation_to_ypr(R)(0); }

double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

}  // namespace

LoopClosure::LoopClosure(const Config& config)
    : cfg_(config.loop),
      cam0_(config.cam0),
      cam1_(config.cam1),
      orb_(cv::ORB::create(config.loop.orb_features)) {
  const CameraExtrinsics cam0 = camera_to_imu(config.cam0);
  R_bc_ = cam0.R_cam_imu;
  t_bc_ = cam0.t_cam_imu;
  R_c1c0_ = config.T_cn_cnm1.topLeftCorner<3, 3>();
  t_c1c0_ = config.T_cn_cnm1.topRightCorner<3, 1>();
}

LoopClosure::KeyFrame LoopClosure::make_keyframe(double t, const cv::Mat& cam0,
                                                 const cv::Mat& cam1) const {
  KeyFrame kf;
  kf.t = t;
  std::vector<cv::KeyPoint> kp0, kp1;
  cv::Mat desc1;
  orb_->detectAndCompute(cam0, cv::noArray(), kp0, kf.descriptors);
  orb_->detectAndCompute(cam1, cv::noArray(), kp1, desc1);
  kf.points = undistort(kp0, cam0_);
  const std::vector<cv::Point2d> points1 = undistort(kp1, cam1_);

  // Stereo landmarks: matches that satisfy the epipolar constraint and give a
  // plausible depth.
  const Eigen::Matrix3d E = skew(t_c1c0_) * R_c1c0_;
  const double max_epipolar = kMaxEpipolarPx / cam0_.intrinsics(0);
  for (const auto& [i0, i1] : match(kf.descriptors, desc1)) {
    const Eigen::Vector2d z0(kf.points[i0].x, kf.points[i0].y);
    const Eigen::Vector2d z1(points1[i1].x, points1[i1].y);
    const Eigen::Vector3d line = E * z0.homogeneous();
    if (std::abs(z1.homogeneous().dot(line)) / line.head<2>().norm() > max_epipolar) {
      continue;
    }
    const double depth = two_view_depth(R_c1c0_, t_c1c0_, z0, z1);
    if (!std::isfinite(depth) || depth < kMinDepth || depth > kMaxDepth) {
      continue;
    }
    kf.landmarks.push_back(Eigen::Vector3d(z0(0), z0(1), 1.0) * depth);
    kf.landmark_descriptors.push_back(kf.descriptors.row(i0));
  }
  return kf;
}

bool LoopClosure::is_keyframe(const Isometry3d& T_w_b) const {
  if (graph_.size() == 0) {
    return true;
  }
  const Isometry3d& last = graph_.vio_pose(graph_.size() - 1);
  const double angle = Eigen::AngleAxisd(last.R.transpose() * T_w_b.R).angle();
  return (T_w_b.t - last.t).norm() > cfg_.kf_distance || angle > cfg_.kf_angle;
}

bool LoopClosure::verify(const KeyFrame& candidate, const KeyFrame& current, Eigen::Matrix3d& R,
                         Eigen::Vector3d& t, int& n_inliers) const {
  const auto matches = match(candidate.landmark_descriptors, current.descriptors);
  if (static_cast<int>(matches.size()) < cfg_.min_inliers) {
    return false;
  }
  std::vector<cv::Point3d> object;
  std::vector<cv::Point2d> image;
  for (const auto& [il, ic] : matches) {
    const Eigen::Vector3d& X = candidate.landmarks[il];
    object.emplace_back(X(0), X(1), X(2));
    image.push_back(current.points[ic]);
  }
  cv::Mat rvec, tvec;
  std::vector<int> inliers;
  const bool ok = cv::solvePnPRansac(object, image, cv::Matx33d::eye(), cv::noArray(), rvec,
                                     tvec, false, 200, cfg_.pnp_px / cam0_.intrinsics(0), 0.99,
                                     inliers, cv::SOLVEPNP_EPNP);
  if (!ok || static_cast<int>(inliers.size()) < cfg_.min_inliers) {
    return false;
  }
  n_inliers = static_cast<int>(inliers.size());
  // Refine on the inliers (EPnP alone is not accurate enough for loop edges).
  std::vector<cv::Point3d> object_in;
  std::vector<cv::Point2d> image_in;
  for (const int k : inliers) {
    object_in.push_back(object[k]);
    image_in.push_back(image[k]);
  }
  cv::solvePnPRefineLM(object_in, image_in, cv::Matx33d::eye(), cv::noArray(), rvec, tvec);
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);
  cv::cv2eigen(R_cv, R);
  cv::cv2eigen(tvec, t);
  return true;
}

void LoopClosure::try_loops(int current) {
  const Isometry3d T_cur = graph_.pose(current);
  const Eigen::Vector3d view_cur = T_cur.R * R_bc_.col(2);   // camera z in the world

  std::vector<std::pair<double, int>> candidates;   // (distance, keyframe)
  for (int i = 0; i < current; ++i) {
    if (keyframes_[current].t - keyframes_[i].t < cfg_.min_time_gap) {
      break;   // keyframes are in time order
    }
    const Isometry3d T_i = graph_.pose(i);
    const double d = (T_i.t - T_cur.t).norm();
    const double view = std::acos(std::clamp((T_i.R * R_bc_.col(2)).dot(view_cur), -1.0, 1.0));
    if (d < cfg_.search_radius && view < cfg_.max_view_angle) {
      candidates.emplace_back(d, i);
    }
  }
  std::sort(candidates.begin(), candidates.end());
  if (static_cast<int>(candidates.size()) > kMaxCandidates) {
    candidates.resize(kMaxCandidates);
  }

  for (const auto& [d, i] : candidates) {
    ++stats_.candidates;
    Eigen::Matrix3d R_cc;   // candidate camera -> current camera
    Eigen::Vector3d t_cc;
    int inliers = 0;
    if (!verify(keyframes_[i], keyframes_[current], R_cc, t_cc, inliers)) {
      continue;
    }
    // Current body in the candidate body frame: T_bc * T_cand_cam_cur_cam * T_cb.
    const Eigen::Matrix3d R_cand_cur_cam = R_cc.transpose();
    const Eigen::Vector3d t_cand_cur_cam = -R_cc.transpose() * t_cc;
    const Eigen::Matrix3d R_ij = R_bc_ * R_cand_cur_cam * R_bc_.transpose();
    const Eigen::Vector3d t_ij = R_bc_ * t_cand_cur_cam + t_bc_ - R_ij * t_bc_;

    // Reject loops that disagree strongly with the current estimate.
    const Isometry3d T_i = graph_.pose(i);
    const Eigen::Matrix3d R_cur_loop = T_i.R * R_ij;
    if ((T_i.t + T_i.R * t_ij - T_cur.t).norm() > cfg_.max_loop_correction ||
        std::abs(wrap(yaw_of(R_cur_loop) - yaw_of(T_cur.R))) > kMaxLoopYaw) {
      continue;
    }
    // Yaw measured as in VINS-Fusion: with the candidate's VIO roll and pitch.
    const Eigen::Matrix3d& R_i_vio = graph_.vio_pose(i).R;
    const double yaw_ij = wrap(yaw_of(R_i_vio * R_ij) - yaw_of(R_i_vio));
    graph_.add_loop(i, current, t_ij, yaw_ij);
    loops_.push_back({keyframes_[i].t, keyframes_[current].t, t_ij, yaw_ij, inliers});
    ++stats_.loops;
  }
}

Isometry3d LoopClosure::drift_at(int keyframe) const {
  const Isometry3d T_corr = graph_.pose(keyframe);
  const Isometry3d& T_vio = graph_.vio_pose(keyframe);
  const double dyaw = yaw_of(T_corr.R) - yaw_of(T_vio.R);
  const Eigen::Matrix3d R = Eigen::AngleAxisd(dyaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return Isometry3d(R, T_corr.t - R * T_vio.t);
}

Isometry3d LoopClosure::process(double t, const Isometry3d& T_w_b, const cv::Mat& cam0,
                                const cv::Mat& cam1) {
  if (is_keyframe(T_w_b)) {
    const int idx = graph_.add_keyframe(T_w_b, drift_ * T_w_b);
    keyframes_.push_back(make_keyframe(t, cam0, cam1));
    ++stats_.keyframes;
    const long loops_before = stats_.loops;
    try_loops(idx);
    if (stats_.loops > loops_before) {
      graph_.optimize();
      drift_ = drift_at(idx);
    }
  }
  frames_.push_back({t, T_w_b, graph_.size() - 1});
  return drift_ * T_w_b;
}

std::vector<std::pair<double, Isometry3d>> LoopClosure::final_trajectory() const {
  std::vector<std::pair<double, Isometry3d>> out;
  out.reserve(frames_.size());
  for (const Frame& f : frames_) {
    out.emplace_back(f.t, drift_at(f.keyframe) * f.vio);
  }
  return out;
}

}  // namespace msckf::loop
