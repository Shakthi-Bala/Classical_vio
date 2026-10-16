#include "msckf/image_processor.h"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include "msckf/math_utils.h"
#include "msckf/profiler.h"

namespace msckf {

namespace {

// Constants of image.py.
constexpr double kImuBeforePrevImage = 0.01;    // [s] gyro window starts this early ...
constexpr double kImuBeforeCurrImage = 0.004;   // [s] ... and ends this early
constexpr float kMaxLeftRightError = 3.0f;      // [px] cam0 -> cam1 -> cam0 round trip
constexpr double kMaxVerticalDisparity = 20.0;  // [px] tracked vs predicted cam1 row
constexpr int kMaskRadius = 3;                  // [px] no new detection this close to a feature

cv::Matx33d to_cv(const Eigen::Matrix3d& m) {
  cv::Matx33d out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = m(i, j);
    }
  }
  return out;
}

cv::Vec4d to_cv(const Eigen::Vector4d& v) { return cv::Vec4d(v(0), v(1), v(2), v(3)); }

Eigen::Matrix3d camera_matrix(const Eigen::Vector4d& intrinsics) {
  Eigen::Matrix3d K;
  K << intrinsics(0), 0.0, intrinsics(2),
       0.0, intrinsics(1), intrinsics(3),
       0.0, 0.0, 1.0;
  return K;
}

// Python's int(a / b) for a pixel coordinate and a grid size. The division is
// done in double: numpy promotes float32 / int to float64.
int grid_index(float coordinate, int grid_size) {
  return static_cast<int>(static_cast<double>(coordinate) / grid_size);
}

// Written as image.py's "outside" test, negated (a NaN point counts as inside).
bool in_image(const cv::Point2f& p, const cv::Mat& img) {
  return !(p.x < 0 || p.x > img.cols - 1 || p.y < 0 || p.y > img.rows - 1);
}

// Strongest first. With std::stable_sort, equal responses keep their original
// order, exactly like Python's sorted(..., reverse=True).
template <typename T>
bool stronger(const T& a, const T& b) {
  return a.response > b.response;
}

// Keeps only the elements whose selector is non-zero (image.py's select()).
template <typename T>
std::vector<T> select(const std::vector<T>& data, const std::vector<unsigned char>& selectors) {
  std::vector<T> out;
  for (size_t i = 0; i < data.size(); ++i) {
    if (selectors[i]) {
      out.push_back(data[i]);
    }
  }
  return out;
}

// Pixel -> undistorted normalized coordinates, optionally rotated by R. The
// output has the precision of the input (float32 or float64), as in OpenCV.
template <typename PointT>
std::vector<PointT> undistort_points(const std::vector<PointT>& pts_in, const CameraConfig& cam,
                                     const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity()) {
  std::vector<PointT> pts_out;
  if (pts_in.empty()) {
    return pts_out;
  }
  const cv::Matx33d K = to_cv(camera_matrix(cam.intrinsics));
  const cv::Matx33d K_new = cv::Matx33d::eye();   // output in normalized coordinates
  if (cam.distortion_model == "equidistant") {
    cv::fisheye::undistortPoints(pts_in, pts_out, K, to_cv(cam.distortion_coeffs), to_cv(R),
                                 K_new);
  } else {   // radtan
    cv::undistortPoints(pts_in, pts_out, K, to_cv(cam.distortion_coeffs), to_cv(R), K_new);
  }
  return pts_out;
}

// Normalized coordinates -> distorted pixels.
template <typename PointT>
std::vector<PointT> distort_points(const std::vector<PointT>& pts_in, const CameraConfig& cam) {
  using Scalar = decltype(PointT::x);
  std::vector<PointT> pts_out;
  if (pts_in.empty()) {
    return pts_out;
  }
  const cv::Matx33d K = to_cv(camera_matrix(cam.intrinsics));
  if (cam.distortion_model == "equidistant") {
    cv::fisheye::distortPoints(pts_in, pts_out, K, to_cv(cam.distortion_coeffs));
  } else {   // radtan: project (x, y, 1) with zero rotation and translation
    std::vector<cv::Point3_<Scalar>> homogeneous;
    homogeneous.reserve(pts_in.size());
    for (const PointT& p : pts_in) {
      homogeneous.emplace_back(p.x, p.y, Scalar(1));
    }
    cv::projectPoints(homogeneous, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K,
                      to_cv(cam.distortion_coeffs), pts_out);
  }
  return pts_out;
}

// Where each point should appear after a pure rotation R_p_c (previous ->
// current camera frame): the homography K R K^-1.
std::vector<cv::Point2f> predict_feature_tracking(const std::vector<cv::Point2f>& input_pts,
                                                  const Eigen::Matrix3d& R_p_c,
                                                  const Eigen::Vector4d& intrinsics) {
  const Eigen::Matrix3d K = camera_matrix(intrinsics);
  const Eigen::Matrix3d H = K * R_p_c * K.inverse();
  std::vector<cv::Point2f> compensated_pts;
  compensated_pts.reserve(input_pts.size());
  for (const cv::Point2f& p : input_pts) {
    const Eigen::Vector3d p2 = H * Eigen::Vector3d(p.x, p.y, 1.0);
    compensated_pts.emplace_back(static_cast<float>(p2(0) / p2(2)),
                                 static_cast<float>(p2(1) / p2(2)));
  }
  return compensated_pts;
}

}  // namespace

ImageProcessor::ImageProcessor(const Config& config)
    : config_(config),
      detector_(cv::FastFeatureDetector::create(config.fast_threshold)),
      prev_features_(config.grid_num()),
      curr_features_(config.grid_num()),
      cam0_(camera_to_imu(config.cam0)),
      cam1_(camera_to_imu(config.cam1)) {}

FeatureMsg ImageProcessor::stereo_callback(const StereoMsg& stereo_msg) {
  MSCKF_PROFILE("frontend");
  curr_timestamp_ = stereo_msg.timestamp;
  curr_cam0_img_ = stereo_msg.cam0_image;
  curr_cam1_img_ = stereo_msg.cam1_image;
  create_image_pyramids();

  if (is_first_img_) {
    grid_height_ = static_cast<int>(std::ceil(static_cast<double>(curr_cam0_img_.rows) /
                                              config_.grid_row));
    grid_width_ = static_cast<int>(std::ceil(static_cast<double>(curr_cam0_img_.cols) /
                                             config_.grid_col));
    initialize_first_frame();
    is_first_img_ = false;
  } else {
    track_features();
    add_new_features();
    prune_features();
  }

  FeatureMsg msg = publish();

  // The current frame becomes the previous one.
  prev_timestamp_ = curr_timestamp_;
  prev_cam0_pyramid_ = std::move(curr_cam0_pyramid_);
  curr_cam0_pyramid_.clear();
  curr_cam1_pyramid_.clear();
  prev_features_ = std::move(curr_features_);
  curr_features_.assign(config_.grid_num(), {});
  return msg;
}

int ImageProcessor::grid_cell(const cv::Point2f& p) const {
  return grid_index(p.y, grid_height_) * config_.grid_col + grid_index(p.x, grid_width_);
}

void ImageProcessor::create_image_pyramids() {
  MSCKF_PROFILE("frontend/build_pyramids");
  const cv::Size win(config_.patch_size, config_.patch_size);
  cv::buildOpticalFlowPyramid(curr_cam0_img_, curr_cam0_pyramid_, win, config_.pyramid_levels,
                              true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
  cv::buildOpticalFlowPyramid(curr_cam1_img_, curr_cam1_pyramid_, win, config_.pyramid_levels,
                              true, cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
}

void ImageProcessor::lk(const std::vector<cv::Mat>& prev_pyramid,
                        const std::vector<cv::Mat>& next_pyramid,
                        const std::vector<cv::Point2f>& prev_pts,
                        std::vector<cv::Point2f>& next_pts,
                        std::vector<unsigned char>& status) const {
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(
      prev_pyramid, next_pyramid, prev_pts, next_pts, status, err,
      cv::Size(config_.patch_size, config_.patch_size), config_.pyramid_levels,
      cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                       config_.max_iteration, config_.track_precision),
      cv::OPTFLOW_USE_INITIAL_FLOW);
}

void ImageProcessor::initialize_first_frame() {
  MSCKF_PROFILE("frontend/initialize_first_frame");
  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(curr_cam0_img_, keypoints);
  add_stereo_features(keypoints);
}

void ImageProcessor::add_stereo_features(const std::vector<cv::KeyPoint>& keypoints) {
  // Keypoint coordinates are Python floats (float64) in image.py.
  std::vector<cv::Point2d> cam0_points;
  for (const cv::KeyPoint& kp : keypoints) {
    cam0_points.emplace_back(kp.pt.x, kp.pt.y);
  }
  std::vector<cv::Point2f> cam1_points;
  std::vector<unsigned char> inlier_markers;
  stereo_match(cam0_points, cam1_points, inlier_markers);

  GridFeatures grid_new_features(config_.grid_num());
  for (size_t i = 0; i < inlier_markers.size(); ++i) {
    if (!inlier_markers[i]) {
      continue;
    }
    FeatureMetaData f;
    f.response = keypoints[i].response;
    f.cam0_point = cv::Point2f(cam0_points[i]);
    f.cam1_point = cam1_points[i];
    grid_new_features[grid_cell(f.cam0_point)].push_back(f);
  }

  for (int i = 0; i < config_.grid_num(); ++i) {
    std::vector<FeatureMetaData>& features = grid_new_features[i];
    std::stable_sort(features.begin(), features.end(), stronger<FeatureMetaData>);
    const size_t n = std::min<size_t>(features.size(), config_.grid_min_feature_num);
    for (size_t k = 0; k < n; ++k) {
      FeatureMetaData f = features[k];
      f.id = next_feature_id_++;
      f.lifetime = 1;
      curr_features_[i].push_back(f);
    }
  }
}

void ImageProcessor::track_features() {
  MSCKF_PROFILE("frontend/track_features");
  Eigen::Matrix3d cam0_R_p_c, cam1_R_p_c;
  integrate_imu_data(cam0_R_p_c, cam1_R_p_c);

  // Flatten the previous features in grid order.
  std::vector<FeatureID> prev_ids;
  std::vector<int> prev_lifetime;
  std::vector<cv::Point2f> prev_cam0_points;
  for (const auto& grid : prev_features_) {
    for (const FeatureMetaData& f : grid) {
      prev_ids.push_back(f.id);
      prev_lifetime.push_back(f.lifetime);
      prev_cam0_points.push_back(f.cam0_point);
    }
  }
  if (prev_cam0_points.empty()) {
    return;
  }

  // Track with KLT, starting from the gyro-predicted locations.
  std::vector<cv::Point2f> curr_cam0_points =
      predict_feature_tracking(prev_cam0_points, cam0_R_p_c, config_.cam0.intrinsics);
  std::vector<unsigned char> track_inliers;
  {
    MSCKF_PROFILE("frontend/track_features/klt_temporal");
    lk(prev_cam0_pyramid_, curr_cam0_pyramid_, prev_cam0_points, curr_cam0_points, track_inliers);
  }
  for (size_t i = 0; i < curr_cam0_points.size(); ++i) {
    if (track_inliers[i] && !in_image(curr_cam0_points[i], curr_cam0_img_)) {
      track_inliers[i] = 0;
    }
  }

  const auto prev_tracked_ids = select(prev_ids, track_inliers);
  const auto prev_tracked_lifetime = select(prev_lifetime, track_inliers);
  const auto curr_tracked_cam0_points = select(curr_cam0_points, track_inliers);

  // Stereo matching (tracked points are float32 in Python).
  std::vector<cv::Point2f> curr_cam1_points;
  std::vector<unsigned char> match_inliers;
  stereo_match(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

  const auto prev_matched_ids = select(prev_tracked_ids, match_inliers);
  const auto prev_matched_lifetime = select(prev_tracked_lifetime, match_inliers);
  const auto curr_matched_cam0_points = select(curr_tracked_cam0_points, match_inliers);
  const auto curr_matched_cam1_points = select(curr_cam1_points, match_inliers);

  // image.py disables its 2-point RANSAC: every stereo match is kept.
  for (size_t i = 0; i < curr_matched_cam0_points.size(); ++i) {
    FeatureMetaData f;
    f.id = prev_matched_ids[i];
    f.lifetime = prev_matched_lifetime[i] + 1;
    f.cam0_point = curr_matched_cam0_points[i];
    f.cam1_point = curr_matched_cam1_points[i];
    curr_features_[grid_cell(f.cam0_point)].push_back(f);
  }
}

void ImageProcessor::add_new_features() {
  MSCKF_PROFILE("frontend/add_new_features");
  const cv::Mat& img = curr_cam0_img_;

  cv::Mat mask(img.rows, img.cols, CV_8U, cv::Scalar(1));
  for (const auto& grid : curr_features_) {
    for (const FeatureMetaData& f : grid) {
      const int x = static_cast<int>(f.cam0_point.x);
      const int y = static_cast<int>(f.cam0_point.y);
      // NOTE (Python parity): numpy's mask[y-3:y+4, x-3:x+4] with a negative
      // start wraps around and selects nothing, so features within 3 px of the
      // top/left border are not masked (quirk 6 in README).
      if (!config_.fixes.mask_clamp && (y - kMaskRadius < 0 || x - kMaskRadius < 0)) {
        continue;
      }
      mask(cv::Range(std::max(y - kMaskRadius, 0), std::min(y + kMaskRadius + 1, img.rows)),
           cv::Range(std::max(x - kMaskRadius, 0), std::min(x + kMaskRadius + 1, img.cols)))
          .setTo(0);
    }
  }

  std::vector<cv::KeyPoint> detected;
  {
    MSCKF_PROFILE("frontend/add_new_features/fast_detect");
    detector_->detect(img, detected, mask);
  }

  // At most grid_max_feature_num of the strongest detections per cell.
  std::vector<std::vector<cv::KeyPoint>> sieve(config_.grid_num());
  for (const cv::KeyPoint& kp : detected) {
    sieve[grid_cell(kp.pt)].push_back(kp);
  }
  std::vector<cv::KeyPoint> candidates;
  for (std::vector<cv::KeyPoint>& cell : sieve) {
    if (static_cast<int>(cell.size()) > config_.grid_max_feature_num) {
      std::stable_sort(cell.begin(), cell.end(), stronger<cv::KeyPoint>);
      cell.resize(config_.grid_max_feature_num);
    }
    candidates.insert(candidates.end(), cell.begin(), cell.end());
  }
  add_stereo_features(candidates);
}

void ImageProcessor::prune_features() {
  MSCKF_PROFILE("frontend/prune_features");
  // Too many features in a cell: keep the longest-tracked ones.
  for (std::vector<FeatureMetaData>& features : curr_features_) {
    if (static_cast<int>(features.size()) <= config_.grid_max_feature_num) {
      continue;
    }
    std::stable_sort(features.begin(), features.end(),
                     [](const FeatureMetaData& a, const FeatureMetaData& b) {
                       return a.lifetime > b.lifetime;
                     });
    features.resize(config_.grid_max_feature_num);
  }
}

FeatureMsg ImageProcessor::publish() const {
  MSCKF_PROFILE("frontend/publish");
  std::vector<FeatureID> ids;
  std::vector<cv::Point2f> cam0_points;
  std::vector<cv::Point2f> cam1_points;
  bool has_new_feature = false;
  for (const auto& grid : curr_features_) {
    for (const FeatureMetaData& f : grid) {
      ids.push_back(f.id);
      cam0_points.push_back(f.cam0_point);
      cam1_points.push_back(f.cam1_point);
      has_new_feature = has_new_feature || f.lifetime == 1;
    }
  }

  // NOTE (Python parity): in image.py a freshly detected cam0 point is a tuple
  // of Python floats while a tracked one is a float32 array. If any new point
  // is present numpy builds a float64 array, otherwise float32, and OpenCV
  // returns the undistorted points in that same precision. cam1 points are
  // always float32 (quirk 8 in README).
  auto undistort_as = [](bool as_double, const std::vector<cv::Point2f>& pts,
                         const CameraConfig& cam) {
    if (as_double) {
      return undistort_points(std::vector<cv::Point2d>(pts.begin(), pts.end()), cam);
    }
    const std::vector<cv::Point2f> und = undistort_points(pts, cam);
    return std::vector<cv::Point2d>(und.begin(), und.end());
  };
  const bool full = config_.fixes.full_precision_publish;
  const std::vector<cv::Point2d> cam0_und =
      undistort_as(has_new_feature || full, cam0_points, config_.cam0);
  const std::vector<cv::Point2d> cam1_und = undistort_as(full, cam1_points, config_.cam1);

  FeatureMsg msg;
  msg.timestamp = curr_timestamp_;
  msg.features.reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    msg.features.push_back({ids[i], cam0_und[i].x, cam0_und[i].y, cam1_und[i].x, cam1_und[i].y});
  }
  return msg;
}

void ImageProcessor::integrate_imu_data(Eigen::Matrix3d& cam0_R_p_c,
                                        Eigen::Matrix3d& cam1_R_p_c) {
  MSCKF_PROFILE("frontend/track_features/integrate_imu");
  cam0_R_p_c.setIdentity();
  cam1_R_p_c.setIdentity();

  // IMU messages between the previous and the current image.
  const auto begin_it = std::find_if(
      imu_msg_buffer_.begin(), imu_msg_buffer_.end(),
      [this](const ImuMsg& m) { return m.timestamp >= prev_timestamp_ - kImuBeforePrevImage; });
  const auto end_it = std::find_if(
      imu_msg_buffer_.begin(), imu_msg_buffer_.end(),
      [this](const ImuMsg& m) { return m.timestamp >= curr_timestamp_ - kImuBeforeCurrImage; });
  if (begin_it == imu_msg_buffer_.end() || end_it == imu_msg_buffer_.end()) {
    return;
  }

  Eigen::Vector3d mean_ang_vel = Eigen::Vector3d::Zero();
  for (auto it = begin_it; it < end_it; ++it) {
    mean_ang_vel += it->angular_velocity;
  }
  if (end_it > begin_it) {
    mean_ang_vel /= static_cast<double>(end_it - begin_it);
  }

  // Mean angular velocity in each camera frame -> relative rotation.
  const double dt = curr_timestamp_ - prev_timestamp_;
  auto rotation_transposed = [](const Eigen::Vector3d& rotation_vector) {
    cv::Matx33d R;
    cv::Rodrigues(cv::Vec3d(rotation_vector(0), rotation_vector(1), rotation_vector(2)), R);
    Eigen::Matrix3d out;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        out(i, j) = R(j, i);
      }
    }
    return out;
  };
  const Eigen::Vector3d cam0_mean_ang_vel = cam0_.R_cam_imu.transpose() * mean_ang_vel;
  const Eigen::Vector3d cam1_mean_ang_vel = cam1_.R_cam_imu.transpose() * mean_ang_vel;
  cam0_R_p_c = rotation_transposed(cam0_mean_ang_vel * dt);
  cam1_R_p_c = rotation_transposed(cam1_mean_ang_vel * dt);

  imu_msg_buffer_.erase(imu_msg_buffer_.begin(), end_it);
}

template <typename PointT>
void ImageProcessor::stereo_match(const std::vector<PointT>& cam0_points_in,
                                  std::vector<cv::Point2f>& cam1_points,
                                  std::vector<unsigned char>& inlier_markers) const {
  MSCKF_PROFILE("frontend/*/stereo_match");
  using Scalar = decltype(PointT::x);
  cam1_points.clear();
  inlier_markers.clear();
  if (cam0_points_in.empty()) {
    return;
  }

  // Predict the cam1 location: undistort in cam0, rotate into cam1, distort
  // with the cam1 model. (image.py derives cam0 -> cam1 from the two
  // IMU-camera extrinsics, not from T_cn_cnm1 like the filter.)
  const Eigen::Matrix3d R_cam0_cam1 = cam1_.R_cam_imu.transpose() * cam0_.R_cam_imu;
  const std::vector<PointT> cam1_points_predicted =
      distort_points(undistort_points(cam0_points_in, config_.cam0, R_cam0_cam1), config_.cam1);

  // KLT works on float32 points.
  const std::vector<cv::Point2f> cam0_points(cam0_points_in.begin(), cam0_points_in.end());
  cam1_points.assign(cam1_points_predicted.begin(), cam1_points_predicted.end());
  lk(curr_cam0_pyramid_, curr_cam1_pyramid_, cam0_points, cam1_points, inlier_markers);

  // Track back from cam1 to cam0 for a left-right consistency check.
  std::vector<cv::Point2f> cam0_points_back = cam0_points;
  std::vector<unsigned char> back_status;
  lk(curr_cam1_pyramid_, curr_cam0_pyramid_, cam1_points, cam0_points_back, back_status);

  for (size_t i = 0; i < cam0_points.size(); ++i) {
    const float dx = cam0_points[i].x - cam0_points_back[i].x;
    const float dy = cam0_points[i].y - cam0_points_back[i].y;
    const float err = std::sqrt(dx * dx + dy * dy);
    const Scalar disparity =
        std::abs(cam1_points_predicted[i].y - static_cast<Scalar>(cam1_points[i].y));
    inlier_markers[i] = inlier_markers[i] && err < kMaxLeftRightError &&
                        disparity < static_cast<Scalar>(kMaxVerticalDisparity) &&
                        in_image(cam1_points[i], curr_cam1_img_);
  }

  // Remove outliers with the known essential matrix.
  const Eigen::Vector3d t_cam0_cam1 =
      cam1_.R_cam_imu.transpose() * (cam0_.t_cam_imu - cam1_.t_cam_imu);
  const Eigen::Matrix3d E = skew(t_cam0_cam1) * R_cam0_cam1;
  const std::vector<cv::Point2f> cam0_und = undistort_points(cam0_points, config_.cam0);
  const std::vector<cv::Point2f> cam1_und = undistort_points(cam1_points, config_.cam1);

  const Eigen::Vector4d& k0 = config_.cam0.intrinsics;
  const Eigen::Vector4d& k1 = config_.cam1.intrinsics;
  const double norm_pixel_unit = 4.0 / (k0(0) + k0(1) + k1(0) + k1(1));

  for (size_t i = 0; i < cam0_und.size(); ++i) {
    if (!inlier_markers[i]) {
      continue;
    }
    const Eigen::Vector3d pt0(cam0_und[i].x, cam0_und[i].y, 1.0);
    const Eigen::Vector3d pt1(cam1_und[i].x, cam1_und[i].y, 1.0);
    const Eigen::Vector3d epipolar_line = E * pt0;
    // NOTE (Python parity): image.py computes (pt1 * epipolar_line)[0], i.e.
    // only the first term of the dot product pt1 . line. Upstream msckf_vio
    // uses the full dot product (quirk 1 in README).
    const double numerator = config_.fixes.epipolar_full_dot ? pt1.dot(epipolar_line)
                                                             : pt1(0) * epipolar_line(0);
    const double error = std::abs(numerator) / epipolar_line.head<2>().norm();
    if (error > config_.stereo_threshold * norm_pixel_unit) {
      inlier_markers[i] = 0;
    }
  }
}

}  // namespace msckf
