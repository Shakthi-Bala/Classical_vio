#pragma once

#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "msckf/config.h"
#include "msckf/messages.h"

// Stereo feature front end: FAST detection on a grid, KLT tracking with gyro
// prediction, stereo matching with a known epipolar geometry.
// C++ port of ImageProcessor in Code/image.py.

namespace msckf {

class ImageProcessor {
 public:
  explicit ImageProcessor(const Config& config);

  // Processes one stereo pair and returns the features to send to the filter.
  FeatureMsg stereo_callback(const StereoMsg& stereo_msg);

  void imu_callback(const ImuMsg& msg) { imu_msg_buffer_.push_back(msg); }

 private:
  struct FeatureMetaData {
    FeatureID id = 0;
    float response = 0.0f;
    int lifetime = 0;
    cv::Point2f cam0_point;
    cv::Point2f cam1_point;
  };
  // One list of features per grid cell, indexed by row * grid_col + col.
  using GridFeatures = std::vector<std::vector<FeatureMetaData>>;

  void create_image_pyramids();
  void initialize_first_frame();
  void track_features();
  void add_new_features();
  void prune_features();
  FeatureMsg publish() const;

  // Stereo-matches new detections, bins the matches into grid cells and adds
  // the strongest grid_min_feature_num of each cell as new features.
  void add_stereo_features(const std::vector<cv::KeyPoint>& keypoints);

  // Mean gyro rotation between the previous and current image, as rotations
  // taking a vector from the previous to the current cam0 / cam1 frame.
  void integrate_imu_data(Eigen::Matrix3d& cam0_R_p_c, Eigen::Matrix3d& cam1_R_p_c);

  // Finds each cam0 point in cam1. PointT is cv::Point2d for freshly detected
  // points and cv::Point2f for tracked ones, because the Python code runs the
  // first undistortion in float64 or float32 depending on where the points
  // came from. Writes cam1 points and a 0/1 inlier flag per point.
  template <typename PointT>
  void stereo_match(const std::vector<PointT>& cam0_points,
                    std::vector<cv::Point2f>& cam1_points,
                    std::vector<unsigned char>& inlier_markers) const;

  // Optical flow with the parameters of config.py's lk_params, on prebuilt pyramids.
  void lk(const std::vector<cv::Mat>& prev_pyramid, const std::vector<cv::Mat>& next_pyramid,
          const std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& next_pts,
          std::vector<unsigned char>& status) const;

  // Grid cell index of a cam0 pixel.
  int grid_cell(const cv::Point2f& p) const;

  Config config_;
  int grid_height_ = 0;
  int grid_width_ = 0;
  bool is_first_img_ = true;
  FeatureID next_feature_id_ = 0;
  cv::Ptr<cv::FastFeatureDetector> detector_;
  std::vector<ImuMsg> imu_msg_buffer_;

  double prev_timestamp_ = 0.0;
  double curr_timestamp_ = 0.0;
  cv::Mat curr_cam0_img_;
  cv::Mat curr_cam1_img_;

  // Image pyramids, built once per image and shared by all KLT calls.
  // (image.py passes raw images, so OpenCV rebuilt both pyramids on each of
  // the 5 KLT calls per frame: 10 builds for 3 distinct images.) The result is
  // identical: calcOpticalFlowPyrLK builds the same pyramid internally.
  std::vector<cv::Mat> prev_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam1_pyramid_;

  GridFeatures prev_features_;
  GridFeatures curr_features_;

  // Camera -> IMU extrinsics (fixed; the filter estimates its own copy).
  CameraExtrinsics cam0_;
  CameraExtrinsics cam1_;
};

}  // namespace msckf
