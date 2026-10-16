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

  void initialize_first_frame();
  void track_features();
  void add_new_features();
  void prune_features();
  FeatureMsg publish() const;

  // Mean gyro rotation between the previous and current image, as rotations
  // taking a vector from the previous to the current cam0 / cam1 frame.
  void integrate_imu_data(Eigen::Matrix3d& cam0_R_p_c, Eigen::Matrix3d& cam1_R_p_c);

  std::vector<cv::Point2f> predict_feature_tracking(const std::vector<cv::Point2f>& input_pts,
                                                    const Eigen::Matrix3d& R_p_c,
                                                    const Eigen::Vector4d& intrinsics) const;

  // Finds each cam0 point in cam1. PointT is cv::Point2d for freshly detected
  // points and cv::Point2f for tracked ones, because the Python code runs the
  // first undistortion in float64 or float32 depending on where the points
  // came from. Writes cam1 points and a 0/1 inlier flag per point.
  template <typename PointT>
  void stereo_match(const std::vector<PointT>& cam0_points,
                    std::vector<cv::Point2f>& cam1_points,
                    std::vector<unsigned char>& inlier_markers) const;

  // Pixel -> undistorted normalized coordinates (optionally rotated by R).
  template <typename PointT>
  std::vector<PointT> undistort_points(const std::vector<PointT>& pts_in,
                                       const Eigen::Vector4d& intrinsics,
                                       const std::string& distortion_model,
                                       const Eigen::Vector4d& distortion_coeffs,
                                       const Eigen::Matrix3d& rectification_matrix =
                                           Eigen::Matrix3d::Identity()) const;

  // Normalized coordinates -> distorted pixels.
  template <typename PointT>
  std::vector<PointT> distort_points(const std::vector<PointT>& pts_in,
                                     const Eigen::Vector4d& intrinsics,
                                     const std::string& distortion_model,
                                     const Eigen::Vector4d& distortion_coeffs) const;

  void get_grid_size(const cv::Mat& img, int& grid_height, int& grid_width) const;

  // Optical flow with the parameters of config.py's lk_params.
  void lk(const cv::Mat& prev_img, const cv::Mat& next_img,
          const std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& next_pts,
          std::vector<unsigned char>& status) const;

  Config config_;
  bool is_first_img_ = true;
  FeatureID next_feature_id_ = 0;
  cv::Ptr<cv::FastFeatureDetector> detector_;
  std::vector<ImuMsg> imu_msg_buffer_;

  // Previous / current images. The Python version does not build pyramids
  // itself (calcOpticalFlowPyrLK builds them internally), and neither do we.
  double prev_timestamp_ = 0.0;
  double curr_timestamp_ = 0.0;
  cv::Mat prev_cam0_img_;
  cv::Mat curr_cam0_img_;
  cv::Mat curr_cam1_img_;

  GridFeatures prev_features_;
  GridFeatures curr_features_;

  // Extrinsics: take a vector from the camera frame to the IMU frame.
  Eigen::Matrix3d R_cam0_imu_, R_cam1_imu_;
  Eigen::Vector3d t_cam0_imu_, t_cam1_imu_;
};

}  // namespace msckf
