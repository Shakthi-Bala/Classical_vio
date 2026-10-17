#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "msckf/config.h"
#include "msckf/loop/pose_graph.h"
#include "msckf/math_utils.h"

// Loop closure on top of either back end (main_euroc --loop).
//
//  - Keyframes: a VIO pose every LoopConfig::kf_distance / kf_angle, with ORB
//    features on cam0, triangulated with cam1 where a stereo match exists.
//  - Candidates: older keyframes (min_time_gap) whose corrected position is
//    within search_radius and whose camera looks the same way. There is no
//    appearance-based place recognition (bag of words): this relies on the
//    drift staying below search_radius, which holds on EuRoC.
//  - Verification: ORB matches from the candidate's 3-D points to the current
//    keyframe, PnP RANSAC; enough inliers make a loop edge.
//  - A 4-DoF pose graph (PoseGraph) is optimized after each loop; the VIO
//    poses are then corrected by the drift of the newest keyframe.

namespace msckf::loop {

class LoopClosure {
 public:
  explicit LoopClosure(const Config& config);

  // VIO pose (body in the VIO world frame) and stereo images of one frame.
  // Returns the loop-corrected pose.
  Isometry3d process(double t, const Isometry3d& T_w_b, const cv::Mat& cam0, const cv::Mat& cam1);

  // Every processed frame corrected with the final pose graph.
  std::vector<std::pair<double, Isometry3d>> final_trajectory() const;

  struct Stats {
    long keyframes = 0;
    long candidates = 0;   // PnP attempts
    long loops = 0;        // accepted loop edges
  };
  const Stats& stats() const { return stats_; }

 private:
  struct KeyFrame {
    double t = 0.0;
    std::vector<cv::Point2d> points;   // cam0 keypoints, undistorted normalized
    cv::Mat descriptors;               // one ORB descriptor per point
    std::vector<Eigen::Vector3d> landmarks;   // stereo-triangulated, cam0 frame
    cv::Mat landmark_descriptors;
  };

  KeyFrame make_keyframe(double t, const cv::Mat& cam0, const cv::Mat& cam1) const;
  bool is_keyframe(const Isometry3d& T_w_b) const;
  // Relative pose of the current camera to the candidate's camera, from PnP:
  // x_cur = R x_cand + t. False if not enough inliers.
  bool verify(const KeyFrame& candidate, const KeyFrame& current, Eigen::Matrix3d& R,
              Eigen::Vector3d& t) const;
  void try_loops(int current);
  // drift_ that makes keyframe's VIO pose its corrected pose.
  Isometry3d drift_at(int keyframe) const;

  LoopConfig cfg_;
  CameraConfig cam0_, cam1_;
  Eigen::Matrix3d R_bc_;   // cam0 -> body
  Eigen::Vector3d t_bc_;
  Eigen::Matrix3d R_c1c0_;   // cam0 -> cam1
  Eigen::Vector3d t_c1c0_;
  cv::Ptr<cv::ORB> orb_;

  PoseGraph graph_;
  std::vector<KeyFrame> keyframes_;
  // corrected = drift * VIO (a yaw rotation and a translation).
  Isometry3d drift_;
  struct Frame {
    double t;
    Isometry3d vio;
    int keyframe;   // newest keyframe at or before this frame
  };
  std::vector<Frame> frames_;
  Stats stats_;
};

}  // namespace msckf::loop
