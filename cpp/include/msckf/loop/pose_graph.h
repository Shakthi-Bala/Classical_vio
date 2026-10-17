#pragma once

#include <vector>

#include <Eigen/Dense>

#include "msckf/math_utils.h"

// 4-DoF pose graph over keyframes (x, y, z, yaw), the loop-closure back end of
// VINS-Fusion: roll and pitch are observable from gravity, so they stay at the
// VIO values and only position and yaw drift.
//
// Each keyframe is tied to the previous kSequentialEdges keyframes by its VIO
// relative pose; loop edges come from place recognition. Residuals are the
// relative position in the first keyframe's frame and the relative yaw.

namespace msckf::loop {

// Rotation from yaw, pitch, roll (Z-Y-X, radians) and back.
Eigen::Matrix3d ypr_to_rotation(double yaw, double pitch, double roll);
Eigen::Vector3d rotation_to_ypr(const Eigen::Matrix3d& R);

class PoseGraph {
 public:
  static constexpr int kSequentialEdges = 4;

  // Adds a keyframe: its VIO pose (body in the VIO world frame) and the
  // initial guess of its corrected pose. Returns its index.
  int add_keyframe(const Isometry3d& T_w_b_vio, const Isometry3d& T_w_b_init);

  // Loop edge: keyframe j's pose in keyframe i's frame (position in i's
  // body frame, and yaw of j minus yaw of i).
  void add_loop(int i, int j, const Eigen::Vector3d& t_ij, double yaw_ij);

  // Optimizes all keyframes; the first one is fixed.
  void optimize(int max_iterations = 20);

  int size() const { return static_cast<int>(nodes_.size()); }
  int num_loops() const { return num_loops_; }
  const Isometry3d& vio_pose(int i) const { return nodes_[i].vio; }
  // Corrected pose of keyframe i.
  Isometry3d pose(int i) const;

 private:
  struct Node {
    Isometry3d vio;
    double p[3];
    double yaw[1];
    double pitch, roll;   // from the VIO, not optimized
  };
  struct Edge {
    int i, j;
    Eigen::Vector3d t_ij;
    double yaw_ij;
    bool loop;
  };
  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  int num_loops_ = 0;
};

}  // namespace msckf::loop
