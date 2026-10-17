// Unit tests of the loop-closure pose graph (4-DoF: x, y, z, yaw).

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "msckf/loop/pose_graph.h"

using namespace msckf;
using namespace msckf::loop;

TEST(Loop, YawPitchRollRoundTrip) {
  for (const Eigen::Vector3d ypr : {Eigen::Vector3d(0.3, -0.2, 0.1), Eigen::Vector3d(-2.5, 0.4, -0.7),
                                    Eigen::Vector3d(3.0, 0.0, 0.0)}) {
    const Eigen::Matrix3d R = ypr_to_rotation(ypr(0), ypr(1), ypr(2));
    EXPECT_NEAR((R * R.transpose() - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
    EXPECT_NEAR((rotation_to_ypr(R) - ypr).norm(), 0.0, 1e-12);
  }
}

TEST(Loop, OneLoopEdgeRemovesDrift) {
  // Keyframes on a circle of radius 2 m (one lap, back to the start) with a
  // little roll and pitch. The VIO poses drift in yaw and position, about as
  // much as the VIO does between loops on EuRoC (1.7 deg, ~0.2 m); one loop
  // edge from the last keyframe to the first, with the true relative pose,
  // must pull the end of the chain most of the way back (one loop does not
  // remove all of it: the drifted VIO edges still carry information). Much
  // larger drift is not closed by one loop: the robust loss treats it as an
  // outlier.
  constexpr int kN = 60;
  std::vector<Isometry3d> truth, vio;
  for (int k = 0; k < kN; ++k) {
    const double a = 2.0 * M_PI * k / (kN - 1);
    const Eigen::Vector3d p(2.0 * std::sin(a), 2.0 * (1.0 - std::cos(a)), 0.1 * std::sin(2 * a));
    truth.emplace_back(ypr_to_rotation(a, 0.05, -0.03), p);
    // Drift grows along the chain: 0.03 deg yaw and 3 mm per keyframe.
    const double drift_yaw = 0.0005 * k;
    const Eigen::Matrix3d R_drift =
        Eigen::AngleAxisd(drift_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    vio.emplace_back(R_drift * truth.back().R,
                     R_drift * truth.back().t + Eigen::Vector3d(0.003 * k, -0.0015 * k, 0.0));
  }

  PoseGraph graph;
  for (int k = 0; k < kN; ++k) {
    graph.add_keyframe(vio[k], vio[k]);
  }
  const Isometry3d& a = truth[0];
  const Isometry3d& b = truth[kN - 1];
  graph.add_loop(0, kN - 1, a.R.transpose() * (b.t - a.t),
                 rotation_to_ypr(b.R)(0) - rotation_to_ypr(a.R)(0));
  graph.optimize(50);

  const double err_vio = (vio[kN - 1].t - truth[kN - 1].t).norm();
  const double err_opt = (graph.pose(kN - 1).t - truth[kN - 1].t).norm();
  EXPECT_GT(err_vio, 0.15);          // the drift is substantial ...
  EXPECT_LT(err_opt, 0.6 * err_vio);   // ... and the loop removes much of it
  const double yaw_err = std::abs(rotation_to_ypr(graph.pose(kN - 1).R.transpose() *
                                                  truth[kN - 1].R)(0));
  EXPECT_LT(yaw_err, 0.4 * 0.0005 * (kN - 1));
  // The first keyframe is fixed.
  EXPECT_NEAR((graph.pose(0).t - vio[0].t).norm(), 0.0, 1e-12);
}
