#pragma once

#include <string>
#include <vector>

#include "msckf/messages.h"

// EuRoC MAV reader (stereo + IMU). C++ port of EuRoCDataset in Code/dataset.py,
// without the threaded real-time publisher: main_euroc replays the data in
// timestamp order instead, which is deterministic and easy to time.

namespace msckf {

class EuRoCDataset {
 public:
  // path: sequence folder that contains mav0/.
  // start_offset: seconds to skip, counted from the first IMU sample
  //               (dataset.py's set_starttime(offset)).
  EuRoCDataset(const std::string& path, double start_offset);

  const std::vector<ImuMsg>& imu() const { return imu_; }

  size_t num_stereo() const { return cam0_files_.size(); }
  double stereo_timestamp(size_t i) const { return cam_timestamps_[i]; }
  StereoMsg load_stereo(size_t i) const;   // reads both images from disk

 private:
  std::vector<ImuMsg> imu_;
  std::vector<std::string> cam0_files_;
  std::vector<std::string> cam1_files_;
  std::vector<double> cam_timestamps_;
};

}  // namespace msckf
