#include "msckf/dataset.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace msckf {

namespace {

// EuRoC stores time as integer nanoseconds. Parse as int64 first, then convert
// once: this gives the same double as Python's float(ns) * 1e-9.
double ns_to_seconds(const std::string& ns) {
  return static_cast<double>(std::stoll(ns)) * 1e-9;
}

std::vector<ImuMsg> read_imu_csv(const std::string& file) {
  std::ifstream in(file);
  if (!in) {
    throw std::runtime_error("cannot open " + file);
  }
  std::vector<ImuMsg> msgs;
  std::string line;
  std::getline(in, line);   // header
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(ss, field, ',')) {
      fields.push_back(field);
    }
    if (fields.size() < 7) {
      continue;
    }
    ImuMsg m;
    m.timestamp = ns_to_seconds(fields[0]);
    m.angular_velocity << std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3]);
    m.linear_acceleration << std::stod(fields[4]), std::stod(fields[5]), std::stod(fields[6]);
    msgs.push_back(m);
  }
  return msgs;
}

// Returns (timestamp_ns, file path) for every .png, sorted by time.
std::vector<std::pair<long long, std::string>> list_images(const std::string& dir) {
  std::vector<std::pair<long long, std::string>> images;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".png") {
      images.emplace_back(std::stoll(entry.path().stem().string()), entry.path().string());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

}  // namespace

EuRoCDataset::EuRoCDataset(const std::string& path, double start_offset) {
  const std::string mav0 = (fs::path(path) / "mav0").string();
  std::vector<ImuMsg> all_imu = read_imu_csv(mav0 + "/imu0/data.csv");
  if (all_imu.empty()) {
    throw std::runtime_error("no IMU data in " + mav0);
  }

  // dataset.py: starttime = max(imu.start_time(), stereo.start_time()), but
  // stereo.start_time() is still -inf at that point, so the reference is the
  // first IMU sample. Messages strictly before the start are skipped.
  const double start = all_imu.front().timestamp + start_offset;
  for (const ImuMsg& m : all_imu) {
    if (m.timestamp >= start) {
      imu_.push_back(m);
    }
  }

  // Pair the two cameras by timestamp. Some sequences drop frames in one
  // camera (MH_04: 1, V1_02: 1, V2_03: 416), so pairing by index would shift
  // every later pair; frames without a partner within 5 ms are skipped.
  const auto cam0 = list_images(mav0 + "/cam0/data");
  const auto cam1 = list_images(mav0 + "/cam1/data");
  constexpr long long kMaxPairOffsetNs = 5'000'000;
  size_t j = 0;
  for (size_t i = 0; i < cam0.size(); ++i) {
    while (j < cam1.size() && cam1[j].first < cam0[i].first - kMaxPairOffsetNs) {
      ++j;
    }
    if (j == cam1.size()) {
      break;
    }
    if (std::llabs(cam1[j].first - cam0[i].first) > kMaxPairOffsetNs) {
      continue;   // no partner for this cam0 frame
    }
    const double t0 = static_cast<double>(cam0[i].first) * 1e-9;
    if (t0 >= start) {
      cam_timestamps_.push_back(t0);
      cam0_files_.push_back(cam0[i].second);
      cam1_files_.push_back(cam1[j].second);
    }
    ++j;
  }
}

StereoMsg EuRoCDataset::load_stereo(size_t i) const {
  StereoMsg msg;
  msg.timestamp = cam_timestamps_[i];
  msg.cam0_image = cv::imread(cam0_files_[i], cv::IMREAD_UNCHANGED);
  msg.cam1_image = cv::imread(cam1_files_[i], cv::IMREAD_UNCHANGED);
  if (msg.cam0_image.empty() || msg.cam1_image.empty()) {
    throw std::runtime_error("cannot read stereo pair " + cam0_files_[i]);
  }
  // TUM-VI's "_16" exports are 16-bit; the front end works on 8-bit images.
  for (cv::Mat* img : {&msg.cam0_image, &msg.cam1_image}) {
    if (img->depth() == CV_16U) {
      img->convertTo(*img, CV_8U, 1.0 / 256.0);
    }
  }
  return msg;
}

}  // namespace msckf
