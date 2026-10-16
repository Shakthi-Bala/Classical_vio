// ROS 2 node around the C++ VIO (MSCKF or sliding-window back end).
//
// Subscribes:  imu0 (sensor_msgs/Imu), cam0/image_raw, cam1/image_raw (mono8)
// Publishes:   odometry (nav_msgs/Odometry), path (nav_msgs/Path), TF world -> imu
//
// Message order. DDS delivers IMU and image messages on separate topics, so
// the order in which they arrive is not the order of their timestamps. The
// node therefore queues stereo pairs and processes a pair only after every IMU
// sample up to its timestamp has been handed to the estimator, which is
// exactly the order main_euroc uses offline. The result does not depend on
// transport timing.
//
// Parameters:  config (YAML, same as main_euroc), backend (msckf | swo),
//              output_dir (TUM trajectory + timing CSV), opencv_threads,
//              skip_seconds (ignore data before first IMU stamp + this).

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "msckf/config.h"
#include "msckf/image_processor.h"
#include "msckf/msckf.h"
#include "msckf/swo/sliding_window.h"

namespace {

using Clock = std::chrono::steady_clock;

double to_seconds(const builtin_interfaces::msg::Time& t) {
  return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
}
int64_t to_ns(const builtin_interfaces::msg::Time& t) {
  return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
}

class VioNode : public rclcpp::Node {
 public:
  VioNode() : Node("msckf_vio") {
    const std::string config_path = declare_parameter<std::string>("config", "");
    const std::string backend = declare_parameter<std::string>("backend", "msckf");
    output_dir_ = declare_parameter<std::string>("output_dir", "");
    skip_seconds_ = declare_parameter<double>("skip_seconds", 0.0);
    cv::setNumThreads(static_cast<int>(declare_parameter<int>("opencv_threads", 1)));
    if (config_path.empty()) {
      throw std::runtime_error("parameter 'config' (YAML file) is required");
    }

    config_ = msckf::load_config(config_path);
    image_processor_ = std::make_unique<msckf::ImageProcessor>(config_);
    if (backend == "swo") {
      estimator_ = std::make_unique<msckf::swo::SlidingWindowEstimator>(config_);
    } else if (backend == "msckf") {
      estimator_ = std::make_unique<msckf::MSCKF>(config_);
    } else {
      throw std::runtime_error("unknown backend '" + backend + "'");
    }

    if (!output_dir_.empty()) {
      std::filesystem::create_directories(output_dir_);
      traj_.open(output_dir_ + "/traj_ros.txt");
      timing_.open(output_dir_ + "/timing_ros.csv");
      timing_ << "timestamp,process_ms,end_to_end_ms,queue_wait_ms\n";
    }

    // Reliable and deep, so bag playback never drops a message.
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable();
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "imu0", qos, [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { on_imu(*m); });
    cam0_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "cam0/image_raw", qos,
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(*m, 0); });
    cam1_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "cam1/image_raw", qos,
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(*m, 1); });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 100);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    path_.header.frame_id = "world";

    RCLCPP_INFO(get_logger(), "backend %s, config %s", backend.c_str(), config_path.c_str());
  }

  ~VioNode() override {
    if (process_ms_.empty()) {
      return;
    }
    auto stats = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
      return std::array<double, 3>{mean, v[v.size() / 2], v[static_cast<size_t>(0.95 * (v.size() - 1))]};
    };
    const auto p = stats(process_ms_), e = stats(end_to_end_ms_);
    std::printf("[msckf_vio] frames %zu | process mean %.2f p50 %.2f p95 %.2f ms | "
                "end-to-end mean %.2f p50 %.2f p95 %.2f ms\n",
                process_ms_.size(), p[0], p[1], p[2], e[0], e[1], e[2]);
  }

 private:
  struct PendingImage {
    cv::Mat image;
    Clock::time_point arrival;
  };
  struct StereoPair {
    int64_t stamp_ns;
    double stamp;
    cv::Mat cam0, cam1;
    Clock::time_point arrival;   // when the later of the two images arrived
  };

  bool skipped(double stamp) {
    if (!have_first_imu_) {
      return true;   // wait for IMU to define the start time
    }
    return stamp < first_imu_stamp_ + skip_seconds_;
  }

  void on_imu(const sensor_msgs::msg::Imu& m) {
    const double t = to_seconds(m.header.stamp);
    if (!have_first_imu_) {
      have_first_imu_ = true;
      first_imu_stamp_ = t;
    }
    if (skipped(t)) {
      return;
    }
    // Offline order: every IMU sample with stamp <= t_image comes before the image.
    process_ready([t](double t_img) { return t_img < t; });
    msckf::ImuMsg imu;
    imu.timestamp = t;
    imu.angular_velocity << m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z;
    imu.linear_acceleration << m.linear_acceleration.x, m.linear_acceleration.y,
        m.linear_acceleration.z;
    image_processor_->imu_callback(imu);
    estimator_->imu_callback(imu);
    last_imu_stamp_ = t;
    process_ready([t](double t_img) { return t_img <= t; });
  }

  void on_image(const sensor_msgs::msg::Image& m, int cam) {
    if (m.encoding != "mono8") {
      RCLCPP_ERROR_ONCE(get_logger(), "expected mono8 images, got %s", m.encoding.c_str());
      return;
    }
    if (skipped(to_seconds(m.header.stamp))) {
      return;
    }
    // Copy into an owned cv::Mat (no cv_bridge: avoids an OpenCV version clash).
    const cv::Mat view(static_cast<int>(m.height), static_cast<int>(m.width), CV_8UC1,
                       const_cast<uint8_t*>(m.data.data()), m.step);
    PendingImage img{view.clone(), Clock::now()};
    const int64_t ns = to_ns(m.header.stamp);
    auto& mine = cam == 0 ? cam0_ : cam1_;
    auto& other = cam == 0 ? cam1_ : cam0_;
    // Pair with the other camera within 5 ms (as in EuRoCDataset).
    constexpr int64_t kTol = 5'000'000;
    for (auto it = other.lower_bound(ns - kTol); it != other.end() && it->first <= ns + kTol;
         ++it) {
      StereoPair pair;
      pair.stamp_ns = cam == 0 ? ns : it->first;
      pair.stamp = 1e-9 * static_cast<double>(pair.stamp_ns);
      pair.cam0 = cam == 0 ? img.image : it->second.image;
      pair.cam1 = cam == 0 ? it->second.image : img.image;
      pair.arrival = std::max(img.arrival, it->second.arrival);
      // Anything older than this pair can never be matched any more.
      other.erase(other.begin(), std::next(it));
      mine.erase(mine.begin(), mine.lower_bound(ns));
      ready_.push_back(std::move(pair));
      std::sort(ready_.begin(), ready_.end(),
                [](const StereoPair& a, const StereoPair& b) { return a.stamp_ns < b.stamp_ns; });
      process_ready([this](double t_img) { return t_img <= last_imu_stamp_; });
      return;
    }
    mine.emplace(ns, std::move(img));
  }

  template <typename Ready>
  void process_ready(Ready is_ready) {
    while (!ready_.empty() && is_ready(ready_.front().stamp)) {
      StereoPair pair = std::move(ready_.front());
      ready_.pop_front();
      process(pair);
    }
  }

  void process(const StereoPair& pair) {
    const auto t0 = Clock::now();
    msckf::StereoMsg stereo;
    stereo.timestamp = pair.stamp;
    stereo.cam0_image = pair.cam0;
    stereo.cam1_image = pair.cam1;
    const msckf::FeatureMsg features = image_processor_->stereo_callback(stereo);
    const std::optional<msckf::VioResult> result = estimator_->feature_callback(features);
    if (result) {
      publish(*result, pair.stamp_ns);
    }
    const auto t1 = Clock::now();

    const double process_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double e2e_ms = std::chrono::duration<double, std::milli>(t1 - pair.arrival).count();
    process_ms_.push_back(process_ms);
    end_to_end_ms_.push_back(e2e_ms);
    if (timing_.is_open()) {
      char line[128];
      std::snprintf(line, sizeof(line), "%.9f,%.4f,%.4f,%.4f\n", pair.stamp, process_ms, e2e_ms,
                    e2e_ms - process_ms);
      timing_ << line;
    }
  }

  void publish(const msckf::VioResult& r, int64_t stamp_ns) {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
    const Eigen::Quaterniond q(r.pose.R);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.child_frame_id = "imu";
    odom.pose.pose.position.x = r.pose.t.x();
    odom.pose.pose.position.y = r.pose.t.y();
    odom.pose.pose.position.z = r.pose.t.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.twist.twist.linear.x = r.velocity.x();   // world frame
    odom.twist.twist.linear.y = r.velocity.y();
    odom.twist.twist.linear.z = r.velocity.z();
    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = odom.header;
    tf.child_frame_id = "imu";
    tf.transform.translation.x = r.pose.t.x();
    tf.transform.translation.y = r.pose.t.y();
    tf.transform.translation.z = r.pose.t.z();
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_->sendTransform(tf);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = odom.header;
    ps.pose = odom.pose.pose;
    path_.poses.push_back(ps);
    path_.header.stamp = stamp;
    if (path_.poses.size() % 10 == 0) {
      path_pub_->publish(path_);
    }

    if (traj_.is_open()) {
      char line[256];
      std::snprintf(line, sizeof(line), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n",
                    1e-9 * static_cast<double>(stamp_ns), r.pose.t.x(), r.pose.t.y(),
                    r.pose.t.z(), q.x(), q.y(), q.z(), q.w());
      traj_ << line;
      traj_.flush();
    }
  }

  msckf::Config config_;
  std::unique_ptr<msckf::ImageProcessor> image_processor_;
  std::unique_ptr<msckf::Estimator> estimator_;

  bool have_first_imu_ = false;
  double first_imu_stamp_ = 0.0;
  double last_imu_stamp_ = -1.0;
  double skip_seconds_ = 0.0;
  std::map<int64_t, PendingImage> cam0_, cam1_;
  std::deque<StereoPair> ready_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam0_sub_, cam1_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  nav_msgs::msg::Path path_;

  std::string output_dir_;
  std::ofstream traj_, timing_;
  std::vector<double> process_ms_, end_to_end_ms_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<VioNode>();
    rclcpp::spin(node);   // single-threaded executor: callbacks never overlap
  }
  rclcpp::shutdown();
  return 0;
}
