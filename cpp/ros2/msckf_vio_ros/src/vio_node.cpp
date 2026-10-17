// ROS 2 node around the C++ VIO (MSCKF or sliding-window back end).
//
// Subscribes:  imu0 (sensor_msgs/Imu), cam0/image_raw, cam1/image_raw (mono8)
// Publishes:   odometry (nav_msgs/Odometry; twist in the body frame, covariances
//              not filled), path (nav_msgs/Path), TF world_frame -> body_frame
//
// Message order. DDS delivers IMU and image messages on separate topics, so
// the order in which they arrive is not the order of their timestamps. The
// node therefore queues stereo pairs and processes a pair only after every IMU
// sample up to its timestamp has been handed to the estimator, which is
// exactly the order main_euroc uses offline.
//
// Parameters:  config (YAML, same as main_euroc), backend (msckf | swo),
//              output_dir (TUM trajectory + timing CSV), opencv_threads,
//              skip_seconds (ignore data before first IMU stamp + this),
//              world_frame, body_frame, reliable_qos (false for drivers that
//              publish best effort, e.g. most camera drivers).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <builtin_interfaces/msg/time.hpp>
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
#include "msckf/profiler.h"
#include "msckf/swo/sliding_window.h"

namespace {

using Clock = std::chrono::steady_clock;

// Stereo images within this much of each other form a pair (as in EuRoCDataset).
constexpr int64_t kStereoPairToleranceNs = 5'000'000;
// The full path is re-sent on every publish, so only every Nth pose triggers one.
constexpr size_t kPathPublishEvery = 10;

int64_t to_ns(const builtin_interfaces::msg::Time& t) {
  return static_cast<int64_t>(t.sec) * 1'000'000'000LL + t.nanosec;
}
// Same conversion as the offline reader (dataset.cpp): sec + 1e-9 * nanosec
// rounds differently in the last bit, and the filter amplifies that.
double to_seconds(const builtin_interfaces::msg::Time& t) {
  return static_cast<double>(to_ns(t)) * 1e-9;
}

class VioNode : public rclcpp::Node {
 public:
  VioNode() : Node("msckf_vio") {
    const auto config_path = declare_parameter<std::string>("config", "");
    const auto backend = declare_parameter<std::string>("backend", "msckf");
    const auto output_dir = declare_parameter<std::string>("output_dir", "");
    const bool reliable = declare_parameter<bool>("reliable_qos", true);
    skip_seconds_ = declare_parameter<double>("skip_seconds", 0.0);
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    body_frame_ = declare_parameter<std::string>("body_frame", "imu");
    cv::setNumThreads(declare_parameter<int>("opencv_threads", 1));
    if (config_path.empty()) {
      throw std::runtime_error("parameter 'config' (YAML file) is required");
    }

    const msckf::Config config = msckf::load_config(config_path);
    image_processor_ = std::make_unique<msckf::ImageProcessor>(config);
    if (backend == "swo") {
      estimator_ = std::make_unique<msckf::swo::SlidingWindowEstimator>(config);
    } else if (backend == "msckf") {
      estimator_ = std::make_unique<msckf::MSCKF>(config);
    } else {
      throw std::runtime_error("unknown backend '" + backend + "'");
    }

    if (!output_dir.empty()) {
      std::filesystem::create_directories(output_dir);
      traj_.open(output_dir + "/traj_ros.txt");
      timing_.open(output_dir + "/timing_ros.csv");
      timing_ << "timestamp,process_ms,end_to_end_ms,queue_wait_ms\n";
    }

    // Deep queues so bag playback never drops a message: 10 s of IMU, 5 s of images.
    auto qos = [reliable](size_t depth) {
      rclcpp::QoS q{rclcpp::KeepLast(depth)};
      return reliable ? q.reliable() : q.best_effort();
    };
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "imu0", qos(2000), [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { on_imu(*m); });
    cam0_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "cam0/image_raw", qos(100),
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(*m, 0); });
    cam1_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "cam1/image_raw", qos(100),
        [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(*m, 1); });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 100);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    path_.header.frame_id = world_frame_;

    RCLCPP_INFO(get_logger(), "backend %s, config %s", backend.c_str(), config_path.c_str());
  }

  ~VioNode() override {
    if (process_ms_.empty()) {
      return;
    }
    auto mean = [](const std::vector<double>& v) {
      return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    using msckf::prof::percentile;
    std::printf("[msckf_vio] frames %zu | process mean %.2f p50 %.2f p95 %.2f ms | "
                "end-to-end mean %.2f p50 %.2f p95 %.2f ms\n",
                process_ms_.size(), mean(process_ms_), percentile(process_ms_, 50),
                percentile(process_ms_, 95), mean(end_to_end_ms_),
                percentile(end_to_end_ms_, 50), percentile(end_to_end_ms_, 95));
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

  bool skipped(double stamp) const {
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
    const auto it = other.lower_bound(ns - kStereoPairToleranceNs);
    if (it == other.end() || it->first > ns + kStereoPairToleranceNs) {
      mine.emplace(ns, std::move(img));
      return;
    }
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
      timing_.flush();   // survives a hard kill at the end of a bag
    }
  }

  void publish(const msckf::VioResult& r, int64_t stamp_ns) {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(stamp_ns / 1'000'000'000LL);
    stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1'000'000'000LL);
    const Eigen::Quaterniond q(r.pose.R);
    // nav_msgs/Odometry: pose in header.frame_id, twist in child_frame_id.
    const Eigen::Vector3d v_body = r.pose.R.transpose() * r.velocity;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = world_frame_;
    odom.child_frame_id = body_frame_;
    odom.pose.pose.position.x = r.pose.t.x();
    odom.pose.pose.position.y = r.pose.t.y();
    odom.pose.pose.position.z = r.pose.t.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.twist.twist.linear.x = v_body.x();
    odom.twist.twist.linear.y = v_body.y();
    odom.twist.twist.linear.z = v_body.z();
    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = odom.header;
    tf.child_frame_id = body_frame_;
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
    if (path_.poses.size() % kPathPublishEvery == 0) {
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

  std::unique_ptr<msckf::ImageProcessor> image_processor_;
  std::unique_ptr<msckf::Estimator> estimator_;

  bool have_first_imu_ = false;
  double first_imu_stamp_ = 0.0;
  double last_imu_stamp_ = -1.0;
  double skip_seconds_ = 0.0;
  std::string world_frame_, body_frame_;
  std::map<int64_t, PendingImage> cam0_, cam1_;
  std::deque<StereoPair> ready_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cam0_sub_, cam1_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  nav_msgs::msg::Path path_;

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
