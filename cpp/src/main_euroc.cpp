// Runs the stereo MSCKF on a EuRoC sequence, offline and deterministic.
//
//   main_euroc --path /path/to/MH_01_easy [--config config/euroc.yaml]
//              [--out results/cpp] [--offset 40] [--max-frames N]
//              [--features results/python/features_python.txt] [--dump-features]
//              [--threads N]   (OpenCV worker threads; default = OpenCV's choice)
//              [--fix NAME]... (turn on a Python-quirk fix; "all" for every fix)
//              [--backend msckf|swo]  (filter, or Ceres sliding-window optimizer)
//
// Outputs in --out:
//   traj_cpp.txt     TUM trajectory: t tx ty tz qx qy qz qw (body in world)
//   timing_cpp.csv   per-frame front-end / back-end time [ms]
//
// --dump-features also writes features_cpp.txt (same format as the Python dump)
// so the two front ends can be compared.
// --features replays a feature file written by scripts/run_python_reference.py
// instead of running the C++ front end. That tests the C++ filter in isolation
// (roadmap milestone M4).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core/utility.hpp>

#include "msckf/config.h"
#include "msckf/dataset.h"
#include "msckf/image_processor.h"
#include "msckf/msckf.h"
#include "msckf/profiler.h"
#include "msckf/swo/sliding_window.h"

namespace {

struct Args {
  std::string path;
  std::string config = "config/euroc.yaml";
  std::string out = "results/cpp";
  std::string features;   // optional: replay Python features
  double offset = 40.0;
  long max_frames = -1;
  bool dump_features = false;
  int threads = -1;
  std::vector<std::string> fixes;
  std::string backend = "msckf";
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
      }
      return argv[++i];
    };
    if (key == "--path") a.path = value();
    else if (key == "--config") a.config = value();
    else if (key == "--out") a.out = value();
    else if (key == "--features") a.features = value();
    else if (key == "--offset") a.offset = std::stod(value());
    else if (key == "--max-frames") a.max_frames = std::stol(value());
    else if (key == "--dump-features") a.dump_features = true;
    else if (key == "--threads") a.threads = std::stoi(value());
    else if (key == "--fix") a.fixes.push_back(value());
    else if (key == "--backend") a.backend = value();
    else throw std::runtime_error("unknown argument " + key);
  }
  if (a.path.empty()) {
    throw std::runtime_error("usage: main_euroc --path /path/to/MH_01_easy [options]");
  }
  return a;
}

// Reads the "F <t> <n>" + n lines format of run_python_reference.py.
std::vector<msckf::FeatureMsg> read_feature_file(const std::string& file) {
  std::ifstream in(file);
  if (!in) {
    throw std::runtime_error("cannot open " + file);
  }
  std::vector<msckf::FeatureMsg> msgs;
  std::string tag;
  while (in >> tag) {
    if (tag != "F") {
      throw std::runtime_error("bad feature file near '" + tag + "'");
    }
    msckf::FeatureMsg msg;
    size_t n = 0;
    in >> msg.timestamp >> n;
    msg.features.resize(n);
    for (msckf::FeatureMeasurement& f : msg.features) {
      in >> f.id >> f.u0 >> f.v0 >> f.u1 >> f.v1;
    }
    msgs.push_back(std::move(msg));
  }
  return msgs;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
  return v[std::min(idx, v.size() - 1)];
}

void print_stats(const char* name, const std::vector<double>& ms) {
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / std::max<size_t>(ms.size(), 1);
  std::printf("  %-9s mean %7.3f  p50 %7.3f  p95 %7.3f  max %8.3f  [ms]\n", name, mean,
              percentile(ms, 50), percentile(ms, 95), percentile(ms, 100));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.threads > 0) {
      cv::setNumThreads(args.threads);
    }
    std::printf("OpenCV threads: %d\n", cv::getNumThreads());
    msckf::Config config = msckf::load_config(args.config);
    for (const std::string& fix : args.fixes) {
      msckf::enable_fix(config, fix);
      std::printf("fix enabled: %s\n", fix.c_str());
    }
    const msckf::EuRoCDataset dataset(args.path, args.offset);

    const bool replay = !args.features.empty();
    std::vector<msckf::FeatureMsg> replay_msgs;
    if (replay) {
      replay_msgs = read_feature_file(args.features);
    }

    msckf::ImageProcessor image_processor(config);
    std::unique_ptr<msckf::Estimator> vio;
    msckf::swo::SlidingWindowEstimator* swo = nullptr;
    if (args.backend == "swo") {
      auto estimator = std::make_unique<msckf::swo::SlidingWindowEstimator>(config);
      swo = estimator.get();
      vio = std::move(estimator);
    } else if (args.backend == "msckf") {
      vio = std::make_unique<msckf::MSCKF>(config);
    } else {
      throw std::runtime_error("unknown backend " + args.backend);
    }
    std::printf("backend: %s\n", args.backend.c_str());

    std::filesystem::create_directories(args.out);
    std::ofstream traj(args.out + "/traj_cpp.txt");
    std::ofstream timing(args.out + "/timing_cpp.csv");
    timing << "timestamp,frontend_ms,backend_ms\n";
    std::ofstream feature_dump;
    if (args.dump_features) {
      feature_dump.open(args.out + "/features_cpp.txt");
    }
    char line[256];

    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };

    const std::vector<msckf::ImuMsg>& imu = dataset.imu();
    size_t imu_idx = 0;
    const size_t n_frames = replay ? replay_msgs.size() : dataset.num_stereo();
    std::vector<double> frontend_ms, backend_ms, total_ms;
    double first_t = 0.0, last_t = 0.0;
    long processed = 0;

    const auto wall_start = Clock::now();
    for (size_t k = 0; k < n_frames; ++k) {
      const double t_img = replay ? replay_msgs[k].timestamp : dataset.stereo_timestamp(k);

      // Feed every IMU message up to (and including) this frame's timestamp.
      while (imu_idx < imu.size() && imu[imu_idx].timestamp <= t_img) {
        image_processor.imu_callback(imu[imu_idx]);
        vio->imu_callback(imu[imu_idx]);
        ++imu_idx;
      }

      msckf::FeatureMsg feature_msg;
      double fe_ms = 0.0;
      if (replay) {
        feature_msg = replay_msgs[k];
      } else {
        const msckf::StereoMsg stereo = dataset.load_stereo(k);   // disk I/O, not timed
        const auto t0 = Clock::now();
        feature_msg = image_processor.stereo_callback(stereo);
        fe_ms = ms_since(t0, Clock::now());
      }

      if (feature_dump.is_open()) {
        std::snprintf(line, sizeof(line), "F %.9f %zu\n", feature_msg.timestamp,
                      feature_msg.features.size());
        feature_dump << line;
        for (const msckf::FeatureMeasurement& f : feature_msg.features) {
          std::snprintf(line, sizeof(line), "%lld %.17g %.17g %.17g %.17g\n",
                        static_cast<long long>(f.id), f.u0, f.v0, f.u1, f.v1);
          feature_dump << line;
        }
      }

      const auto t1 = Clock::now();
      const std::optional<msckf::VioResult> result = vio->feature_callback(feature_msg);
      const double be_ms = ms_since(t1, Clock::now());

      std::snprintf(line, sizeof(line), "%.9f,%.4f,%.4f\n", t_img, fe_ms, be_ms);
      timing << line;
      frontend_ms.push_back(fe_ms);
      backend_ms.push_back(be_ms);
      total_ms.push_back(fe_ms + be_ms);

      if (result) {
        const Eigen::Quaterniond q(result->pose.R);   // Hamilton, as TUM expects
        const Eigen::Vector3d& p = result->pose.t;
        std::snprintf(line, sizeof(line), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n",
                      result->timestamp, p(0), p(1), p(2), q.x(), q.y(), q.z(), q.w());
        traj << line;
      }

      if (processed == 0) first_t = t_img;
      last_t = t_img;
      ++processed;
      if (args.max_frames > 0 && processed >= args.max_frames) {
        break;
      }
    }
    const double wall_s = std::chrono::duration<double>(Clock::now() - wall_start).count();
    const double compute_s = std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / 1e3;
    const double data_s = last_t - first_t;

    std::printf("frames: %ld  dataset: %.2f s  wall (incl. image loading): %.2f s\n",
                processed, data_s, wall_s);
    std::printf("real-time factor (dataset s / compute s): %.2f\n", data_s / compute_s);
    std::printf("latency per frame%s:\n", replay ? " (back end only, replayed features)" : "");
    if (!replay) print_stats("frontend", frontend_ms);
    print_stats("backend", backend_ms);
    print_stats("total", total_ms);

    if (swo != nullptr) {
      const auto& st = swo->stats();
      std::printf("sliding window: %ld frames, %ld keyframes, %ld marginalizations, "
                  "%.1f landmarks per solve, %ld outlier landmarks removed, %ld resets\n",
                  st.frames, st.keyframes, st.marginalizations,
                  st.keyframes > 1 ? double(st.landmarks_optimized) / (st.keyframes - 1) : 0.0,
                  st.outliers_removed, st.resets);
    }

    // Per-stage breakdown (inclusive times; "*" = called from several stages).
    std::ofstream stages(args.out + "/profile_stages.csv");
    stages << "stage,total_ms,calls,ms_per_frame\n";
    std::printf("stage breakdown (ms per frame, inclusive):\n");
    for (const auto& [name, st] : msckf::prof::registry()) {
      const double per_frame = st.total_ms / std::max<long>(processed, 1);
      std::printf("  %-48s %8.3f  (%ld calls)\n", name.c_str(), per_frame, st.calls);
      std::snprintf(line, sizeof(line), "%s,%.3f,%ld,%.4f\n", name.c_str(), st.total_ms,
                    st.calls, per_frame);
      stages << line;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
