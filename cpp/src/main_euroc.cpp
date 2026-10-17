// Runs a stereo VIO back end (MSCKF or sliding window) on a EuRoC sequence,
// offline and deterministic.
//
//   main_euroc --path /path/to/MH_01_easy [--config config/euroc.yaml]
//              [--out results/cpp] [--offset 40] [--max-frames N]
//              [--features results/python/features_python.txt] [--dump-features]
//              [--threads N]   (OpenCV worker threads; default = OpenCV's choice)
//              [--fix NAME]... (turn on a Python-quirk fix; "all" for every fix)
//              [--backend msckf|swo]  (filter, or Ceres sliding-window optimizer)
//              [--loop]  (loop closure on top of the back end)
//
// Outputs in --out:
//   traj_cpp.txt        TUM trajectory: t tx ty tz qx qy qz qw (body in world)
//   timing_cpp.csv      per-frame front-end / back-end time [ms]
//   profile_stages.csv  per-stage time (MSCKF_PROFILE scopes)
//
// --loop also writes traj_loop.txt (each pose corrected with the loops found
// so far), traj_loop_final.txt (all poses corrected with the final pose graph)
// and loop_timing.csv.
// --dump-features also writes features_cpp.txt (same format as the Python dump)
// so the two front ends can be compared.
// --features replays a feature file written by scripts/run_python_reference.py
// instead of running the C++ front end, to test the back end in isolation.

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
#include "msckf/loop/loop_closure.h"
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
  bool loop = false;
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
    else if (key == "--loop") a.loop = true;
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
      // cam1 of a cam0-only feature is written as "nan", which >> rejects.
      std::string u1, v1;
      in >> f.id >> f.u0 >> f.v0 >> u1 >> v1;
      f.u1 = std::stod(u1);
      f.v1 = std::stod(v1);
    }
    msgs.push_back(std::move(msg));
  }
  return msgs;
}

// One TUM line: t tx ty tz qx qy qz qw (Hamilton quaternion).
void write_pose(std::ofstream& out, double t, const msckf::Isometry3d& pose) {
  const Eigen::Quaterniond q(pose.R);
  char line[256];
  std::snprintf(line, sizeof(line), "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", t, pose.t(0),
                pose.t(1), pose.t(2), q.x(), q.y(), q.z(), q.w());
  out << line;
}

void print_stats(const char* name, const std::vector<double>& ms) {
  using msckf::prof::percentile;
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
    std::unique_ptr<msckf::loop::LoopClosure> loop;
    if (args.loop) {
      if (replay) {
        throw std::runtime_error("--loop needs images; it cannot be used with --features");
      }
      loop = std::make_unique<msckf::loop::LoopClosure>(config);
    }

    std::filesystem::create_directories(args.out);
    std::ofstream traj(args.out + "/traj_cpp.txt");
    std::ofstream timing(args.out + "/timing_cpp.csv");
    timing << "timestamp,frontend_ms,backend_ms\n";
    std::ofstream feature_dump;
    if (args.dump_features) {
      feature_dump.open(args.out + "/features_cpp.txt");
    }
    std::ofstream traj_loop, loop_timing;
    if (loop) {
      traj_loop.open(args.out + "/traj_loop.txt");
      loop_timing.open(args.out + "/loop_timing.csv");
      loop_timing << "timestamp,loop_ms\n";
    }
    std::vector<double> loop_ms;
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
      msckf::StereoMsg stereo;
      double fe_ms = 0.0;
      if (replay) {
        feature_msg = replay_msgs[k];
      } else {
        stereo = dataset.load_stereo(k);   // disk I/O, not timed
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
        write_pose(traj, result->timestamp, result->pose);
        if (loop) {
          const auto t2 = Clock::now();
          const msckf::Isometry3d corrected =
              loop->process(result->timestamp, result->pose, stereo.cam0_image, stereo.cam1_image);
          loop_ms.push_back(ms_since(t2, Clock::now()));
          std::snprintf(line, sizeof(line), "%.9f,%.4f\n", t_img, loop_ms.back());
          loop_timing << line;
          write_pose(traj_loop, result->timestamp, corrected);
        }
      }

      if (processed == 0) {
        first_t = t_img;
      }
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
    if (!replay) {
      print_stats("frontend", frontend_ms);
    }
    print_stats("backend", backend_ms);
    print_stats("total", total_ms);

    if (loop) {
      std::ofstream traj_final(args.out + "/traj_loop_final.txt");
      for (const auto& [t, pose] : loop->final_trajectory()) {
        write_pose(traj_final, t, pose);
      }
      std::ofstream loops_csv(args.out + "/loops.csv");
      loops_csv << "t_candidate,t_current,tx,ty,tz,yaw,inliers\n";
      for (const auto& l : loop->loops()) {
        std::snprintf(line, sizeof(line), "%.9f,%.9f,%.6f,%.6f,%.6f,%.6f,%d\n", l.t_candidate,
                      l.t_current, l.t_ij(0), l.t_ij(1), l.t_ij(2), l.yaw_ij, l.inliers);
        loops_csv << line;
      }
      const auto& st = loop->stats();
      std::printf("loop closure: %ld keyframes, %ld candidates checked, %ld loops\n",
                  st.keyframes, st.candidates, st.loops);
      print_stats("loop", loop_ms);
    }

    if (swo != nullptr) {
      const auto& st = swo->stats();
      std::printf("sliding window: %ld frames, %ld keyframes, %ld marginalizations, "
                  "%.1f landmarks per solve, %ld outlier landmarks removed\n",
                  st.frames, st.keyframes, st.marginalizations,
                  st.keyframes > 1
                      ? static_cast<double>(st.landmarks_optimized) / (st.keyframes - 1)
                      : 0.0,
                  st.outliers_removed);
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
