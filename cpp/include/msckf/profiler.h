#pragma once

#include <chrono>
#include <map>
#include <string>

// Minimal per-stage wall-clock profiler.
//
//   void track_features() {
//     MSCKF_PROFILE("frontend/track_features");
//     ...
//   }
//
// Each call site gets its own accumulator (a function-local static), so the
// overhead is two clock reads per scope. Times are inclusive: a nested scope
// is also counted in its parent. main_euroc prints the table at the end.

namespace msckf::prof {

struct Stat {
  double total_ms = 0.0;
  long calls = 0;
};

// All stats, sorted by name.
inline std::map<std::string, Stat>& registry() {
  static std::map<std::string, Stat> stats;
  return stats;
}

inline Stat& stat(const char* name) { return registry()[name]; }

class Scope {
 public:
  explicit Scope(Stat& s) : stat_(s), t0_(std::chrono::steady_clock::now()) {}
  ~Scope() {
    stat_.total_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    ++stat_.calls;
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  Stat& stat_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace msckf::prof

#define MSCKF_PROF_CAT2(a, b) a##b
#define MSCKF_PROF_CAT(a, b) MSCKF_PROF_CAT2(a, b)
#define MSCKF_PROFILE(name)                                                              \
  static ::msckf::prof::Stat& MSCKF_PROF_CAT(msckf_prof_stat_, __LINE__) =                \
      ::msckf::prof::stat(name);                                                         \
  ::msckf::prof::Scope MSCKF_PROF_CAT(msckf_prof_scope_, __LINE__)(                       \
      MSCKF_PROF_CAT(msckf_prof_stat_, __LINE__))
