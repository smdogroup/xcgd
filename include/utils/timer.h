#ifndef XCGD_UTILS_H
#define XCGD_UTILS_H

#include <chrono>

class StopWatch {
 public:
  StopWatch() { t_start = std::chrono::steady_clock::now(); }
  double lap() {
    auto now = std::chrono::steady_clock::now();
    double t_elapse =
        1e-9 *
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_start)
            .count();  // in s
    return t_elapse;
  }
  void reset_start() { t_start = std::chrono::steady_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::steady_clock> t_start;
};

#endif  // XCGD_UTILS_H
