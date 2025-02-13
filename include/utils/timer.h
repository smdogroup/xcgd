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

  std::string format_time(double t) {
    int h = t / 3600.0;
    t -= h * 3600.0;
    int m = t / 60.0;
    t -= m * 60.0;
    char msg[256];
    // std::snprintf(msg, 256, "%02d:%02d:%04.1f", h, m, t);
    std::snprintf(msg, 256, "%02d:%02d:%02d", h, m, int(t));
    return msg;
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> t_start;
};

#endif  // XCGD_UTILS_H
