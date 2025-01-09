#ifndef XCGD_MISC_H
#define XCGD_MISC_H

#include <complex>
#include <cstdio>
#include <ctime>
#include <numeric>
#include <string>

inline double freal(double a) { return a; }
inline double freal(std::complex<double> a) { return a.real(); }

// inline double fabs(double a) { return a >= 0 ? a : -a; }
inline double fabs(std::complex<double> a) {
  return a.real() >= 0 ? a.real() : -a.real();
}

template <typename T>
double hard_max(std::vector<T> vals) {
  return *std::max_element(vals.begin(), vals.end());
}

template <typename T>
double hard_min(std::vector<T> vals) {
  return *std::min_element(vals.begin(), vals.end());
}

template <typename T>
double ks_max(std::vector<T> vals, double ksrho = 50.0) {
  double umax = hard_max(vals);
  std::vector<T> eta(vals.size());
  std::transform(vals.begin(), vals.end(), eta.begin(),
                 [umax, ksrho](T x) { return exp(ksrho * (x - umax)); });
  return log(std::accumulate(eta.begin(), eta.end(), 0.0)) / ksrho + umax;
}

template <typename T>
double ks_min(std::vector<T> vals, double ksrho = 50.0) {
  double umin = hard_min(vals);
  std::vector<T> eta(vals.size());
  std::transform(vals.begin(), vals.end(), eta.begin(),
                 [umin, ksrho](T x) { return exp(ksrho * (umin - x)); });
  return umin - log(std::accumulate(eta.begin(), eta.end(), 0.0)) / ksrho;
}

/**
 * @brief Check if a type is a specialization of a template
 *
 * Note: This won't work for non-type template arguments (int, bool, etc.)
 *
 * Usage: bool is_spec = is_specialization<type, template>::value;
 */
template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

/**
 * A horrid hack that could cause compilation to blow up, don't use it unless
 * you know what you're doing
 *
 * Usage:
 * auto f = [](){...;};  // Use a lambda expression to encapsulate the callee
 *                       // function
 * switcher<10>::run(f, some_runtime_value);
 */
template <int Max>
struct switcher {
  template <class Functor>
  static void run(const Functor& f, int rtval) {
    if (rtval < 0) {
      char msg[256];
      std::snprintf(msg, 256,
                    "only positive runtime values are supported, got %d",
                    rtval);
      throw std::runtime_error(msg);
    } else if (rtval > Max) {
      char msg[256];
      std::snprintf(
          msg, 256,
          "runtime value %d exceeds the maximum pre-compiled value %d, if this "
          "is intended, change the Max template argument for the switch in "
          "source code",
          rtval, Max);
      throw std::runtime_error(msg);
    } else if (rtval == Max) {
      f.template operator()<Max>();
    } else {
      switcher<Max - 1>::run(f, rtval);
    }
  }
};

// Prevent the infinite recursion, will never be envoked
template <>
struct switcher<-1> {
  template <class Functor>
  static void run(const Functor& f, int rtval) {}
};

// Get local time in YYYYMMDDHHMMSS
// Note: not thread-safe because std::localtime() is not not thread-safe
inline std::string get_local_time() {
  std::time_t rawtime;
  std::time(&rawtime);

  std::tm* timeinfo = std::localtime(&rawtime);

  char buffer[80];
  std::strftime(buffer, 80, "%Y%m%d%H%M%S", timeinfo);

  return std::string(buffer);
}

template <typename Vec>
void write_vec(const std::string fname, int size, const Vec& vec) {
  std::FILE* fp = std::fopen(fname.c_str(), "w");
  for (int i = 0; i < size; i++) {
    std::fprintf(fp, "%30.20e\n", vec[i]);
  }
  std::fclose(fp);
}

inline void xcgd_assert(bool condition, std::string message) {
  if (not condition) {
    throw std::runtime_error(message);
  }
}

#endif  // XCGD_MISC_H
