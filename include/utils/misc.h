#ifndef XCGD_MISC_H
#define XCGD_MISC_H

#include <complex>
#include <ctime>
#include <string>

inline double freal(double a) { return a; }
inline double freal(std::complex<double> a) { return a.real(); }

// inline double fabs(double a) { return a >= 0 ? a : -a; }
inline double fabs(std::complex<double> a) {
  return a.real() >= 0 ? a.real() : -a.real();
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

// Get local time in YYYYMMDDHHMMSS
// Note: not thread-safe because std::localtime() is not not thread-safe
std::string get_local_time() {
  std::time_t rawtime;
  std::time(&rawtime);

  std::tm* timeinfo = std::localtime(&rawtime);

  char buffer[80];
  std::strftime(buffer, 80, "%Y%m%d%H%M%S", timeinfo);

  return std::string(buffer);
}

#endif  // XCGD_MISC_H