#ifndef XCGD_MISC_H
#define XCGD_MISC_H

#include <complex>

inline double freal(double a) { return a; }
inline double freal(std::complex<double> a) { return a.real(); }

// inline double fabs(double a) { return a >= 0 ? a : -a; }
inline double fabs(std::complex<double> a) {
  return a.real() >= 0 ? a.real() : -a.real();
}

#endif  // XCGD_MISC_H