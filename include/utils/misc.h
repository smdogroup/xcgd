#ifndef XCGD_MISC_H
#define XCGD_MISC_H

#include <complex>

inline double freal(double a) { return a; }
inline double freal(std::complex<double> a) { return a.real(); }

#endif  // XCGD_MISC_H