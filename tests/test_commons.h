#ifndef TEST_COMMONS_H
#define TEST_COMMONS_H

#include <gtest/gtest.h>

#include <iostream>

template <int n, class VecType>
void print_vec(const VecType& vec) {
  for (int j = 0; j < n; j++) {
    std::cout << std::setw(5) << vec[j];
  }
  std::cout << std::endl;
}

template <int m, int n, class MatType>
void print_mat(const MatType& mat) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << std::setw(15) << mat(i, j);
    }
    std::cout << std::endl;
  }
}

// Helper macros
#define _EXPECT_VAL_NEAR(val1, val2) EXPECT_NEAR(val1, val2, 1e-15)

#define _EXPECT_VAL_NEAR_TOL(val1, val2, abs_err) \
  EXPECT_NEAR(val1, val2, abs_err)

#define _EXPECT_VEC_NEAR(m, vec, vals)   \
  for (int i = 0; i < m; i++) {          \
    EXPECT_NEAR(vec[i], vals[i], 1e-15); \
  }

#define _EXPECT_CPLX_VEC_NEAR(m, vec, vals)            \
  for (int i = 0; i < m; i++) {                        \
    EXPECT_NEAR(vec[i].real(), vals[i].real(), 1e-15); \
    EXPECT_NEAR(vec[i].imag(), vals[i].imag(), 1e-15); \
  }

#define _EXPECT_VEC_NEAR_TOL(m, vec, vals, abs_err) \
  for (int i = 0; i < m; i++) {                     \
    EXPECT_NEAR(vec[i], vals[i], abs_err);          \
  }

#define _EXPECT_CPLX_VEC_NEAR_TOL(m, vec, vals, abs_err) \
  for (int i = 0; i < m; i++) {                          \
    EXPECT_NEAR(vec[i].real(), vals[i].real(), abs_err); \
    EXPECT_NEAR(vec[i].imag(), vals[i].imag(), abs_err); \
  }

#define _EXPECT_MAT_NEAR(m, n, mat, vals)             \
  for (int i = 0; i < m; i++) {                       \
    for (int j = 0; j < n; j++) {                     \
      EXPECT_NEAR(mat(i, j), vals[n * i + j], 1e-15); \
    }                                                 \
  }

#define _EXPECT_MAT_NEAR_TOL(m, n, mat, vals, abs_err)  \
  for (int i = 0; i < m; i++) {                         \
    for (int j = 0; j < n; j++) {                       \
      EXPECT_NEAR(mat(i, j), vals[n * i + j], abs_err); \
    }                                                   \
  }

#define _GET_EXPECT_VAL_MACRO(_1, _2, _3, FUNC, ...) FUNC
#define _GET_EXPECT_VEC_MACRO(_1, _2, _3, _4, FUNC, ...) FUNC
#define _GET_EXPECT_MAT_MACRO(_1, _2, _3, _4, _5, FUNC, ...) FUNC

// Usage:
// - EXPECT_VAL_NEAR(val1, val2), or
// - EXPECT_VAL_NEAR(val1, val2, abs_err)
#define EXPECT_VAL_NEAR(...)                                                 \
  _GET_EXPECT_VAL_MACRO(__VA_ARGS__, _EXPECT_VAL_NEAR_TOL, _EXPECT_VAL_NEAR) \
  (__VA_ARGS__)

// Usage:
// - EXPECT_VEC_NEAR(m, vec, vals), or
// - EXPECT_VEC_NEAR(m, vec, vals, abs_err)
#define EXPECT_VEC_NEAR(...)                                                 \
  _GET_EXPECT_VEC_MACRO(__VA_ARGS__, _EXPECT_VEC_NEAR_TOL, _EXPECT_VEC_NEAR) \
  (__VA_ARGS__)
#define EXPECT_CPLX_VEC_NEAR(...)                               \
  _GET_EXPECT_VEC_MACRO(__VA_ARGS__, _EXPECT_CPLX_VEC_NEAR_TOL, \
                        _EXPECT_CPLX_VEC_NEAR)                  \
  (__VA_ARGS__)

// Usage:
// - EXPECT_MAT_NEAR(m, n, mat, vals), or
// - EXPECT_MAT_NEAR(m, n, mat, vals, abs_err)
#define EXPECT_MAT_NEAR(...)                                                 \
  _GET_EXPECT_MAT_MACRO(__VA_ARGS__, _EXPECT_MAT_NEAR_TOL, _EXPECT_MAT_NEAR) \
  (__VA_ARGS__)

// Usage:
// - EXPECT_VEC_EQ(m, vec, vals)
#define EXPECT_VEC_EQ(m, vec, vals) \
  for (int i = 0; i < m; i++) {     \
    EXPECT_EQ(vec[i], vals[i]);     \
  }

#endif  // TEST_COMMONS_H