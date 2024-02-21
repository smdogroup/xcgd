#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <vector>

#include "utils/linalg.h"

// The structured ground grid
template <typename T, T lx, T ly, int nx, int ny>
class GDGrid {};

/**
 * Galerkin difference basis given a set of stencil nodes
 *
 * @tparam Np_1d number of nodes along one dimension, number of stencil nodes
 *               should be Np_1d^2, Np_1d >= 2
 */
template <int Np_1d>
class GD2DBasis {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int nodes_per_element = Np_1d * Np_1d;

  template <typename T>
  static void eval_basis_grad(const T pt[], T Nxi[]) {
    static constexpr int Np = nodes_per_element;
    static constexpr int Nk = nodes_per_element;

    T Ck[Nk * Np];
    std::vector<T> xpows(Np_1d);
    std::vector<T> ypows(Np_1d);

    for (int i = 0; i < Nk; i++) {
      T x = pt[spatial_dim * i];
      T y = pt[spatial_dim * i + 1];

      std::generate(xpows.begin(), xpows.end(), []() {
        static int __i = 0;
        return pow(x, __i++);
      });

      std::generate(ypows.begin(), ypows.end(), []() {
        static int __i = 0;
        return pow(y, __i++);
      });

      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          Ck[i + Nk * idx] = xpows[j] * ypows[k];  // (i, idx) entry
        }
      }
    }

    direct_inverse(Nk, Ck);

    std::vector<T> xpows(Np_1d);
    std::vector<T> ypows(Np_1d);
    std::vector<T> dxpows(Np_1d);
    std::vector<T> dypows(Np_1d);

    for (int i = 0; i < Nk; i++) {
      T x = pt[spatial_dim * i];
      T y = pt[spatial_dim * i + 1];

      std::generate(dxpows.begin(), dxpows.end(), []() {
        static int __i = 0;
        return T(__i) * pow(x, __i++ - 1);
      });

      std::generate(dypows.begin(), dypows.end(), []() {
        static int __i = 0;
        return T(__i) * pow(y, __i++ - 1);
      });

      Nxi[spatial_dim * i] = 0.0;
      Nxi[spatial_dim * i + 1] = 0.0;

      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          Nxi[spatial_dim * i] += Ck[i + Nk * idx] = dxpows[j] * ypows[k];
          Nxi[spatial_dim * i + 1] += Ck[i + Nk * idx] = xpows[j] * dypows[k];
        }
      }
    }
  }
};

template <int q>
class GD2DQuadrature {};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
