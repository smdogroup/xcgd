#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <complex>
#include <vector>

#include "utils/linalg.h"

// The structured ground grid
template <typename T, T lx, T ly, int nx, int ny>
class GDGrid {};

/**
 * Galerkin difference basis given a set of stencil nodes
 *
 * @tparam Np_1d number of nodes along one dimension, number of stencil nodes
 *               should be Np_1d^2, Np_1d >= 2, Np_1d should be even
 */
template <int Np_1d>
class GD2DBasis {
 public:
  static_assert(Np_1d % 2 == 0);
  static constexpr int spatial_dim = 2;
  static constexpr int nodes_per_element = Np_1d * Np_1d;

  template <typename T>
  static void eval_basis_grad(const T* pt, T* N, T* Nxi) {
    static constexpr int Np = nodes_per_element;
    static constexpr int Nk = nodes_per_element;

    T Ck[Nk * Np];
    std::vector<T> xpows(Np_1d);
    std::vector<T> ypows(Np_1d);

    for (int i = 0; i < Nk; i++) {
      T x = 2.0 * (i % Np_1d) / (Np_1d - 1) - 1.0;
      T y = 2.0 * (i / Np_1d) / (Np_1d - 1) - 1.0;

      for (int ii = 0; ii < Np_1d; ii++) {
        xpows[ii] = pow(x, ii);
        ypows[ii] = pow(y, ii);
      }

      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          Ck[i + Np * idx] = xpows[j] * ypows[k];  // (i, idx) entry
        }
      }
    }

    direct_inverse(Nk, Ck);

    std::vector<T> dxpows(Np_1d);
    std::vector<T> dypows(Np_1d);

    T x = pt[0];
    T y = pt[1];

    for (int ii = 0; ii < Np_1d; ii++) {
      xpows[ii] = pow(x, ii);
      ypows[ii] = pow(y, ii);
      dxpows[ii] = T(ii) * pow(x, ii - 1);
      dypows[ii] = T(ii) * pow(y, ii - 1);
    }

    for (int i = 0; i < Nk; i++) {
      if (N) {
        N[i] = 0.0;
      }
      if (Nxi) {
        Nxi[spatial_dim * i] = 0.0;
        Nxi[spatial_dim * i + 1] = 0.0;
      }

      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          if (N) {
            N[i] += Ck[idx + Nk * i] * xpows[j] * ypows[k];
          }
          if (Nxi) {
            Nxi[spatial_dim * i] += Ck[idx + Nk * i] * dxpows[j] * ypows[k];
            Nxi[spatial_dim * i + 1] += Ck[idx + Nk * i] * xpows[j] * dypows[k];
          }
        }
      }
    }
  }
};

template <int q>
class GD2DQuadrature {};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
