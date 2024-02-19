#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>

/**
 * Galerkin difference basis given a set of stencil nodes
 *
 * @tparam Np_1d number of nodes along one dimension, number of stencil nodes
 *               should be Np_1d^2
 */
template <int Np_1d>
class GalerkinDiff2DBasis {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int nodes_per_element = Np_1d * Np_1d;

  template <typename T>
  static void eval_basis_grad(const T pt[], T Nxi[]) {
    static constexpr int Np = nodes_per_element;
    static constexpr int Nk = nodes_per_element;

    T Vk[Nk * Np];

    for (int i = 0; i < Nk; i++) {
      T x = pt[spatial_dim * i];
      T y = pt[spatial_dim * i + 1];
      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          Vk[i + Nk * idx] = pow(x, j) * pow(y, k);  // (i, idx) entry
        }
      }
    }
  }
};

template <int q>
class GalerkinDiff2DQuadrature {};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
