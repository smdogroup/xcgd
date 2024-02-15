#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

/**
 * @tparam q p + 1 = 2q, p is polynomial degree, q > 0
 */
template <int q>
class GalerkinDiff2DBasis {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int nodes_per_element = 4 * q * q;

  template <typename T>
  static void eval_basis_grad(const T pt[], T Nxi[]) {}
};

template <int q>
class GalerkinDiff2DQuadrature {};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
