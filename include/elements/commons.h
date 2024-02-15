#ifndef XCGD_COMMONS_H
#define XCGD_COMMONS_H

#include "a2dcore.h"

template <typename T, class Basis, int dim>
static void eval_grad(const T pt[], const T dof[],
                      A2D::Mat<T, dim, Basis::spatial_dim>& grad) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  T Nxi[spatial_dim * nodes_per_element];
  Basis::template eval_basis_grad<T>(pt, Nxi);

  for (int k = 0; k < spatial_dim * dim; k++) {
    grad[k] = 0.0;
  }

  for (int i = 0; i < nodes_per_element; i++) {
    for (int k = 0; k < dim; k++) {
      for (int j = 0; j < spatial_dim; j++) {
        grad(k, j) += Nxi[spatial_dim * i + j] * dof[dim * i + k];
      }
    }
  }
}

template <typename T, class Basis, int dim>
static void add_grad(const T pt[],
                     const A2D::Mat<T, dim, Basis::spatial_dim>& coef,
                     T elem_res[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  T Nxi[spatial_dim * nodes_per_element];
  Basis::template eval_basis_grad<T>(pt, Nxi);

  for (int i = 0; i < nodes_per_element; i++) {
    for (int k = 0; k < dim; k++) {
      for (int j = 0; j < spatial_dim; j++) {
        elem_res[dim * i + k] += (coef(k, j) * Nxi[spatial_dim * i + j]);
      }
    }
  }
}

template <typename T, class Basis, int dim>
static void add_matrix(
    const T pt[],
    const A2D::Mat<T, dim * Basis::spatial_dim, dim * Basis::spatial_dim>& coef,
    T elem_jac[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  T Nxi[spatial_dim * nodes_per_element];
  Basis::template eval_basis_grad(pt, Nxi);

  constexpr int dof_per_element = dim * nodes_per_element;

  for (int i = 0; i < nodes_per_element; i++) {
    std::vector<T> ni(&Nxi[spatial_dim * i],
                      &Nxi[spatial_dim * i] + spatial_dim);

    for (int j = 0; j < nodes_per_element; j++) {
      std::vector<T> nj(&Nxi[spatial_dim * j],
                        &Nxi[spatial_dim * j] + spatial_dim);

      for (int ii = 0; ii < dim; ii++) {
        int row = dim * i + ii;
        for (int jj = 0; jj < dim; jj++) {
          int col = dim * j + jj;

          T val = 0.0;
          for (int kk = 0; kk < dim; kk++) {
            for (int ll = 0; ll < dim; ll++) {
              val +=
                  coef(spatial_dim * ii + kk, spatial_dim * jj + ll) * nj[ll];
            }
            val *= ni[kk];
          }
          elem_jac[col + row * dof_per_element] += val;
        }
      }
    }
  }
}

#endif  // XCGD_COMMONS_H