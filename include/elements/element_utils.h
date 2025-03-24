#ifndef XCGD_ELEMENT_UTILS_H
#define XCGD_ELEMENT_UTILS_H

#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

#include "a2dcore.h"
#include "a2ddefs.h"
#include "element_commons.h"
#include "physics/physics_commons.h"
#include "utils/vtk.h"

/**
 * @brief Transform gradient from reference coordinates to physical coordinates
 *
 *   ∇_x u = J^{-T} ∇_ξ u
 *
 * @tparam T numeric type
 * @tparam spatial_dim spatial dimension
 * @param J coordinate transformation matrix
 * @param grad_ref ∇_ξ u
 * @param grad ∇_x u, output
 */
template <typename T, int spatial_dim>
inline void transform(const A2D::Mat<T, spatial_dim, spatial_dim> &J,
                      const A2D::Vec<T, spatial_dim> &grad_ref,
                      A2D::Vec<T, spatial_dim> &grad) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatVecMult<A2D::MatOp::TRANSPOSE>(Jinv, grad_ref, grad);
}
template <typename T, int dim, int spatial_dim>
inline void transform(const A2D::Mat<T, spatial_dim, spatial_dim> &J,
                      const A2D::Mat<T, dim, spatial_dim> &grad_ref,
                      A2D::Mat<T, dim, spatial_dim> &grad) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatMatMult(grad_ref, Jinv, grad);
}

/**
 * @brief Transform gradient from physical coordinates to reference coordinates
 *
 *   ∂e/∂∇_ξ = ∂e/∂∇_x J^{-T}
 *
 * @tparam T numeric type
 * @tparam spatial_dim spatial dimension
 * @param J coordinate transformation matrix
 * @param grad ∂e/∂∇_x
 * @param grad_ref ∂e/∂∇_ξ, output
 */
template <typename T, int spatial_dim>
inline void rtransform(const A2D::Mat<T, spatial_dim, spatial_dim> &J,
                       const A2D::Vec<T, spatial_dim> &grad,
                       A2D::Vec<T, spatial_dim> &grad_ref) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatVecMult(Jinv, grad, grad_ref);
}
template <typename T, int dim, int spatial_dim>
inline void rtransform(const A2D::Mat<T, spatial_dim, spatial_dim> &J,
                       const A2D::Mat<T, dim, spatial_dim> &grad,
                       A2D::Mat<T, dim, spatial_dim> &grad_ref) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatMatMult<A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(grad, Jinv,
                                                             grad_ref);
}

/**
 * @brief Transform the mixed Hessian from physical coordinates to reference
 * coordinates
 *
 * @tparam T numeric type
 * @tparam dim dof dimension
 * @tparam spatial_dim spatial dimension
 * @param J coordinate transformation matrix
 * @param hess_mixed ∂/∂(∇_x)uq(∂e/∂uq)
 * @param hess_mixed_ref ∂/∂(∇_ξ)uq(∂e/∂uq), output
 */
template <typename T, int dim, int spatial_dim>
inline void mtransform(const A2D::Mat<T, spatial_dim, spatial_dim> &J,
                       const A2D::Mat<T, dim, spatial_dim * dim> &hess_mixed,
                       A2D::Mat<T, dim, spatial_dim * dim> &hess_mixed_ref) {
  // throw std::runtime_error(
  //     "Uncomment the following code to verify and use this function");

  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  hess_mixed_ref.zero();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int ii = 0; ii < spatial_dim; ii++) {
        for (int jj = 0; jj < spatial_dim; jj++) {
          hess_mixed_ref(i, j * spatial_dim + jj) +=
              hess_mixed(i, j * spatial_dim + ii) * Jinv(jj, ii);
        }
      }
    }
  }
}
template <typename T, int spatial_dim>
inline void mtransform(const A2D::Mat<T, spatial_dim, spatial_dim> &J,
                       const A2D::Vec<T, spatial_dim> &hess_mixed,
                       A2D::Vec<T, spatial_dim> &hess_mixed_ref) {
  rtransform<T, spatial_dim>(J, hess_mixed, hess_mixed_ref);
}

/**
 * @brief Transform Hessian from physical coordinates to reference coordinates
 *
 *   ∂^2e/∂∇_ξ^2 = J^{-1} ∂^2e/∂∇_x^2 J^{-T}
 *
 * @tparam T numeric type
 * @tparam spatial_dim spatial dimension
 * @param J coordinate transformation matrix
 * @param hess ∂^2e/∂∇_x^2
 * @param hess_ref ∂^2e/∂∇_ξ^2, output
 */
template <typename T, int dim, int spatial_dim>
inline void jtransform(
    const A2D::Mat<T, spatial_dim, spatial_dim> &J,
    const A2D::Mat<T, dim * spatial_dim, dim * spatial_dim> &hess,
    A2D::Mat<T, dim * spatial_dim, dim * spatial_dim> &hess_ref) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  hess_ref.zero();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      for (int ii = 0; ii < spatial_dim; ii++) {
        for (int jj = 0; jj < spatial_dim; jj++) {
          for (int kk = 0; kk < spatial_dim; kk++) {
            for (int ll = 0; ll < spatial_dim; ll++) {
              // hess(ii, jj) for dof (i, j)
              hess_ref(spatial_dim * i + ii, spatial_dim * j + jj) +=
                  Jinv(ii, kk) *
                  hess(spatial_dim * i + kk, spatial_dim * j + ll) *
                  Jinv(jj, ll);
            }
          }
        }
      }
    }
  }
}

template <typename T, class Mesh, class Basis>
void get_element_xloc(const Mesh &mesh, int e, T element_xloc[]) {
  int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  std::fill(element_xloc, element_xloc + spatial_dim * max_nnodes_per_element,
            T(0.0));
  int nodes[max_nnodes_per_element];
  int nnodes = mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nnodes; j++) {
    mesh.get_node_xloc(nodes[j], element_xloc);
    element_xloc += spatial_dim;
  }
}

template <typename T, int dim, class Mesh, int max_nnodes_per_element,
          int spatial_dim>
void get_element_vars(const Mesh &mesh, int e, const T dof[], T element_dof[]) {
  std::fill(element_dof, element_dof + max_nnodes_per_element * dim, T(0.0));
  int nodes[max_nnodes_per_element];
  int nnodes = mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nnodes; j++) {
    for (int k = 0; k < dim; k++, element_dof++) {
      element_dof[0] = dof[dim * nodes[j] + k];
    }
  }
}

template <typename T, int dim, class Mesh, class Basis>
void get_element_vars(const Mesh &mesh, int e, const T dof[], T element_dof[]) {
  int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  std::fill(element_dof, element_dof + max_nnodes_per_element * dim, T(0.0));
  int nodes[max_nnodes_per_element];
  int nnodes = mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nnodes; j++) {
    for (int k = 0; k < dim; k++, element_dof++) {
      element_dof[0] = dof[dim * nodes[j] + k];
    }
  }
}

template <typename T, int dim, class Basis>
void get_element_vars(int nnodes, int *nodes, const T dof[], T element_dof[]) {
  int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  std::fill(element_dof, element_dof + max_nnodes_per_element * dim, T(0.0));
  for (int j = 0; j < nnodes; j++) {
    for (int k = 0; k < dim; k++, element_dof++) {
      element_dof[0] = dof[dim * nodes[j] + k];
    }
  }
}

template <typename T, int dim, class Mesh, class Basis>
void add_element_res(const Mesh &mesh, int e, const T element_res[], T res[]) {
  int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[max_nnodes_per_element];
  int nnodes = mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nnodes; j++) {
    for (int k = 0; k < dim; k++, element_res++) {
      res[dim * nodes[j] + k] += element_res[0];
    }
  }
}

template <typename T, int dim, class Basis>
void add_element_res(int nnodes, int *nodes, const T element_res[], T res[]) {
  for (int j = 0; j < nnodes; j++) {
    for (int k = 0; k < dim; k++, element_res++) {
      res[dim * nodes[j] + k] += element_res[0];
    }
  }
}

template <typename T, class Mesh, class Basis>
void add_element_dfdx(const Mesh &mesh, int e, const T element_dfdx[],
                      T dfdx[]) {
  int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  int nodes[max_nnodes_per_element];
  int nnodes = mesh.get_elem_dof_nodes(e, nodes);

  for (int j = 0; j < nnodes; j++) {
    dfdx[nodes[j]] += element_dfdx[j];
  }
}

template <typename T, class Basis>
void add_element_dfdx(int nnodes, int *nodes, const T element_dfdx[],
                      T dfdx[]) {
  for (int j = 0; j < nnodes; j++) {
    dfdx[nodes[j]] += element_dfdx[j];
  }
}

template <typename T, class Mesh, class Basis>
void add_element_dfdphi(const Mesh &lsf_mesh, int c, const T element_dfdphi[],
                        T dfdphi[]) {
  int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  int nodes[max_nnodes_per_element];
  int nnodes = lsf_mesh.get_elem_dof_nodes(c, nodes);

  for (int j = 0; j < nnodes; j++) {
    dfdphi[nodes[j]] += element_dfdphi[j];
  }
}

template <typename T>
void add_element_dfdphi(int nnodes, int *nodes, const T element_dfdphi[],
                        T dfdphi[]) {
  for (int j = 0; j < nnodes; j++) {
    dfdphi[nodes[j]] += element_dfdphi[j];
  }
}

template <typename T, class Mesh, int max_nnodes_per_element>
void get_element_dfdphi(const Mesh &lsf_mesh, int c, const T dfdphi[],
                        T element_dfdphi[]) {
  int nodes[max_nnodes_per_element];
  int nnodes = lsf_mesh.get_elem_dof_nodes(c, nodes);

  for (int j = 0; j < nnodes; j++) {
    element_dfdphi[j] = dfdphi[nodes[j]];
  }
}

// Helper functions
inline double *get_ptr(double &val) { return &val; }
inline std::complex<double> *get_ptr(std::complex<double> &val) { return &val; }

template <typename T, int N>
inline T *get_ptr(A2D::Vec<T, N> &vec) {
  return vec.get_data();
}

template <typename T, int M, int N>
inline T *get_ptr(A2D::Mat<T, M, N> &vec) {
  return vec.get_data();
}

/**
 * @brief Unified element interpolation function
 *
 * @param dof[dim * max_nnodes_per_element]
 * @param N[max_nnodes_per_element]
 * @param Nxi[spatial_dim * max_nnodes_per_element]
 * @param vals[dim]
 * @param grad[dim * spatial_dim]
 */
template <typename T, int spatial_dim, int max_nnodes_per_element, int dim>
void interp_val_grad(const T dof[], const T N[], const T Nxi[], T *vals,
                     T *grad) {
  if (vals) {
    for (int k = 0; k < dim; k++) {
      vals[k] = 0.0;
    }
  }

  if (grad) {
    for (int k = 0; k < spatial_dim * dim; k++) {
      grad[k] = 0.0;
    }
  }

  for (int i = 0; i < max_nnodes_per_element; i++) {
    for (int k = 0; k < dim; k++) {
      if (vals) {
        vals[k] += N[i] * dof[dim * i + k];
      }
      if (grad) {
        for (int j = 0; j < spatial_dim; j++) {
          grad[spatial_dim * k + j] +=
              Nxi[spatial_dim * i + j] * dof[dim * i + k];
        }
      }
    }
  }
}

#if 0
/**
 * The following two functions evaluate u and ∇u at a quadrature point given the
 * shape function and shape gradient evaluations N and Nxi at this quadrature
 * point.
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param dof node dof values of size max_nnodes_per_element * dim
 * @param N shape function values, size of max_nnodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates, size of
 * max_nnodes_per_element * spatial_dim
 * @param vals interpolated dof
 * @param grad gradients of vals w.r.t. computational coordinates dv/dxi
 */
template <typename T, int spatial_dim, int max_nnodes_per_element, int dim>
void interp_val_grad_deprecated(const T dof[], const T N[], const T Nxi[],
                                A2D::Vec<T, dim> *vals,
                                A2D::Mat<T, dim, spatial_dim> *grad) {
  if (vals) {
    for (int k = 0; k < dim; k++) {
      (*vals)(k) = 0.0;
    }
  }

  if (grad) {
    for (int k = 0; k < spatial_dim * dim; k++) {
      (*grad)[k] = 0.0;
    }
  }

  for (int i = 0; i < max_nnodes_per_element; i++) {
    for (int k = 0; k < dim; k++) {
      if (vals) {
        (*vals)(k) += N[i] * dof[dim * i + k];
      }
      if (grad) {
        for (int j = 0; j < spatial_dim; j++) {
          (*grad)(k, j) += Nxi[spatial_dim * i + j] * dof[dim * i + k];
        }
      }
    }
  }
}

// Back compatibility
template <typename T, class Basis, int dim>
inline void interp_val_grad_deprecated(
    const T dof[], const T N[], const T Nxi[], A2D::Vec<T, dim> *vals,
    A2D::Mat<T, dim, Basis::spatial_dim> *grad) {
  return interp_val_grad_deprecated<T, Basis::spatial_dim,
                                    Basis::max_nnodes_per_element, dim>(
      dof, N, Nxi, vals, grad);
}

// dim == 1
template <typename T, int spatial_dim, int max_nnodes_per_element>
void interp_val_grad_deprecated(const T *dof, const T *N, const T *Nxi, T *val,
                                A2D::Vec<T, spatial_dim> *grad) {
  if (val) {
    *val = 0.0;
  }

  if (grad) {
    for (int k = 0; k < spatial_dim; k++) {
      (*grad)[k] = 0.0;
    }
  }

  for (int i = 0; i < max_nnodes_per_element; i++) {
    if (val) {
      *val += N[i] * dof[i];
    }
    if (grad) {
      for (int j = 0; j < spatial_dim; j++) {
        (*grad)(j) += Nxi[spatial_dim * i + j] * dof[i];
      }
    }
  }
}

// Back compatibility
template <typename T, class Basis>
inline void interp_val_grad_deprecated(const T *dof, const T *N, const T *Nxi,
                                       T *val,
                                       A2D::Vec<T, Basis::spatial_dim> *grad) {
  return interp_val_grad_deprecated<T, Basis::spatial_dim,
                                    Basis::max_nnodes_per_element>(dof, N, Nxi,
                                                                   val, grad);
}
#endif

/**
 * The following two functions evaluate u and ∇2u at a quadrature point given
 * the shape function and shape gradient evaluations N and Nxi at this
 * quadrature point.
 *
 * Note: this is a unified element interpolation function regardless different
 * types of hess given dim = 1 or dim > 1
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param dof node dof values of size max_nnodes_per_element * dim
 * @param Nxixi shape function Hessians, concatenation of (∂2/∂ξξ, ∂2/∂ξη,
 * ∂2/∂ηξ, ∂2/∂ηη) N_q
 * @param hess[dim * spatial_dim * spatial_dim] ∇2u, hess(i, :) = (∂2u[i]/∂ξξ,
 * ∂2u[i]/∂ξη, ∂2u[i]/∂ηξ, ∂2u[i]/∂ηη)
 */
template <typename T, int spatial_dim, int max_nnodes_per_element, int dim>
void interp_hess(const T *dof, const T *Nxixi, T *hess) {
  constexpr int s2 = spatial_dim * spatial_dim;
  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < max_nnodes_per_element; i++) {
      int offset = s2 * i;
      for (int d1 = 0; d1 < spatial_dim; d1++) {
        for (int d2 = 0; d2 < spatial_dim; d2++) {
          int index = d1 * spatial_dim + d2;
          hess[s2 * j + index] += Nxixi[offset + index] * dof[dim * i + j];
        }
      }
    }
  }
}

/**
 * The following two functions evaluate u and ∇2u at a quadrature point given
 * the shape function and shape gradient evaluations N and Nxi at this
 * quadrature point.
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param dof node dof values of size max_nnodes_per_element * dim
 * @param Nxixi shape function Hessians, concatenation of (∂2/∂ξξ, ∂2/∂ξη,
 * ∂2/∂ηξ, ∂2/∂ηη) N_q
 * @param hess ∇2u, hess(i, :) = (∂2u[i]/∂ξξ, ∂2u[i]/∂ξη, ∂2u[i]/∂ηξ,
 * ∂2u[i]/∂ηη)
 */
template <typename T, class Basis, int dim>
void interp_hess_deprecated(
    const T *dof, const T *Nxixi,
    A2D::Mat<T, dim, Basis::spatial_dim * Basis::spatial_dim> &hess) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  hess.zero();

  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < max_nnodes_per_element; i++) {
      int offset = spatial_dim * spatial_dim * i;
      for (int d1 = 0; d1 < spatial_dim; d1++) {
        for (int d2 = 0; d2 < spatial_dim; d2++) {
          int index = d1 * spatial_dim + d2;
          hess(j, index) += Nxixi[offset + index] * dof[dim * i + j];
        }
      }
    }
  }
}

// dim == 1
template <typename T, class Basis>
void interp_hess_deprecated(
    const T *dof, const T *Nxixi,
    A2D::Vec<T, Basis::spatial_dim * Basis::spatial_dim> &hess) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  hess.zero();

  for (int i = 0; i < max_nnodes_per_element; i++) {
    int offset = spatial_dim * spatial_dim * i;
    for (int d1 = 0; d1 < spatial_dim; d1++) {
      for (int d2 = 0; d2 < spatial_dim; d2++) {
        int index = d1 * spatial_dim + d2;
        hess(index) += Nxixi[offset + index] * dof[i];
      }
    }
  }
}
/**
 * @brief The following two functions assemble the total derivatives of the
 * energy functional e:
 *
 *   de/du = ∂e/∂u + ∂e/∂(∇u) * ∂(∇u)/∂u
 *         = ∂e/∂uq * ∂uq/∂u + ∂e/∂(∇uq) * ∂(∇uq)/∂u
 *
 * where u are the element dof, uq, ∇uq are quantities on quadrature points,
 * uq = Nu, ∇uq = ∇Nu, so we have ∂uq/∂u = N, ∂(∇uq)/∂u = ∇N, ∇ = ∇_ξ, as a
 * result:
 *
 *   de/du = ∂e/∂uq * N + ∂e/∂(∇uq) * ∇N
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param N shape function values, size of max_nnodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ,
 * size of max_nnodes_per_element * spatial_dim
 * @param coef_vals ∂e/∂uq
 * @param coef_grad ∂e/∂((∇_ξ)uq)
 * @param elem_res de/du
 */
template <typename T, class Basis, int dim>
void add_grad(const T N[], const T Nxi[], const A2D::Vec<T, dim> &coef_vals,
              A2D::Mat<T, dim, Basis::spatial_dim> &coef_grad, T elem_res[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  for (int i = 0; i < max_nnodes_per_element; i++) {
    for (int k = 0; k < dim; k++) {
      elem_res[dim * i + k] += coef_vals(k) * N[i];
      for (int j = 0; j < spatial_dim; j++) {
        elem_res[dim * i + k] += (coef_grad(k, j) * Nxi[spatial_dim * i + j]);
      }
    }
  }
}

// dim == 1
template <typename T, class Basis>
void add_grad(const T N[], const T Nxi[], const T &coef_val,
              const A2D::Vec<T, Basis::spatial_dim> &coef_grad, T elem_res[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  for (int i = 0; i < max_nnodes_per_element; i++) {
    elem_res[i] += coef_val * N[i];
    for (int j = 0; j < spatial_dim; j++) {
      elem_res[i] += (coef_grad(j) * Nxi[spatial_dim * i + j]);
    }
  }
}

/**
 * @brief Assemble the total Hessian of the energy functional e:
 *
 *   d^2e/du^2 =   [∂uq/∂u]^T    * ∂^2e/∂(uq)^2  * ∂uq/∂u
 *               + [∂(∇uq)/∂u]^T * ∂^2e/∂(∇uq)^2 * ∂(∇uq)/∂u
 *
 * where u are the element dof, uq, ∇uq are quantities on quadrature points,
 * uq = Nu, ∇uq = ∇Nu, ∇ = ∇_ξ, as a result:
 *
 *   d^2e/du^2 =   N^T    * ∂^2e/∂(uq)^2  * N
 *               + [∇N]^T * ∂^2e/∂(∇uq)^2 * ∇N
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param N shape function values, size of max_nnodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ,
 * size of max_nnodes_per_element * spatial_dim
 * @param coef_vals ∂^2e/∂(uq)^2
 * @param coef_mixed ∂/∂(∇_ξ)uq(∂e/∂uq)
 * @param coef_hess ∂^2e/∂((∇_ξ)uq)^2
 * @param elem_jac d^2e/du^2
 */
template <typename T, int spatial_dim, int max_nnodes_per_element, int dim>
void add_matrix(
    const T N[], const T Nxi[], const A2D::Mat<T, dim, dim> &coef_vals,
    const A2D::Mat<T, dim, spatial_dim * dim> &coef_mixed,
    const A2D::Mat<T, dim * spatial_dim, dim * spatial_dim> &coef_hess,
    T elem_jac[]) {
  constexpr int max_dof_per_element = dim * max_nnodes_per_element;

  for (int i = 0; i < max_nnodes_per_element; i++) {
    T ni = N[i];
    std::vector<T> nxi(&Nxi[spatial_dim * i],
                       &Nxi[spatial_dim * i] + spatial_dim);

    for (int j = 0; j < max_nnodes_per_element; j++) {
      T nj = N[j];
      std::vector<T> nxj(&Nxi[spatial_dim * j],
                         &Nxi[spatial_dim * j] + spatial_dim);

      for (int ii = 0; ii < dim; ii++) {
        int row = dim * i + ii;
        for (int jj = 0; jj < dim; jj++) {
          int col = dim * j + jj;

          T val = 0.0;
          for (int kk = 0; kk < spatial_dim; kk++) {
            for (int ll = 0; ll < spatial_dim; ll++) {
              val += coef_hess(spatial_dim * ii + kk, spatial_dim * jj + ll) *
                     nxi[kk] * nxj[ll];
            }
          }

          for (int ll = 0; ll < spatial_dim; ll++) {
            val += coef_mixed(ii, spatial_dim * jj + ll) * ni * nxj[ll] +
                   coef_mixed(jj, spatial_dim * ii + ll) * nj * nxi[ll];
          }

          elem_jac[col + row * max_dof_per_element] +=
              val + coef_vals(ii, jj) * ni * nj;
        }
      }
    }
  }
}

// dim == 1
template <typename T, int spatial_dim, int max_nnodes_per_element>
void add_matrix(const T N[], const T Nxi[], const T &coef_val,
                const A2D::Vec<T, spatial_dim> &coef_mixed,
                const A2D::Mat<T, spatial_dim, spatial_dim> &coef_hess,
                T elem_jac[]) {
  constexpr int max_dof_per_element = max_nnodes_per_element;

  for (int i = 0; i < max_nnodes_per_element; i++) {
    T ni = N[i];
    std::vector<T> nxi(&Nxi[spatial_dim * i],
                       &Nxi[spatial_dim * i] + spatial_dim);

    for (int j = 0; j < max_nnodes_per_element; j++) {
      T nj = N[j];
      std::vector<T> nxj(&Nxi[spatial_dim * j],
                         &Nxi[spatial_dim * j] + spatial_dim);

      T val = 0.0;
      for (int kk = 0; kk < spatial_dim; kk++) {
        for (int ll = 0; ll < spatial_dim; ll++) {
          val += coef_hess(kk, ll) * nxi[kk] * nxj[ll];
        }
      }

      for (int ll = 0; ll < spatial_dim; ll++) {
        val += coef_mixed(ll) * (ni * nxj[ll] + nj * nxi[ll]);
      }

      elem_jac[j + i * max_dof_per_element] += val + coef_val * ni * nj;
    }
  }
}

/**
 * @brief Compute ∂|J|q/∂ξq: the derivatives of the Jacobian determinant w.r.t.
 * reference coordinates ξ at a quadrature point
 *
 * Note: for the GD applications with structural grid and mesh, this may be
 * effectively zero because the Jacobian transformation is constant throughout
 * the computational/physical coordinates
 *
 * @tparam T numeric type
 * @tparam Basis the Basis type
 * @param elem_xloc element nodal locations
 * @param Nxixi shape function Hessians, concatenation of (∂2/∂ξξ, ∂2/∂ξη,
 * ∂2/∂ηξ, ∂2/∂ηη) N_q
 * @param J Jacobian transformation matrix
 * @param grad, output
 */
template <typename T, class Basis>
void det_deriv(const T *elem_xloc, const T *Nxixi,
               const A2D::Mat<T, Basis::spatial_dim, Basis::spatial_dim> &J,
               A2D::Vec<T, Basis::spatial_dim> &grad) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  // Get derivatives of detJ w.r.t. J
  T detJ, detJb = 1.0;
  A2D::ADObj<T &> detJ_obj(detJ, detJb);
  A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);
  auto stack = A2D::MakeStack(A2D::MatDet(J_obj, detJ_obj));
  stack.reverse();

  auto Jb = J_obj.bvalue();

  grad.zero();

  // sum_j ∇^2_ξ N_j * Jbar^T * xj
  for (int j = 0; j < max_nnodes_per_element; j++) {
    int Nxixi_offset = j * spatial_dim * spatial_dim;
    int xloc_offset = j * spatial_dim;
    for (int ii = 0; ii < spatial_dim; ii++) {
      for (int jj = 0; jj < spatial_dim; jj++) {
        for (int kk = 0; kk < spatial_dim; kk++) {
          grad(ii) += Nxixi[Nxixi_offset + ii * spatial_dim + jj] * Jb(kk, jj) *
                      elem_xloc[xloc_offset + kk];
        }
      }
    }
  }
}

/**
 * @brief The following two functions add ψ^T * dR/dx contribution from a single
 * quadrature point
 */
template <typename T, class Basis>
void add_jac_adj_product_bulk(const T N[], const T &x_val, T elem_dfdx[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  for (int i = 0; i < max_nnodes_per_element; i++) {
    elem_dfdx[i] += x_val * N[i];
  }
}

/**
 * @brief The following two functions add ψ^T * dR/dφ contribution from a single
 * quadrature point
 *
 * Note: this function only works for the Galerkin difference basis
 *
 * Note: this is intended to work with bulk integration, for interface
 * integration use add_jac_adj_product_interface instead
 *
 * @tparam T numeric type
 * @tparam GDBasis a GD Basis specialization
 * @tparam dim number of dof components at each dof node
 * @param [in] weight quadrature weight
 * @param [in] detJ determinant of Jacobian transformation matrix a the quad pt
 * @param [in] wts_grad derivatives of quadrature weight w.r.t. nodal phi
 * @param [in] pts_grad derivatives of the quadrature point w.r.t. nodal phi
 * @param [in] psiq adjoint variable at the quadrature point, (dim,)
 * @param [in] ugrad_ref (∇_ξ)uq, (dim, spatial_dim)
 * @param [in] pgrad_ref (∇_ξ)ψq, (dim, spatial_dim)
 * @param [in] uhess_ref (∇2_ξ)uq, (dim, spatial_dim * spatial_dim)
 * @param [in] phess_ref (∇2_ξ)ψq, (dim, spatial_dim * spatial_dim)
 * @param [in] coef_uq ∂e/∂uq, (dim,)
 * @param [in] coef_ugrad_ref ∂e/∂(∇_ξ)uq, (dim, spatial_dim)
 * @param [in] jp_uq ∂2e/∂uq2 * ψq, (dim)
 * @param [in] jp_ugrad_ref ∂2e/∂(∇_ξ)uq2 * ψq, (dim, spatial_dim)
 * @param [out] elem_dfdphi element vector of ψ^T * dR/dφ,
 * (max_nnodes_per_element,)
 */
template <typename T, class GDBasis, int dim>
void add_jac_adj_product_bulk(
    T weight, const T wts_grad[], const T pts_grad[],
    const A2D::Vec<T, dim> &psiq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &ugrad_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &pgrad_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &uhess_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &phess_ref,
    const A2D::Vec<T, dim> &coef_uq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &coef_ugrad_ref,
    const A2D::Vec<T, dim> &jp_uq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &jp_ugrad_ref,
    T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int max_nnodes_per_element = GDBasis::max_nnodes_per_element;

  // ∂e/∂u * ψ
  // = ∂e/∂uq * ψq + ∂e/∂(∇_ξ)uq * (∇_ξ)ψq
  T dedu_psi = 0.0;
  for (int i = 0; i < dim; i++) {
    dedu_psi += coef_uq(i) * psiq(i);
    for (int d = 0; d < spatial_dim; d++) {
      dedu_psi += coef_ugrad_ref(i, d) * pgrad_ref(i, d);
    }
  }

  // Jacobian-vector product times ugrad
  // ∂2e/∂uq2 * ψq * ∇uq
  A2D::Vec<T, spatial_dim> jvp_ugrad{};
  for (int j = 0; j < dim; j++) {
    for (int d = 0; d < spatial_dim; d++) {
      jvp_ugrad(d) += jp_uq(j) * ugrad_ref(j, d);
    }
  }

  // Jacobian-vector product times hess
  // ∂2e/∂uq2 * ∇ψq * ∇2uq
  A2D::Vec<T, spatial_dim> jvp_uhess{};
  for (int j = 0; j < dim; j++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        jvp_uhess(dd) +=
            jp_ugrad_ref(j, d) * uhess_ref(j, d * spatial_dim + dd);
      }
    }
  }

  // ∂e/∂uq * ∇ψq
  A2D::Vec<T, spatial_dim> deriv_grad{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      deriv_grad(d) += coef_uq(i) * pgrad_ref(i, d);
    }
  }

  // ∂e/∂∇uq * ∇2ψq
  A2D::Vec<T, spatial_dim> deriv_hess{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        deriv_hess(dd) +=
            coef_ugrad_ref(i, d) * phess_ref(i, d * spatial_dim + dd);
      }
    }
  }

  for (int n = 0; n < max_nnodes_per_element; n++) {
    // AJP_{1,n}
    elem_dfdphi[n] += dedu_psi * wts_grad[n] / weight;

    // AJP_{2,n} is assumed zero

    // AJP_{3,n}
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdphi[n] +=
          (jvp_ugrad(d) + jvp_uhess(d) + deriv_grad(d) + deriv_hess(d)) *
          pts_grad[spatial_dim * n + d];
    }
  }
}

// dim == 1
template <typename T, class GDBasis>
void add_jac_adj_product_bulk(
    T weight, const T wts_grad[], const T pts_grad[], T psiq,
    const A2D::Vec<T, GDBasis::spatial_dim> &ugrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim> &pgrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim * GDBasis::spatial_dim> &uhess_ref,
    const A2D::Vec<T, GDBasis::spatial_dim * GDBasis::spatial_dim> &phess_ref,
    T coef_uq, const A2D::Vec<T, GDBasis::spatial_dim> &coef_ugrad_ref, T jp_uq,
    const A2D::Vec<T, GDBasis::spatial_dim> &jp_ugrad_ref, T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int max_nnodes_per_element = GDBasis::max_nnodes_per_element;

  // ∂e/∂u * ψ
  // = ∂e/∂uq * ψq + ∂e/∂(∇_ξ)uq * (∇_ξ)ψq
  T dedu_psi = coef_uq * psiq;
  for (int d = 0; d < spatial_dim; d++) {
    dedu_psi += coef_ugrad_ref(d) * pgrad_ref(d);
  }

  // Jacobian-vector product times ugrad
  // ∂2e/∂uq2 * ψq * ∇uq
  A2D::Vec<T, spatial_dim> jvp_ugrad{};
  for (int d = 0; d < spatial_dim; d++) {
    jvp_ugrad(d) += jp_uq * ugrad_ref(d);
  }

  // Jacobian-vector product times hess
  // ∂2e/∂uq2 * ∇ψq * ∇2uq
  A2D::Vec<T, spatial_dim> jvp_uhess{};
  for (int d = 0; d < spatial_dim; d++) {
    for (int dd = 0; dd < spatial_dim; dd++) {
      jvp_uhess(dd) += jp_ugrad_ref(d) * uhess_ref(d * spatial_dim + dd);
    }
  }

  // ∂e/∂uq * ∇ψq
  A2D::Vec<T, spatial_dim> deriv_grad{};
  for (int d = 0; d < spatial_dim; d++) {
    deriv_grad(d) += coef_uq * pgrad_ref(d);
  }

  // ∂e/∂∇uq * ∇2ψq
  A2D::Vec<T, spatial_dim> deriv_hess{};
  for (int d = 0; d < spatial_dim; d++) {
    for (int dd = 0; dd < spatial_dim; dd++) {
      deriv_hess(dd) += coef_ugrad_ref(d) * phess_ref(d * spatial_dim + dd);
    }
  }

  for (int n = 0; n < max_nnodes_per_element; n++) {
    // AJP_{1,n}
    elem_dfdphi[n] += dedu_psi * wts_grad[n] / weight;

    // AJP_{2,n} is assumed zero

    // AJP_{3,n}
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdphi[n] +=
          (jvp_ugrad(d) + jvp_uhess(d) + deriv_grad(d) + deriv_hess(d)) *
          pts_grad[spatial_dim * n + d];
    }
  }
}

template <typename T, int spatial_dim, int max_nnodes_per_element, int dim>
void add_jac_adj_product_bulk_deprecated(
    T weight, T detJ, const T wts_grad[], const T pts_grad[], const T *psiq,
    const T *ugrad_ref, const T *pgrad_ref, const T *uhess_ref,
    const T *phess_ref, const T *coef_uq, const T *coef_ugrad_ref,
    const T *jp_uq, const T *jp_ugrad_ref, T elem_dfdphi[]) {
  constexpr int s2 = spatial_dim * spatial_dim;

  // ∂e/∂u * ψ
  // = ∂e/∂uq * ψq + ∂e/∂(∇_ξ)uq * (∇_ξ)ψq
  T dedu_psi = 0.0;
  for (int i = 0; i < dim; i++) {
    dedu_psi += coef_uq[i] * psiq[i];
    for (int d = 0; d < spatial_dim; d++) {
      dedu_psi +=
          coef_ugrad_ref[i * spatial_dim + d] * pgrad_ref[i * spatial_dim + d];
    }
  }

  // Jacobian-vector product times ugrad
  // ∂2e/∂uq2 * ψq * ∇uq
  A2D::Vec<T, spatial_dim> jvp_ugrad{};
  for (int j = 0; j < dim; j++) {
    for (int d = 0; d < spatial_dim; d++) {
      jvp_ugrad(d) += jp_uq[j] * ugrad_ref[j * spatial_dim + d];
    }
  }

  // Jacobian-vector product times hess
  // ∂2e/∂uq2 * ∇ψq * ∇2uq
  A2D::Vec<T, spatial_dim> jvp_uhess{};
  for (int j = 0; j < dim; j++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        jvp_uhess(dd) += jp_ugrad_ref[j * spatial_dim + d] *
                         uhess_ref[j * s2 + d * spatial_dim + dd];
      }
    }
  }

  // ∂e/∂uq * ∇ψq
  A2D::Vec<T, spatial_dim> deriv_grad{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      deriv_grad(d) += coef_uq[i] * pgrad_ref[i * spatial_dim + d];
    }
  }

  // ∂e/∂∇uq * ∇2ψq
  A2D::Vec<T, spatial_dim> deriv_hess{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        deriv_hess(dd) += coef_ugrad_ref[i * spatial_dim + d] *
                          phess_ref[i * s2 + d * spatial_dim + dd];
      }
    }
  }

  T wdetJ = weight * detJ;

  for (int n = 0; n < max_nnodes_per_element; n++) {
    // AJP_{1,n}
    elem_dfdphi[n] += detJ * dedu_psi * wts_grad[n];

    // AJP_{2,n} is assumed zero

    // AJP_{3,n}
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdphi[n] +=
          wdetJ *
          (jvp_ugrad(d) + jvp_uhess(d) + deriv_grad(d) + deriv_hess(d)) *
          pts_grad[spatial_dim * n + d];
    }
  }
}

template <typename T, class GDBasis, int dim>
void add_jac_adj_product_interface(
    T weight, const T wts_grad[], const T pts_grad[], const T wns_grad[],
    const A2D::Vec<T, dim> &psiq_primary,
    const A2D::Vec<T, dim> &psiq_secondary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &ugrad_ref_primary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &ugrad_ref_secondary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &pgrad_ref_primary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &pgrad_ref_secondary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &uhess_ref_primary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &uhess_ref_secondary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &phess_ref_primary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &phess_ref_secondary,
    const A2D::Vec<T, dim> &coef_uq_primary,
    const A2D::Vec<T, dim> &coef_uq_secondary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &coef_ugrad_ref_primary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &coef_ugrad_ref_secondary,
    const A2D::Vec<T, dim> &jp_uq_primary,
    const A2D::Vec<T, dim> &jp_uq_secondary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &jp_ugrad_ref_primary,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &jp_ugrad_ref_secondary,
    const A2D::Vec<T, GDBasis::spatial_dim> &jp_nrm_ref, T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int max_nnodes_per_element = GDBasis::max_nnodes_per_element;

  // ∂e/∂u * ψ
  // = ∂e/∂uq * ψq + ∂e/∂(∇_ξ)uq * (∇_ξ)ψq
  T dedu_psi = 0.0;
  for (int i = 0; i < dim; i++) {
    dedu_psi += coef_uq_primary(i) * psiq_primary(i) +
                coef_uq_secondary(i) * psiq_secondary(i);
    for (int d = 0; d < spatial_dim; d++) {
      dedu_psi += coef_ugrad_ref_primary(i, d) * pgrad_ref_primary(i, d) +
                  coef_ugrad_ref_secondary(i, d) * pgrad_ref_secondary(i, d);
    }
  }

  A2D::Vec<T, spatial_dim> rTp_deriv{};

  // Jacobian-vector product times ugrad
  // ∂2e/∂uq2 * ψq * ∇uq
  for (int j = 0; j < dim; j++) {
    for (int d = 0; d < spatial_dim; d++) {
      rTp_deriv(d) += jp_uq_primary(j) * ugrad_ref_primary(j, d) +
                      jp_uq_secondary(j) * ugrad_ref_secondary(j, d);
    }
  }

  // Jacobian-vector product times hess
  // ∂2e/∂uq2 * ∇ψq * ∇2uq
  for (int j = 0; j < dim; j++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        rTp_deriv(dd) += jp_ugrad_ref_primary(j, d) *
                             uhess_ref_primary(j, d * spatial_dim + dd) +
                         jp_ugrad_ref_secondary(j, d) *
                             uhess_ref_secondary(j, d * spatial_dim + dd);
      }
    }
  }

  // ∂e/∂uq * ∇ψq
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      rTp_deriv(d) += coef_uq_primary(i) * pgrad_ref_primary(i, d) +
                      coef_uq_secondary(i) * pgrad_ref_secondary(i, d);
    }
  }

  // ∂e/∂∇uq * ∇2ψq
  A2D::Vec<T, spatial_dim> deriv_hess_primary{};
  A2D::Vec<T, spatial_dim> deriv_hess_secondary{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        rTp_deriv(dd) += coef_ugrad_ref_primary(i, d) *
                             phess_ref_primary(i, d * spatial_dim + dd) +
                         coef_ugrad_ref_secondary(i, d) *
                             phess_ref_secondary(i, d * spatial_dim + dd);
      }
    }
  }

  for (int n = 0; n < max_nnodes_per_element; n++) {
    elem_dfdphi[n] += dedu_psi * wts_grad[n];
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdphi[n] += weight * rTp_deriv(d) * pts_grad[spatial_dim * n + d];
      elem_dfdphi[n] += weight * jp_nrm_ref(d) * wns_grad[spatial_dim * n + d];
    }
  }
}

/**
 * @brief The following two functions add partial derivatives of the energy with
 * respect to the level-set dof ∂e/∂φ contribution from a single quadrature
 * point
 *
 * Note: this function only works for the Galerkin difference basis
 *
 * @tparam T numeric type
 * @tparam GDBasis a GD Basis specialization
 * @tparam dim number of dof components at each dof node
 * @param weight quadrature weight
 * @param detJ determinant of the Jacobian transformation matrix a the quad pt
 * @param wts_grad derivatives of quadrature weight w.r.t. nodal phi
 * @param pts_grad derivatives of the quadrature point w.r.t. nodal phi
 * @param ugrad_ref (∇_ξ)uq
 * @param uhess_ref (∇2_ξ)uq
 * @param coef_uq ∂e/∂uq
 * @param coef_ugrad_ref ∂e/∂(∇_ξ)uq
 * @param unity_quad_wts if true, then the energy is evaluated with quadrature
 * weights being 1, one of such use cases is to compute discrete KS aggration
 * via existing energy evaluation mechanism
 * @param elem_dfphi output, element vector of ∂e/∂φ
 */
template <typename T, class GDBasis, int dim>
void add_energy_partial_deriv(
    T weight, T energy, const T wts_grad[], const T pts_grad[],
    const T ns_grad[], const A2D::Mat<T, dim, GDBasis::spatial_dim> &ugrad_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>
        &uhess_ref,
    const A2D::Vec<T, dim> &coef_uq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim> &coef_ugrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim> &coef_nrm_ref, T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int max_nnodes_per_element = GDBasis::max_nnodes_per_element;

  A2D::Vec<T, spatial_dim> deduq_ugrad{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      deduq_ugrad(d) += coef_uq(i) * ugrad_ref(i, d);
    }
  }

  A2D::Vec<T, spatial_dim> deduq_uhess{};
  for (int i = 0; i < dim; i++) {
    for (int d = 0; d < spatial_dim; d++) {
      for (int dd = 0; dd < spatial_dim; dd++) {
        deduq_uhess(dd) +=
            coef_ugrad_ref(i, d) * uhess_ref(i, d * spatial_dim + dd);
      }
    }
  }

  for (int n = 0; n < max_nnodes_per_element; n++) {
    // dedphi_{1,n}
    if (wts_grad) {
      elem_dfdphi[n] += energy * wts_grad[n] / weight;
    }

    // dedphi_{2,n} is assumed zero (dJdphi = 0)

    // dedphi_{3,n}
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdphi[n] +=
          (deduq_ugrad(d) + deduq_uhess(d)) * pts_grad[spatial_dim * n + d];
    }

    if (ns_grad) {
      for (int d = 0; d < spatial_dim; d++) {
        elem_dfdphi[n] += coef_nrm_ref(d) * ns_grad[spatial_dim * n + d];
      }
    }
  }
}

// dim = 1
// TODO: added coef_nrm_ref
template <typename T, class GDBasis>
void add_energy_partial_deriv(
    T weight, T energy, const T wts_grad[], const T pts_grad[],
    const A2D::Vec<T, GDBasis::spatial_dim> &ugrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim * GDBasis::spatial_dim> &uhess_ref,
    T coef_uq, const A2D::Vec<T, GDBasis::spatial_dim> &coef_ugrad_ref,
    T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int max_nnodes_per_element = GDBasis::max_nnodes_per_element;

  A2D::Vec<T, spatial_dim> deduq_ugrad{};
  for (int d = 0; d < spatial_dim; d++) {
    deduq_ugrad(d) += coef_uq * ugrad_ref(d);
  }

  A2D::Vec<T, spatial_dim> deduq_uhess{};
  for (int d = 0; d < spatial_dim; d++) {
    for (int dd = 0; dd < spatial_dim; dd++) {
      deduq_uhess(dd) += coef_ugrad_ref(d) * uhess_ref(d * spatial_dim + dd);
    }
  }

  for (int n = 0; n < max_nnodes_per_element; n++) {
    // dedphi_{1,n}
    if (wts_grad) {
      elem_dfdphi[n] += energy * wts_grad[n] / weight;
    }

    // dedphi_{2,n} is assumed zero (dJdphi = 0)

    // dedphi_{3,n}
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdphi[n] +=
          (deduq_ugrad(d) + deduq_uhess(d)) * pts_grad[spatial_dim * n + d];
    }
  }
}

/**
 * @brief convert pstencil to polynomial term indices
 *
 * pstencil is a Np_1d-by-Np_1d boolean matrix, given a stencil vertex
 * represented by the index coordinates (i, j), boolearn pstencil[i][j]
 * indicates whether the vertex is active or not. e.g. for a regular internal
 * stencil, entries of pstencil are all 1.
 *
 * Given a pstencil, this function determines which polynomial terms (1, x, y,
 * xy, x^2, y^2, x^2y, xy^2, etc.) to use to construct the basis function by
 * the following process:
 *   1. count number of active stencils for each column
 *   2. sort the counts in descending order
 *
 * For example, for the pstencil below,
 *
 *      0    0    1    1
 *
 *      0    1    0    1
 *
 *      1    1    1    0
 *
 *      0    1    1    0
 *
 * Number of active stencils per each column in descending order is
 *
 *      3    3    2    1
 *
 * As a result, the polynomial terms used could be represented by the
 * following table
 *
 *      1    x   x^2  x^3
 *      -----------------
 *   1 |✓    ✓    ✓    ✓
 *     |
 *   y |✓    ✓    ✓
 *     |
 *  y^2|✓    ✓
 *     |
 *  y^3|
 *
 * which are the following terms, to explicitly enumerate:
 *
 *      1    x    x^2   x^3
 *      y    xy   x^2y
 *      y^2  xy^2
 *
 */
template <int Np_1d>
std::vector<std::pair<int, int>> pstencil_to_pterms_deprecated(
    const std::vector<std::vector<bool>> &pstencil) {
  // Populate count for each column
  std::vector<int> counts(Np_1d, 0);
  for (int i = 0; i < Np_1d; i++) {
    for (int j = 0; j < Np_1d; j++) {
      if (pstencil[i][j]) {
        counts[i]++;
      }
    }
  }

  // Sort count in descending order
  std::sort(counts.begin(), counts.end(), std::greater<>());

  // Populate polynomial terms, note that x^m y^n is represented by tuple (m, n)
  std::vector<std::pair<int, int>> pterms;
  for (int m = 0; m < counts.size(); m++) {
    for (int n = 0; n < counts[m]; n++) {
      pterms.push_back({m, n});
    }
  }

  return pterms;
}

inline std::vector<std::pair<int, int>> verts_to_pterms(
    const std::vector<std::pair<int, int>> &verts, bool vertical_first = true) {
  std::map<int, int> ix_counts, iy_counts;

  for (auto &[ix, iy] : verts) {
    ix_counts[ix]++;  // how many verts for each ix index
    iy_counts[iy]++;  // how many verts for each iy index
  }

  std::vector<int> ix_counts_v, iy_counts_v;
  ix_counts_v.reserve(ix_counts.size());
  iy_counts_v.reserve(iy_counts.size());

  for (auto &[_, v] : ix_counts) {
    ix_counts_v.push_back(v);
  }

  for (auto &[_, v] : iy_counts) {
    iy_counts_v.push_back(v);
  }

  // Sort count in descending order
  auto &counts = vertical_first ? ix_counts_v : iy_counts_v;
  std::sort(counts.begin(), counts.end(), std::greater<>());

  std::vector<std::pair<int, int>> pterms;
  pterms.reserve(verts.size());
  for (int m = 0; m < counts.size(); m++) {
    for (int n = 0; n < counts[m]; n++) {
      if (vertical_first) {
        pterms.push_back({m, n});
      } else {
        pterms.push_back({n, m});
      }
    }
  }
  return pterms;
}

template <typename T, int samples_1d, class Mesh>
class GDSampler2D final : public QuadratureBase<T> {
 private:
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int samples = samples_1d * samples_1d;

 public:
  GDSampler2D(const Mesh &mesh, double t1 = 0.05, double t2 = 0.05)
      : t1(t1), t2(t2), mesh(mesh) {}

  int get_quadrature_pts(int elem, std::vector<T> &pts, std::vector<T> &_,
                         std::vector<T> &__) const {
    pts.resize(spatial_dim * samples);

    T lxy[2], xy0[2];
    for (int d = 0; d < spatial_dim; d++) {
      xy0[d] = t1;
      lxy[d] = (1.0 - t1 - t2);
    }
    int nxy[2] = {samples_1d - 1, samples_1d - 1};
    StructuredGrid2D<T> grid(nxy, lxy, xy0);

    T *pts_ptr = pts.data();
    for (int i = 0; i < samples; i++) {
      grid.get_vert_xloc(i, pts_ptr);
      pts_ptr += spatial_dim;
    }

    return samples;
  }

 private:
  double t1;  // normalized gap before the first sampling point
  double t2;  // normalized gap after the last sampling point
  const Mesh &mesh;
};

/**
 * @brief Given a mesh and dof, interpolate the field.
 *
 * Note: This is useful for sanity check and debugging.
 */
template <typename T, class Sampler, class Basis>
class Interpolator final {
  using Mesh = typename Basis::Mesh;

  static int constexpr dof_per_node = 1;
  static int constexpr data_per_node = 0;
  static int constexpr spatial_dim = Mesh::spatial_dim;

  using Physics = PhysicsBase<T, spatial_dim, data_per_node, dof_per_node>;

 public:
  Interpolator(const Mesh &mesh, const Sampler &sampler, const Basis &basis)
      : mesh(mesh), basis(basis), sampler(sampler) {}

  void to_vtk(const std::string name, T *dof = (T *)nullptr) const {
    FieldToVTK<T, spatial_dim> field_vtk(name);

    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      std::vector<T> element_dof;
      if (dof) {
        element_dof.resize(Mesh::max_nnodes_per_element);
        get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, elem, dof,
                                                       element_dof.data());
      }

      std::vector<T> element_xloc(Mesh::max_nnodes_per_element *
                                  Basis::spatial_dim);
      get_element_xloc<T, Mesh, Basis>(mesh, elem, element_xloc.data());

      std::vector<T> pts, _, __;
      int nsamples = sampler.get_quadrature_pts(elem, pts, _, __);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(elem, pts, N, Nxi);

      std::vector<T> N_xloc, Nxi_xloc;
      basis.eval_basis_grad(elem, pts, N_xloc, Nxi_xloc);

      std::vector<T> vals(nsamples);
      std::vector<T> ptx(nsamples * Basis::spatial_dim);

      for (int i = 0; i < nsamples; i++) {
        int offset_n = i * Basis::max_nnodes_per_element;
        T val = 0.0;
        if (dof) {
          interp_val_grad<T, spatial_dim, Basis::max_nnodes_per_element, 1>(
              element_dof.data(), &N[offset_n], nullptr, &val, nullptr);
        }
        vals[i] = val;
        A2D::Vec<T, Basis::spatial_dim> xloc;
        interp_val_grad<T, Basis::spatial_dim, Basis::max_nnodes_per_element,
                        Basis::spatial_dim>(element_xloc.data(),
                                            &N_xloc[offset_n], nullptr,
                                            get_ptr(xloc), nullptr);
        for (int d = 0; d < Basis::spatial_dim; d++) {
          ptx[i * Basis::spatial_dim + d] = xloc[d];
        }
      }
      field_vtk.add_scalar_field(ptx, vals);
    }
    field_vtk.write_vtk();
  }

 private:
  const Mesh &mesh;
  const Basis &basis;
  const Sampler &sampler;
  Physics physics;
};

#endif  // XCGD_ELEMENT_UTILS_H
