#ifndef XCGD_ELEMENT_UTILS_H
#define XCGD_ELEMENT_UTILS_H

#include <array>
#include <vector>

#include "a2dcore.h"
#include "element_commons.h"
#include "gd_mesh.h"
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
inline void transform(const A2D::Mat<T, spatial_dim, spatial_dim>& J,
                      const A2D::Vec<T, spatial_dim>& grad_ref,
                      A2D::Vec<T, spatial_dim>& grad) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatVecMult<A2D::MatOp::TRANSPOSE>(Jinv, grad_ref, grad);
}
template <typename T, int dim, int spatial_dim>
inline void transform(const A2D::Mat<T, spatial_dim, spatial_dim>& J,
                      const A2D::Mat<T, dim, spatial_dim>& grad_ref,
                      A2D::Mat<T, dim, spatial_dim>& grad) {
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
inline void rtransform(const A2D::Mat<T, spatial_dim, spatial_dim>& J,
                       const A2D::Vec<T, spatial_dim>& grad,
                       A2D::Vec<T, spatial_dim>& grad_ref) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatVecMult(Jinv, grad, grad_ref);
}
template <typename T, int dim, int spatial_dim>
inline void rtransform(const A2D::Mat<T, spatial_dim, spatial_dim>& J,
                       const A2D::Mat<T, dim, spatial_dim>& grad,
                       A2D::Mat<T, dim, spatial_dim>& grad_ref) {
  A2D::Mat<T, spatial_dim, spatial_dim> Jinv;
  A2D::MatInv(J, Jinv);
  A2D::MatMatMult<A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(grad, Jinv,
                                                             grad_ref);
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
    const A2D::Mat<T, spatial_dim, spatial_dim>& J,
    const A2D::Mat<T, dim * spatial_dim, dim * spatial_dim>& hess,
    A2D::Mat<T, dim * spatial_dim, dim * spatial_dim>& hess_ref) {
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
void get_element_xloc(const Mesh& mesh, int e, T element_xloc[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    mesh.get_node_xloc(nodes[j], element_xloc);
    element_xloc += spatial_dim;
  }
}

template <typename T, int dim, class Mesh, class Basis>
void get_element_vars(const Mesh& mesh, int e, const T dof[], T element_dof[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    for (int k = 0; k < dim; k++, element_dof++) {
      element_dof[0] = dof[dim * nodes[j] + k];
    }
  }
}

template <typename T, int dim, class Mesh, class Basis>
void add_element_res(const Mesh& mesh, int e, const T element_res[], T res[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    for (int k = 0; k < dim; k++, element_res++) {
      res[dim * nodes[j] + k] += element_res[0];
    }
  }
}

template <typename T, class Mesh, class Basis>
void add_element_dfdx(const Mesh& mesh, int e, const T element_dfdx[],
                      T dfdx[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);

  for (int j = 0; j < nodes_per_element; j++) {
    dfdx[nodes[j]] += element_dfdx[j];
  }
}

template <typename T, class Mesh, class Basis>
void add_element_dfdphi(const Mesh& lsf_mesh, int c, const T element_dfdphi[],
                        T dfdphi[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int nodes[nodes_per_element];
  lsf_mesh.get_elem_dof_nodes(c, nodes);

  for (int j = 0; j < nodes_per_element; j++) {
    dfdphi[nodes[j]] += element_dfdphi[j];
  }
}

/**
 * The following two functions evaluate u and ∇u at a quadrature point given the
 * shape function and shape gradient evaluations N and Nxi at this quadrature
 * point.
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param dof node dof values of size nodes_per_element * dim
 * @param N shape function values, size of nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates, size of
 * nodes_per_element * spatial_dim
 * @param vals interpolated dof
 * @param grad gradients of vals w.r.t. computational coordinates dv/dxi
 */
template <typename T, class Basis, int dim>
void interp_val_grad(const T dof[], const T N[], const T Nxi[],
                     A2D::Vec<T, dim>* vals,
                     A2D::Mat<T, dim, Basis::spatial_dim>* grad) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

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

  for (int i = 0; i < nodes_per_element; i++) {
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

// dim == 1
template <typename T, class Basis>
void interp_val_grad(const T* dof, const T* N, const T* Nxi, T* val,
                     A2D::Vec<T, Basis::spatial_dim>* grad) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  if (val) {
    *val = 0.0;
  }

  if (grad) {
    for (int k = 0; k < spatial_dim; k++) {
      (*grad)[k] = 0.0;
    }
  }

  for (int i = 0; i < nodes_per_element; i++) {
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

/**
 * The following two functions evaluate u and ∇2u at a quadrature point given
 * the shape function and shape gradient evaluations N and Nxi at this
 * quadrature point.
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param dof node dof values of size nodes_per_element * dim
 * @param Nxixi shape function Hessians, concatenation of (∂2/∂ξξ, ∂2/∂ξη,
 * ∂2/∂ηξ, ∂2/∂ηη) N_q
 * @param hess ∇2u, hess(i, :) = (∂2u[i]/∂ξξ, ∂2u[i]/∂ξη, ∂2u[i]/∂ηξ,
 * ∂2u[i]/∂ηη)
 */
template <typename T, class Basis, int dim>
void interp_hess(
    const T* dof, const T* Nxixi,
    A2D::Mat<T, dim, Basis::spatial_dim * Basis::spatial_dim>& hess) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  hess.zero();

  for (int j = 0; j < dim; j++) {
    for (int i = 0; i < nodes_per_element; i++) {
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
void interp_hess(const T* dof, const T* Nxixi,
                 A2D::Vec<T, Basis::spatial_dim * Basis::spatial_dim>& hess) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  hess.zero();

  for (int i = 0; i < nodes_per_element; i++) {
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
 * @param N shape function values, size of nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ,
 * size of nodes_per_element * spatial_dim
 * @param coef_vals ∂e/∂uq
 * @param coef_grad ∂e/∂((∇_ξ)uq)
 * @param elem_res de/du
 */
template <typename T, class Basis, int dim>
void add_grad(const T N[], const T Nxi[], const A2D::Vec<T, dim>& coef_vals,
              A2D::Mat<T, dim, Basis::spatial_dim>& coef_grad, T elem_res[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  for (int i = 0; i < nodes_per_element; i++) {
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
void add_grad(const T N[], const T Nxi[], const T& coef_val,
              const A2D::Vec<T, Basis::spatial_dim>& coef_grad, T elem_res[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  for (int i = 0; i < nodes_per_element; i++) {
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
 * @param N shape function values, size of nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ,
 * size of nodes_per_element * spatial_dim
 * @param coef_vals ∂^2e/∂(uq)^2
 * @param coef_hess ∂^2e/∂((∇_ξ)uq)^2
 * @param elem_jac d^2e/du^2
 */
template <typename T, class Basis, int dim>
void add_matrix(const T N[], const T Nxi[],
                const A2D::Mat<T, dim, dim>& coef_vals,
                const A2D::Mat<T, dim * Basis::spatial_dim,
                               dim * Basis::spatial_dim>& coef_hess,
                T elem_jac[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  constexpr int dof_per_element = dim * nodes_per_element;

  for (int i = 0; i < nodes_per_element; i++) {
    T ni = N[i];
    std::vector<T> nxi(&Nxi[spatial_dim * i],
                       &Nxi[spatial_dim * i] + spatial_dim);

    for (int j = 0; j < nodes_per_element; j++) {
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
          elem_jac[col + row * dof_per_element] +=
              val + coef_vals(ii, jj) * ni * nj;
        }
      }
    }
  }
}

// dim == 1
template <typename T, class Basis>
void add_matrix(
    const T N[], const T Nxi[], const T& coef_val,
    const A2D::Mat<T, Basis::spatial_dim, Basis::spatial_dim>& coef_grad,
    T elem_jac[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  constexpr int dof_per_element = nodes_per_element;

  for (int i = 0; i < nodes_per_element; i++) {
    T ni = N[i];
    std::vector<T> nxi(&Nxi[spatial_dim * i],
                       &Nxi[spatial_dim * i] + spatial_dim);

    for (int j = 0; j < nodes_per_element; j++) {
      T nj = N[j];
      std::vector<T> nxj(&Nxi[spatial_dim * j],
                         &Nxi[spatial_dim * j] + spatial_dim);

      T val = 0.0;
      for (int kk = 0; kk < spatial_dim; kk++) {
        for (int ll = 0; ll < spatial_dim; ll++) {
          val += coef_grad(kk, ll) * nxi[kk] * nxj[ll];
        }
      }
      elem_jac[j + i * dof_per_element] += val + coef_val * ni * nj;
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
void det_deriv(const T* elem_xloc, const T* Nxixi,
               const A2D::Mat<T, Basis::spatial_dim, Basis::spatial_dim>& J,
               A2D::Vec<T, Basis::spatial_dim>& grad) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  // Get derivatives of detJ w.r.t. J
  T detJ, detJb = 1.0;
  A2D::ADObj<T&> detJ_obj(detJ, detJb);
  A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);
  auto stack = A2D::MakeStack(A2D::MatDet(J_obj, detJ_obj));
  stack.reverse();

  auto Jb = J_obj.bvalue();

  grad.zero();

  // sum_j ∇^2_ξ N_j * Jbar^T * xj
  for (int j = 0; j < nodes_per_element; j++) {
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
void add_jac_adj_product(const T N[], const T& x_val, T elem_dfdx[]) {
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  for (int i = 0; i < nodes_per_element; i++) {
    elem_dfdx[i] += x_val * N[i];
  }
}

/**
 * @brief The following two functions add ψ^T * dR/dφ contribution from a single
 * quadrature point
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
 * @param psiq adjoint variable at the quadrature point
 * @param ugrad_ref (∇_ξ)uq
 * @param pgrad_ref (∇_ξ)ψq
 * @param uhess_ref (∇2_ξ)uq
 * @param phess_ref (∇2_ξ)ψq
 * @param coef_uq ∂e/∂uq
 * @param coef_ugrad_ref ∂e/∂(∇_ξ)uq
 * @param jp_uq ∂2e/∂uq2 * ψq
 * @param jp_ugrad_ref ∂2e/∂(∇_ξ)uq2 * ψq
 * @param elem_dfdphi output, element vector of ψ^T * dR/dφ
 */
template <typename T, class GDBasis, int dim>
void add_jac_adj_product(
    T weight, T detJ, const T wts_grad[], const T pts_grad[],
    const A2D::Vec<T, dim>& psiq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& ugrad_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& pgrad_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>&
        uhess_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim * GDBasis::spatial_dim>&
        phess_ref,
    const A2D::Vec<T, dim>& coef_uq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& coef_ugrad_ref,
    const A2D::Vec<T, dim>& jp_uq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& jp_ugrad_ref,
    T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int nodes_per_element = GDBasis::nodes_per_element;

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

  T wdetJ = weight * detJ;

  for (int n = 0; n < nodes_per_element; n++) {
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

// dim == 1
template <typename T, class GDBasis>
void add_jac_adj_product(
    T weight, T detJ, const T wts_grad[], const T pts_grad[], T psiq,
    const A2D::Vec<T, GDBasis::spatial_dim>& ugrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim>& pgrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim * GDBasis::spatial_dim>& uhess_ref,
    const A2D::Vec<T, GDBasis::spatial_dim * GDBasis::spatial_dim>& phess_ref,
    T coef_uq, const A2D::Vec<T, GDBasis::spatial_dim>& coef_ugrad_ref, T jp_uq,
    const A2D::Vec<T, GDBasis::spatial_dim>& jp_ugrad_ref, T elem_dfdphi[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");

  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int nodes_per_element = GDBasis::nodes_per_element;

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

  T wdetJ = weight * detJ;

  for (int n = 0; n < nodes_per_element; n++) {
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

template <typename T, int samples_1d, class Mesh>
class GDSampler2D final : public QuadratureBase<T> {
 private:
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int samples = samples_1d * samples_1d;

 public:
  GDSampler2D(const Mesh& mesh) : mesh(mesh) {}

  int get_quadrature_pts(int elem, std::vector<T>& pts,
                         std::vector<T>& _) const {
    pts.resize(spatial_dim * samples);

    T xymin[2], xymax[2];
    get_computational_coordinates_limits(mesh, elem, xymin, xymax);

    T lxy[2], xy0[2];
    for (int d = 0; d < spatial_dim; d++) {
      xy0[d] = xymin[d] + 0.05 * (xymax[d] - xymin[d]);
      lxy[d] = 0.9 * (xymax[d] - xymin[d]);
    }
    int nxy[2] = {samples_1d - 1, samples_1d - 1};
    StructuredGrid2D<T> grid(nxy, lxy, xy0);

    T* pts_ptr = pts.data();
    for (int i = 0; i < samples; i++) {
      grid.get_vert_xloc(i, pts_ptr);
      pts_ptr += spatial_dim;
    }

    return samples;
  }

 private:
  const Mesh& mesh;
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
  Interpolator(const Mesh& mesh, const Sampler& sampler, const Basis& basis)
      : mesh(mesh), basis(basis), sampler(sampler) {}

  void to_vtk(const std::string name, T* dof = (T*)nullptr) const {
    FieldToVTK<T, spatial_dim> field_vtk(name);

    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      std::vector<T> element_dof;
      if (dof) {
        element_dof.resize(Mesh::nodes_per_element);
        get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, elem, dof,
                                                       element_dof.data());
      }

      std::vector<T> element_xloc(Mesh::nodes_per_element * Basis::spatial_dim);
      get_element_xloc<T, Mesh, Basis>(mesh, elem, element_xloc.data());

      std::vector<T> pts, wts;
      int nsamples = sampler.get_quadrature_pts(elem, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(elem, pts, N, Nxi);

      std::vector<T> N_xloc, Nxi_xloc;
      basis.eval_basis_grad(elem, pts, N_xloc, Nxi_xloc);

      std::vector<T> vals(nsamples);
      std::vector<T> ptx(nsamples * Basis::spatial_dim);

      for (int i = 0; i < nsamples; i++) {
        int offset_n = i * Basis::nodes_per_element;
        T val = 0.0;
        if (dof) {
          interp_val_grad<T, Basis>(element_dof.data(), &N[offset_n], nullptr,
                                    &val, nullptr);
        }
        vals[i] = val;
        A2D::Vec<T, Basis::spatial_dim> xloc;
        interp_val_grad<T, Basis, Basis::spatial_dim>(
            element_xloc.data(), &N_xloc[offset_n], nullptr, &xloc, nullptr);
        for (int d = 0; d < Basis::spatial_dim; d++) {
          ptx[i * Basis::spatial_dim + d] = xloc[d];
        }
      }
      field_vtk.add_scalar_field(ptx, vals);
    }
    field_vtk.write_vtk();
  }

 private:
  const Mesh& mesh;
  const Basis& basis;
  const Sampler& sampler;
  Physics physics;
};

#endif  // XCGD_ELEMENT_UTILS_H