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

template <typename T, class Basis>
void get_element_xloc(const typename Basis::Mesh& mesh, int e,
                      T element_xloc[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    mesh.get_node_xloc(nodes[j], element_xloc);
    element_xloc += spatial_dim;
  }
}

template <typename T, int dim, class Basis>
void get_element_vars(const typename Basis::Mesh& mesh, int e, const T dof[],
                      T element_dof[]) {
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

template <typename T, int dim, class Basis>
void add_element_res(const typename Basis::Mesh& mesh, int e,
                     const T element_res[], T res[]) {
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

/**
 * The following two functions evaluate f and ∇f about a quadrature point given
 * its shape function and shape gradient evaluations N and Nxi.
 *
 * @tparam T numeric type
 * @tparam Basis Basis type
 * @tparam dim number of dof components at each dof node
 * @param elem element index
 * @param dof node dof values of size nodes_per_element * dim
 * @param N shape function values, size of nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates, size of
 * nodes_per_element * spatial_dim
 * @param vals interpolated dof
 * @param grad gradients of vals w.r.t. computational coordinates dv/dxi
 */
template <typename T, class Basis, int dim>
void interp_val_grad(int elem, const T dof[], const T N[], const T Nxi[],
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
void interp_val_grad(int elem, const T dof[], const T N[], const T Nxi[],
                     T* val, A2D::Vec<T, Basis::spatial_dim>* grad) {
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
 * @param elem element index
 * @param N shape function values, size of nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ, size
 * of nodes_per_element * spatial_dim
 * @param coef_vals ∂e/∂uq
 * @param coef_grad ∂e/∂((∇_ξ)uq)
 * @param elem_res de/du
 */
template <typename T, class Basis, int dim>
void add_grad(int elem, const T N[], const T Nxi[],
              const A2D::Vec<T, dim>& coef_vals,
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
void add_grad(int elem, const T N[], const T Nxi[], const T& coef_val,
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
 * @param elem element index
 * @param N shape function values, size of nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ, size
 * of nodes_per_element * spatial_dim
 * @param coef_vals ∂^2e/∂(uq)^2
 * @param coef_hess ∂^2e/∂((∇_ξ)uq)^2
 * @param elem_jac d^2e/du^2
 */
template <typename T, class Basis, int dim>
void add_matrix(int elem, const T N[], const T Nxi[],
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
    int elem, const T N[], const T Nxi[], const T& coef_val,
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
 * @brief Assemble psi^T * dR/dphi for an element, this class only works for
 * GDBasis
 *
 * @tparam T numeric type
 * @tparam GDBasis a GD Basis specialization
 * @tparam dim number of dof components at each dof node
 * @param elem element index
 * @param N shape function values, size: nodes_per_element
 * @param Nxi shape function gradients w.r.t. computational coordinates ξ, size:
 * nodes_per_element * spatial_dim
 * @param Nxixi shape function Hessians, concatenation of (∇_xi_xi N_q,
 * ∇_xi_eta N_q, ∇_eta_xi N_q, ∇_eta_eta N_q)
 * @param dwdphi ∂q/∂phi, size: nodes_per_element
 * @param dxidphi ∂xi/∂phi, size: spatial_dim * nodes_per_element
 * @param coef_uq ∂e/∂uq
 * @param jp_uq ∂2e/∂uq2 * psiq
 * @param coef_ugrad_ref ∂e/∂(∇_ξ)uq
 * @param jp_ugrad_ref ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq
 * @param elem_dfdx output, element vector of psi^T * dR/dphi
 */
template <typename T, class GDBasis, int dim>
void add_jac_adj_product(
    int elem, const T elem_xloc[], const T elem_dof[], const T Nxixi[],
    const T dwdphi[], const T dxidphi[], T weight, T detJ,
    const A2D::Mat<T, GDBasis::spatial_dim, GDBasis::spatial_dim>& Jb,
    const A2D::Vec<T, dim>& psiq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& pgrad_ref,
    const A2D::Vec<T, dim>& coef_uq, const A2D::Vec<T, dim>& jp_uq,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& coef_ugrad_ref,
    const A2D::Mat<T, dim, GDBasis::spatial_dim>& jp_ugrad_ref, T elem_dfdx[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");
  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int nodes_per_element = GDBasis::nodes_per_element;

  T dot = 0.0;  // ∂e/∂uq * psiq + ∂e/∂(∇_ξ)uq * (∇_ξ)psiq
  for (int i = 0; i < dim; i++) {
    dot += coef_uq(i) * psiq(i);
    for (int d = 0; d < spatial_dim; d++) {
      dot += coef_ugrad_ref(i, d) * pgrad_ref(i, d);
    }
  }

  A2D::Vec<T, spatial_dim> vec{};
  for (int j = 0; j < nodes_per_element; j++) {
    int Nxixi_offset = j * spatial_dim * spatial_dim;
    int xloc_offset = j * spatial_dim;
    for (int ii = 0; ii < spatial_dim; ii++) {
      for (int jj = 0; jj < spatial_dim; jj++) {
        for (int kk = 0; kk < spatial_dim; kk++) {
          vec(ii) += Nxixi[Nxixi_offset + ii * spatial_dim + jj] * Jb(kk, jj) *
                     elem_xloc[xloc_offset + kk];
        }
      }
    }
  }

  A2D::Vec<T, spatial_dim> vec2{};
  for (int j = 0; j < nodes_per_element; j++) {
    int Nxixi_offset = j * spatial_dim * spatial_dim;
    int dof_offset = j * dim;
    for (int k = 0; k < dim; k++) {
      for (int ii = 0; ii < spatial_dim; ii++) {
        for (int jj = 0; jj < spatial_dim; jj++) {
          vec2(jj) += elem_dof[dof_offset + k] *
                      Nxixi[Nxixi_offset + jj * spatial_dim + ii] *
                      jp_ugrad_ref(k, jj);
        }
      }
    }
  }

  for (int j = 0; j < nodes_per_element; j++) {
    // First term
    elem_dfdx[j] += detJ * dot * dwdphi[j];

    // Second term
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdx[j] += weight * dot * dxidphi[spatial_dim * j + d] * vec(d);
    }

    // Third term
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdx[j] += weight * detJ * dxidphi[spatial_dim * j + d] * vec2(d);
    }
  }
}

// dim == 1
template <typename T, class GDBasis>
void add_jac_adj_product(
    int elem, const T elem_xloc[], const T elem_dof[], const T Nxixi[],
    const T dwdphi[], const T dxidphi[], T weight, T detJ,
    const A2D::Mat<T, GDBasis::spatial_dim, GDBasis::spatial_dim>& Jb, T psiq,
    const A2D::Vec<T, GDBasis::spatial_dim>& pgrad_ref, T coef_uq, T jp_uq,
    const A2D::Vec<T, GDBasis::spatial_dim>& coef_ugrad_ref,
    const A2D::Vec<T, GDBasis::spatial_dim>& jp_ugrad_ref, T elem_dfdx[]) {
  static_assert(GDBasis::is_gd_basis, "This method only works with GD Basis");
  static constexpr int spatial_dim = GDBasis::spatial_dim;
  static constexpr int nodes_per_element = GDBasis::nodes_per_element;

  T dot = 0.0;  // ∂e/∂uq * psiq + ∂e/∂(∇_ξ)uq * (∇_ξ)psiq
  dot += coef_uq * psiq;
  for (int d = 0; d < spatial_dim; d++) {
    dot += coef_ugrad_ref(d) * pgrad_ref(d);
  }

  A2D::Vec<T, spatial_dim> vec{};
  for (int j = 0; j < nodes_per_element; j++) {
    int Nxixi_offset = j * spatial_dim * spatial_dim;
    int xloc_offset = j * spatial_dim;
    for (int ii = 0; ii < spatial_dim; ii++) {
      for (int jj = 0; jj < spatial_dim; jj++) {
        for (int kk = 0; kk < spatial_dim; kk++) {
          vec(ii) += Nxixi[Nxixi_offset + ii * spatial_dim + jj] * Jb(kk, jj) *
                     elem_xloc[xloc_offset + kk];
        }
      }
    }
  }

  A2D::Vec<T, spatial_dim> vec2{};
  for (int j = 0; j < nodes_per_element; j++) {
    int Nxixi_offset = j * spatial_dim * spatial_dim;
    for (int ii = 0; ii < spatial_dim; ii++) {
      for (int jj = 0; jj < spatial_dim; jj++) {
        vec2(jj) += elem_dof[j] * Nxixi[Nxixi_offset + jj * spatial_dim + ii] *
                    jp_ugrad_ref(jj);
      }
    }
  }

  for (int j = 0; j < nodes_per_element; j++) {
    // First term
    elem_dfdx[j] += detJ * dot * dwdphi[j];

    // Second term
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdx[j] += weight * dot * dxidphi[spatial_dim * j + d] * vec(d);
    }

    // Third term
    for (int d = 0; d < spatial_dim; d++) {
      elem_dfdx[j] += weight * detJ * dxidphi[spatial_dim * j + d] * vec2(d);
    }
  }
}

template <typename T, int Np_1d, int samples_1d>
class GDSampler2D final : public QuadratureBase<T> {
 private:
  using Mesh =
      GDMesh2D<T, Np_1d>;  // Np_1d and samples_1d can be different, which is
                           // the whole point of using this Sampler clas
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
      lxy[d] = 0.95 * (xymax[d] - xymin[d]);
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

  void to_vtk(const std::string name, T* dof) const {
    FieldToVTK<T, spatial_dim> field_vtk(name);

    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      std::vector<T> element_dof(Mesh::nodes_per_element);
      get_element_vars<T, dof_per_node, Basis>(mesh, elem, dof,
                                               element_dof.data());

      std::vector<T> element_xloc(Mesh::nodes_per_element * Basis::spatial_dim);
      get_element_xloc<T, Basis>(mesh, elem, element_xloc.data());

      std::vector<T> pts, wts;
      int nsamples = sampler.get_quadrature_pts(elem, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(elem, pts, N, Nxi);

      std::vector<T> vals(nsamples);
      std::vector<T> ptx(nsamples * Basis::spatial_dim);

      for (int i = 0; i < nsamples; i++) {
        int offset_n = i * Basis::nodes_per_element;
        T val = 0.0;
        interp_val_grad<T, Basis>(elem, element_dof.data(), &N[offset_n],
                                  nullptr, &val, nullptr);
        vals[i] = val;
        A2D::Vec<T, Basis::spatial_dim> xloc;
        interp_val_grad<T, Basis, Basis::spatial_dim>(
            elem, element_xloc.data(), &N[offset_n], nullptr, &xloc, nullptr);
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