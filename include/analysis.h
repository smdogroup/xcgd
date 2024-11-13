#ifndef XCGD_ANALYSIS_H
#define XCGD_ANALYSIS_H

#include <vector>

#include "a2dcore.h"
#include "ad/a2dvecnorm.h"
#include "elements/element_utils.h"
#include "physics/volume.h"
#include "sparse_utils/sparse_matrix.h"
#include "utils/linalg.h"
#include "utils/misc.h"

template <typename T, class Mesh, class Quadrature, class Basis, class Physics>
class GalerkinAnalysis final {
 public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static constexpr int max_dof_per_element =
      dof_per_node * max_nnodes_per_element;

  GalerkinAnalysis(const Mesh& mesh, const Quadrature& quadrature,
                   const Basis& basis, const Physics& physics)
      : mesh(mesh), quadrature(quadrature), basis(basis), physics(physics) {}

  T energy(const T x[], const T dof[]) const {
    T total_energy = 0.0;
    T xq = 0.0;
    std::vector<T> element_x(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Mesh, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &vals, &grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Add the energy contributions
        if (x) {
          interp_val_grad<T, Basis>(element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        total_energy +=
            physics.energy(wts[j], xq, xloc, nrm_ref, J, vals, grad);
      }
    }

    return total_energy;
  }

  void residual(const T x[], const T dof[], T res[]) const {
    T xq = 0.0;
    std::vector<T> element_x(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Mesh, Basis>(mesh, i, x, element_x.data());
      }

      // Create the element residual
      T element_res[max_dof_per_element];
      for (int j = 0; j < max_dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &vals, &grad_ref);
        if (x) {
          interp_val_grad<T, Basis>(element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals{};
        typename Physics::grad_t coef_grad{}, coef_grad_ref{};
        physics.residual(wts[j], xq, xloc, nrm_ref, J, vals, grad, coef_vals,
                         coef_grad);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad, coef_grad_ref);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals,
                           coef_grad_ref, element_res);
      }

      add_element_res<T, dof_per_node, Mesh, Basis>(mesh, i, element_res, res);
    }
  }

  void jacobian_product(const T x[], const T dof[], const T direct[],
                        T res[]) const {
    T xq = 0.0;
    std::vector<T> element_x(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Mesh, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      // Get the element directions for the Jacobian-vector product
      T element_direct[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, direct,
                                                     element_direct);

      // Create the element residual
      T element_res[max_dof_per_element];
      for (int j = 0; j < max_dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &vals, &grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        typename Physics::dof_t direct_vals{};
        typename Physics::grad_t direct_grad{}, direct_grad_ref{};
        interp_val_grad<T, Basis>(element_direct, &N[offset_n],
                                  &Nxi[offset_nxi], &direct_vals,
                                  &direct_grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, direct_grad_ref, direct_grad);

        if (x) {
          interp_val_grad<T, Basis>(element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals{};
        typename Physics::grad_t coef_grad{}, coef_grad_ref{};
        physics.jacobian_product(wts[j], xq, xloc, nrm_ref, J, vals, grad,
                                 direct_vals, direct_grad, coef_vals,
                                 coef_grad);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad, coef_grad_ref);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals,
                           coef_grad_ref, element_res);
      }

      add_element_res<T, dof_per_node, Mesh, Basis>(mesh, i, element_res, res);
    }
  }

  /*
    Evaluate the matrix vector product dR/dx * psi, where x are the nodal data,
    psi are the adjoint variables
  */
  void jacobian_adjoint_product(const T x[], const T dof[], const T psi[],
                                T dfdx[]) const {
    T xq = 0.0;
    std::vector<T> element_x(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Mesh, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      // Get the element psi for the Jacobian-vector product
      T element_psi[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, psi, element_psi);

      // Create the element residual
      T element_dfdx[max_nnodes_per_element];
      for (int j = 0; j < max_nnodes_per_element; j++) {
        element_dfdx[j] = 0.0;
      }

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &vals, &grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the derivative of the psi in the computational
        // coordinates
        typename Physics::dof_t psi_vals{};
        typename Physics::grad_t psi_grad{}, psi_grad_ref{};
        interp_val_grad<T, Basis>(element_psi, &N[offset_n], &Nxi[offset_nxi],
                                  &psi_vals, &psi_grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, psi_grad_ref, psi_grad);

        if (x) {
          interp_val_grad<T, Basis>(element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Evaluate the residuals at the quadrature points
        typename Physics::dv_t dv_val{};
        physics.adjoint_jacobian_product(wts[j], xq, xloc, nrm_ref, J, vals,
                                         grad, psi_vals, psi_grad, dv_val);

        add_jac_adj_product<T, Basis>(&N[offset_n], dv_val, element_dfdx);
      }

      add_element_dfdx<T, Mesh, Basis>(mesh, i, element_dfdx, dfdx);
    }
  }

  void jacobian(const T x[], const T dof[],
                GalerkinBSRMat<T, dof_per_node>* mat,
                bool zero_jac = true) const {
    if (zero_jac) {
      mat->zero();
    }

    T xq = 0.0;
    std::vector<T> element_x(max_nnodes_per_element);

    std::vector<T> element_jac(
        mesh.get_num_elements() * max_dof_per_element * max_dof_per_element,
        T(0.0));

#pragma omp parallel for
    for (int i = 0; i < mesh.get_num_elements(); i++) {
      int element_jac_offset = i * max_dof_per_element * max_dof_per_element;

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Mesh, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad_ref{}, grad{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &vals, &grad_ref);
        if (x) {
          interp_val_grad<T, Basis>(element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the residuals at the quadrature points
        typename Physics::jac_t jac_vals{};
        typename Physics::jac_mixed_t jac_mixed{}, jac_mixed_ref{};
        typename Physics::jac_grad_t jac_grad{}, jac_grad_ref{};

        physics.jacobian(wts[j], xq, xloc, nrm_ref, J, vals, grad, jac_vals,
                         jac_mixed, jac_grad);

        // Transform hessian from physical coordinates back to ref coordinates
        jtransform<T, dof_per_node, spatial_dim>(J, jac_grad, jac_grad_ref);
        mtransform(J, jac_mixed, jac_mixed_ref);

        // Add the contributions to the element Jacobian
        add_matrix<T, Basis>(&N[offset_n], &Nxi[offset_nxi], jac_vals,
                             jac_mixed_ref, jac_grad_ref,
                             element_jac.data() + element_jac_offset);
      }
    }

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      int element_jac_offset = i * max_dof_per_element * max_dof_per_element;
      mat->add_block_values(i, mesh, element_jac.data() + element_jac_offset);
    }
  }

  /*
    Evaluate the matrix vector product dR/dphi * psi, where phi are the
    LSF dof, psi are the adjoint variables

    Note: This only works for Galerkin Difference method combined with the
    level-set mesh
  */
  void LSF_jacobian_adjoint_product(const T dof[], const T psi[],
                                    T dfdphi[]) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    static_assert(Mesh::is_cut_mesh,
                  "This method requires a level-set-cut mesh");

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element states and adjoints
      T element_dof[max_dof_per_element], element_psi[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, psi, element_psi);

      // Create the element dfdphi
      std::vector<T> element_dfdphi(max_nnodes_per_element, 0.0);

      std::vector<T> pts, wts, ns, pts_grad, wts_grad;
      int num_quad_pts = quadrature.get_quadrature_pts_grad(i, pts, wts, ns,
                                                            pts_grad, wts_grad);

      std::vector<T> N, Nxi, Nxixi;
      basis.eval_basis_grad(i, pts, N, Nxi, Nxixi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;
        int offset_nxixi =
            j * max_nnodes_per_element * spatial_dim * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t uq{}, psiq{};           // uq, psiq
        typename Physics::grad_t ugrad{}, ugrad_ref{};  // (∇_x)uq, (∇_ξ)uq
        typename Physics::grad_t pgrad{}, pgrad_ref{};  // (∇_x)psiq, (∇_ξ)psiq
        typename Physics::hess_t uhess_ref{};           //(∇2_ξ)uq
        typename Physics::hess_t phess_ref{};           //(∇2_ξ)psiq

        // Interpolate the quantities at the quadrature point
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &uq, &ugrad_ref);
        interp_val_grad<T, Basis>(element_psi, &N[offset_n], &Nxi[offset_nxi],
                                  &psiq, &pgrad_ref);
        interp_hess<T, Basis>(element_dof, &Nxixi[offset_nxixi], uhess_ref);
        interp_hess<T, Basis>(element_psi, &Nxixi[offset_nxixi], phess_ref);

        transform(J, ugrad_ref, ugrad);
        transform(J, pgrad_ref, pgrad);

        typename Physics::dof_t coef_uq{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad{};  // ∂e/∂(∇_x)uq
        typename Physics::dof_t jp_uq{};        // ∂2e/∂uq2 * psiq
        typename Physics::grad_t jp_ugrad{};  // ∂2e/∂(∇_x)uq2 * (∇_x)psiq

        T detJ;
        A2D::MatDet(J, detJ);

        physics.residual(1.0 / detJ, 0.0, xloc, nrm_ref, J, uq, ugrad, coef_uq,
                         coef_ugrad);
        physics.jacobian_product(1.0 / detJ, 0.0, xloc, nrm_ref, J, uq, ugrad,
                                 psiq, pgrad, jp_uq, jp_ugrad);

        typename Physics::grad_t coef_ugrad_ref{};  // ∂e/∂(∇_ξ)uq
        typename Physics::grad_t jp_ugrad_ref{};  // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq

        // Transform gradient from physical coordinates back to ref
        // coordinates
        rtransform(J, coef_ugrad, coef_ugrad_ref);
        rtransform(J, jp_ugrad, jp_ugrad_ref);

        int offset_wts = j * max_nnodes_per_element;
        int offset_pts = j * max_nnodes_per_element * spatial_dim;

        add_jac_adj_product<T, Basis>(
            wts[j], detJ, &wts_grad[offset_wts], &pts_grad[offset_pts], psiq,
            ugrad_ref, pgrad_ref, uhess_ref, phess_ref, coef_uq, coef_ugrad_ref,
            jp_uq, jp_ugrad_ref, element_dfdphi.data());
      }

      const auto& lsf_mesh = mesh.get_lsf_mesh();
      int c = mesh.get_elem_cell(i);
      add_element_dfdphi<T, decltype(lsf_mesh), Basis>(
          lsf_mesh, c, element_dfdphi.data(), dfdphi);
    }
  }

  /*
    Evaluate the derivatives of the volume defined by the LSF with respect to
    the LSF dofs

   * Note: This only works for Galerkin Difference method combined with the
   * level-set mesh
  */
  void LSF_volume_derivatives(T dfdphi[]) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    static_assert(Mesh::is_cut_mesh,
                  "This method requires a level-set-cut mesh");

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Create the element dfdphi
      std::vector<T> element_dfdphi(max_nnodes_per_element, 0.0);

      std::vector<T> pts, wts, ns, pts_grad, wts_grad;
      int num_quad_pts = quadrature.get_quadrature_pts_grad(i, pts, wts, ns,
                                                            pts_grad, wts_grad);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, nullptr,
                                               &Nxi[offset_nxi], nullptr, &J);

        T detJ;
        A2D::MatDet(J, detJ);
        int offset_wts = j * max_nnodes_per_element;
        for (int n = 0; n < max_nnodes_per_element; n++) {
          element_dfdphi[n] += wts_grad[offset_wts + n] * detJ;
        }
      }

      const auto& lsf_mesh = mesh.get_lsf_mesh();
      int c = mesh.get_elem_cell(i);
      add_element_dfdphi<T, decltype(lsf_mesh), Basis>(
          lsf_mesh, c, element_dfdphi.data(), dfdphi);
    }
  }

  // Interpolate dof on quadrature points, intended to be used for
  // debug or post-process
  std::pair<std::vector<T>, std::vector<T>> interpolate_dof(
      const T dof[]) const {
    std::vector<T> xloc_q, dof_q;

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               nullptr, &xloc, nullptr);

        typename Physics::dof_t vals{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], nullptr, &vals,
                                  nullptr);

        for (int d = 0; d < spatial_dim; d++) {
          xloc_q.push_back(xloc(d));
        }

        if constexpr (Physics::dof_per_node > 1) {
          for (int k = 0; k < Physics::dof_per_node; k++) {
            dof_q.push_back(vals(k));
          }
        } else {
          dof_q.push_back(vals);
        }
      }
    }
    return {xloc_q, dof_q};
  }

  // Evaluate the energy on quadrature points, intended to be used for
  // debug or post-process
  std::pair<std::vector<T>, std::vector<T>> interpolate_energy(
      const T dof[]) const {
    std::vector<T> xloc_q, energy_q;

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Mesh, Basis>(mesh, i, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(element_xloc, &N[offset_n],
                                               &Nxi[offset_nxi], &xloc, &J);

        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(element_dof, &N[offset_n], &Nxi[offset_nxi],
                                  &vals, &grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        for (int d = 0; d < spatial_dim; d++) {
          xloc_q.push_back(xloc(d));
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        energy_q.push_back(
            physics.energy(1.0, 0.0, xloc, nrm_ref, J, vals, grad));
      }
    }
    return {xloc_q, energy_q};
  }

 private:
  const Mesh& mesh;
  const Quadrature& quadrature;
  const Basis& basis;
  const Physics& physics;
};

#endif  // XCGD_ANALYSIS_H
