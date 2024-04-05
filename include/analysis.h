#ifndef XCGD_ANALYSIS_H
#define XCGD_ANALYSIS_H

#include <vector>

#include "a2dcore.h"
#include "elements/element_utils.h"
#include "sparse_utils/sparse_matrix.h"
#include "utils/linalg.h"

template <typename T, class Mesh, class Quadrature, class Basis, class Physics>
class GalerkinAnalysis final {
 public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int nodes_per_element = Basis::nodes_per_element;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static constexpr int dof_per_element = dof_per_node * nodes_per_element;

  GalerkinAnalysis(const Mesh& mesh, const Quadrature& quadrature,
                   const Basis& basis, const Physics& physics)
      : mesh(mesh), quadrature(quadrature), basis(basis), physics(physics) {}

  T energy(const T x[], const T dof[]) const {
    T total_energy = 0.0;
    T xq = 0.0;
    std::vector<T> element_x(nodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc<T, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(mesh, i, dof, element_dof);

      std::vector<T> pts, wts;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(i, element_xloc, nullptr,
                                               &Nxi[offset_nxi], nullptr, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(i, element_dof, &N[offset_n],
                                  &Nxi[offset_nxi], &vals, &grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Add the energy contributions
        if (x) {
          interp_val_grad<T, Basis>(i, element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }
        total_energy += physics.energy(wts[j], xq, J, vals, grad);
      }
    }

    return total_energy;
  }

  void residual(const T x[], const T dof[], T res[]) const {
    T xq = 0.0;
    std::vector<T> element_x(nodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc<T, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(mesh, i, dof, element_dof);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(mesh, i, x, element_x.data());
      }

      // Create the element residual
      T element_res[dof_per_element];
      for (int j = 0; j < dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      std::vector<T> pts, wts;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(i, element_xloc, nullptr,
                                               &Nxi[offset_nxi], nullptr, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(i, element_dof, &N[offset_n],
                                  &Nxi[offset_nxi], &vals, &grad_ref);
        if (x) {
          interp_val_grad<T, Basis>(i, element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals{};
        typename Physics::grad_t coef_grad{}, coef_grad_ref{};
        physics.residual(wts[j], xq, J, vals, grad, coef_vals, coef_grad);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad, coef_grad_ref);

        // Add the contributions to the element residual
        add_grad<T, Basis>(i, &N[offset_n], &Nxi[offset_nxi], coef_vals,
                           coef_grad_ref, element_res);
      }

      add_element_res<T, dof_per_node, Basis>(mesh, i, element_res, res);
    }
  }

  void jacobian_product(const T x[], const T dof[], const T direct[],
                        T res[]) const {
    T xq = 0.0;
    std::vector<T> element_x(nodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc<T, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(mesh, i, dof, element_dof);

      // Get the element directions for the Jacobian-vector product
      T element_direct[dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(mesh, i, direct, element_direct);

      // Create the element residual
      T element_res[dof_per_element];
      for (int j = 0; j < dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      std::vector<T> pts, wts;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(i, element_xloc, nullptr,
                                               &Nxi[offset_nxi], nullptr, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, Basis>(i, element_dof, &N[offset_n],
                                  &Nxi[offset_nxi], &vals, &grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        typename Physics::dof_t direct_vals{};
        typename Physics::grad_t direct_grad{}, direct_grad_ref{};
        interp_val_grad<T, Basis>(i, element_direct, &N[offset_n],
                                  &Nxi[offset_nxi], &direct_vals,
                                  &direct_grad_ref);

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, direct_grad_ref, direct_grad);

        if (x) {
          interp_val_grad<T, Basis>(i, element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals{};
        typename Physics::grad_t coef_grad{}, coef_grad_ref{};
        physics.jacobian_product(wts[j], xq, J, vals, grad, direct_vals,
                                 direct_grad, coef_vals, coef_grad);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad, coef_grad_ref);

        // Add the contributions to the element residual
        add_grad<T, Basis>(i, &N[offset_n], &Nxi[offset_nxi], coef_vals,
                           coef_grad_ref, element_res);
      }

      add_element_res<T, dof_per_node, Basis>(mesh, i, element_res, res);
    }
  }

  void jacobian(const T x[], const T dof[],
                GalerkinBSRMat<T, dof_per_node>* mat) const {
    T xq = 0.0;
    std::vector<T> element_x(nodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc<T, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(mesh, i, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(mesh, i, dof, element_dof);

      // Create the element Jacobian
      T element_jac[dof_per_element * dof_per_element];
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        element_jac[j] = 0.0;
      }

      std::vector<T> pts, wts;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, Basis, spatial_dim>(i, element_xloc, nullptr,
                                               &Nxi[offset_nxi], nullptr, &J);

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad_ref{}, grad{};
        interp_val_grad<T, Basis>(i, element_dof, &N[offset_n],
                                  &Nxi[offset_nxi], &vals, &grad_ref);
        if (x) {
          interp_val_grad<T, Basis>(i, element_x.data(), &N[offset_n], nullptr,
                                    &xq, nullptr);
        }

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the residuals at the quadrature points
        typename Physics::jac_t jac_vals{};
        A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>
            jac_grad;
        A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>
            jac_grad_ref;
        physics.jacobian(wts[j], xq, J, vals, grad, jac_vals, jac_grad);

        // Transform hessian from physical coordinates back to ref coordinates
        jtransform<T, dof_per_node, spatial_dim>(J, jac_grad, jac_grad_ref);

        // Add the contributions to the element residual
        add_matrix<T, Basis>(i, &N[offset_n], &Nxi[offset_nxi], jac_vals,
                             jac_grad_ref, element_jac);
      }

      mat->add_block_values(i, nodes_per_element, mesh, element_jac);
    }
  }

  /*
    Evaluate the matrix vector product dR/dphi * psi, where phi are the
    LSF dof, psi are the adjoint variables

    Note: This only works for Galerkin Difference method combined with the
    level-set mesh
  */
  void LSF_jacobian_adjoint_product(const T dof[], const T psi[],
                                    T dfdx[]) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    if (!mesh.has_lsf()) {
      throw std::runtime_error(
          "This method only works with the level-set mesh");
    }

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc<T, Basis>(mesh, i, element_xloc);

      // Get the element states and adjoints
      T element_dof[dof_per_element], element_psi[dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(mesh, i, dof, element_dof);
      get_element_vars<T, dof_per_node, Basis>(mesh, i, psi, element_psi);

      // Create the element dfdx
      T element_dfdx[dof_per_element];
      for (int j = 0; j < dof_per_element; j++) element_dfdx[j] = 0.0;

      std::vector<T> pts, wts, pts_grad, wts_grad;
      int num_quad_pts =
          quadrature.get_quadrature_pts_grad(i, pts, wts, pts_grad, wts_grad);

      std::vector<T> N, Nxi, Nxixi;
      basis.eval_basis_grad(i, pts, N, Nxi, Nxixi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;
        int offset_nxixi = j * nodes_per_element * spatial_dim * spatial_dim;

        T detJ, detJb = 1.0;
        A2D::Mat<T, spatial_dim, spatial_dim> J, Jb;
        interp_val_grad<T, Basis, spatial_dim>(i, element_xloc, nullptr,
                                               &Nxi[offset_nxi], nullptr, &J);

        // Get derivatives of detJ w.r.t. J
        A2D::ADObj<T&> detJ_obj(detJ, detJb);
        A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>&> J_obj(J, Jb);
        auto stack = A2D::MakeStack(A2D::MatDet(J_obj, detJ_obj));
        stack.reverse();

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t uq{}, psiq{};           // uq, psiq
        typename Physics::grad_t ugrad{}, ugrad_ref{};  // (∇_x)uq, (∇_ξ)uq
        typename Physics::grad_t pgrad{}, pgrad_ref{};  // (∇_x)psiq, (∇_ξ)psiq

        // Interpolate the quantities at the quadrature point
        interp_val_grad<T, Basis>(i, element_dof, &N[offset_n],
                                  &Nxi[offset_nxi], &uq, &ugrad_ref);
        interp_val_grad<T, Basis>(i, element_psi, &N[offset_n],
                                  &Nxi[offset_nxi], &psiq, &pgrad_ref);

        transform(J, ugrad_ref, ugrad);
        transform(J, pgrad_ref, pgrad);

        typename Physics::dof_t coef_uq{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad{};  // ∂e/∂(∇_x)uq
        typename Physics::dof_t jp_uq{};        // ∂2e/∂uq2 * psiq
        typename Physics::grad_t jp_ugrad{};  // ∂2e/∂(∇_x)uq2 * (∇_x)psiq

        physics.residual(1.0 / detJ, 0.0, J, uq, ugrad, coef_uq, coef_ugrad);
        physics.jacobian_product(1.0 / detJ, 0.0, J, uq, ugrad, psiq, pgrad,
                                 jp_uq, jp_ugrad);

        typename Physics::grad_t coef_ugrad_ref{};  // ∂e/∂(∇_ξ)uq
        typename Physics::grad_t jp_ugrad_ref{};  // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq

        // Transform gradient from physical coordinates back to ref
        // coordinates
        rtransform(J, coef_ugrad, coef_ugrad_ref);
        rtransform(J, jp_ugrad, jp_ugrad_ref);

        // TODO: finish from here
        std::vector<T> dwdphi(nodes_per_element, 1.2);
        std::vector<T> dxidphi(spatial_dim * nodes_per_element, 3.4);

        add_jac_adj_product<T, Basis>(
            i, element_xloc, element_dof, Nxixi.data(), dwdphi.data(),
            dxidphi.data(), wts[j], detJ, Jb, psiq, pgrad_ref, coef_uq, jp_uq,
            coef_ugrad_ref, jp_ugrad_ref, element_dfdx);
      }

      add_element_res<T, dof_per_node, Basis>(mesh, i, element_dfdx, dfdx);
    }
  }

 private:
  const Mesh& mesh;
  const Quadrature& quadrature;
  const Basis& basis;
  const Physics& physics;
};

#endif  // XCGD_ANALYSIS_H