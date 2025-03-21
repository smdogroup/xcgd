#ifndef XCGD_ANALYSIS_H
#define XCGD_ANALYSIS_H

#include <type_traits>
#include <vector>

#include "a2dcore.h"
#include "ad/a2dvecnorm.h"
#include "elements/element_utils.h"
#include "physics/volume.h"
#include "sparse_utils/sparse_matrix.h"
#include "utils/exceptions.h"
#include "utils/linalg.h"
#include "utils/misc.h"

/**
 *  ...
 * @tparam from_to_grid_mesh if true, then the global vectors (dof, rhs, phi,
 * etc) and global matrices (Jacobian, etc) are defined on the grid mesh,
 * regardless if the Mesh itself is a cut mesh or not. This is useful for
 * two-sided problems such as the elasticity with ersatz material.
 */
template <typename T, class Mesh, class Quadrature, class Basis, class Physics,
          bool from_to_grid_mesh = false>
class GalerkinAnalysis final {
 public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;
  static constexpr int data_per_node = Physics::data_per_node;

  static_assert(data_per_node == 0 or data_per_node == 1,
                "we only support data_per_node == 0 or 1 for now");
  static_assert(not Physics::is_interface_physics,
                "GalerkinAnalysis does not work with interface physics");

  // Derived static data
  static constexpr int max_dof_per_element =
      dof_per_node * max_nnodes_per_element;

  // Constructor for regular analysis
  GalerkinAnalysis(const Mesh& mesh, const Quadrature& quadrature,
                   const Basis& basis, const Physics& physics)
      : mesh(mesh), quadrature(quadrature), basis(basis), physics(physics) {}

  T energy(const T x[], const T dof[], int node_offset = 0) const {
    T total_energy = 0.0;
    T xq = 0.0;
    std::vector<T> element_x = std::vector<T>(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(nnodes, nodes, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Add the energy contributions
        if (x) {
          interp_val_grad<T, spatial_dim, max_nnodes_per_element,
                          data_per_node>(element_x.data(), &N[offset_n],
                                         nullptr, get_ptr(xq), nullptr);
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

  void residual(const T x[], const T dof[], T res[],
                int node_offset = 0) const {
    T xq = 0.0;
    std::vector<T> element_x = std::vector<T>(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(nnodes, nodes, x, element_x.data());
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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));
        if (x) {
          interp_val_grad<T, spatial_dim, max_nnodes_per_element,
                          data_per_node>(element_x.data(), &N[offset_n],
                                         nullptr, get_ptr(xq), nullptr);
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

      add_element_res<T, dof_per_node, Basis>(nnodes, nodes, element_res, res);
    }
  }

  void jacobian_product(const T x[], const T dof[], const T direct[], T res[],
                        int node_offset = 0) const {
    T xq = 0.0;
    std::vector<T> element_x = std::vector<T>(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(nnodes, nodes, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      // Get the element directions for the Jacobian-vector product
      T element_direct[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, direct,
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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        typename Physics::dof_t direct_vals{};
        typename Physics::grad_t direct_grad{}, direct_grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_direct, &N[offset_n], &Nxi[offset_nxi],
            get_ptr(direct_vals), get_ptr(direct_grad_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, direct_grad_ref, direct_grad);

        if (x) {
          interp_val_grad<T, spatial_dim, max_nnodes_per_element,
                          data_per_node>(element_x.data(), &N[offset_n],
                                         nullptr, get_ptr(xq), nullptr);
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

      add_element_res<T, dof_per_node, Basis>(nnodes, nodes, element_res, res);
    }
  }

  /*
    Evaluate the matrix vector product dR/dx * psi, where x are the nodal data,
    psi are the adjoint variables
  */
  void jacobian_adjoint_product(const T x[], const T dof[], const T psi[],
                                T dfdx[], int node_offset = 0) const {
    T xq = 0.0;
    std::vector<T> element_x = std::vector<T>(max_nnodes_per_element);

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(nnodes, nodes, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      // Get the element psi for the Jacobian-vector product
      T element_psi[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, psi, element_psi);

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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the derivative of the psi in the computational
        // coordinates
        typename Physics::dof_t psi_vals{};
        typename Physics::grad_t psi_grad{}, psi_grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi, &N[offset_n], &Nxi[offset_nxi], get_ptr(psi_vals),
            get_ptr(psi_grad_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, psi_grad_ref, psi_grad);

        if (x) {
          interp_val_grad<T, spatial_dim, max_nnodes_per_element,
                          data_per_node>(element_x.data(), &N[offset_n],
                                         nullptr, get_ptr(xq), nullptr);
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

        add_jac_adj_product_bulk<T, Basis>(&N[offset_n], dv_val, element_dfdx);
      }

      add_element_dfdx<T, Basis>(nnodes, nodes, element_dfdx, dfdx);
    }
  }

  void jacobian(const T x[], const T dof[],
                GalerkinBSRMat<T, dof_per_node>* mat, bool zero_jac = true,
                int node_offset = 0) const {
    if (zero_jac) {
      mat->zero();
    }

    T xq = 0.0;
    std::vector<T> element_x = std::vector<T>(max_nnodes_per_element);

    std::vector<T> element_jac(
        mesh.get_num_elements() * max_dof_per_element * max_dof_per_element,
        T(0.0));

    // #pragma omp parallel for
    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      int element_jac_offset = i * max_dof_per_element * max_dof_per_element;

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(nnodes, nodes, x, element_x.data());
      }

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;

      try {
        basis.eval_basis_grad(i, pts, N, Nxi);
      } catch (const LapackFailed& e) {
        std::printf(
            "jacobian() called failed at basis.eval_basis_grad() for element: "
            "%d\n",
            i);
        throw;
      }

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));
        if (x) {
          interp_val_grad<T, spatial_dim, max_nnodes_per_element,
                          data_per_node>(element_x.data(), &N[offset_n],
                                         nullptr, get_ptr(xq), nullptr);
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
        add_matrix<T, spatial_dim, max_nnodes_per_element>(
            &N[offset_n], &Nxi[offset_nxi], jac_vals, jac_mixed_ref,
            jac_grad_ref, element_jac.data() + element_jac_offset);
      }
    }

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      int element_jac_offset = i * max_dof_per_element * max_dof_per_element;
      mat->template add_block_values<Mesh::max_nnodes_per_element>(
          nnodes, nodes, element_jac.data() + element_jac_offset);
    }
  }

  /*
    Evaluate the matrix vector product psi^T * dR/dphi, where phi are the
    LSF dof, psi are the adjoint variables

    Note: This only works for Galerkin Difference method combined with the
    level-set mesh
  */
  void LSF_jacobian_adjoint_product(const T dof[], const T psi[], T dfdphi[],
                                    int node_offset = 0) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    static_assert(Mesh::is_cut_mesh,
                  "This method requires a level-set-cut mesh");

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element states and adjoints
      T element_dof[max_dof_per_element], element_psi[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, psi, element_psi);

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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(uq),
            get_ptr(ugrad_ref));
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi, &N[offset_n], &Nxi[offset_nxi], get_ptr(psiq),
            get_ptr(pgrad_ref));

        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &Nxixi[offset_nxixi], get_ptr(uhess_ref));
        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi, &Nxixi[offset_nxixi], get_ptr(phess_ref));

        transform(J, ugrad_ref, ugrad);
        transform(J, pgrad_ref, pgrad);

        typename Physics::dof_t coef_uq{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad{};  // ∂e/∂(∇_x)uq
        typename Physics::dof_t jp_uq{};        // ∂2e/∂uq2 * psiq
        typename Physics::grad_t jp_ugrad{};    // ∂2e/∂(∇_x)uq2 * (∇_x)psiq

        physics.residual(wts[j], 0.0, xloc, nrm_ref, J, uq, ugrad, coef_uq,
                         coef_ugrad);
        physics.jacobian_product(wts[j], 0.0, xloc, nrm_ref, J, uq, ugrad, psiq,
                                 pgrad, jp_uq, jp_ugrad);

        typename Physics::grad_t coef_ugrad_ref{};  // ∂e/∂(∇_ξ)uq
        typename Physics::grad_t jp_ugrad_ref{};    // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq

        // Transform gradient from physical coordinates back to ref
        // coordinates
        rtransform(J, coef_ugrad, coef_ugrad_ref);
        rtransform(J, jp_ugrad, jp_ugrad_ref);

        int offset_wts = j * max_nnodes_per_element;
        int offset_pts = j * max_nnodes_per_element * spatial_dim;

        add_jac_adj_product_bulk<T, Basis>(
            wts[j], &wts_grad[offset_wts], &pts_grad[offset_pts], psiq,
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
  void LSF_volume_derivatives(T dfdphi[], int node_offset = 0) const {
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
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, nullptr, &Nxi[offset_nxi], nullptr, get_ptr(J));

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

  /**
   * @brief Evaluate ∂f/∂phi where f is an integration of the energy
   *
   * @param dof state variables
   * @param dfdphi partial derivatives ∂f/∂phi
   * @param unity_quad_wts if true, then the energy is evaluated with quadrature
   * weights being 1, one of such use cases is to compute discrete KS aggration
   * via existing energy evaluation mechanism
   */
  void LSF_energy_derivatives(const T dof[], T dfdphi[],
                              bool unity_quad_wts = false,
                              int node_offset = 0) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    static_assert(Mesh::is_cut_mesh,
                  "This method requires a level-set-cut mesh");

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

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

        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));
        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t uq{};
        typename Physics::grad_t ugrad{}, ugrad_ref{};
        typename Physics::hess_t uhess_ref{};  //(∇2_ξ)uq
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(uq),
            get_ptr(ugrad_ref));
        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &Nxixi[offset_nxixi], get_ptr(uhess_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, ugrad_ref, ugrad);

        typename Physics::dof_t coef_uq{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad{};  // ∂e/∂(∇_x)uq

        T detJ;
        A2D::MatDet(J, detJ);

        T energy = physics.energy(1.0 / detJ, 0.0, xloc, nrm_ref, J, uq, ugrad);
        physics.residual(1.0 / detJ, 0.0, xloc, nrm_ref, J, uq, ugrad, coef_uq,
                         coef_ugrad);

        typename Physics::grad_t coef_ugrad_ref{};  // ∂e/∂(∇_ξ)uq

        // Transform gradient from physical coordinates back to ref
        // coordinates
        rtransform(J, coef_ugrad, coef_ugrad_ref);

        int offset_wts = j * max_nnodes_per_element;
        int offset_pts = j * max_nnodes_per_element * spatial_dim;

        if (unity_quad_wts) {
          add_energy_partial_deriv<T, Basis>(
              1.0, 1.0, energy, nullptr, &pts_grad[offset_pts], ugrad_ref,
              uhess_ref, coef_uq, coef_ugrad_ref, element_dfdphi.data());
        } else {
          add_energy_partial_deriv<T, Basis>(
              wts[j], detJ, energy, &wts_grad[offset_wts],
              &pts_grad[offset_pts], ugrad_ref, uhess_ref, coef_uq,
              coef_ugrad_ref, element_dfdphi.data());
        }
      }

      const auto& lsf_mesh = mesh.get_lsf_mesh();
      int c = mesh.get_elem_cell(i);
      add_element_dfdphi<T, decltype(lsf_mesh), Basis>(
          lsf_mesh, c, element_dfdphi.data(), dfdphi);
    }
  }

  // Interpolate scalar or vector on quadrature points, intended to be used for
  // debug or post-process
  template <int ncomp_per_node = Physics::dof_per_node>
  std::pair<std::vector<T>, std::vector<T>> interpolate(
      const T vals[], int node_offset = 0) const {
    std::vector<T> xloc_q, vals_q;

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_vals[max_nnodes_per_element * ncomp_per_node];
      get_element_vars<T, ncomp_per_node, Basis>(nnodes, nodes, vals,
                                                 element_vals);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], nullptr, get_ptr(xloc), nullptr);

        typename std::conditional<ncomp_per_node == 1, T,
                                  A2D::Vec<T, ncomp_per_node>>::type vq{};

        interp_val_grad<T, spatial_dim, max_nnodes_per_element, ncomp_per_node>(
            element_vals, &N[offset_n], nullptr, get_ptr(vq), nullptr);

        for (int d = 0; d < spatial_dim; d++) {
          xloc_q.push_back(xloc(d));
        }

        if constexpr (ncomp_per_node > 1) {
          for (int k = 0; k < ncomp_per_node; k++) {
            vals_q.push_back(vq(k));
          }
        } else {
          vals_q.push_back(vq);
        }
      }
    }
    return {xloc_q, vals_q};
  }

  // Evaluate the energy on quadrature points, intended to be used for
  // debug or post-process
  std::pair<std::vector<T>, std::vector<T>> interpolate_energy(
      const T dof[], int node_offset = 0) const {
    std::vector<T> xloc_q, energy_q;

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));

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

        T detJ;
        A2D::MatDet(J, detJ);
        energy_q.push_back(
            physics.energy(1.0 / detJ, 0.0, xloc, nrm_ref, J, vals, grad));
      }
    }
    return {xloc_q, energy_q};
  }

  // identical to interpolate_energy(), except that the quadrature points and
  // values are returned in a map instead of vector
  // ret.first: map: element -> quad points
  // ret.second: map: element -> quad values
  //
  std::pair<std::map<int, std::vector<T>>, std::map<int, std::vector<T>>>
  interpolate_energy_map(const T dof[], int node_offset = 0) const {
    std::map<int, std::vector<T>> xloc_q_map, energy_q_map;

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      if constexpr (from_to_grid_mesh) {
        nnodes = mesh.get_cell_dof_verts(mesh.get_elem_cell(i), nodes);
      } else {
        nnodes = mesh.get_elem_dof_nodes(i, nodes);
      }

      if (node_offset != 0) {
        for (int ii = 0; ii < nnodes; ii++) {
          nodes[ii] += node_offset;
        }
      }

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        for (int d = 0; d < spatial_dim; d++) {
          xloc_q_map[i].push_back(xloc(d));
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        T detJ;
        A2D::MatDet(J, detJ);
        energy_q_map[i].push_back(
            physics.energy(1.0 / detJ, 0.0, xloc, nrm_ref, J, vals, grad));
      }
    }
    return {xloc_q_map, energy_q_map};
  }

  // Get the normal vector for surface quadrature points, intended to be used
  // for debug or post-process
  auto get_quadrature_normals() const {
    static_assert(Quadrature::quad_type == QuadPtType::SURFACE,
                  "get_quadrature_normals only works with surface quadrature");

    std::vector<T> xloc_q, normal_ref_q, normal_q;

    for (int i = 0; i < mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        for (int d = 0; d < spatial_dim; d++) {
          nrm_ref[d] = ns[spatial_dim * j + d];
        }

        A2D::Vec<T, spatial_dim> nrm;
        A2D::MatVecMult(J, nrm_ref, nrm);
        A2D::VecNormalize(nrm, nrm);

        for (int d = 0; d < spatial_dim; d++) {
          xloc_q.push_back(xloc(d));
          normal_ref_q.push_back(nrm_ref(d));
          normal_q.push_back(nrm(d));
        }
      }
    }
    return std::make_tuple(xloc_q, normal_ref_q, normal_q);
  }

  const Mesh& get_mesh() { return mesh; }
  const Quadrature& get_quadrature() { return quadrature; }
  const Basis& get_basis() { return basis; }
  const Physics& get_physics() { return physics; }

 private:
  const Mesh& mesh;
  const Quadrature& quadrature;
  const Basis& basis;
  const Physics& physics;
};

#endif  // XCGD_ANALYSIS_H
