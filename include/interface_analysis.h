#pragma once

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

template <typename T, class Mesh, class Quadrature, class Basis, class Physics>
class InterfaceGalerkinAnalysis final {
 public:
  // Static data taken from the element basis
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;
  static constexpr int dof_per_vert = Physics::dof_per_vert;
  static constexpr int data_per_node = Physics::data_per_node;

  static_assert(Mesh::is_finite_cell_mesh,
                "InterfaceProblem only works with FiniteCellMesh");
  static_assert(data_per_node == 0 or data_per_node == 1,
                "we only support data_per_node == 0 or 1 for now");
  static_assert(Quadrature::quad_type == QuadPtType::SURFACE,
                "interface analysis only works with surface quadrature");
  static_assert(Physics::is_interface_physics,
                "InterfaceGalerkinAnalysis only works with interface physics");
  static_assert(Mesh::is_cut_mesh, "This method requires a level-set-cut mesh");

  // Derived static data
  static constexpr int max_dof_per_element =
      dof_per_node * max_nnodes_per_element;

  // Constructor for interface regular analysis
  InterfaceGalerkinAnalysis(Mesh& mesh_primary, Mesh& mesh_secondary,
                            Quadrature& quadrature_interface,
                            Basis& basis_primary, Physics& physics_interface)
      : mesh_primary(mesh_primary),
        mesh_secondary(mesh_secondary),
        quadrature(quadrature_interface),
        basis(basis_primary),
        physics(physics_interface),
        cell_primary_elems(mesh_primary.get_cell_elems()),
        cell_secondary_elems(mesh_secondary.get_cell_elems()),
        secondary_node_offset(mesh_primary.get_num_nodes()) {
    update_mesh();
  }

  void update_mesh() {
    interface_cells.clear();
    const auto& cut_elems = mesh_primary.get_cut_elems();
    for (int elem_primary = 0; elem_primary < mesh_primary.get_num_elements();
         elem_primary++) {
      if (cut_elems.count(elem_primary)) {
        interface_cells.push_back(mesh_primary.get_elem_cell(elem_primary));
      }
    }
  }

  T energy(const T x[], const T dof[]) const {
    T total_energy = 0.0;

    for (int cell : interface_cells) {
      auto [nnodes_primary, nnodes_secondary, num_quad_pts, nodes_primary,
            nodes_secondary, element_xloc, N, Nxi, element_dof_primary,
            element_dof_secondary, element_x, wts, ns] =
          interpolate_for_element(cell, x, dof);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc.data(), &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals_primary{};
        typename Physics::grad_t grad_primary{}, grad_ref_primary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_primary), get_ptr(grad_ref_primary));

        typename Physics::dof_t vals_secondary{};
        typename Physics::grad_t grad_secondary{}, grad_ref_secondary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_secondary), get_ptr(grad_ref_secondary));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_primary, grad_primary);
        transform(J, grad_ref_secondary, grad_secondary);

        // Add the energy contributions
        T xq = T(0.0);
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

        // // TODO: delete
        // nrm_ref(0) = 0.2;
        // nrm_ref(1) = 0.7;

        total_energy +=
            physics.energy(wts[j], xq, xloc, nrm_ref, J, vals_primary,
                           vals_secondary, grad_primary, grad_secondary);
      }
    }

    return total_energy;
  }

  void residual(const T x[], const T dof[], T res[],
                std::vector<T> debug_phi_primary = {}) const {
    // TODO: delete
    std::vector<T> phi_primary = mesh_primary.get_lsf_dof();
    if (debug_phi_primary.size()) {
      xcgd_assert(debug_phi_primary.size() == phi_primary.size(),
                  "incompatible lsf_dof size");
    }

    for (int cell : interface_cells) {
      auto [nnodes_primary, nnodes_secondary, num_quad_pts, nodes_primary,
            nodes_secondary, element_xloc, N, Nxi, element_dof_primary,
            element_dof_secondary, element_x, wts, ns] =
          interpolate_for_element(cell, x, dof);

      if (debug_phi_primary.size()) {
        mesh_primary.get_lsf_dof() = debug_phi_primary;
        mesh_primary.update_mesh();
        auto [_nnodes_primary, _nnodes_secondary, _num_quad_pts, _nodes_primary,
              _nodes_secondary, _element_xloc, _N, _Nxi, _element_dof_primary,
              _element_dof_secondary, _element_x, _wts, _ns] =
            interpolate_for_element(cell, x, dof);
        mesh_primary.get_lsf_dof() = phi_primary;
      }

      std::vector<T> element_res_primary(max_dof_per_element, T(0.0));
      std::vector<T> element_res_secondary(max_dof_per_element, T(0.0));

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc.data(), &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals_primary{};
        typename Physics::grad_t grad_primary{}, grad_ref_primary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_primary), get_ptr(grad_ref_primary));

        typename Physics::dof_t vals_secondary{};
        typename Physics::grad_t grad_secondary{}, grad_ref_secondary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_secondary), get_ptr(grad_ref_secondary));

        T xq = T(0.0);
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

        // // TODO: delete
        // nrm_ref(0) = 0.2;
        // nrm_ref(1) = 0.7;

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_primary, grad_primary);
        transform(J, grad_ref_secondary, grad_secondary);

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals_primary{};
        typename Physics::grad_t coef_grad_primary{}, coef_grad_ref_primary{};

        typename Physics::dof_t coef_vals_secondary{};
        typename Physics::grad_t coef_grad_secondary{},
            coef_grad_ref_secondary{};

        physics.residual(wts[j], xq, xloc, nrm_ref, J, vals_primary,
                         vals_secondary, grad_primary, grad_secondary,
                         coef_vals_primary, coef_vals_secondary,
                         coef_grad_primary, coef_grad_secondary);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad_primary, coef_grad_ref_primary);
        rtransform(J, coef_grad_secondary, coef_grad_ref_secondary);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_primary,
                           coef_grad_ref_primary, element_res_primary.data());
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_secondary,
                           coef_grad_ref_secondary,
                           element_res_secondary.data());
      }

      add_element_res<T, dof_per_node, Basis>(nnodes_primary,
                                              nodes_primary.data(),
                                              element_res_primary.data(), res);
      // add_element_res<T, dof_per_node, Basis>(
      //     nnodes_secondary, nodes_secondary.data(),
      //     element_res_secondary.data(), res);
    }
  }

  void jacobian_product(const T x[], const T dof[], const T direct[],
                        T res[]) const {
    for (int cell : interface_cells) {
      auto [nnodes_primary, nnodes_secondary, num_quad_pts, nodes_primary,
            nodes_secondary, element_xloc, N, Nxi, element_dof_primary,
            element_dof_secondary, element_x, wts, ns] =
          interpolate_for_element(cell, x, dof);

      // Get the element directions for the Jacobian-vector product
      std::vector<T> element_direct_primary(max_dof_per_element, T(0.0));
      get_element_vars<T, dof_per_node, Basis>(nnodes_primary,
                                               nodes_primary.data(), direct,
                                               element_direct_primary.data());
      std::vector<T> element_direct_secondary(max_dof_per_element, T(0.0));
      get_element_vars<T, dof_per_node, Basis>(nnodes_secondary,
                                               nodes_secondary.data(), direct,
                                               element_direct_secondary.data());

      // Create the element residual
      std::vector<T> element_res_primary(max_dof_per_element, T(0.0));
      std::vector<T> element_res_secondary(max_dof_per_element, T(0.0));

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc.data(), &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals_primary{};
        typename Physics::grad_t grad_primary{}, grad_ref_primary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_primary), get_ptr(grad_ref_primary));

        typename Physics::dof_t vals_secondary{};
        typename Physics::grad_t grad_secondary{}, grad_ref_secondary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_secondary), get_ptr(grad_ref_secondary));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_primary, grad_primary);
        transform(J, grad_ref_secondary, grad_secondary);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        typename Physics::dof_t direct_vals_primary{};
        typename Physics::grad_t direct_grad_primary{},
            direct_grad_ref_primary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_direct_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(direct_vals_primary), get_ptr(direct_grad_ref_primary));

        typename Physics::dof_t direct_vals_secondary{};
        typename Physics::grad_t direct_grad_secondary{},
            direct_grad_ref_secondary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_direct_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(direct_vals_secondary), get_ptr(direct_grad_ref_secondary));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, direct_grad_ref_primary, direct_grad_primary);
        transform(J, direct_grad_ref_secondary, direct_grad_secondary);

        T xq = T(0.0);
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
        typename Physics::dof_t coef_vals_primary{};
        typename Physics::grad_t coef_grad_primary{}, coef_grad_ref_primary{};
        typename Physics::dof_t coef_vals_secondary{};
        typename Physics::grad_t coef_grad_secondary{},
            coef_grad_ref_secondary{};

        typename Physics::nrm_t coef_nrm_ref_dummy{};

        physics.extended_jacobian_product(
            wts[j], xq, xloc, nrm_ref, J, vals_primary, vals_secondary,
            grad_primary, grad_secondary, direct_vals_primary,
            direct_vals_secondary, direct_grad_primary, direct_grad_secondary,
            coef_vals_primary, coef_vals_secondary, coef_grad_primary,
            coef_grad_secondary, coef_nrm_ref_dummy);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad_primary, coef_grad_ref_primary);
        rtransform(J, coef_grad_secondary, coef_grad_ref_secondary);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_primary,
                           coef_grad_ref_primary, element_res_primary.data());
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_secondary,
                           coef_grad_ref_secondary,
                           element_res_secondary.data());
      }

      add_element_res<T, dof_per_node, Basis>(nnodes_primary,
                                              nodes_primary.data(),
                                              element_res_primary.data(), res);
      add_element_res<T, dof_per_node, Basis>(
          nnodes_secondary, nodes_secondary.data(),
          element_res_secondary.data(), res);
    }
  }

  void jacobian(const T x[], const T dof[],
                GalerkinBSRMat<T, dof_per_node>* mat,
                bool zero_jac = true) const {
    if (zero_jac) {
      mat->zero();
    }

    for (int cell : interface_cells) {
      auto [nnodes_primary, nnodes_secondary, num_quad_pts, nodes_primary,
            nodes_secondary, element_xloc, N, Nxi, element_dof_primary,
            element_dof_secondary, element_x, wts, ns] =
          interpolate_for_element(cell, x, dof);

      std::vector<T> element_jac(
          2 * 2 * max_dof_per_element * max_dof_per_element, T(0.0));

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc.data(), &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals_primary{};
        typename Physics::grad_t grad_primary{}, grad_ref_primary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_primary), get_ptr(grad_ref_primary));

        typename Physics::dof_t vals_secondary{};
        typename Physics::grad_t grad_secondary{}, grad_ref_secondary{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_secondary), get_ptr(grad_ref_secondary));

        T xq = T(0.0);
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

        // // TODO: delete
        // nrm_ref(0) = 0.2;
        // nrm_ref(1) = 0.7;

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_primary, grad_primary);
        transform(J, grad_ref_secondary, grad_secondary);

        // Evaluate the residuals at the quadrature points
        typename Physics::jac_t jac_vals{};
        typename Physics::jac_mixed_t jac_mixed{}, jac_mixed_ref{};
        typename Physics::jac_grad_t jac_grad{}, jac_grad_ref{};

        physics.jacobian(wts[j], xq, xloc, nrm_ref, J, vals_primary,
                         vals_secondary, grad_primary, grad_secondary, jac_vals,
                         jac_mixed, jac_grad);

        // Transform hessian from physical coordinates back to ref coordinates
        jtransform<T, dof_per_vert, spatial_dim>(J, jac_grad, jac_grad_ref);
        mtransform(J, jac_mixed, jac_mixed_ref);

        // Add the contributions to the element Jacobian
        add_matrix<T, spatial_dim, max_nnodes_per_element>(
            &N[offset_n], &Nxi[offset_nxi], jac_vals, jac_mixed_ref,
            jac_grad_ref, element_jac.data());
      }

      xcgd_assert(nnodes_primary == nnodes_secondary,
                  "number of primary dof nodes (" +
                      std::to_string(nnodes_primary) +
                      ") should be equal to number of "
                      "secondary dof nodes(" +
                      std::to_string(nnodes_secondary) + ")");
      int nnodes_all = nnodes_primary + nnodes_secondary;
      std::vector<int> nodes_all(nnodes_all, 0);
      for (int t = 0; t < nnodes_primary; t++) {
        nodes_all[2 * t] = nodes_primary[t];
        nodes_all[2 * t + 1] = nodes_secondary[t];
      }

      // I really don't like this quite devil hard-coded 2 here, but it works
      mat->template add_block_values<2 * max_nnodes_per_element>(
          nnodes_all, nodes_all.data(), element_jac.data());
    }
  }

  void LSF_jacobian_adjoint_product(const T dof[], const T psi[],
                                    T dfdphi[]) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    static_assert(Mesh::is_cut_mesh,
                  "This method requires a level-set-cut mesh");

    for (int cell : interface_cells) {
      constexpr bool need_Nxixi_and_quad_grads = true;
      auto [nnodes_primary, nnodes_secondary, num_quad_pts, nodes_primary,
            nodes_secondary, element_xloc, N, Nxi, Nxixi, element_dof_primary,
            element_dof_secondary, element_x, wts, ns, pts_grad, wts_grad,
            wns_grad] =
          interpolate_for_element<need_Nxixi_and_quad_grads>(cell, nullptr,
                                                             dof);

      std::vector<T> element_psi_primary(max_dof_per_element, T(0.0));
      get_element_vars<T, dof_per_node, Basis>(nnodes_primary,
                                               nodes_primary.data(), psi,
                                               element_psi_primary.data());
      std::vector<T> element_psi_secondary(max_dof_per_element, T(0.0));
      get_element_vars<T, dof_per_node, Basis>(nnodes_secondary,
                                               nodes_secondary.data(), psi,
                                               element_psi_secondary.data());

      // Create the element dfdphi
      std::vector<T> element_dfdphi(max_nnodes_per_element, 0.0);

      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;
        int offset_nxixi =
            j * max_nnodes_per_element * spatial_dim * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc.data(), &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t uq_primary{}, psiq_primary{};  // uq, psiq
        typename Physics::grad_t ugrad_primary{},
            ugrad_ref_primary{};  // (∇_x)uq, (∇_ξ)uq
        typename Physics::grad_t pgrad_primary{},
            pgrad_ref_primary{};                       // (∇_x)psiq, (∇_ξ)psiq
        typename Physics::hess_t uhess_ref_primary{};  //(∇2_ξ)uq
        typename Physics::hess_t phess_ref_primary{};  //(∇2_ξ)psiq

        typename Physics::dof_t uq_secondary{}, psiq_secondary{};  // uq, psiq
        typename Physics::grad_t ugrad_secondary{},
            ugrad_ref_secondary{};  // (∇_x)uq, (∇_ξ)uq
        typename Physics::grad_t pgrad_secondary{},
            pgrad_ref_secondary{};                       // (∇_x)psiq, (∇_ξ)psiq
        typename Physics::hess_t uhess_ref_secondary{};  //(∇2_ξ)uq
        typename Physics::hess_t phess_ref_secondary{};  //(∇2_ξ)psiq

        // Interpolate the quantities at the quadrature point
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(uq_primary), get_ptr(ugrad_ref_primary));
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(uq_secondary), get_ptr(ugrad_ref_secondary));

        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi_primary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(psiq_primary), get_ptr(pgrad_ref_primary));
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi_secondary.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(psiq_secondary), get_ptr(pgrad_ref_secondary));

        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_primary.data(), &Nxixi[offset_nxixi],
            get_ptr(uhess_ref_primary));
        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_secondary.data(), &Nxixi[offset_nxixi],
            get_ptr(uhess_ref_secondary));

        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi_primary.data(), &Nxixi[offset_nxixi],
            get_ptr(phess_ref_primary));
        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi_secondary.data(), &Nxixi[offset_nxixi],
            get_ptr(phess_ref_secondary));

        transform(J, ugrad_ref_primary, ugrad_primary);
        transform(J, ugrad_ref_secondary, ugrad_secondary);

        transform(J, pgrad_ref_primary, pgrad_primary);
        transform(J, pgrad_ref_secondary, pgrad_secondary);

        typename Physics::dof_t coef_uq_primary{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad_primary{};  // ∂e/∂(∇_x)uq
        typename Physics::dof_t jp_uq_primary{};        // ∂2e/∂uq2 * psiq
        typename Physics::grad_t
            jp_ugrad_primary{};  // ∂2e/∂(∇_x)uq2 * (∇_x)psiq
                                 //
        typename Physics::dof_t coef_uq_secondary{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad_secondary{};  // ∂e/∂(∇_x)uq
        typename Physics::dof_t jp_uq_secondary{};        // ∂2e/∂uq2 * psiq
        typename Physics::grad_t
            jp_ugrad_secondary{};  // ∂2e/∂(∇_x)uq2 * (∇_x)psiq
        typename Physics::nrm_t jp_nrm_ref{};

        // // TODO: delete
        // nrm_ref(0) = 0.2;
        // nrm_ref(1) = 0.7;
        // for (auto& val : wns_grad) val = 0.0;

        static_assert(spatial_dim == 2,
                      "InterfaceGalerkinAnalysis is only implemented for 2D");
        // Prepare passive quantities: Evaluate dt
        A2D::Vec<T, spatial_dim> tan_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> rot;
        rot(0, 1) = -1.0;
        rot(1, 0) = 1.0;
        A2D::MatVecMult(rot, nrm_ref, tan_ref);

        T cq;  // frame transform scaling
        A2D::Vec<T, spatial_dim> Jdt;
        A2D::MatVecMult(J, tan_ref, Jdt);
        A2D::VecNorm(Jdt, cq);

        A2D::Vec<T, spatial_dim> JTJdt, RTJTJdt;
        A2D::MatVecMult<A2D::MatOp::TRANSPOSE>(J, Jdt, JTJdt);
        A2D::MatVecMult<A2D::MatOp::TRANSPOSE>(rot, JTJdt, RTJTJdt);

        T dummy_x = 0.0;
        physics.residual(1.0, dummy_x, xloc, nrm_ref, J, uq_primary,
                         uq_secondary, ugrad_primary, ugrad_secondary,
                         coef_uq_primary, coef_uq_secondary, coef_ugrad_primary,
                         coef_ugrad_secondary);

        // // TODO: delete
        // psiq_primary = {};
        // psiq_secondary = {};
        // pgrad_primary = {};
        // pgrad_secondary = {};
        //
        // psiq_primary(0) = 1.0;
        // // pgrad_primary(0, 0) = 1.0;

        physics.extended_jacobian_product(
            1.0, dummy_x, xloc, nrm_ref, J, uq_primary, uq_secondary,
            ugrad_primary, ugrad_secondary, psiq_primary, psiq_secondary,
            pgrad_primary, pgrad_secondary, jp_uq_primary, jp_uq_secondary,
            jp_ugrad_primary, jp_ugrad_secondary, jp_nrm_ref);

        typename Physics::grad_t coef_ugrad_ref_primary{};    // ∂e/∂(∇_ξ)uq
        typename Physics::grad_t coef_ugrad_ref_secondary{};  // ∂e/∂(∇_ξ)uq

        typename Physics::grad_t
            jp_ugrad_ref_primary{};  // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq
        typename Physics::grad_t
            jp_ugrad_ref_secondary{};  // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq

        // Transform gradient from physical coordinates back to ref
        // coordinates
        rtransform(J, coef_ugrad_primary, coef_ugrad_ref_primary);
        rtransform(J, coef_ugrad_secondary, coef_ugrad_ref_secondary);

        rtransform(J, jp_ugrad_primary, jp_ugrad_ref_primary);
        rtransform(J, jp_ugrad_secondary, jp_ugrad_ref_secondary);

        int offset_wts = j * max_nnodes_per_element;
        int offset_pts = j * max_nnodes_per_element * spatial_dim;
        int offset_wns = j * max_nnodes_per_element * spatial_dim;

        // std::printf("[cell:%2d][quad:%2d]", cell, j);
        add_jac_adj_product_surf<T, Basis>(
            wts[j], cq, &wts_grad[offset_wts], &pts_grad[offset_pts],
            &wns_grad[offset_wns], RTJTJdt, psiq_primary, ugrad_ref_primary,
            pgrad_ref_primary, uhess_ref_primary, phess_ref_primary,
            coef_uq_primary, coef_ugrad_ref_primary, jp_uq_primary,
            jp_ugrad_ref_primary, jp_nrm_ref, element_dfdphi.data());

        // jp_nrm_ref = {};
        // add_jac_adj_product_surf<T, Basis>(
        //     wts[j], cq, &wts_grad[offset_wts], &pts_grad[offset_pts],
        //     &wns_grad[offset_wns], RTJTJdt, psiq_secondary,
        //     ugrad_ref_secondary, pgrad_ref_secondary, uhess_ref_secondary,
        //     phess_ref_secondary, coef_uq_secondary, coef_ugrad_ref_secondary,
        //     jp_uq_secondary, jp_ugrad_ref_secondary, jp_nrm_ref,
        //     element_dfdphi.data());
      }

      const auto& lsf_mesh = mesh_primary.get_lsf_mesh();
      add_element_dfdphi<T, decltype(lsf_mesh), Basis>(
          lsf_mesh, cell, element_dfdphi.data(), dfdphi);
    }
  }

  auto debug_wc_and_grad(const T dof[]) {
    const auto& lsf_mesh = mesh_primary.get_lsf_mesh();

    T wc_sum = 0.0;
    std::vector<T> wc_grad(lsf_mesh.get_num_nodes(), 0.0);

    for (int cell : interface_cells) {
      constexpr bool need_Nxixi_and_quad_grads = true;
      auto [nnodes_primary, nnodes_secondary, num_quad_pts, nodes_primary,
            nodes_secondary, element_xloc, N, Nxi, Nxixi, element_dof_primary,
            element_dof_secondary, element_x, wts, ns, pts_grad, wts_grad,
            wns_grad] =
          interpolate_for_element<need_Nxixi_and_quad_grads>(cell, nullptr,
                                                             dof);

      std::vector<T> element_wc_grad(max_nnodes_per_element, 0.0);
      for (int j = 0; j < num_quad_pts; j++) {
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;
        int offset_nxixi =
            j * max_nnodes_per_element * spatial_dim * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc.data(), &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        A2D::Vec<T, spatial_dim> tan_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> rot;
        rot(0, 1) = -1.0;
        rot(1, 0) = 1.0;
        A2D::MatVecMult(rot, nrm_ref, tan_ref);

        T cq;  // frame transform scaling
        A2D::Vec<T, spatial_dim> Jdt;
        A2D::MatVecMult(J, tan_ref, Jdt);
        A2D::VecNorm(Jdt, cq);

        wc_sum += cq * wts[j];

        A2D::Vec<T, spatial_dim> JTJdt, RTJTJdt;
        A2D::MatVecMult<A2D::MatOp::TRANSPOSE>(J, Jdt, JTJdt);
        A2D::MatVecMult<A2D::MatOp::TRANSPOSE>(rot, JTJdt, RTJTJdt);

        int offset_wts = j * max_nnodes_per_element;
        int offset_pts = j * max_nnodes_per_element * spatial_dim;
        int offset_wns = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t uq_primary{}, psiq_primary{};  // uq, psiq
        typename Physics::grad_t ugrad_primary{},
            ugrad_ref_primary{};  // (∇_x)uq, (∇_ξ)uq
        typename Physics::grad_t pgrad_primary{},
            pgrad_ref_primary{};                       // (∇_x)psiq, (∇_ξ)psiq
        typename Physics::hess_t uhess_ref_primary{};  //(∇2_ξ)uq
        typename Physics::hess_t phess_ref_primary{};  //(∇2_ξ)psiq

        typename Physics::dof_t coef_uq_primary{};          // ∂e/∂uq
        typename Physics::grad_t coef_ugrad_ref_primary{};  // ∂e/∂(∇_ξ)uq
        typename Physics::grad_t
            jp_ugrad_ref_primary{};               // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq
        typename Physics::dof_t jp_uq_primary{};  // ∂2e/∂uq2 * psiq
        typename Physics::nrm_t jp_nrm_ref{};

        coef_uq_primary(0) = 1.0;
        psiq_primary(0) = 1.0;

        add_jac_adj_product_surf<T, Basis>(
            wts[j], cq, &wts_grad[offset_wts], &pts_grad[offset_pts],
            &wns_grad[offset_wns], RTJTJdt, psiq_primary, ugrad_ref_primary,
            pgrad_ref_primary, uhess_ref_primary, phess_ref_primary,
            coef_uq_primary, coef_ugrad_ref_primary, jp_uq_primary,
            jp_ugrad_ref_primary, jp_nrm_ref, element_wc_grad.data());
      }
      add_element_dfdphi<T, decltype(lsf_mesh), Basis>(
          lsf_mesh, cell, element_wc_grad.data(), wc_grad.data());
    }

    return std::make_tuple(wc_sum, wc_grad);
  }

  const Mesh& get_primary_mesh() { return mesh_primary; }
  const Mesh& get_secondary_mesh() { return mesh_secondary; }
  const Quadrature& get_quadrature() { return quadrature; }
  const Basis& get_basis() { return basis; }
  const Physics& get_physics() { return physics; }

 private:
  template <bool need_Nxixi_and_quad_grads = false>
  auto interpolate_for_element(int cell, const T x[], const T dof[]) const {
    // Get elem indices for this interface cell in both the primary mesh and
    // the secondary mesh
    int elem_primary = cell_primary_elems.at(cell);
    int elem_secondary = cell_secondary_elems.at(cell);

    // Get nodes associated to this element
    std::vector<int> nodes_primary(Mesh::max_nnodes_per_element, 0);
    int nnodes_primary =
        mesh_primary.get_elem_dof_nodes(elem_primary, nodes_primary.data());

    std::vector<int> nodes_secondary(Mesh::max_nnodes_per_element, 0);
    int nnodes_secondary = mesh_secondary.get_elem_dof_nodes(
        elem_secondary, nodes_secondary.data());
    for (int ii = 0; ii < nnodes_secondary; ii++) {
      nodes_secondary[ii] += secondary_node_offset;
    }

    // Get the element node locations
    std::vector<T> element_xloc(spatial_dim * max_nnodes_per_element, T(0.0));
    get_element_xloc<T, Mesh, Basis>(mesh_primary, elem_primary,
                                     element_xloc.data());

    // Get element design variable if needed
    std::vector<T> element_x(max_dof_per_element, T(0.0));
    if (x) {
      get_element_vars<T, 1, Basis>(nnodes_primary, nodes_primary.data(), x,
                                    element_x.data());
    }

    // Get the element degrees of freedom
    std::vector<T> element_dof_primary(max_dof_per_element, 0.0);
    get_element_vars<T, dof_per_node, Basis>(
        nnodes_primary, nodes_primary.data(), dof, element_dof_primary.data());
    std::vector<T> element_dof_secondary(max_dof_per_element, 0.0);
    get_element_vars<T, dof_per_node, Basis>(nnodes_secondary,
                                             nodes_secondary.data(), dof,
                                             element_dof_secondary.data());

    if constexpr (not need_Nxixi_and_quad_grads) {
      std::vector<T> pts, wts, ns;
      int num_quad_pts =
          quadrature.get_quadrature_pts(elem_primary, pts, wts, ns);
      std::vector<T> N, Nxi;
      basis.eval_basis_grad(elem_primary, pts, N, Nxi);
      return std::make_tuple(nnodes_primary, nnodes_secondary, num_quad_pts,
                             nodes_primary, nodes_secondary, element_xloc, N,
                             Nxi, element_dof_primary, element_dof_secondary,
                             element_x, wts, ns);
    } else {
      std::vector<T> pts, wts, ns, pts_grad, wts_grad, wns_grad;
      int num_quad_pts = quadrature.get_quadrature_pts_grad(
          elem_primary, pts, wts, ns, pts_grad, wts_grad, wns_grad);
      std::vector<T> N, Nxi, Nxixi;
      basis.eval_basis_grad(elem_primary, pts, N, Nxi, Nxixi);
      return std::make_tuple(nnodes_primary, nnodes_secondary, num_quad_pts,
                             nodes_primary, nodes_secondary, element_xloc, N,
                             Nxi, Nxixi, element_dof_primary,
                             element_dof_secondary, element_x, wts, ns,
                             pts_grad, wts_grad, wns_grad);
    }
  }

  Mesh& mesh_primary;
  Mesh& mesh_secondary;
  Quadrature& quadrature;
  Basis& basis;
  Physics& physics;

  const std::map<int, int>& cell_primary_elems;
  const std::map<int, int>& cell_secondary_elems;
  int secondary_node_offset;
  std::vector<int> interface_cells;
};
