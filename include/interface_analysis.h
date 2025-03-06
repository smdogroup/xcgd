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
  InterfaceGalerkinAnalysis(const Mesh& mesh_master, const Mesh& mesh_slave,
                            const Quadrature& quadrature_interface,
                            const Basis& basis_master,
                            const Physics& physics_interface)
      : mesh_master(mesh_master),
        mesh_slave(mesh_slave),
        quadrature(quadrature_interface),
        basis(basis_master),
        physics(physics_interface),
        cell_master_elems(mesh_master.get_cell_elems()),
        cell_slave_elems(mesh_slave.get_cell_elems()),
        slave_node_offset(mesh_master.get_num_nodes()) {
    const auto& cut_elems = mesh_master.get_cut_elems();
    for (int elem_master = 0; elem_master < mesh_master.get_num_elements();
         elem_master++) {
      if (cut_elems.count(elem_master)) {
        interface_cells.push_back(mesh_master.get_elem_cell(elem_master));
      }
    }
  }

  T energy(const T x[], const T dof[]) const {
    T total_energy = 0.0;

    for (int cell : interface_cells) {
      auto [nnodes_master, nnodes_slave, num_quad_pts, nodes_master,
            nodes_slave, element_xloc, N, Nxi, element_dof_master,
            element_dof_slave, element_x, wts, ns] =
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
        typename Physics::dof_t vals_master{};
        typename Physics::grad_t grad_master{}, grad_ref_master{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_master.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_master), get_ptr(grad_ref_master));

        typename Physics::dof_t vals_slave{};
        typename Physics::grad_t grad_slave{}, grad_ref_slave{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_slave.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_slave), get_ptr(grad_ref_slave));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_master, grad_master);
        transform(J, grad_ref_slave, grad_slave);

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

        total_energy +=
            physics.energy(wts[j], xq, xloc, nrm_ref, J, vals_master,
                           vals_slave, grad_master, grad_slave);
      }
    }

    return total_energy;
  }

  void residual(const T x[], const T dof[], T res[]) const {
    for (int cell : interface_cells) {
      auto [nnodes_master, nnodes_slave, num_quad_pts, nodes_master,
            nodes_slave, element_xloc, N, Nxi, element_dof_master,
            element_dof_slave, element_x, wts, ns] =
          interpolate_for_element(cell, x, dof);

      std::vector<T> element_res_master(max_dof_per_element, T(0.0));
      std::vector<T> element_res_slave(max_dof_per_element, T(0.0));

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
        typename Physics::dof_t vals_master{};
        typename Physics::grad_t grad_master{}, grad_ref_master{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_master.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_master), get_ptr(grad_ref_master));

        typename Physics::dof_t vals_slave{};
        typename Physics::grad_t grad_slave{}, grad_ref_slave{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_slave.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_slave), get_ptr(grad_ref_slave));

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

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_master, grad_master);
        transform(J, grad_ref_slave, grad_slave);

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals_master{};
        typename Physics::grad_t coef_grad_master{}, coef_grad_ref_master{};

        typename Physics::dof_t coef_vals_slave{};
        typename Physics::grad_t coef_grad_slave{}, coef_grad_ref_slave{};

        physics.residual(wts[j], xq, xloc, nrm_ref, J, vals_master, vals_slave,
                         grad_master, grad_slave, coef_vals_master,
                         coef_vals_slave, coef_grad_master, coef_grad_slave);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad_master, coef_grad_ref_master);
        rtransform(J, coef_grad_slave, coef_grad_ref_slave);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_master,
                           coef_grad_ref_master, element_res_master.data());
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_slave,
                           coef_grad_ref_slave, element_res_slave.data());
      }

      add_element_res<T, dof_per_node, Basis>(
          nnodes_master, nodes_master.data(), element_res_master.data(), res);
      add_element_res<T, dof_per_node, Basis>(nnodes_slave, nodes_slave.data(),
                                              element_res_slave.data(), res);
    }
  }

  void jacobian_product(const T x[], const T dof[], const T direct[],
                        T res[]) const {
    for (int cell : interface_cells) {
      auto [nnodes_master, nnodes_slave, num_quad_pts, nodes_master,
            nodes_slave, element_xloc, N, Nxi, element_dof_master,
            element_dof_slave, element_x, wts, ns] =
          interpolate_for_element(cell, x, dof);

      // Get the element directions for the Jacobian-vector product
      std::vector<T> element_direct_master(max_dof_per_element, T(0.0));
      get_element_vars<T, dof_per_node, Basis>(nnodes_master,
                                               nodes_master.data(), direct,
                                               element_direct_master.data());
      std::vector<T> element_direct_slave(max_dof_per_element, T(0.0));
      get_element_vars<T, dof_per_node, Basis>(nnodes_slave, nodes_slave.data(),
                                               direct,
                                               element_direct_slave.data());

      // Create the element residual
      std::vector<T> element_res_master(max_dof_per_element, T(0.0));
      std::vector<T> element_res_slave(max_dof_per_element, T(0.0));

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
        typename Physics::dof_t vals_master{};
        typename Physics::grad_t grad_master{}, grad_ref_master{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_master.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_master), get_ptr(grad_ref_master));

        typename Physics::dof_t vals_slave{};
        typename Physics::grad_t grad_slave{}, grad_ref_slave{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_slave.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_slave), get_ptr(grad_ref_slave));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_master, grad_master);
        transform(J, grad_ref_slave, grad_slave);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        typename Physics::dof_t direct_vals_master{};
        typename Physics::grad_t direct_grad_master{}, direct_grad_ref_master{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_direct_master.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(direct_vals_master), get_ptr(direct_grad_ref_master));

        typename Physics::dof_t direct_vals_slave{};
        typename Physics::grad_t direct_grad_slave{}, direct_grad_ref_slave{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_direct_slave.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(direct_vals_slave), get_ptr(direct_grad_ref_slave));

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, direct_grad_ref_master, direct_grad_master);
        transform(J, direct_grad_ref_slave, direct_grad_slave);

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
        typename Physics::dof_t coef_vals_master{};
        typename Physics::grad_t coef_grad_master{}, coef_grad_ref_master{};
        typename Physics::dof_t coef_vals_slave{};
        typename Physics::grad_t coef_grad_slave{}, coef_grad_ref_slave{};

        physics.jacobian_product(
            wts[j], xq, xloc, nrm_ref, J, vals_master, vals_slave, grad_master,
            grad_slave, direct_vals_master, direct_vals_slave,
            direct_grad_master, direct_grad_slave, coef_vals_master,
            coef_vals_slave, coef_grad_master, coef_grad_slave);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad_master, coef_grad_ref_master);
        rtransform(J, coef_grad_slave, coef_grad_ref_slave);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_master,
                           coef_grad_ref_master, element_res_master.data());
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals_slave,
                           coef_grad_ref_slave, element_res_slave.data());
      }

      add_element_res<T, dof_per_node, Basis>(
          nnodes_master, nodes_master.data(), element_res_master.data(), res);
      add_element_res<T, dof_per_node, Basis>(nnodes_slave, nodes_slave.data(),
                                              element_res_slave.data(), res);
    }
  }

  void jacobian(const T x[], const T dof[],
                GalerkinBSRMat<T, dof_per_node>* mat,
                bool zero_jac = true) const {
    if (zero_jac) {
      mat->zero();
    }

    for (int cell : interface_cells) {
      auto [nnodes_master, nnodes_slave, num_quad_pts, nodes_master,
            nodes_slave, element_xloc, N, Nxi, element_dof_master,
            element_dof_slave, element_x, wts, ns] =
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
        typename Physics::dof_t vals_master{};
        typename Physics::grad_t grad_master{}, grad_ref_master{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_master.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_master), get_ptr(grad_ref_master));

        typename Physics::dof_t vals_slave{};
        typename Physics::grad_t grad_slave{}, grad_ref_slave{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof_slave.data(), &N[offset_n], &Nxi[offset_nxi],
            get_ptr(vals_slave), get_ptr(grad_ref_slave));

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

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref_master, grad_master);
        transform(J, grad_ref_slave, grad_slave);

        // Evaluate the residuals at the quadrature points
        typename Physics::jac_t jac_vals{};
        typename Physics::jac_mixed_t jac_mixed{}, jac_mixed_ref{};
        typename Physics::jac_grad_t jac_grad{}, jac_grad_ref{};

        physics.jacobian(wts[j], xq, xloc, nrm_ref, J, vals_master, vals_slave,
                         grad_master, grad_slave, jac_vals, jac_mixed,
                         jac_grad);

        // Transform hessian from physical coordinates back to ref coordinates
        jtransform<T, dof_per_vert, spatial_dim>(J, jac_grad, jac_grad_ref);
        mtransform(J, jac_mixed, jac_mixed_ref);

        // Add the contributions to the element Jacobian
        add_matrix<T, spatial_dim, max_nnodes_per_element>(
            &N[offset_n], &Nxi[offset_nxi], jac_vals, jac_mixed_ref,
            jac_grad_ref, element_jac.data());
      }

      xcgd_assert(nnodes_master == nnodes_slave,
                  "number of master dof nodes (" +
                      std::to_string(nnodes_master) +
                      ") should be equal to number of "
                      "slave dof nodes(" +
                      std::to_string(nnodes_slave) + ")");
      int nnodes_all = nnodes_master + nnodes_slave;
      std::vector<int> nodes_all(nnodes_all, 0);
      for (int t = 0; t < nnodes_master; t++) {
        nodes_all[2 * t] = nodes_master[t];
        nodes_all[2 * t + 1] = nodes_slave[t];
      }

      // I really don't like this quite devil hard-coded 2 here, but it works
      mat->template add_block_values<2 * max_nnodes_per_element>(
          nnodes_all, nodes_all.data(), element_jac.data());
    }
  }

  const Mesh& get_master_mesh() { return mesh_master; }
  const Mesh& get_slave_mesh() { return mesh_slave; }
  const Quadrature& get_quadrature() { return quadrature; }
  const Basis& get_basis() { return basis; }
  const Physics& get_physics() { return physics; }

 private:
  auto interpolate_for_element(int cell, const T x[], const T dof[]) const {
    // Get elem indices for this interface cell in both the master mesh and
    // the slave mesh
    int elem_master = cell_master_elems.at(cell);
    int elem_slave = cell_slave_elems.at(cell);

    // Get nodes associated to this element
    std::vector<int> nodes_master(Mesh::max_nnodes_per_element, 0);
    int nnodes_master =
        mesh_master.get_elem_dof_nodes(elem_master, nodes_master.data());

    std::vector<int> nodes_slave(Mesh::max_nnodes_per_element, 0);
    int nnodes_slave =
        mesh_slave.get_elem_dof_nodes(elem_slave, nodes_slave.data());
    for (int ii = 0; ii < nnodes_slave; ii++) {
      nodes_slave[ii] += slave_node_offset;
    }

    // Get the element node locations
    std::vector<T> element_xloc(spatial_dim * max_nnodes_per_element, T(0.0));
    get_element_xloc<T, Mesh, Basis>(mesh_master, elem_master,
                                     element_xloc.data());

    // Get element design variable if needed
    std::vector<T> element_x(max_dof_per_element, T(0.0));
    if (x) {
      get_element_vars<T, 1, Basis>(nnodes_master, nodes_master.data(), x,
                                    element_x.data());
    }

    // Get the element degrees of freedom
    std::vector<T> element_dof_master(max_dof_per_element, 0.0);
    get_element_vars<T, dof_per_node, Basis>(nnodes_master, nodes_master.data(),
                                             dof, element_dof_master.data());
    std::vector<T> element_dof_slave(max_dof_per_element, 0.0);
    get_element_vars<T, dof_per_node, Basis>(nnodes_slave, nodes_slave.data(),
                                             dof, element_dof_slave.data());

    std::vector<T> pts, wts, ns;
    int num_quad_pts = quadrature.get_quadrature_pts(elem_master, pts, wts, ns);

    std::vector<T> N, Nxi;
    basis.eval_basis_grad(elem_master, pts, N, Nxi);

    return std::make_tuple(nnodes_master, nnodes_slave, num_quad_pts,
                           nodes_master, nodes_slave, element_xloc, N, Nxi,
                           element_dof_master, element_dof_slave, element_x,
                           wts, ns);
  }

  const Mesh& mesh_master;
  const Mesh& mesh_slave;
  const Quadrature& quadrature;
  const Basis& basis;
  const Physics& physics;

  const std::map<int, int>& cell_master_elems;
  const std::map<int, int>& cell_slave_elems;
  int slave_node_offset;
  std::vector<int> interface_cells;
};
