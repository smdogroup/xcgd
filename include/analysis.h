#ifndef XCGD_ANALYSIS_H
#define XCGD_ANALYSIS_H

#include "a2dcore.h"
#include "elements/commons.h"
#include "sparse_utils/sparse_matrix.h"
#include "utils/linalg.h"

template <typename T, class Basis, class Physics>
class FEAnalysis final {
 public:
  using Quadrature = typename Basis::Quadrature;

  // Static data taken from the element basis
  static const int spatial_dim = Basis::spatial_dim;
  static const int nodes_per_element = Basis::nodes_per_element;

  // Static data from the qaudrature
  static const int num_quadrature_pts = Quadrature::num_quadrature_pts;

  // Static data taken from the physics
  static const int dof_per_node = Physics::dof_per_node;

  // Derived static data
  static const int dof_per_element = dof_per_node * nodes_per_element;

  FEAnalysis(const Basis& basis, const Physics& physics)
      : basis(basis), physics(physics) {}

  void get_element_xloc(int e, T element_xloc[]) const {
    int nodes[nodes_per_element];
    basis.mesh.get_elem_dof_nodes(e, nodes);
    for (int j = 0; j < nodes_per_element; j++) {
      basis.mesh.get_node_xloc(nodes[j], element_xloc);
      element_xloc += spatial_dim;
    }
  }

  template <int dim>
  void get_element_dof(int e, const T dof[], T element_dof[]) const {
    int nodes[nodes_per_element];
    basis.mesh.get_elem_dof_nodes(e, nodes);
    for (int j = 0; j < nodes_per_element; j++) {
      for (int k = 0; k < dim; k++, element_dof++) {
        element_dof[0] = dof[dim * nodes[j] + k];
      }
    }
  }

  template <int dim>
  void add_element_res_new(int e, const T element_res[], T res[]) const {
    int nodes[nodes_per_element];
    basis.mesh.get_elem_dof_nodes(e, nodes);
    for (int j = 0; j < nodes_per_element; j++) {
      for (int k = 0; k < dim; k++, element_res++) {
        res[dim * nodes[j] + k] += element_res[0];
      }
    }
  }

  template <int dim>
  void add_element_res(const int nodes[], const T element_res[],
                       T res[]) const {
    for (int j = 0; j < nodes_per_element; j++) {
      int node = nodes[j];
      for (int k = 0; k < dim; k++, element_res++) {
        res[dim * node + k] += element_res[0];
      }
    }
  }

  T energy(const T dof[]) const {
    T total_energy = 0.0;

    for (int i = 0; i < basis.mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc(i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_dof<dof_per_node>(i, dof, element_dof);

      T pts[spatial_dim * Quadrature::num_quadrature_pts];
      T wts[Quadrature::num_quadrature_pts];
      basis.quadrature.get_quadrature_pts(pts, wts);

      T N[nodes_per_element * num_quadrature_pts];
      T Nxi[spatial_dim * nodes_per_element * num_quadrature_pts];
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quadrature_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        eval_grad<T, Basis, spatial_dim>(i, pts, element_xloc, &Nxi[offset_nxi],
                                         J);

        // Evaluate the derivative of the dof in the computational coordinates
        A2D::Vec<T, dof_per_node> vals;
        A2D::Mat<T, dof_per_node, spatial_dim> grad;
        eval_val_grad<T, Basis, dof_per_node>(i, pts, element_dof, &N[offset_n],
                                              &Nxi[offset_nxi], vals, grad);

        // Add the energy contributions
        total_energy += physics.energy(wts[j], J, vals, grad);
      }
    }

    return total_energy;
  }

  void residual(const T dof[], T res[]) const {
    for (int i = 0; i < basis.mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc(i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_dof<dof_per_node>(i, dof, element_dof);

      // Create the element residual
      T element_res[dof_per_element];
      for (int j = 0; j < dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      T pts[spatial_dim * Quadrature::num_quadrature_pts];
      T wts[Quadrature::num_quadrature_pts];
      basis.quadrature.get_quadrature_pts(pts, wts);

      T N[nodes_per_element * num_quadrature_pts];
      T Nxi[spatial_dim * nodes_per_element * num_quadrature_pts];
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quadrature_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        eval_grad<T, Basis, spatial_dim>(i, pts, element_xloc, &Nxi[offset_nxi],
                                         J);

        // Evaluate the derivative of the dof in the computational coordinates
        A2D::Vec<T, dof_per_node> vals;
        A2D::Mat<T, dof_per_node, spatial_dim> grad;
        eval_val_grad<T, Basis, dof_per_node>(i, pts, element_dof, &N[offset_n],
                                              &Nxi[offset_nxi], vals, grad);

        // Evaluate the residuals at the quadrature points
        A2D::Vec<T, dof_per_node> coef_vals;
        A2D::Mat<T, dof_per_node, spatial_dim> coef_grad;
        physics.residual(wts[j], J, vals, grad, coef_vals, coef_grad);

        // Add the contributions to the element residual
        add_grad<T, Basis, dof_per_node>(i, pts, &N[offset_n], &Nxi[offset_nxi],
                                         coef_vals, coef_grad, element_res);
      }

      add_element_res_new<dof_per_node>(i, element_res, res);
    }
  }

  void jacobian_product(const T dof[], const T direct[], T res[]) const {
    for (int i = 0; i < basis.mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc(i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_dof<dof_per_node>(i, dof, element_dof);

      // Get the element directions for the Jacobian-vector product
      T element_direct[dof_per_element];
      get_element_dof<dof_per_node>(i, direct, element_direct);

      // Create the element residual
      T element_res[dof_per_element];
      for (int j = 0; j < dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      T pts[spatial_dim * Quadrature::num_quadrature_pts];
      T wts[Quadrature::num_quadrature_pts];
      basis.quadrature.get_quadrature_pts(pts, wts);

      T N[nodes_per_element * num_quadrature_pts];
      T Nxi[spatial_dim * nodes_per_element * num_quadrature_pts];
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quadrature_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        eval_grad<T, Basis, spatial_dim>(i, pts, element_xloc, &Nxi[offset_nxi],
                                         J);

        // Evaluate the derivative of the dof in the computational coordinates
        A2D::Vec<T, dof_per_node> vals;
        A2D::Mat<T, dof_per_node, spatial_dim> grad;
        eval_val_grad<T, Basis, dof_per_node>(i, pts, element_dof, &N[offset_n],
                                              &Nxi[offset_nxi], vals, grad);

        // Evaluate the derivative of the direction in the computational
        // coordinates
        A2D::Vec<T, dof_per_node> direct_vals;
        A2D::Mat<T, dof_per_node, spatial_dim> direct_grad;
        eval_val_grad<T, Basis, dof_per_node>(i, pts, element_direct,
                                              &N[offset_n], &Nxi[offset_nxi],
                                              direct_vals, direct_grad);

        // Evaluate the residuals at the quadrature points
        A2D::Vec<T, dof_per_node> coef_vals;
        A2D::Mat<T, dof_per_node, spatial_dim> coef_grad;
        physics.jacobian_product(wts[j], J, vals, grad, direct_vals,
                                 direct_grad, coef_vals, coef_grad);

        // Add the contributions to the element residual
        add_grad<T, Basis, dof_per_node>(i, pts, &N[offset_n], &Nxi[offset_nxi],
                                         coef_vals, coef_grad, element_res);
      }

      add_element_res_new<dof_per_node>(i, element_res, res);
    }
  }

  void jacobian(const T dof[], GalerkinBSRMat<T, dof_per_node>* mat) const {
    for (int i = 0; i < basis.mesh.get_num_elements(); i++) {
      // Get the element node locations
      T element_xloc[spatial_dim * nodes_per_element];
      get_element_xloc(i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[dof_per_element];
      get_element_dof<dof_per_node>(i, dof, element_dof);

      // Create the element Jacobian
      T element_jac[dof_per_element * dof_per_element];
      for (int j = 0; j < dof_per_element * dof_per_element; j++) {
        element_jac[j] = 0.0;
      }

      T pts[spatial_dim * Quadrature::num_quadrature_pts];
      T wts[Quadrature::num_quadrature_pts];
      basis.quadrature.get_quadrature_pts(pts, wts);

      T N[nodes_per_element * num_quadrature_pts];
      T Nxi[spatial_dim * nodes_per_element * num_quadrature_pts];
      basis.eval_basis_grad(i, pts, N, Nxi);

      for (int j = 0; j < num_quadrature_pts; j++) {
        int offset_n = j * nodes_per_element;
        int offset_nxi = j * nodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        eval_grad<T, Basis, spatial_dim>(i, pts, element_xloc, &Nxi[offset_nxi],
                                         J);

        // Evaluate the derivative of the dof in the computational coordinates
        A2D::Vec<T, dof_per_node> vals;
        A2D::Mat<T, dof_per_node, spatial_dim> grad;
        eval_val_grad<T, Basis, dof_per_node>(i, pts, element_dof, &N[offset_n],
                                              &Nxi[offset_nxi], vals, grad);

        // Evaluate the residuals at the quadrature points
        A2D::Mat<T, dof_per_node, dof_per_node> coef_vals;
        A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>
            coef_grad;
        physics.jacobian(wts[j], J, vals, grad, coef_vals, coef_grad);

        // Add the contributions to the element residual
        add_matrix<T, Basis, dof_per_node>(i, pts, &N[offset_n],
                                           &Nxi[offset_nxi], coef_vals,
                                           coef_grad, element_jac);
      }

      mat->add_block_values(i, nodes_per_element, basis, element_jac);
    }
  }

 private:
  const Basis& basis;
  const Physics& physics;
};

#endif  // XCGD_ANALYSIS_H