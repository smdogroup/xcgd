#ifndef XCGD_PHYSICS_COMMONS_H
#define XCGD_PHYSICS_COMMONS_H

#include "a2dcore.h"

template <typename T, int spatial_dim_, int dof_per_node_>
class PhysicsBase {
 public:
  static constexpr int dof_per_node = dof_per_node_;
  static constexpr int spatial_dim = spatial_dim_;

  virtual T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
                   A2D::Vec<T, dof_per_node>& vals,
                   A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    return 0.0;
  }

  virtual void residual(
      T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
      A2D::Vec<T, dof_per_node>& vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& coef_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {}

  virtual void jacobian_product(
      T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
      A2D::Vec<T, dof_per_node>& vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& direct_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
      A2D::Vec<T, dof_per_node>& coef_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {}

  virtual void jacobian(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                        A2D::Vec<T, dof_per_node>& vals,
                        A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                        A2D::Mat<T, dof_per_node, dof_per_node>& jac_vals,
                        A2D::Mat<T, dof_per_node * spatial_dim,
                                 dof_per_node * spatial_dim>& jac_grad) const {}
};

#endif  //  XCGD_PHYSICS_COMMONS_H