#ifndef XCGD_PHYSICS_COMMONS_H
#define XCGD_PHYSICS_COMMONS_H

#include <type_traits>

#include "a2dcore.h"

template <typename T, int spatial_dim_, int data_per_node_, int dof_per_node_>
class PhysicsBase {
 public:
  static constexpr int dof_per_node = dof_per_node_;
  static constexpr int data_per_node = data_per_node_;
  static constexpr int spatial_dim = spatial_dim_;

  using dof_t = typename std::conditional<dof_per_node == 1, T,
                                          A2D::Vec<T, dof_per_node>>::type;
  using grad_t =
      typename std::conditional<dof_per_node == 1, A2D::Vec<T, spatial_dim>,
                                A2D::Mat<T, dof_per_node, spatial_dim>>::type;
  using jac_t =
      typename std::conditional<dof_per_node == 1, T,
                                A2D::Mat<T, dof_per_node, dof_per_node>>::type;

  virtual T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
                   dof_t& vals, grad_t& grad) const {
    return 0.0;
  }

  virtual void residual(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                        dof_t& vals, grad_t& grad, dof_t& coef_vals,
                        grad_t& coef_grad) const {}

  virtual void jacobian_product(T weight,
                                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                                dof_t& vals, grad_t& grad, dof_t& direct_vals,
                                grad_t& direct_grad, dof_t& coef_vals,
                                grad_t& coef_grad) const {}

  virtual void jacobian(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                        dof_t& vals, grad_t& grad, jac_t& jac_vals,
                        A2D::Mat<T, dof_per_node * spatial_dim,
                                 dof_per_node * spatial_dim>& jac_grad) const {}
};

#endif  //  XCGD_PHYSICS_COMMONS_H