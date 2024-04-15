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

  using x_t = T;
  using dof_t = typename std::conditional<dof_per_node == 1, T,
                                          A2D::Vec<T, dof_per_node>>::type;
  using grad_t =
      typename std::conditional<dof_per_node == 1, A2D::Vec<T, spatial_dim>,
                                A2D::Mat<T, dof_per_node, spatial_dim>>::type;
  using hess_t = typename std::conditional<
      dof_per_node == 1, A2D::Vec<T, spatial_dim * spatial_dim>,
      A2D::Mat<T, dof_per_node, spatial_dim * spatial_dim>>::type;
  using jac_t =
      typename std::conditional<dof_per_node == 1, T,
                                A2D::Mat<T, dof_per_node, dof_per_node>>::type;
  using jac_grad_t =
      A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>;

  /**
   * @brief At a quadrature point, evaluate the energy functional
   *
   * @param weight quadrature weight
   * @param x design variable, optional
   * @param J coordinate transformation matrix ∂x/∂ξ
   * @param vals uq, state variable at the quadrature point
   * @param grad (∇_x)uq, gradients of state w.r.t. x at the quadrature point
   * @return T energy functional scalar
   */
  virtual T energy(T weight, x_t x,
                   const A2D::Mat<T, spatial_dim, spatial_dim>& J, dof_t& vals,
                   grad_t& grad) const {
    return 0.0;
  }

  /**
   * @brief At a quadrature point, evaluate the partials of the energy
   * functional e
   *
   * @param weight quadrature weight
   * @param x design variable, optional
   * @param J coordinate transformation matrix ∂x/∂ξ
   * @param vals uq, state variable at the quadrature point
   * @param grad (∇_x)uq, gradients of state w.r.t. x at the quadrature point
   * @param coef_vals ∂e/∂uq
   * @param coef_grad ∂e/∂(∇_x)uq
   */
  virtual void residual(T weight, x_t x,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, dof_t& vals,
                        grad_t& grad, dof_t& coef_vals,
                        grad_t& coef_grad) const {}

  /**
   * @brief At a quadrature point, evaluate the matrix-vector product of second
   * order derivatives of the energy functional
   *
   * @param weight quadrature weight
   * @param x design variable, optional
   * @param J coordinate transformation matrix ∂x/∂ξ
   * @param vals uq, state variable at the quadrature point
   * @param grad (∇_x)uq, gradients of state w.r.t. x at the quadrature point
   * @param direct_vals pq, state variable perturbation at the quadrature point
   * @param direct_grad (∇_x)pq, gradients of the state perturbation
   * @param coef_vals ∂2e/∂uq2 * pq
   * @param coef_grad ∂2e/∂(∇_x)uq2 * (∇_x)pq
   */
  virtual void jacobian_product(T weight, x_t x,
                                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                                dof_t& vals, grad_t& grad, dof_t& direct_vals,
                                grad_t& direct_grad, dof_t& coef_vals,
                                grad_t& coef_grad) const {}

  /**
   * @brief At a quadrature point, evaluate ∂/∂x(ψ^T * ∂e/∂u)
   *
   * @param weight quadrature weight
   * @param x design variable at the quadrature point
   * @param J coordinate transformation matrix ∂x/∂ξ
   * @param vals uq, state variable at the quadrature point
   * @param grad (∇_x)uq, gradients of state w.r.t. x at the quadrature point
   * @param psi_vals ψq, state variable perturbation at the quadrature point
   * @param psi_grad (∇_x)ψq, gradients of the state perturbation
   * @param x_coef output, ∂/∂xq(ψ^T * ∂e/∂u)
   */
  virtual void adjoint_jacobian_product(
      T weight, x_t x, A2D::Mat<T, spatial_dim, spatial_dim>& J, dof_t& vals,
      grad_t& grad, dof_t& psi_vals, grad_t& psi_grad, T& x_coef) const {}

  /**
   * @brief At a quadrature point, evaluate the second order derivatives of the
   * energy functional
   *
   * @param weight quadrature weight
   * @param x design variable, optional
   * @param J coordinate transformation matrix ∂x/∂ξ
   * @param vals uq, state variable at the quadrature point
   * @param grad (∇_x)uq, gradients of state w.r.t. x at the quadrature point
   * @param jac_vals ∂2e/∂uq2
   * @param jac_grad ∂2e/∂(∇_x)uq2
   */
  virtual void jacobian(T weight, x_t x,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, dof_t& vals,
                        grad_t& grad, jac_t& jac_vals,
                        A2D::Mat<T, dof_per_node * spatial_dim,
                                 dof_per_node * spatial_dim>& jac_grad) const {}
};

#endif  //  XCGD_PHYSICS_COMMONS_H