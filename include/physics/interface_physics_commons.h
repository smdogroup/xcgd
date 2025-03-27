#pragma once

#include <type_traits>

#include "a2dcore.h"
#include "utils/exceptions.h"

template <typename T, int spatial_dim_, int data_per_node_, int dof_per_node_>
class InterfacePhysicsBase {
 public:
  static constexpr int dof_per_node = dof_per_node_;
  static constexpr int data_per_node = data_per_node_;
  static constexpr int spatial_dim = spatial_dim_;
  static constexpr bool is_interface_physics = true;

  static_assert(data_per_node <= 1,
                "we only support data_per_node = 0 or 1 now");
  using dv_t = T;
  using nrm_t = A2D::Vec<T, spatial_dim>;
  using xloc_t = A2D::Vec<T, spatial_dim>;
  using J_t = A2D::Mat<T, spatial_dim, spatial_dim>;
  using dof_t = typename std::conditional<dof_per_node == 1, T,
                                          A2D::Vec<T, dof_per_node>>::type;
  using grad_t =
      typename std::conditional<dof_per_node == 1, A2D::Vec<T, spatial_dim>,
                                A2D::Mat<T, dof_per_node, spatial_dim>>::type;
  using hess_t = typename std::conditional<
      dof_per_node == 1, A2D::Vec<T, spatial_dim * spatial_dim>,
      A2D::Mat<T, dof_per_node, spatial_dim * spatial_dim>>::type;

  // Note that these types have different shape than PhysicsBase
  using jac_t = A2D::Mat<T, dof_per_node, dof_per_node>;
  using jac_mixed_t = A2D::Mat<T, dof_per_node, spatial_dim * dof_per_node>;
  using jac_grad_t =
      A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>;

  /**
   * @brief At a quadrature point, evaluate the energy functional
   *
   * @param [in] weight quadrature weight
   * @param [in] dv design variable, optional
   * @param [in] xloc coordinates of the analyzed point in physical frame
   * @param [in] nrm_ref surface normal vector in reference frame, most useful
   * for surface physics
   * @param [in] J coordinate transformation matrix ∂x/∂ξ
   * @param [in] vals uq, state variable at the quadrature point
   * @param [in] grad (∇_x)uq, gradients of state w.r.t. x at quadrature point
   * @return T energy functional scalar
   */
  virtual T energy(T weight, dv_t dv, xloc_t& xloc, nrm_t& nrm_ref, J_t& J,
                   dof_t& vals_m, dof_t& vals_s, grad_t& grad_m,
                   grad_t& grad_s) const {
    throw NotImplemented("energy() for your physics is not implemented");
  }

  /**
   * @brief At a quadrature point, evaluate the partials of the energy
   * functional e
   *
   * @param [in] weight quadrature weight
   * @param [in] dv design variable, optional
   * @param [in] xloc coordinates of the analyzed point in physical frame
   * @param [in] nrm_ref surface normal vector in reference frame, most useful
   * for surface physics
   * @param [in] J coordinate transformation matrix ∂x/∂ξ
   * @param [in] vals uq, state variable at the quadrature point
   * @param [in] grad (∇_x)uq, gradients of state w.r.t. x at the quadrature
   * point
   * @param [out] coef_vals ∂e/∂uq
   * @param [out] coef_grad ∂e/∂(∇_x)uq
   */
  virtual void residual(T weight, dv_t dv, xloc_t& xloc, nrm_t& nrm_ref, J_t& J,
                        dof_t& vals_m, dof_t& vals_s, grad_t& grad_m,
                        grad_t& grad_s, dof_t& coef_vals_m, dof_t& coef_vals_s,
                        grad_t& coef_grad_m, grad_t& coef_grad_s) const {
    throw NotImplemented("residual() for your physics is not implemented");
  }

  /**
   * @brief At a quadrature point, evaluate the matrix-vector product of second
   * order derivatives of the energy functional
   *
   * @param [in] weight quadrature weight
   * @param [in] dv design variable, optional
   * @param [in] xloc coordinates of the analyzed point in physical frame
   * @param [in] nrm_ref surface normal vector in reference frame, most useful
   * for surface physics
   * @param [in] J coordinate transformation matrix ∂x/∂ξ
   * @param [in] vals uq, state variable at the quadrature point
   * @param [in] grad (∇_x)uq, gradients of state w.r.t. x at quadrature point
   * @param [in] direct_vals pq, state variable perturbation at quadrature point
   * @param [in] direct_grad (∇_x)pq, gradients of the state perturbation
   * @param [out] coef_vals ∂2e/∂uq2 * pq
   * @param [out] coef_grad ∂2e/∂(∇_x)uq2 * (∇_x)pq
   */
  virtual void jacobian_product(T weight, dv_t dv, xloc_t& xloc, nrm_t& nrm_ref,
                                J_t& J, dof_t& vals_m, dof_t& vals_s,
                                grad_t& grad_m, grad_t& grad_s,
                                dof_t& direct_vals_m, dof_t& direct_vals_s,
                                grad_t& direct_grad_m, grad_t& direct_grad_s,
                                dof_t& coef_vals_m, dof_t& coef_vals_s,
                                grad_t& coef_grad_m,
                                grad_t& coef_grad_s) const {
    throw NotImplemented(
        "jacobian_product() for your physics is not implemented");
  }

  // similar to jacobian_product but evaluates derivatives with respect to
  // nrm_ref in additional to vals and grad
  virtual void extended_jacobian_product(
      T weight, dv_t dv, xloc_t& xloc, nrm_t& nrm_ref, J_t& J, dof_t& vals_m,
      dof_t& vals_s, grad_t& grad_m, grad_t& grad_s, dof_t& direct_vals_m,
      dof_t& direct_vals_s, grad_t& direct_grad_m, grad_t& direct_grad_s,
      dof_t& coef_vals_m, dof_t& coef_vals_s, grad_t& coef_grad_m,
      grad_t& coef_grad_s, nrm_t& coef_nrm_ref) const {
    throw NotImplemented(
        "jacobian_product() for your physics is not implemented");
  }

  /**
   * @brief At a quadrature point, evaluate ∂/∂x(ψ^T * ∂e/∂u)
   *
   * @param [in] weight quadrature weight
   * @param [in] dv design variable at the quadrature point
   * @param [in] xloc coordinates of the analyzed point in physical frame
   * @param [in] nrm_ref surface normal vector in reference frame, most useful
   * for surface physics
   * @param [in] J coordinate transformation matrix ∂x/∂ξ
   * @param [in] vals uq, state variable at the quadrature point
   * @param [in] grad (∇_x)uq, gradients of state w.r.t. x at quadrature point
   * @param [in] psi_vals ψq, state variable perturbation at quadrature point
   * @param [in] psi_grad (∇_x)ψq, gradients of the state perturbation
   * @param [out] x_coef output, ∂/∂xq(ψ^T * ∂e/∂u)
   */
  virtual void adjoint_jacobian_product(T weight, dv_t dv, xloc_t& xloc,
                                        nrm_t& nrm_ref, J_t& J, dof_t& vals_m,
                                        dof_t& vals_s, grad_t& grad_m,
                                        grad_t& grad_s, dof_t& psi_vals_m,
                                        dof_t& psi_vals_s, grad_t& psi_grad_m,
                                        grad_t& psi_grad_s, T& x_coef_m,
                                        T& x_coef_s) const {
    throw NotImplemented(
        "adjoint_jacobian_product() for your physics is not implemented");
  }

  /**
   * @brief At a quadrature point, evaluate the second order derivatives of the
   * energy functional
   *
   * @param [in] weight quadrature weight
   * @param [in] dv design variable, optional
   * @param [in] xloc coordinates of the analyzed point in physical frame
   * @param [in] nrm_ref surface normal vector in reference frame, most useful
   * for surface physics
   * @param [in] J coordinate transformation matrix ∂x/∂ξ
   * @param [in] vals uq, state variable at the quadrature point
   * @param [in] grad (∇_x)uq, gradients of state w.r.t. x at quadrature point
   * @param [out] jac_vals_mm ∂2e/∂uqm∂uqm
   * @param [out] jac_vals_ms ∂2e/∂uqm∂uqs
   * @param [out] jac_vals_ss ∂2e/∂uqs∂uqs
   * @param [out] jac_mixed ∂/∂(∇_x)uq(∂e/∂uq), shape (dim(uq), dim(∂(∇_x)uq))
   * @param [out] jac_grad ∂2e/∂(∇_x)uq2
   */
  virtual void jacobian(T weight, dv_t dv, xloc_t& xloc, nrm_t& nrm_ref, J_t& J,
                        dof_t& vals_m, dof_t& vals_s, grad_t& grad_m,
                        grad_t& grad_s, jac_t& jac_vals_mm, jac_t& jac_vals_ms,
                        jac_t& jac_vals_ss, jac_mixed_t& jac_mixed_mm,
                        jac_mixed_t& jac_mixed_ms, jac_mixed_t& jac_mixed_ss,
                        jac_grad_t& jac_grad_mm, jac_grad_t& jac_grad_ms,
                        jac_grad_t& jac_grad_ss) const {
    throw NotImplemented("jacobian() for your physics is not implemented");
  }
};
