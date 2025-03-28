#ifndef XCGD_PHYSICS_COMMONS_H
#define XCGD_PHYSICS_COMMONS_H

#include <type_traits>

#include "a2dcore.h"
#include "utils/exceptions.h"

template <typename T, int spatial_dim_, int data_per_node_, int dof_per_node_>
class PhysicsBase {
 public:
  static constexpr int dof_per_node = dof_per_node_;
  static constexpr int data_per_node = data_per_node_;
  static constexpr int spatial_dim = spatial_dim_;
  static constexpr bool is_interface_physics = false;

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
  using jac_t =
      typename std::conditional<dof_per_node == 1, T,
                                A2D::Mat<T, dof_per_node, dof_per_node>>::type;
  using jac_mixed_t = typename std::conditional<
      dof_per_node == 1, A2D::Vec<T, spatial_dim>,
      A2D::Mat<T, dof_per_node, spatial_dim * dof_per_node>>::type;

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
                   dof_t& vals, grad_t& grad) const {
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
                        dof_t& vals, grad_t& grad, dof_t& coef_vals,
                        grad_t& coef_grad) const {
    throw NotImplemented("residual() for your physics is not implemented");
  }

  // Similar to residual, with an extra output ∂e/∂n_ξ
  virtual void extended_residual(T weight, dv_t dv, xloc_t& xloc,
                                 nrm_t& nrm_ref, J_t& J, dof_t& vals,
                                 grad_t& grad, dof_t& coef_vals,
                                 grad_t& coef_grad, nrm_t& coef_nrm_ref) const {
    throw NotImplemented(
        "extended_residual() for your physics is not implemented");
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
                                J_t& J, dof_t& vals, grad_t& grad,
                                dof_t& direct_vals, grad_t& direct_grad,
                                dof_t& coef_vals, grad_t& coef_grad) const {
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
                                        nrm_t& nrm_ref, J_t& J, dof_t& vals,
                                        grad_t& grad, dof_t& psi_vals,
                                        grad_t& psi_grad, T& x_coef) const {
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
   * @param [out] jac_vals ∂2e/∂uq2
   * @param [out] jac_mixed ∂/∂(∇_x)uq(∂e/∂uq), shape (dim(uq), dim(∂(∇_x)uq))
   * @param [out] jac_grad ∂2e/∂(∇_x)uq2
   */
  virtual void jacobian(T weight, dv_t dv, xloc_t& xloc, nrm_t& nrm_ref, J_t& J,
                        dof_t& vals, grad_t& grad, jac_t& jac_vals,
                        jac_mixed_t& jac_mixed, jac_grad_t& jac_grad) const {
    throw NotImplemented("jacobian() for your physics is not implemented");
  }
};

#endif  //  XCGD_PHYSICS_COMMONS_H
