#ifndef XCGD_HELMHOLTZ_H
#define XCGD_HELMHOLTZ_H

#include "physics_commons.h"

template <typename T, int spatial_dim_>
class HelmholtzPhysics final : public PhysicsBase<T, spatial_dim_, 1, 1> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim_, 1, 1>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  HelmholtzPhysics(T r0) : r0(r0) {}

  T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
           dof_t& vals, grad_t& grad) const {
    T detJ, output, dot, rho = vals(0);
    A2D::Vec<T, spatial_dim> grad_v(grad.get_data());
    A2D::MatDet(J, detJ);
    A2D::VecDot(grad_v, grad_v, dot);
    return 0.5 * weight * detJ * (rho * rho + r0 * r0 * dot - 2.0 * rho * x);
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

 private:
  T r0;
};

#endif  // XCGD_HELMHOLTZ_H