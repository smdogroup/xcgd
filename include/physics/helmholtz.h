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

  T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ, dot;
    A2D::MatDet(J, detJ);
    A2D::VecDot(grad, grad, dot);
    return 0.5 * weight * detJ * (val * val + r0 * r0 * dot - 2.0 * val * x);
  }

  void residual(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& coef_val,
                A2D::Vec<T, spatial_dim>& coef_grad) const {}

  void jacobian_product(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                        T& val, A2D::Vec<T, spatial_dim>& grad, T& direct_val,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_val,
                        A2D::Vec<T, spatial_dim>& coef_grad) const {}

  void jacobian(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& jac_val,
                A2D::Mat<T, dof_per_node * spatial_dim,
                         dof_per_node * spatial_dim>& jac_grad) const {}

 private:
  T r0;
};

#endif  // XCGD_HELMHOLTZ_H