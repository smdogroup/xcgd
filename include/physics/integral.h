#ifndef XCGD_INTEGRAL_H
#define XCGD_INTEGRAL_H

#include "physics_commons.h"

template <typename T, int spatial_dim>
class IntegralPhysics final : public PhysicsBase<T, spatial_dim, 0, 1> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim, 0, 1>;

 public:
  T energy(T weight, T _, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
           T& val, A2D::Vec<T, spatial_dim>& grad) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * val;
  }

  void residual(T weight, T _, A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& coef_val,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    T detJ;
    A2D::MatDet(J, detJ);
    coef_val = weight * detJ;
  }
};

#endif  // XCGD_INTEGRAL_H