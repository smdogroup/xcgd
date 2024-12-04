#ifndef XCGD_GRAD_PENALIZATION_H
#define XCGD_GRAD_PENALIZATION_H

#include "physics_commons.h"

template <typename T, int spatial_dim_>
class GradPenalization final : public PhysicsBase<T, spatial_dim_, 0, 1> {
 private:
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, 1>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  GradPenalization(T coeff) : coeff(coeff) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ, dot;
    A2D::MatDet(J, detJ);
    A2D::VecDot(grad, grad, dot);
    return 0.5 * coeff * weight * (sqrt(dot) - 1.0) * (sqrt(dot) - 1.0);
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& __,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& coef_val,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    A2D::ADObj<T> dot_obj, output_obj, detJ_obj;
    A2D::ADObj<T&> u_obj(val, coef_val);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(0.5 * coeff * weight * (sqrt(dot_obj) - 1.0) *
                      (sqrt(dot_obj) - 1.0),
                  output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                        A2D::Vec<T, spatial_dim>& __,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_val,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_val,
                        A2D::Vec<T, spatial_dim>& coef_grad) const {}

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& jac_val,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& jac_grad) const {}

 private:
  T coeff;
};
#endif
