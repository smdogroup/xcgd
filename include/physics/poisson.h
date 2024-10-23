#ifndef XCGD_POISSON_H
#define XCGD_POISSON_H

#include "physics_commons.h"

template <typename T, int spatial_dim, class SourceFunc>
class PoissonPhysics final : public PhysicsBase<T, spatial_dim, 0, 1> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim, 0, 1>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  /**
   * @param source_fun [in] the source term callable that takes in
   * const A2D::Vec<T, spatial_dim>& and returns T
   */
  PoissonPhysics(SourceFunc source_fun) : source_fun(source_fun) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& __,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ, output, dot, u = val;
    A2D::MatDet(J, detJ);
    A2D::VecDot(grad, grad, dot);
    return weight * detJ * (0.5 * dot - source_fun(xloc) * u);
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& coef_val,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    A2D::ADObj<T> dot_obj, output_obj, detJ_obj;
    A2D::ADObj<T&> u_obj(val, coef_val);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            weight * detJ_obj * (0.5 * dot_obj - source_fun(xloc) * u_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                        A2D::Vec<T, spatial_dim>& __,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_val,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_val,
                        A2D::Vec<T, spatial_dim>& coef_grad) const {
    A2D::Vec<T, spatial_dim> bgrad;
    T ub = 0.0;

    A2D::A2DObj<T> dot_obj, output_obj, detJ_obj;
    A2D::A2DObj<T&> u_obj(val, ub, direct_val, coef_val);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, direct_grad,
                                                    coef_grad);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            weight * detJ_obj * (0.5 * dot_obj - source_fun(xloc) * u_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& jac_val,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& jac_grad) const {
    A2D::Vec<T, spatial_dim> bgrad, pgrad, hgrad;
    T ub = 0.0, up = 1.0;

    A2D::A2DObj<T> dot_obj, output_obj, detJ_obj;
    A2D::A2DObj<T&> u_obj(val, ub, up, jac_val);

    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, pgrad, hgrad);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            weight * detJ_obj * (0.5 * dot_obj - source_fun(xloc) * u_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.hextract(pgrad, hgrad, jac_grad);
  }

 private:
  SourceFunc source_fun;
};

#endif  // XCGD_POISSON_H
