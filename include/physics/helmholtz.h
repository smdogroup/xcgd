#ifndef XCGD_HELMHOLTZ_H
#define XCGD_HELMHOLTZ_H

#include "physics_commons.h"

template <typename T, int spatial_dim>
class HelmholtzPhysics final : public PhysicsBase<T, spatial_dim, 1, 1> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim, 1, 1>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  HelmholtzPhysics(T r0) : r0square(r0 * r0) {}

  T energy(T weight, T x, A2D::Vec<T, spatial_dim>& _,
           A2D::Vec<T, spatial_dim>& __,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ, dot;
    A2D::MatDet(J, detJ);
    A2D::VecDot(grad, grad, dot);
    return 0.5 * weight * detJ * (r0square * dot + val * val - 2.0 * val * x);
  }

  void residual(T weight, T x, A2D::Vec<T, spatial_dim>& _,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& coef_val,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    A2D::ADObj<T> dot_obj, output_obj, detJ_obj, x_obj(x);
    A2D::ADObj<T&> u_obj(val, coef_val);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            0.5 * weight * detJ_obj *
                (r0square * dot_obj + u_obj * u_obj - 2.0 * u_obj * x_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T x, A2D::Vec<T, spatial_dim>& _,
                        A2D::Vec<T, spatial_dim>& __,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_val,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_val,
                        A2D::Vec<T, spatial_dim>& coef_grad) const {
    A2D::Vec<T, spatial_dim> bgrad;
    T ub = 0.0;

    A2D::A2DObj<T> dot_obj, output_obj, detJ_obj, x_obj(x);
    A2D::A2DObj<T&> u_obj(val, ub, direct_val, coef_val);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, direct_grad,
                                                    coef_grad);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            0.5 * weight * detJ_obj *
                (r0square * dot_obj + u_obj * u_obj - 2.0 * u_obj * x_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void adjoint_jacobian_product(T weight, T x, A2D::Vec<T, spatial_dim>& _,
                                A2D::Vec<T, spatial_dim>& __,
                                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                                T& val, A2D::Vec<T, spatial_dim>& grad,
                                T& psi_val, A2D::Vec<T, spatial_dim>& psi_grad,
                                T& x_coef) const {
    A2D::Vec<T, spatial_dim> bgrad, coef_grad;
    T ub = 0.0, xb = 0.0, xp = 0.0, coef_val;

    A2D::A2DObj<T> dot_obj, output_obj, detJ_obj;
    A2D::A2DObj<T&> x_obj(x, xb, xp, x_coef);
    A2D::A2DObj<T&> u_obj(val, ub, psi_val, coef_val);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, psi_grad,
                                                    coef_grad);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            0.5 * weight * detJ_obj *
                (r0square * dot_obj + u_obj * u_obj - 2.0 * u_obj * x_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T x, A2D::Vec<T, spatial_dim>& _,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& jac_val,
                A2D::Mat<T, dof_per_node * spatial_dim,
                         dof_per_node * spatial_dim>& jac_grad) const {
    A2D::Vec<T, spatial_dim> bgrad, pgrad, hgrad;
    T ub = 0.0, up = 1.0;

    A2D::A2DObj<T> dot_obj, output_obj, detJ_obj, x_obj;
    A2D::A2DObj<T&> u_obj(val, ub, up, jac_val);

    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, pgrad, hgrad);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj), A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(
            0.5 * weight * detJ_obj *
                (r0square * dot_obj + u_obj * u_obj - 2.0 * u_obj * x_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.hextract(pgrad, hgrad, jac_grad);
  }

 private:
  T r0square;
};

#endif  // XCGD_HELMHOLTZ_H
