#ifndef XCGD_POISSON_H
#define XCGD_POISSON_H

#include "a2dcore.h"

template <typename T, int spatial_dim = 3>
class PoissonPhysics {
 public:
  static constexpr int dof_per_node = 1;

  PoissonPhysics() = default;

  T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& vals,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ, output, dot, u = vals(0);
    A2D::Vec<T, spatial_dim> grad_v(grad.get_data());

    A2D::MatDet(J, detJ);
    A2D::VecDot(grad_v, grad_v, dot);
    output = weight * detJ * (0.5 * dot - u);
    return output;
  }

  void residual(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    A2D::Vec<T, spatial_dim> grad_v(grad.get_data());
    A2D::Vec<T, spatial_dim> coef_v;

    A2D::ADObj<T> u(vals(0)), dot, output, detJ;
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad_v, coef_v);
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ), A2D::VecDot(grad_obj, grad_obj, dot),
        A2D::Eval(weight * detJ * (0.5 * dot - u), output));

    output.bvalue() = 1.0;
    stack.reverse();

    coef_vals(0) = u.bvalue();
    for (int i = 0; i < spatial_dim; i++) {
      coef_grad(0, i) = coef_v(i);
    }
  }

  void jacobian_product(
      T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
      A2D::Vec<T, dof_per_node>& vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& direct_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
      A2D::Vec<T, dof_per_node>& coef_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    A2D::Vec<T, spatial_dim> grad_v(grad.get_data());
    A2D::Vec<T, spatial_dim> bgrad_v;
    A2D::Vec<T, spatial_dim> pgrad_v(direct_grad.get_data());
    A2D::Vec<T, spatial_dim> hgrad_v;

    A2D::A2DObj<T> dot, output, detJ;
    T ub = 0.0, uh = 0.0;
    A2D::A2DObj<T&> u(vals(0), ub, direct_vals(0), uh);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad_v, bgrad_v, pgrad_v,
                                                    hgrad_v);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ), A2D::VecDot(grad_obj, grad_obj, dot),
        A2D::Eval(weight * detJ * (0.5 * dot - u), output));

    output.bvalue() = 1.0;
    stack.hproduct();
    coef_vals(0) = u.hvalue();  // This is 0.0
    for (int i = 0; i < spatial_dim; i++) {
      coef_grad(0, i) = hgrad_v(i);
    }
  }

  void jacobian(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Mat<T, dof_per_node, dof_per_node>& jac_vals,
                A2D::Mat<T, dof_per_node * spatial_dim,
                         dof_per_node * spatial_dim>& jac_grad) const {
    A2D::Vec<T, spatial_dim> grad_v(grad.get_data());
    A2D::Vec<T, spatial_dim> bgrad_v;
    A2D::Vec<T, spatial_dim> pgrad_v;
    A2D::Vec<T, spatial_dim> hgrad_v;

    A2D::A2DObj<T> dot, output, detJ;
    T ub = 0.0, up = 1.0, uh = 0.0;
    A2D::A2DObj<T&> u(vals(0), ub, up, uh);

    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad_v, bgrad_v, pgrad_v,
                                                    hgrad_v);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ), A2D::VecDot(grad_obj, grad_obj, dot),
        A2D::Eval(weight * detJ * (0.5 * dot - u), output));

    output.bvalue() = 1.0;
    stack.hextract(pgrad_v, hgrad_v, jac_grad);
    jac_vals(0, 0) = u.hvalue();  // This is 0.0
  }
};

#endif  // XCGD_POISSON_H