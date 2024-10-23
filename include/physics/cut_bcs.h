#pragma once

#include "physics_commons.h"

/**
 * @brief The Dirichlet boundary condition applied on the cut boundary using
 * Nitsche's method.
 */
template <typename T, int spatial_dim, class BCFunc>
class CutDirichlet final : public PhysicsBase<T, spatial_dim, 0, 1> {
 private:
  static_assert(spatial_dim == 2,
                "This is only implemented for 2d problems for now");
  using PhysicsBase = PhysicsBase<T, spatial_dim, 0, 1>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  CutDirichlet(double eta, const BCFunc& bc_func)
      : eta(eta), bc_func(bc_func) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    A2D::Vec<T, spatial_dim> tan_ref, nrm;

    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    A2D::MatVecMult(J, nrm_ref, nrm);

    // Compute the scaling from ref frame to physical frame
    T scale, u = val;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale);

    T ngrad;
    A2D::VecDot(grad, nrm, ngrad);

    return weight * sqrt(scale) *
           (-ngrad * u + 0.5 * eta * u * u + bc_func(xloc) * (ngrad - eta * u));
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& coef_val,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::ADObj<T> ngrad_obj, scale_obj, output_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::ADObj<T&> u_obj(val, coef_val);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale_obj) *
                      (-ngrad_obj * u_obj + 0.5 * eta * u_obj * u_obj +
                       bc_func(xloc) * (ngrad_obj - eta * u_obj)),
                  output_obj));
    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                        A2D::Vec<T, spatial_dim>& nrm_ref,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_val,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_val,
                        A2D::Vec<T, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    T ub = 0.0;
    A2D::Vec<T, spatial_dim> bgrad;
    A2D::A2DObj<T> ngrad_obj, scale_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(val, ub, direct_val, coef_val);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, direct_grad,
                                                    coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale_obj) *
                      (-ngrad_obj * u_obj + 0.5 * eta * u_obj * u_obj +
                       bc_func(xloc) * (ngrad_obj - eta * u_obj)),
                  output_obj));
    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
                A2D::Vec<T, spatial_dim>& grad, T& jac_val,
                A2D::Vec<T, spatial_dim>& jac_mixed,
                A2D::Mat<T, spatial_dim, spatial_dim>& jac_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    T ub = 0.0, up = 0.0, uh = 0.0;
    A2D::Vec<T, spatial_dim> bgrad, pgrad, hgrad;
    A2D::A2DObj<T> ngrad_obj, scale_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(val, ub, up, uh);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, pgrad, hgrad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale_obj) *
                      (-ngrad_obj * u_obj + 0.5 * eta * u_obj * u_obj +
                       bc_func(xloc) * (ngrad_obj - eta * u_obj)),
                  output_obj));
    output_obj.bvalue() = 1.0;

    // Note: this part is zero for this physics
    // // Extract Hessian w.r.t. ∇u: ∂2e/∂(∇_x)uq2
    // stack.hextract(pgrad, hgrad, jac_grad);

    stack.reverse();

    // Extract the mixed Hessian w.r.t. u and ∇u:
    hgrad.zero();
    stack.hzero();
    up = 1.0;

    stack.hforward();
    stack.hreverse();

    for (int d = 0; d < spatial_dim; d++) {
      jac_mixed(d) = hgrad(d);
    }

    // Extract the Hessian w.r.t. u
    jac_val = uh;
  }

 private:
  double eta;             // A sufficiently largeN itsche parameter
  const BCFunc& bc_func;  // the Dirichlet boundary value evaluator
};
