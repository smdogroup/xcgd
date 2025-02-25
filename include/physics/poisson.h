#ifndef XCGD_POISSON_H
#define XCGD_POISSON_H

#include "physics_commons.h"

// Poisson's equation:  Δu = f
template <typename T, int spatial_dim_, class SourceFunc>
class PoissonPhysics final : public PhysicsBase<T, spatial_dim_, 0, 1> {
 private:
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, 1>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  /**
   * @param source_fun [in] the source term callable that takes in
   * const A2D::Vec<T, spatial_dim>& and returns T
   */
  PoissonPhysics(const SourceFunc& source_fun) : source_fun(source_fun) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& __,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ, output, dot, u = val;
    A2D::MatDet(J, detJ);
    A2D::VecDot(grad, grad, dot);
    return weight * detJ * (0.5 * dot + source_fun(xloc) * u);
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
            weight * detJ_obj * (0.5 * dot_obj + source_fun(xloc) * u_obj),
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
            weight * detJ_obj * (0.5 * dot_obj + source_fun(xloc) * u_obj),
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
            weight * detJ_obj * (0.5 * dot_obj + source_fun(xloc) * u_obj),
            output_obj));

    output_obj.bvalue() = 1.0;
    stack.hextract(pgrad, hgrad, jac_grad);
  }

 private:
  const SourceFunc& source_fun;
};

/**
 * @brief The Dirichlet boundary condition for Poisson's equation applied on
 * the cut boundary using Nitsche's method.
 */
template <typename T, int spatial_dim_, class BCFunc>
class PoissonCutDirichlet final : public PhysicsBase<T, spatial_dim_, 0, 1> {
 private:
  static_assert(spatial_dim_ == 2,
                "This is only implemented for 2d problems for now");
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, 1>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  /**
   * @param eta Nitsche parameter
   * @param bc_func functor with the following signature:
   *        T bc_func(A2D::Vec<T, spatial_dim> xloc)
   */
  PoissonCutDirichlet(double eta, const BCFunc& bc_func)
      : eta(eta), bc_func(bc_func) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
           A2D::Vec<T, spatial_dim>& grad) const {
    A2D::Vec<T, spatial_dim> tan_ref, nrm;

    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    A2D::MatVecMult(J, nrm_ref, nrm);

    // Compute the scaling from ref frame to physical frame
    T scale2;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale2);

    T ngrad;
    A2D::VecDot(grad, nrm, ngrad);

    return weight * sqrt(scale2) *
           (-ngrad * u + 0.5 * eta * u * u + bc_func(xloc) * (ngrad - eta * u));
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                A2D::Vec<T, spatial_dim>& grad, T& coef_u,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::ADObj<T> ngrad_obj, scale2_obj, output_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::ADObj<T&> u_obj(u, coef_u);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale2_obj) *
                      (-ngrad_obj * u_obj + 0.5 * eta * u_obj * u_obj +
                       bc_func(xloc) * (ngrad_obj - eta * u_obj)),
                  output_obj));
    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                        A2D::Vec<T, spatial_dim>& nrm_ref,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_u,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_u,
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
    A2D::A2DObj<T> ngrad_obj, scale2_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, direct_u, coef_u);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, direct_grad,
                                                    coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale2_obj) *
                      (-ngrad_obj * u_obj + 0.5 * eta * u_obj * u_obj +
                       bc_func(xloc) * (ngrad_obj - eta * u_obj)),
                  output_obj));
    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                A2D::Vec<T, spatial_dim>& grad, T& jac_u,
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
    A2D::A2DObj<T> ngrad_obj, scale2_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, up, uh);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, pgrad, hgrad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale2_obj) *
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
    jac_u = uh;
  }

 private:
  double eta;             // A sufficiently largeN itsche parameter
  const BCFunc& bc_func;  // the Dirichlet boundary value evaluator
};

/**
 * @brief The physics to assemble K matrix eigenvalue system that determines the
 * Nitsche parameter
 */
template <typename T, int spatial_dim_>
class PoissonCutDirichletNitscheParameterEigK final
    : public PhysicsBase<T, spatial_dim_, 0, 1> {
 private:
  static_assert(spatial_dim_ == 2,
                "This is only implemented for 2d problems for now");
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, 1>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
           A2D::Vec<T, spatial_dim>& grad) const {
    A2D::Vec<T, spatial_dim> tan_ref, nrm;

    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    A2D::MatVecMult(J, nrm_ref, nrm);

    // Compute the scaling from ref frame to physical frame
    T scale2;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale2);

    T ngrad;
    A2D::VecDot(grad, nrm, ngrad);

    return weight * sqrt(scale2) * 0.5 * (ngrad * ngrad);
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                A2D::Vec<T, spatial_dim>& grad, T& coef_u,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::ADObj<T> ngrad_obj, scale2_obj, output_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::ADObj<T&> u_obj(u, coef_u);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale2_obj) * 0.5 * ngrad_obj * ngrad_obj,
                  output_obj));
    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                        A2D::Vec<T, spatial_dim>& nrm_ref,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_u,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_u,
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
    A2D::A2DObj<T> ngrad_obj, scale2_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, direct_u, coef_u);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, direct_grad,
                                                    coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale2_obj) * 0.5 * ngrad_obj * ngrad_obj,
                  output_obj));
    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                A2D::Vec<T, spatial_dim>& grad, T& jac_u,
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
    A2D::A2DObj<T> ngrad_obj, scale2_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, up, uh);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, pgrad, hgrad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, nrm, ngrad_obj),
        A2D::Eval(weight * sqrt(scale2_obj) * 0.5 * ngrad_obj * ngrad_obj,
                  output_obj));
    output_obj.bvalue() = 1.0;

    // Extract Hessian w.r.t. ∇u: ∂2e/∂(∇_x)uq2
    stack.hextract(pgrad, hgrad, jac_grad);

    stack.reverse();

    // Note: this part is zero for this physics
    // // Extract the mixed Hessian w.r.t. u and ∇u:
    // hgrad.zero();
    // stack.hzero();
    // up = 1.0;
    //
    // stack.hforward();
    // stack.hreverse();
    //
    // for (int d = 0; d < spatial_dim; d++) {
    //   jac_mixed(d) = hgrad(d);
    // }
    //
    // // Extract the Hessian w.r.t. u
    // jac_u = uh;
  }
};

/**
 * @brief The physics to assemble M matrix eigenvalue system that determines the
 * Nitsche parameter
 */
template <typename T, int spatial_dim_>
class PoissonCutDirichletNitscheParameterEigM final
    : public PhysicsBase<T, spatial_dim_, 0, 1> {
 private:
  static_assert(spatial_dim_ == 2,
                "This is only implemented for 2d problems for now");
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, 1>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
           A2D::Vec<T, spatial_dim>& grad) const {
    A2D::Vec<T, spatial_dim> tan_ref, nrm;

    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    A2D::MatVecMult(J, nrm_ref, nrm);

    // Compute the scaling from ref frame to physical frame
    T scale2;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale2);

    T dot;
    A2D::VecDot(grad, grad, dot);

    return weight * sqrt(scale2) * 0.5 * dot;
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                A2D::Vec<T, spatial_dim>& grad, T& coef_u,
                A2D::Vec<T, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::ADObj<T> dot_obj, scale2_obj, output_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::ADObj<T&> u_obj(u, coef_u);
    A2D::ADObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(weight * sqrt(scale2_obj) * 0.5 * dot_obj, output_obj));
    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                        A2D::Vec<T, spatial_dim>& nrm_ref,
                        A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                        A2D::Vec<T, spatial_dim>& grad, T& direct_u,
                        A2D::Vec<T, spatial_dim>& direct_grad, T& coef_u,
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
    A2D::A2DObj<T> dot_obj, scale2_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, direct_u, coef_u);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, direct_grad,
                                                    coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(weight * sqrt(scale2_obj) * 0.5 * dot_obj, output_obj));
    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J, T& u,
                A2D::Vec<T, spatial_dim>& grad, T& jac_u,
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
    A2D::A2DObj<T> dot_obj, scale2_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, up, uh);
    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> grad_obj(grad, bgrad, pgrad, hgrad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(grad_obj, grad_obj, dot_obj),
        A2D::Eval(weight * sqrt(scale2_obj) * 0.5 * dot_obj, output_obj));
    output_obj.bvalue() = 1.0;

    // Extract Hessian w.r.t. ∇u: ∂2e/∂(∇_x)uq2
    stack.hextract(pgrad, hgrad, jac_grad);

    // Note: this part is zero for this physics
    // stack.reverse();
    //
    // // Extract the mixed Hessian w.r.t. u and ∇u:
    // hgrad.zero();
    // stack.hzero();
    // up = 1.0;
    //
    // stack.hforward();
    // stack.hreverse();
    //
    // for (int d = 0; d < spatial_dim; d++) {
    //   jac_mixed(d) = hgrad(d);
    // }
    //
    // // Extract the Hessian w.r.t. u
    // jac_u = uh;
  }
};

// Energy norm error ||e|| = [∫[(u - uh)^2 + (du - duh)^T(du - duh)]dΩ]^0.5
template <typename T, int spatial_dim, class u_fun_t, class stress_fun_t>
class PoissonEnergyNormError final : public PhysicsBase<T, spatial_dim, 0, 1> {
 public:
  PoissonEnergyNormError(const u_fun_t& u_fun, const stress_fun_t& stress_fun,
                         double c1 = 1.0, double c2 = 1.0)
      : u_fun(u_fun), stress_fun(stress_fun), c1(c1), c2(c2) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ;
    A2D::MatDet(J, detJ);

    T u_diff = u_fun(xloc) - val;

    A2D::Vec<T, spatial_dim> du_diff;
    for (int d = 0; d < spatial_dim; d++) {
      du_diff(d) = stress_fun(xloc)(d) - grad(d);
    }
    T du_diff_dot = 0.0;
    A2D::VecDot(du_diff, du_diff, du_diff_dot);

    return weight * detJ * (c1 * u_diff * u_diff + c2 * du_diff_dot);
  }

 private:
  const u_fun_t& u_fun;
  const stress_fun_t& stress_fun;
  double c1, c2;
};

#endif  // XCGD_POISSON_H
