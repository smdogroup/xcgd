#pragma once

#include "physics_commons.h"

/**
 * @brief The Dirichlet boundary condition for scalar problem (such as
 * Poisson's equation) applied on the cut boundary using Nitsche's method.
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

  /**
   * @param eta Nitsche parameter
   * @param bc_func functor with the following signature:
   *        T bc_func(A2D::Vec<T, spatial_dim> xloc)
   */
  CutDirichlet(double eta, const BCFunc& bc_func)
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
    T scale;
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
    A2D::ADObj<T> ngrad_obj, scale_obj, output_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::ADObj<T&> u_obj(u, coef_u);
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
    A2D::A2DObj<T> ngrad_obj, scale_obj, output_obj;
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
    A2D::A2DObj<T> ngrad_obj, scale_obj, output_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<T&> u_obj(u, ub, up, uh);
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
    jac_u = uh;
  }

 private:
  double eta;             // A sufficiently largeN itsche parameter
  const BCFunc& bc_func;  // the Dirichlet boundary value evaluator
};

/**
 * @brief The Dirichlet boundary condition for vector problem (such as
 * elasticity problem) applied on the cut boundary using Nitsche's method.
 */
template <typename T, int spatial_dim, int dim, class BCFunc>
class VectorCutDirichlet final : public PhysicsBase<T, spatial_dim, 0, dim> {
 private:
  static_assert(spatial_dim == 2,
                "This is only implemented for 2d problems for now");
  using PhysicsBase = PhysicsBase<T, spatial_dim, 0, dim>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  /**
   * @param eta Nitsche parameter
   * @param bc_func functor with the following signature:
   *        A2D::Vec<T, dof_per_node> bc_func(A2D::Vec<T, spatial_dim> xloc)
   */
  VectorCutDirichlet(double eta, const BCFunc& bc_func)
      : eta(eta), bc_func(bc_func) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& u,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Vec<T, dim> g(bc_func(xloc));

    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    A2D::MatVecMult(J, nrm_ref, nrm);

    // Compute the scaling from ref frame to physical frame
    T scale;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale);

    A2D::Vec<T, dof_per_node> ngrad;  // ∂u/∂n
    A2D::MatVecMult(grad, nrm, ngrad);

    T ngradu = 0.0, uu = 0.0, ngradg = 0.0, ug = 0.0;
    A2D::VecDot(ngrad, u, ngradu);
    A2D::VecDot(u, u, uu);
    A2D::VecDot(ngrad, g, ngradg);
    A2D::VecDot(u, g, ug);

    return weight * sqrt(scale) * (-ngradu + ngradg + eta * (0.5 * uu - ug));
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& u,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_u,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Vec<T, dim> g(bc_func(xloc));
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::ADObj<T> scale_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj, ug_obj;
    A2D::ADObj<A2D::Vec<T, dof_per_node>> ngrad_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::ADObj<A2D::Vec<T, dof_per_node>&> u_obj(u, coef_u);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
        A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
        A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
        A2D::VecDot(u_obj, u_obj, uu_obj),
        A2D::VecDot(ngrad_obj, g, ngradg_obj), A2D::VecDot(u_obj, g, ug_obj),
        A2D::Eval(
            weight * sqrt(scale_obj) *
                (-ngradu_obj + ngradg_obj + eta * (0.5 * uu_obj - ug_obj)),
            output_obj));
    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(
      T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
      A2D::Vec<T, spatial_dim>& nrm_ref,
      A2D::Mat<T, spatial_dim, spatial_dim>& J, A2D::Vec<T, dof_per_node>& u,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& direct_u,
      A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
      A2D::Vec<T, dof_per_node>& coef_u,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Vec<T, dim> g(bc_func(xloc));
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::Vec<T, dof_per_node> ub;
    A2D::Mat<T, dof_per_node, spatial_dim> bgrad;
    A2D::A2DObj<T> scale_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj,
        ug_obj;
    A2D::A2DObj<A2D::Vec<T, dof_per_node>> ngrad_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<A2D::Vec<T, dof_per_node>&> u_obj(u, ub, direct_u, coef_u);
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(
        grad, bgrad, direct_grad, coef_grad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
        A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
        A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
        A2D::VecDot(u_obj, u_obj, uu_obj),
        A2D::VecDot(ngrad_obj, g, ngradg_obj), A2D::VecDot(u_obj, g, ug_obj),
        A2D::Eval(
            weight * sqrt(scale_obj) *
                (-ngradu_obj + ngradg_obj + eta * (0.5 * uu_obj - ug_obj)),
            output_obj));
    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(
      T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
      A2D::Vec<T, spatial_dim>& nrm_ref,
      A2D::Mat<T, spatial_dim, spatial_dim>& J, A2D::Vec<T, dof_per_node>& u,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Mat<T, dof_per_node, dof_per_node>& jac_u,
      A2D::Mat<T, dof_per_node, dof_per_node * spatial_dim>& jac_mixed,
      A2D::Mat<T, dof_per_node * spatial_dim, dof_per_node * spatial_dim>&
          jac_grad) const {
    // Prepare quantities
    A2D::Vec<T, spatial_dim> tan_ref, nrm;
    A2D::Vec<T, dim> g(bc_func(xloc));
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);
    A2D::MatVecMult(J, nrm_ref, nrm);

    // Create quantites
    A2D::Vec<T, dof_per_node> ub, up, uh;
    A2D::Mat<T, dof_per_node, spatial_dim> bgrad, pgrad, hgrad;
    A2D::A2DObj<T> scale_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj,
        ug_obj;
    A2D::A2DObj<A2D::Vec<T, dof_per_node>> ngrad_obj;
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;

    A2D::A2DObj<A2D::Vec<T, dof_per_node>&> u_obj(u, ub, up, uh);
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad, bgrad,
                                                                  pgrad, hgrad);

    // Compute
    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale_obj),
        A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
        A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
        A2D::VecDot(u_obj, u_obj, uu_obj),
        A2D::VecDot(ngrad_obj, g, ngradg_obj), A2D::VecDot(u_obj, g, ug_obj),
        A2D::Eval(
            weight * sqrt(scale_obj) *
                (-ngradu_obj + ngradg_obj + eta * (0.5 * uu_obj - ug_obj)),
            output_obj));
    output_obj.bvalue() = 1.0;

    // Note: Extract Hessian w.r.t. ∇u: ∂2e/∂(∇_x)uq2 is zero
    // stack.hextract(pgrad, hgrad, jac_grad);
    //
    // Extract the Hessian w.r.t. u
    stack.hextract(up, uh, jac_u);

    for (int i = 0; i < dof_per_node; i++) {
      up.zero();
      hgrad.zero();
      stack.hzero();

      up[i] = 1.0;

      stack.hforward();
      stack.hreverse();

      // Extract the mixed Hessian w.r.t. u and ∇u:
      for (int j = 0; j < dof_per_node; j++) {
        for (int k = 0; k < spatial_dim; k++) {
          jac_mixed(i, j * spatial_dim + k) = hgrad(j, k);
        }
      }
    }
  }

 private:
  double eta;             // A sufficiently largeN itsche parameter
  const BCFunc& bc_func;  // the Dirichlet boundary value evaluator
};
