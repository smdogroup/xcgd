#ifndef XCGD_LINEAR_ELASTICITY_H
#define XCGD_LINEAR_ELASTICITY_H

#include "physics_commons.h"

// IntFunc: type of the internal (body) force functor
template <typename T, int spatial_dim_, class IntFunc>
class LinearElasticity final
    : public PhysicsBase<T, spatial_dim_, 1, spatial_dim_> {
 private:
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 1, spatial_dim_>;
  static_assert(spatial_dim_ == 3 or spatial_dim_ == 2,
                "LinearElasticity is only implemented for 2D and 3D problems");

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  LinearElasticity(T E, T nu, const IntFunc& int_func)
      : mu(0.5 * E / (1.0 + nu)),
        lambda(spatial_dim == 3 ? E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                                : E * nu / ((1.0 + nu) * (1.0 - nu))),
        int_func(int_func) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& u,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ, energy, potential;
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::Vec<T, dof_per_node> g(int_func(xloc));

    A2D::MatDet(J, detJ);
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);
    A2D::SymMatMultTrace(E, S, energy);
    A2D::VecDot(g, u, potential);
    T output = weight * detJ * (0.5 * energy - potential);
    return output;
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& u,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_u,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    A2D::Vec<T, dof_per_node> g(int_func(xloc));
    A2D::ADObj<T> detJ_obj, energy_obj, potential_obj, output_obj;
    A2D::ADObj<A2D::Vec<T, dof_per_node>&> u_obj(u, coef_u);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>> J_obj(J);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj),
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::SymMatMultTrace(E_obj, S_obj, energy_obj),
        A2D::VecDot(g, u_obj, potential_obj),
        A2D::Eval(weight * detJ_obj * (0.5 * energy_obj - potential_obj),
                  output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(
      T weight, T _, A2D::Vec<T, spatial_dim>& __,
      A2D::Vec<T, spatial_dim>& ___, A2D::Mat<T, spatial_dim, spatial_dim>& J,
      A2D::Vec<T, dof_per_node>& vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& direct_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
      A2D::Vec<T, dof_per_node>& coef_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    A2D::Mat<T, dof_per_node, spatial_dim> bgrad;
    A2D::A2DObj<T> detJ_obj, energy_obj, output_obj;
    A2D::A2DObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>> J_obj(J);
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(
        grad, bgrad, direct_grad, coef_grad);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj),
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::SymMatMultTrace(E_obj, S_obj, energy_obj),
        A2D::Eval(0.5 * weight * detJ_obj * energy_obj, output_obj));

    output_obj.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& __,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Mat<T, dof_per_node, dof_per_node>& jac_vals,
                A2D::Mat<T, dof_per_node, dof_per_node * spatial_dim>& ____,
                A2D::Mat<T, dof_per_node * spatial_dim,
                         dof_per_node * spatial_dim>& jac_grad) const {
    A2D::Mat<T, dof_per_node, spatial_dim> bgrad, pgrad, hgrad;
    A2D::A2DObj<T> detJ_obj, energy_obj, output_obj;
    A2D::A2DObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>> J_obj(J);
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad, bgrad,
                                                                  pgrad, hgrad);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj),
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::SymMatMultTrace(E_obj, S_obj, energy_obj),
        A2D::Eval(0.5 * weight * detJ_obj * energy_obj, output_obj));

    output_obj.bvalue() = 1.0;
    stack.hextract(pgrad, hgrad, jac_grad);
  }

 private:
  T mu, lambda;  // Lame parameters
  const IntFunc& int_func;
};

template <typename T, int spatial_dim_, class LoadFunc>
class ElasticityExternalLoad final
    : public PhysicsBase<T, spatial_dim_, 1, spatial_dim_> {
 private:
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 1, spatial_dim_>;
  static_assert(spatial_dim_ == 3 or spatial_dim_ == 2,
                "LinearElasticity is only implemented for 2D and 3D problems");

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  ElasticityExternalLoad(const LoadFunc& load_func) : load_func(load_func) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& u,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    A2D::Vec<T, dof_per_node> load(load_func(xloc));

    // Get tangent direction in the reference frame
    A2D::Vec<T, spatial_dim> tan_ref;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    // Compute the scaling from ref frame to physical frame
    T scale2;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale2);

    T dot;
    A2D::VecDot(load, u, dot);
    T output = -weight * sqrt(scale2) * dot;

    return output;
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& u,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_u,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    A2D::Vec<T, dof_per_node> load(load_func(xloc));

    // Get tangent direction in the reference frame
    A2D::Vec<T, spatial_dim> tan_ref;
    A2D::Mat<T, spatial_dim, spatial_dim> rot;
    rot(0, 1) = -1.0;
    rot(1, 0) = 1.0;
    A2D::MatVecMult(rot, nrm_ref, tan_ref);

    // Create AD objects
    A2D::ADObj<T> scale2_obj, dot_obj, output_obj;
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J), JTJ_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> JTJdt_obj;
    A2D::ADObj<A2D::Vec<T, dof_per_node>&> u_obj(u, coef_u);

    auto stack = A2D::MakeStack(
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J_obj, J_obj,
                                                                   JTJ_obj),
        A2D::MatVecMult(JTJ_obj, tan_ref, JTJdt_obj),
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::VecDot(load, u_obj, dot_obj),
        A2D::Eval(-weight * sqrt(scale2_obj) * dot_obj, output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(
      T weight, T _, A2D::Vec<T, spatial_dim>& __,
      A2D::Vec<T, spatial_dim>& ___, A2D::Mat<T, spatial_dim, spatial_dim>& J,
      A2D::Vec<T, dof_per_node>& vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& direct_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
      A2D::Vec<T, dof_per_node>& coef_vals,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {}

 private:
  const LoadFunc& load_func;
};

/**
 * @brief The Dirichlet boundary condition for linear elasticity problem applied
 * on the cut boundary using Nitsche's method.
 */
template <typename T, int spatial_dim_, int dim, class BCFunc>
class LinearElasticityCutDirichlet final
    : public PhysicsBase<T, spatial_dim_, 0, dim> {
 private:
  static_assert(spatial_dim_ == 2,
                "This is only implemented for 2d problems for now");
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, dim>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  /**
   * @param eta Nitsche parameter
   * @param bc_func functor with the following signature:
   *        A2D::Vec<T, dof_per_node> bc_func(A2D::Vec<T, spatial_dim> xloc)
   */
  LinearElasticityCutDirichlet(double eta, const BCFunc& bc_func)
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
    T scale2;
    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> JTJdt;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, tan_ref, JTJdt);
    A2D::VecDot(tan_ref, JTJdt, scale2);

    A2D::Vec<T, dof_per_node> ngrad;  // ∂u/∂n
    A2D::MatVecMult(grad, nrm, ngrad);

    T ngradu = 0.0, uu = 0.0, ngradg = 0.0, ug = 0.0;
    A2D::VecDot(ngrad, u, ngradu);
    A2D::VecDot(u, u, uu);
    A2D::VecDot(ngrad, g, ngradg);
    A2D::VecDot(u, g, ug);

    return weight * sqrt(scale2) * (-ngradu + ngradg + eta * (0.5 * uu - ug));
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
    A2D::ADObj<T> scale2_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj,
        ug_obj;
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
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
        A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
        A2D::VecDot(u_obj, u_obj, uu_obj),
        A2D::VecDot(ngrad_obj, g, ngradg_obj), A2D::VecDot(u_obj, g, ug_obj),
        A2D::Eval(
            weight * sqrt(scale2_obj) *
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
    A2D::A2DObj<T> scale2_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj,
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
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
        A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
        A2D::VecDot(u_obj, u_obj, uu_obj),
        A2D::VecDot(ngrad_obj, g, ngradg_obj), A2D::VecDot(u_obj, g, ug_obj),
        A2D::Eval(
            weight * sqrt(scale2_obj) *
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
    A2D::A2DObj<T> scale2_obj, output_obj, ngradu_obj, uu_obj, ngradg_obj,
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
        A2D::VecDot(tan_ref, JTJdt_obj, scale2_obj),
        A2D::MatVecMult(grad_obj, nrm, ngrad_obj),
        A2D::VecDot(ngrad_obj, u_obj, ngradu_obj),
        A2D::VecDot(u_obj, u_obj, uu_obj),
        A2D::VecDot(ngrad_obj, g, ngradg_obj), A2D::VecDot(u_obj, g, ug_obj),
        A2D::Eval(
            weight * sqrt(scale2_obj) *
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

// Energy norm error ||e|| = [∫(s - sh)^T(s - sh)dΩ]^0.5
template <typename T, int spatial_dim_, class StressFun>
class LinearElasticityEnergyNormError final
    : public PhysicsBase<T, spatial_dim_, 1, spatial_dim_> {
 private:
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 1, spatial_dim_>;
  static_assert(spatial_dim_ == 3 or spatial_dim_ == 2,
                "LinearElasticityEnergyNormError is only implemented for 2D "
                "and 3D problems");

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  // stress_fun returns a symmetric stress tensor
  LinearElasticityEnergyNormError(T E, T nu, const StressFun& stress_fun)
      : mu(0.5 * E / (1.0 + nu)),
        lambda(spatial_dim == 3 ? E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                                : E * nu / ((1.0 + nu) * (1.0 - nu))),
        stress_fun(stress_fun) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& u,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    // det
    T detJ;
    A2D::MatDet(J, detJ);

    // numerical stress
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);

    // Exact stress
    A2D::SymMat<T, spatial_dim> S_exact = stress_fun(xloc);

    // Diff in "energy"
    // Note: we take a shortcut here and are not using the exact strain energy
    // tr(E^TS) instead, we compute tr(S^TS) for simplicity
    A2D::SymMat<T, spatial_dim> S_diff;
    A2D::MatSum(1.0, S, -1.0, S_exact, S_diff);
    T energy_diff;
    A2D::SymMatMultTrace(S_diff, S_diff, energy_diff);

    return weight * detJ * energy_diff;
  }

 private:
  T mu, lambda;  // Lame parameters
  const StressFun& stress_fun;
};

#endif  // XCGD_LINEAR_ELASTICITY_H
