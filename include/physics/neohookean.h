#ifndef XCGD_NEOHOOKEAN_H
#define XCGD_NEOHOOKEAN_H

#include <cmath>

#include "physics_commons.h"

/* Neohookean physics implemented using A2D, i.e. residual and Jacobian are
automatically differentiated */
template <typename T, int spatial_dim>
class NeohookeanPhysics final
    : public PhysicsBase<T, spatial_dim, 0, spatial_dim> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim, 0, spatial_dim>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  NeohookeanPhysics(T C1, T D1) : C1(C1), D1(D1) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& vals,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    A2D::Mat<T, spatial_dim, spatial_dim> Jinv, F, FTF, I;

    // Identity matrix
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    T detJ;
    A2D::MatDet(J, detJ);
    A2D::MatInv(J, Jinv);
    A2D::MatMatMult(grad, Jinv, F);
    A2D::MatSum(F, I, F);

    // Compute the invariants
    T detF;
    A2D::MatDet(F, detF);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F, F, FTF);
    T trace;
    A2D::MatTrace(FTF, trace);

    // Compute the energy density for the model
    T energy_density = C1 * (trace - 3.0 - 2.0 * std::log(detF)) +
                       D1 * (detF - 1.0) * (detF - 1.0);

    return weight * detJ * energy_density;
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& __,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    // Identity matrix
    A2D::Mat<T, spatial_dim, spatial_dim> I;
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> Jinv_obj, F_obj, FTF_obj,
        I_obj(I);

    A2D::ADObj<T> detJ, detF, trace, energy_density, output;

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ), A2D::MatInv(J_obj, Jinv_obj),
        A2D::MatMatMult(grad_obj, Jinv_obj, F_obj),
        A2D::MatSum(F_obj, I_obj, F_obj), A2D::MatDet(F_obj, detF),
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_obj, F_obj,
                                                                   FTF_obj),
        A2D::MatTrace(FTF_obj, trace),
        A2D::Eval(C1 * (trace - 3.0 - 2.0 * A2D::log(detF)) +
                      D1 * (detF - 1.0) * (detF - 1.0),
                  energy_density),
        A2D::Eval(0.5 * weight * energy_density * detJ, output));

    output.bvalue() = 1.0;
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
    // Identity matrix
    A2D::Mat<T, spatial_dim, spatial_dim> I;
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    A2D::Mat<T, dof_per_node, spatial_dim> bgrad;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(
        grad, bgrad, direct_grad, coef_grad);

    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> Jinv_obj, F_obj, FTF_obj,
        I_obj(I);

    A2D::A2DObj<T> detJ, detF, trace, energy_density, output;

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ), A2D::MatInv(J_obj, Jinv_obj),
        A2D::MatMatMult(grad_obj, Jinv_obj, F_obj),
        A2D::MatSum(F_obj, I_obj, F_obj), A2D::MatDet(F_obj, detF),
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_obj, F_obj,
                                                                   FTF_obj),
        A2D::MatTrace(FTF_obj, trace),
        A2D::Eval(C1 * (trace - 3.0 - 2.0 * A2D::log(detF)) +
                      D1 * (detF - 1.0) * (detF - 1.0),
                  energy_density),
        A2D::Eval(0.5 * weight * energy_density * detJ, output));

    output.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, T _, A2D::Vec<T, spatial_dim>& __,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Mat<T, dof_per_node, dof_per_node>& jac_vals,
                A2D::Mat<T, dof_per_node * spatial_dim,
                         dof_per_node * spatial_dim>& jac_grad) const {
    // Identity matrix
    A2D::Mat<T, spatial_dim, spatial_dim> I;
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    A2D::Mat<T, dof_per_node, spatial_dim> bgrad, direct, coef;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad, bgrad,
                                                                  direct, coef);

    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_obj(J);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> Jinv_obj, F_obj, FTF_obj,
        I_obj(I);

    A2D::A2DObj<T> detJ, detF, trace, energy_density, output;

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ), A2D::MatInv(J_obj, Jinv_obj),
        A2D::MatMatMult(grad_obj, Jinv_obj, F_obj),
        A2D::MatSum(F_obj, I_obj, F_obj), A2D::MatDet(F_obj, detF),
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_obj, F_obj,
                                                                   FTF_obj),
        A2D::MatTrace(FTF_obj, trace),
        A2D::Eval(C1 * (trace - 3.0 - 2.0 * A2D::log(detF)) +
                      D1 * (detF - 1.0) * (detF - 1.0),
                  energy_density),
        A2D::Eval(0.5 * weight * energy_density * detJ, output));

    output.bvalue() = 1.0;
    stack.hextract(direct, coef, jac_grad);
  }

 private:
  T C1, D1;  // Constitutive data
};

#endif  // XCGD_NEOHOOKEAN_H
