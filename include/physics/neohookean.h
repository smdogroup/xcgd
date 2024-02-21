#ifndef XCGD_NEOHOOKEAN_H
#define XCGD_NEOHOOKEAN_H

#include <cmath>

#include "a2dcore.h"

/* Neohookean physics implemented using A2D, i.e. residual and Jacobian are
automatically differentiated */
template <typename T, int spatial_dim = 3>
class NeohookeanPhysics {
 public:
  static constexpr int dof_per_node = spatial_dim;

  T C1, D1;  // Constitutitive data

  NeohookeanPhysics(T C1, T D1) : C1(C1), D1(D1) {}

  T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J_mat,
           A2D::Vec<T, dof_per_node>& vals,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad_mat) {
    A2D::Mat<T, spatial_dim, spatial_dim> Jinv_mat, F_mat, FTF_mat, I_mat;

    // Identity matrix
    for (int i = 0; i < spatial_dim; i++) {
      I_mat(i, i) = 1.0;
    }

    T detJ;
    A2D::MatDet(J_mat, detJ);
    A2D::MatInv(J_mat, Jinv_mat);

    A2D::MatMatMult(grad_mat, Jinv_mat, F_mat);
    A2D::MatSum(F_mat, I_mat, F_mat);

    // Compute the invariants
    T detF;
    A2D::MatDet(F_mat, detF);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_mat, F_mat,
                                                               FTF_mat);
    T trace;
    A2D::MatTrace(FTF_mat, trace);

    // Compute the energy density for the model
    T energy_density = C1 * (trace - 3.0 - 2.0 * std::log(detF)) +
                       D1 * (detF - 1.0) * (detF - 1.0);

    return weight * detJ * energy_density;
  }

  void residual(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) {
    // Identity matrix
    A2D::Mat<T, spatial_dim, spatial_dim> I;
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_mat(J);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_mat(grad,
                                                                 coef_grad);
    A2D::ADObj<A2D::Mat<T, spatial_dim, spatial_dim>> Jinv_mat, F_mat, FTF_mat,
        I_mat(I);

    A2D::ADObj<T> detJ, detF, trace, energy_density, output;

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_mat, detJ), A2D::MatInv(J_mat, Jinv_mat),
        A2D::MatMatMult(grad_mat, Jinv_mat, F_mat),
        A2D::MatSum(F_mat, I_mat, F_mat), A2D::MatDet(F_mat, detF),
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_mat, F_mat,
                                                                   FTF_mat),
        A2D::MatTrace(FTF_mat, trace),
        A2D::Eval(C1 * (trace - 3.0 - 2.0 * A2D::log(detF)) +
                      D1 * (detF - 1.0) * (detF - 1.0),
                  energy_density),
        A2D::Eval(0.5 * weight * energy_density * detJ, output));

    output.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                        A2D::Vec<T, dof_per_node>& vals,
                        A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                        A2D::Vec<T, dof_per_node>& direct_vals,
                        A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
                        A2D::Vec<T, dof_per_node>& coef_vals,
                        A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) {
    // Identity matrix
    A2D::Mat<T, spatial_dim, spatial_dim> I;
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    A2D::Mat<T, dof_per_node, spatial_dim> bgrad;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_mat(
        grad, bgrad, direct_grad, coef_grad);

    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_mat(J);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> Jinv_mat, F_mat, FTF_mat,
        I_mat(I);

    A2D::A2DObj<T> detJ, detF, trace, energy_density, output;

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_mat, detJ), A2D::MatInv(J_mat, Jinv_mat),
        A2D::MatMatMult(grad_mat, Jinv_mat, F_mat),
        A2D::MatSum(F_mat, I_mat, F_mat), A2D::MatDet(F_mat, detF),
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_mat, F_mat,
                                                                   FTF_mat),
        A2D::MatTrace(FTF_mat, trace),
        A2D::Eval(C1 * (trace - 3.0 - 2.0 * A2D::log(detF)) +
                      D1 * (detF - 1.0) * (detF - 1.0),
                  energy_density),
        A2D::Eval(0.5 * weight * energy_density * detJ, output));

    output.bvalue() = 1.0;
    stack.hproduct();
  }

  void jacobian(T weight, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Mat<T, dof_per_node, dof_per_node>& jac_vals,
                A2D::Mat<T, dof_per_node * spatial_dim,
                         dof_per_node * spatial_dim>& jac_grad) {
    // Identity matrix
    A2D::Mat<T, spatial_dim, spatial_dim> I;
    for (int i = 0; i < spatial_dim; i++) {
      I(i, i) = 1.0;
    }

    A2D::Mat<T, dof_per_node, spatial_dim> bgrad, direct, coef;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_mat(grad, bgrad,
                                                                  direct, coef);

    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> J_mat(J);
    A2D::A2DObj<A2D::Mat<T, spatial_dim, spatial_dim>> Jinv_mat, F_mat, FTF_mat,
        I_mat(I);

    A2D::A2DObj<T> detJ, detF, trace, energy_density, output;

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_mat, detJ), A2D::MatInv(J_mat, Jinv_mat),
        A2D::MatMatMult(grad_mat, Jinv_mat, F_mat),
        A2D::MatSum(F_mat, I_mat, F_mat), A2D::MatDet(F_mat, detF),
        A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(F_mat, F_mat,
                                                                   FTF_mat),
        A2D::MatTrace(FTF_mat, trace),
        A2D::Eval(C1 * (trace - 3.0 - 2.0 * A2D::log(detF)) +
                      D1 * (detF - 1.0) * (detF - 1.0),
                  energy_density),
        A2D::Eval(0.5 * weight * energy_density * detJ, output));

    output.bvalue() = 1.0;
    stack.hextract(direct, coef, jac_grad);
  }
};

#endif  // XCGD_NEOHOOKEAN_H