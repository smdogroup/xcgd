#ifndef XCGD_LINEAR_ELASTICITY_H
#define XCGD_LINEAR_ELASTICITY_H

#include "physics_commons.h"

template <typename T, int spatial_dim>
class LinearElasticity final
    : public PhysicsBase<T, spatial_dim, 1, spatial_dim> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim, 1, spatial_dim>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  LinearElasticity(T E, T nu) {
    mu = 0.5 * E / (1.0 + nu);
    lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  }

  T energy(T weight, T _, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& __,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ, energy;
    A2D::SymMat<T, spatial_dim> E, S;

    A2D::MatDet(J, detJ);
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);
    A2D::SymMatMultTrace(E, S, energy);
    T output = 0.5 * weight * detJ * energy;
    return output;
  }

  void residual(T weight, T _, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& __,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    A2D::ADObj<T> detJ_obj, energy_obj, output_obj;
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>> J_obj(J);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);

    auto stack = A2D::MakeStack(
        A2D::MatDet(J_obj, detJ_obj),
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::SymMatMultTrace(E_obj, S_obj, energy_obj),
        A2D::Eval(0.5 * weight * detJ_obj * energy_obj, output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(
      T weight, T _, A2D::Mat<T, spatial_dim, spatial_dim>& J,
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

  void jacobian(T weight, T _, A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Mat<T, dof_per_node, dof_per_node>& jac_vals,
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
};

#endif  // XCGD_LINEAR_ELASTICITY_H