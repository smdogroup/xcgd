#pragma once

#include "ad/a2dmattrace.h"
#include "physics_commons.h"

template <typename T>
class LinearElasticity2DVonMisesStress final : public PhysicsBase<T, 2, 0, 2> {
 private:
  using PhysicsBase_s = PhysicsBase<T, 2, 0, 2>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  LinearElasticity2DVonMisesStress(T E, T nu)
      : mu(0.5 * E / (1.0 + nu)), lambda(E * nu / ((1.0 + nu) * (1.0 - nu))) {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& ____,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ, trS, detS, von_mises;
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatDet(J, detJ);
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);
    A2D::MatTrace(S, trS);
    A2D::MatDet(S, detS);
    von_mises = sqrt(trS * trS - 3.0 * detS);
    return von_mises;
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& u,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_u,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {}

 private:
  T mu, lambda;  // Lame parameters
  T vs_max;      // maximum Von Mises stress
};

template <typename T>
class LinearElasticity2DVonMisesStressAggregation final
    : public PhysicsBase<T, 2, 0, 2> {
 private:
  using PhysicsBase_s = PhysicsBase<T, 2, 0, 2>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  LinearElasticity2DVonMisesStressAggregation(double ksrho, T E, T nu,
                                              T vm_max = 0.0)
      : ksrho(ksrho),
        mu(0.5 * E / (1.0 + nu)),
        lambda(E * nu / ((1.0 + nu) * (1.0 - nu))),
        vm_max(vm_max) {}

  void set_von_mises_max_stress(T vm_max_) { vm_max = vm_max_; }
  double get_ksrho() { return ksrho; }

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& vals,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ, trS, detS, von_mises;
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatDet(J, detJ);
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);
    A2D::MatTrace(S, trS);
    A2D::MatDet(S, detS);
    von_mises = sqrt(trS * trS - 3.0 * detS);
    return weight * detJ * exp(ksrho * (von_mises - vm_max));
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& __,
                A2D::Vec<T, spatial_dim>& ___,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_vals,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    T detJ;
    A2D::MatDet(J, detJ);

    A2D::ADObj<T> trS, detS, output;
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj), A2D::MatTrace(S_obj, trS),
        A2D::MatDet(S_obj, detS),
        A2D::Eval(weight * detJ *
                      exp(ksrho * (sqrt(trS * trS - 3.0 * detS) - vm_max)),
                  output));

    output.bvalue() = 1.0;
    stack.reverse();
  }

  void jacobian_product(
      T weight, T _, A2D::Vec<T, spatial_dim>& __,
      A2D::Vec<T, spatial_dim>& ___, A2D::Mat<T, spatial_dim, spatial_dim>& J,
      A2D::Vec<T, dof_per_node>& ____,
      A2D::Mat<T, dof_per_node, spatial_dim>& grad,
      A2D::Vec<T, dof_per_node>& _____,
      A2D::Mat<T, dof_per_node, spatial_dim>& direct_grad,
      A2D::Vec<T, dof_per_node>& ______,
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    T detJ;
    A2D::MatDet(J, detJ);

    A2D::A2DObj<T> trS, detS, output;
    A2D::Mat<T, dof_per_node, spatial_dim> bgrad;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(
        grad, bgrad, direct_grad, coef_grad);
    A2D::A2DObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj), A2D::MatTrace(S_obj, trS),
        A2D::MatDet(S_obj, detS),
        A2D::Eval(weight * detJ *
                      exp(ksrho * (sqrt(trS * trS - 3.0 * detS) - vm_max)),
                  output));
    output.bvalue() = 1.0;
    stack.hproduct();
  }

 private:
  double ksrho;  // KS aggregation parameter
  T mu, lambda;  // Lame parameters
  T vm_max;      // maximum Von Mises stress
};
