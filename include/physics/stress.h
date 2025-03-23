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

  T energy(T _____, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& ____,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T trS, detS, von_mises;
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);
    A2D::MatTrace(S, trS);
    A2D::MatDet(S, detS);
    von_mises = sqrt(trS * trS - 3.0 * detS);
    return von_mises;
  }

  void residual(T _____, T _, A2D::Vec<T, spatial_dim>& xloc,
                A2D::Vec<T, spatial_dim>& __,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& u,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& coef_u,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {}

 private:
  T mu, lambda;  // Lame parameters
};

/*
 * Evaluates the approximated maximum Von Mises stress over a domain using
 * continuous KS aggregation:
 *
 *   max s ~= m + 1/ρ * ln(∫exp(ρ * (s - m))dΩ)
 *
 * where m = max(s) to prevent floating-point number overflow for
 * exponential, ρ is the ks parameter, s = Von Mises stress / yield stress
 *
 * */
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
                                              T yield_stress,
                                              T max_stress_ratio = 1.0,
                                              bool use_discrete_ks = false)
      : ksrho(ksrho),
        mu(0.5 * E / (1.0 + nu)),
        lambda(E * nu / ((1.0 + nu) * (1.0 - nu))),
        yield_stress(yield_stress),
        max_stress_ratio(max_stress_ratio),
        use_discrete_ks(use_discrete_ks) {}

  // Does not effect result, but set a proper value can help preventing
  // floating-point overflow
  void set_max_stress_ratio(T max_stress_ratio_) {
    max_stress_ratio = max_stress_ratio_;
  }
  T get_yield_stress() const { return yield_stress; }

  void set_ksrho(double ksrho_) { ksrho = ksrho_; }
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

    T coef = use_discrete_ks ? 1.0 : weight * detJ;
    return coef * exp(ksrho * (von_mises / yield_stress - max_stress_ratio));
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
    T coef = use_discrete_ks ? 1.0 : weight * detJ;

    A2D::ADObj<T> trS, detS, output;
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj), A2D::MatTrace(S_obj, trS),
        A2D::MatDet(S_obj, detS),
        A2D::Eval(
            coef * exp(ksrho * (sqrt(trS * trS - 3.0 * detS) / yield_stress -
                                max_stress_ratio)),
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
    T coef = use_discrete_ks ? 1.0 : weight * detJ;

    A2D::A2DObj<T> trS, detS, output;
    A2D::Mat<T, dof_per_node, spatial_dim> bgrad;
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(
        grad, bgrad, direct_grad, coef_grad);
    A2D::A2DObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj), A2D::MatTrace(S_obj, trS),
        A2D::MatDet(S_obj, detS),
        A2D::Eval(
            coef * exp(ksrho * (sqrt(trS * trS - 3.0 * detS) / yield_stress -
                                max_stress_ratio)),
            output));
    output.bvalue() = 1.0;
    stack.hproduct();
  }

 private:
  double ksrho;        // KS aggregation parameter
  const T mu, lambda;  // Lame parameters
  const T yield_stress;
  T max_stress_ratio;  // maximum Von Mises stress / yield stress
  bool use_discrete_ks;
};

enum class StrainStressType { sx, sy, sxy, ex, ey, exy };

template <typename T>
class LinearElasticity2DStrainStress final : public PhysicsBase<T, 2, 0, 2> {
 private:
  using PhysicsBase_s = PhysicsBase<T, 2, 0, 2>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  LinearElasticity2DStrainStress(
      T E, T nu, StrainStressType strain_stress_type = StrainStressType::sx)
      : mu(0.5 * E / (1.0 + nu)),
        lambda(E * nu / ((1.0 + nu) * (1.0 - nu))),
        strain_stress_type(strain_stress_type) {}

  void set_type(StrainStressType strain_stress_type_) {
    strain_stress_type = strain_stress_type_;
  }

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& ____,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T trS, detS, von_mises;
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);

    switch (strain_stress_type) {
      case StrainStressType::sx: {
        return S(0, 0);
      }
      case StrainStressType::sy: {
        return S(1, 1);
      }
      case StrainStressType::sxy: {
        return S(0, 1);
      }
      case StrainStressType::ex: {
        return E(0, 0);
      }
      case StrainStressType::ey: {
        return E(1, 1);
      }
      case StrainStressType::exy: {
        return E(0, 1);
      }
    }
    return T(0.0);
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
  StrainStressType strain_stress_type;
};

// normal is outer normal
// tangent is outer normal rotating 90 degrees counterclockwise
enum class SurfStressType { normal, tangent };

template <typename T>
class LinearElasticity2DSurfStress final : public PhysicsBase<T, 2, 0, 2> {
 private:
  using PhysicsBase_s = PhysicsBase<T, 2, 0, 2>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  LinearElasticity2DSurfStress(
      T E, T nu, SurfStressType surf_stress_type = SurfStressType::normal)
      : mu(0.5 * E / (1.0 + nu)),
        lambda(E * nu / ((1.0 + nu) * (1.0 - nu))),
        surf_stress_type(surf_stress_type) {}

  void set_type(SurfStressType surf_stress_type_) {
    surf_stress_type = surf_stress_type_;
  }

  SurfStressType get_type() { return surf_stress_type; }

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& ____,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    // Evaluate stress tensor in cartesian coordinates
    T trS, detS, von_mises;
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);

    // Compute sin and cos
    A2D::Vec<T, spatial_dim> nrm;
    A2D::MatVecMult(J, nrm_ref, nrm);
    A2D::VecNormalize(nrm, nrm);
    T Cos = nrm(0);
    T Sin = nrm(1);

    switch (surf_stress_type) {
      case SurfStressType::normal: {
        return S(0, 0) * Cos * Cos + S(1, 1) * Sin * Sin +
               2.0 * S(0, 1) * Sin * Cos;
      }
      case SurfStressType::tangent: {
        return (S(0, 0) - S(1, 1)) * Sin * Cos +
               S(0, 1) * (Sin * Sin - Cos * Cos);
      }
    }
  }

 private:
  T mu, lambda;  // Lame parameters
  SurfStressType surf_stress_type;
};

template <typename T>
class LinearElasticity2DSurfStressAggregation final
    : public PhysicsBase<T, 2, 0, 2> {
 private:
  using PhysicsBase_s = PhysicsBase<T, 2, 0, 2>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  LinearElasticity2DSurfStressAggregation(
      double ksrho, T E, T nu, T yield_stress, T max_stress_ratio = 1.0,
      SurfStressType surf_stress_type = SurfStressType::normal)
      : ksrho(ksrho),
        mu(0.5 * E / (1.0 + nu)),
        lambda(E * nu / ((1.0 + nu) * (1.0 - nu))),
        yield_stress(yield_stress),
        max_stress_ratio(max_stress_ratio),
        surf_stress_type(surf_stress_type) {}

  void set_type(SurfStressType surf_stress_type_) {
    surf_stress_type = surf_stress_type_;
  }

  SurfStressType get_type() { return surf_stress_type; }

  // Does not effect result, but set a proper value can help preventing
  // floating-point overflow
  void set_max_stress_ratio(T max_stress_ratio_) {
    max_stress_ratio = max_stress_ratio_;
  }

  T get_yield_stress() const { return yield_stress; }

  void set_ksrho(double ksrho_) { ksrho = ksrho_; }
  double get_ksrho() { return ksrho; }

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& ___,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ;
    A2D::MatDet(J, detJ);

    // Evaluate stress tensor in cartesian coordinates
    A2D::SymMat<T, spatial_dim> E, S;
    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);

    // clockwise rotation matrix
    A2D::Mat<T, spatial_dim, spatial_dim> cw_rot;
    cw_rot(0, 1) = 1.0;
    cw_rot(1, 0) = -1.0;

    A2D::Vec<T, spatial_dim> nrm, tan;
    A2D::MatVecMult(J, nrm_ref, nrm);
    A2D::VecNormalize(nrm, nrm);
    A2D::MatVecMult(cw_rot, nrm, tan);

    T Cos = nrm(0);
    T Sin = nrm(1);

    T stress = 0.0;
    switch (surf_stress_type) {
      case SurfStressType::normal: {
        stress = S(0, 0) * Cos * Cos + S(1, 1) * Sin * Sin +
                 2.0 * S(0, 1) * Sin * Cos;
        break;
      }
      case SurfStressType::tangent: {
        stress =
            (S(0, 0) - S(1, 1)) * Sin * Cos + S(0, 1) * (Sin * Sin - Cos * Cos);
        break;
      }
    }

    return weight * detJ *  // FIXME: use cq instead of detJ
           exp(ksrho * (stress / yield_stress - max_stress_ratio));
  }

  void residual(T weight, T _, A2D::Vec<T, spatial_dim>& __,
                A2D::Vec<T, spatial_dim>& nrm_ref,
                A2D::Mat<T, spatial_dim, spatial_dim>& J,
                A2D::Vec<T, dof_per_node>& ___,
                A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                A2D::Vec<T, dof_per_node>& ____,
                A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {
    T detJ;
    A2D::MatDet(J, detJ);

    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);

    // selection matrix
    A2D::Mat<T, spatial_dim, spatial_dim> select;
    if (surf_stress_type == SurfStressType::normal) {
      // Identity matrix
      select(0, 0) = 1.0;
      select(1, 1) = 1.0;
    } else {
      // Clockwise rotation matrix
      select(0, 1) = 1.0;
      select(1, 0) = -1.0;
    }

    A2D::Vec<T, spatial_dim> nrm, left;
    A2D::MatVecMult(J, nrm_ref, nrm);
    A2D::VecNormalize(nrm, nrm);
    A2D::MatVecMult(select, nrm, left);

    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> Sn_obj;
    A2D::ADObj<T> stress_obj, output_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::MatVecMult(S_obj, nrm, Sn_obj),
        A2D::VecDot(left, Sn_obj, stress_obj),
        A2D::Eval(
            weight * detJ *
                exp(ksrho * (stress_obj / yield_stress - max_stress_ratio)),
            output_obj));
    output_obj.bvalue() = 1.0;
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
      A2D::Mat<T, dof_per_node, spatial_dim>& coef_grad) const {}

 private:
  double ksrho;  // KS aggregation parameter
  T mu, lambda;  // Lame parameters
  const T yield_stress;
  T max_stress_ratio;  // maximum Von Mises stress / yield stress
  SurfStressType surf_stress_type;
};
