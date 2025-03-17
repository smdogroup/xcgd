#include <complex>

#include "a2dcore.h"
#include "test_commons.h"
#include "utils/misc.h"

template <typename T, int dof_per_node, int spatial_dim>
class SymMatTestPhysics {
 public:
  T energy(A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) {
    T output;
    A2D::SymMat<T, dof_per_node> E, S;
    A2D::Vec<T, dof_per_node> Sn;

    A2D::Vec<T, dof_per_node> t;
    t(0) = 1.2;
    t(1) = 0.345;

    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, E);
    A2D::SymIsotropic(mu, lambda, E, S);
    A2D::MatVecMult(S, nrm_ref, Sn);
    A2D::VecDot(Sn, nrm_ref, output);
    // A2D::VecDot(Sn, Sn, output);
    // A2D::SymMatMultTrace(E, S, output);
    return output;
  }

  T residual(A2D::Vec<T, spatial_dim>& nrm_ref,
             A2D::Mat<T, dof_per_node, spatial_dim>& grad,
             A2D::Vec<T, spatial_dim>& q_nrm_ref,
             A2D::Mat<T, dof_per_node, spatial_dim>& q_grad) {
    A2D::ADObj<T> output_obj;
    A2D::ADObj<A2D::SymMat<T, dof_per_node>> E_obj, S_obj;
    A2D::ADObj<A2D::Vec<T, dof_per_node>> Sn_obj;

    A2D::Vec<T, dof_per_node> t;
    t(0) = 1.2;
    t(1) = 0.345;

    A2D::Vec<T, spatial_dim> coef_nrm_ref;
    A2D::Mat<T, dof_per_node, spatial_dim> coef_grad;

    A2D::ADObj<A2D::Vec<T, spatial_dim>&> nrm_ref_obj(nrm_ref, coef_nrm_ref);
    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(grad,
                                                                 coef_grad);

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::MatVecMult(S_obj, nrm_ref_obj, Sn_obj),
        A2D::VecDot(Sn_obj, nrm_ref_obj, output_obj)
        // A2D::VecDot(Sn_obj, Sn_obj, output_obj)
        // A2D::SymMatMultTrace(E_obj, S_obj, output_obj)
    );

    output_obj.bvalue() = 1.0;
    stack.reverse();

    T rq = 0.0;
    for (int i = 0; i < spatial_dim; i++) {
      rq += coef_nrm_ref[i] * q_nrm_ref[i];
    }
    for (int i = 0; i < dof_per_node * spatial_dim; i++) {
      rq += coef_grad[i] * q_grad[i];
    }

    return rq;
  }

  T jacobian_product(A2D::Vec<T, spatial_dim>& nrm_ref,
                     A2D::Mat<T, dof_per_node, spatial_dim>& grad,
                     A2D::Vec<T, spatial_dim>& p_nrm_ref,
                     A2D::Mat<T, dof_per_node, spatial_dim>& p_grad,
                     A2D::Vec<T, spatial_dim>& q_nrm_ref,
                     A2D::Mat<T, dof_per_node, spatial_dim>& q_grad) {
    A2D::A2DObj<T> output_obj;
    A2D::A2DObj<A2D::SymMat<T, dof_per_node>> E_obj, S_obj;
    A2D::A2DObj<A2D::Vec<T, dof_per_node>> Sn_obj;

    A2D::Vec<T, dof_per_node> t;
    t(0) = 1.2;
    t(1) = 0.345;

    // work objs
    A2D::Vec<T, spatial_dim> b_nrm_ref, jp_nrm_ref;
    A2D::Mat<T, dof_per_node, spatial_dim> b_grad, jp_grad;

    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> nrm_ref_obj(nrm_ref, b_nrm_ref,
                                                       p_nrm_ref, jp_nrm_ref);
    A2D::A2DObj<A2D::Mat<T, dof_per_node, spatial_dim>&> grad_obj(
        grad, b_grad, p_grad, jp_grad);

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj),
        A2D::MatVecMult(S_obj, nrm_ref_obj, Sn_obj),
        A2D::VecDot(Sn_obj, nrm_ref_obj, output_obj)
        // A2D::VecDot(Sn_obj, Sn_obj, output_obj)
        // A2D::SymMatMultTrace(E_obj, S_obj, output_obj)
    );

    output_obj.bvalue() = 1.0;
    stack.hproduct();

    T qJp = 0.0;
    for (int i = 0; i < spatial_dim; i++) {
      qJp += jp_nrm_ref[i] * q_nrm_ref[i];
    }
    for (int i = 0; i < dof_per_node * spatial_dim; i++) {
      qJp += jp_grad[i] * q_grad[i];
    }

    return qJp;
  }

 private:
  T mu = 0.32, lambda = 0.54;
};

TEST(a2d, SymMat) {
  constexpr int dof_per_node = 2;
  constexpr int spatial_dim = 2;
  using T = double;
  using Tc = std::complex<double>;

  // Inputs and collectors
  A2D::Vec<T, spatial_dim> nrm_ref, q_nrm_ref, p_nrm_ref;
  A2D::Mat<T, dof_per_node, spatial_dim> grad, q_grad, p_grad;

  // random initialization
  srand(42);
  for (int i = 0; i < spatial_dim; i++) {
    nrm_ref[i] = (T)rand() / RAND_MAX;
    p_nrm_ref[i] = (T)rand() / RAND_MAX;
    q_nrm_ref[i] = (T)rand() / RAND_MAX;
    // p_nrm_ref[i] = 0.0;
    // q_nrm_ref[i] = 0.0;
  }
  for (int i = 0; i < dof_per_node * spatial_dim; i++) {
    grad[i] = (T)rand() / RAND_MAX;
    p_grad[i] = (T)rand() / RAND_MAX;
    q_grad[i] = (T)rand() / RAND_MAX;
    // p_grad[i] = 0.0;
    // q_grad[i] = 0.0;
  }

  SymMatTestPhysics<T, dof_per_node, spatial_dim> phy;

  // Prepare for complex step
  SymMatTestPhysics<Tc, dof_per_node, spatial_dim> phy_c;
  double h = 1e-30;

  A2D::Vec<Tc, spatial_dim> nrm_ref_c, p_nrm_ref_c;
  A2D::Mat<Tc, dof_per_node, spatial_dim> grad_c, p_grad_c;

  for (int i = 0; i < spatial_dim; i++) {
    nrm_ref_c[i] = Tc(nrm_ref[i], h * q_nrm_ref[i]);
    p_nrm_ref_c[i] = Tc(p_nrm_ref[i], 0.0);
  }

  for (int i = 0; i < dof_per_node * spatial_dim; i++) {
    grad_c[i] = Tc(grad[i], h * q_grad[i]);
    p_grad_c[i] = Tc(p_grad[i], 0.0);
  }

  T rq_exact = phy.residual(nrm_ref, grad, q_nrm_ref, q_grad);

  T rq_cs = phy_c.energy(nrm_ref_c, grad_c).imag() / h;

  std::printf("rq_exact: %30.20f\n", rq_exact);
  std::printf("rq_cs:    %30.20f\n", rq_cs);
  double relerr = (rq_exact - rq_cs) / rq_cs;
  EXPECT_NEAR(relerr, 0.0, 1e-15);

  T qJp_exact =
      phy.jacobian_product(nrm_ref, grad, p_nrm_ref, p_grad, q_nrm_ref, q_grad);

  T qJp_cs =
      phy_c.residual(nrm_ref_c, grad_c, p_nrm_ref_c, p_grad_c).imag() / h;

  std::printf("qJp_exact: %30.20f\n", qJp_exact);
  std::printf("qJp_cs:    %30.20f\n", qJp_cs);
  relerr = (qJp_exact - qJp_cs) / qJp_cs;
  EXPECT_NEAR(relerr, 0.0, 1e-15);
}
