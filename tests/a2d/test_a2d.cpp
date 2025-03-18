#include <complex>

#include "a2dcore.h"
#include "test_commons.h"
#include "utils/misc.h"

template <typename T, int spatial_dim>
class SymMatTestPhysics {
  static_assert(spatial_dim == 2, "");

 public:
  T energy(A2D::Vec<T, spatial_dim>& nrm_ref, A2D::SymMat<T, spatial_dim>& S) {
    T output;
    A2D::Vec<T, spatial_dim> Sn;

    A2D::Vec<T, spatial_dim> t;
    t(0) = 1.0;
    t(1) = 0.0;

    A2D::SymMat<T, spatial_dim> s;
    s[0] = 0.45;
    s[1] = 0.213;
    s[2] = 0.336;
    s[3] = 0.503;

    // This works
    // a2d::matvecmult(s, nrm_ref, sn);
    // a2d::vecdot(sn, nrm_ref, output);

    A2D::MatVecMult(S, nrm_ref, Sn);
    A2D::VecDot(Sn, t, output);

    return output;
  }

  T residual(A2D::Vec<T, spatial_dim>& nrm_ref, A2D::SymMat<T, spatial_dim>& S,
             A2D::Vec<T, spatial_dim>& q_nrm_ref,
             A2D::SymMat<T, spatial_dim>& q_S) {
    A2D::ADObj<T> output_obj;
    A2D::ADObj<A2D::Vec<T, spatial_dim>> Sn_obj;

    A2D::Vec<T, spatial_dim> t;
    t(0) = 1.0;
    t(1) = 0.0;

    A2D::SymMat<T, spatial_dim> s;
    s[0] = 0.45;
    s[1] = 0.213;
    s[2] = 0.336;
    s[3] = 0.503;

    A2D::Vec<T, spatial_dim> coef_nrm_ref;
    A2D::SymMat<T, spatial_dim> coef_S;

    A2D::ADObj<A2D::Vec<T, spatial_dim>&> nrm_ref_obj(nrm_ref, coef_nrm_ref);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>&> S_obj(S, coef_S);

    // This works
    // auto stack = A2D::MakeStack(A2D::MatVecMult(s, nrm_ref_obj, Sn_obj),
    //                             A2D::VecDot(Sn_obj, nrm_ref_obj,
    //                             output_obj));

    auto stack = A2D::MakeStack(A2D::MatVecMult(S_obj, t, Sn_obj),
                                A2D::VecDot(Sn_obj, nrm_ref_obj, output_obj));

    output_obj.bvalue() = 1.0;
    stack.reverse();

    T rq = 0.0;
    for (int i = 0; i < spatial_dim; i++) {
      rq += coef_nrm_ref[i] * q_nrm_ref[i];
    }
    for (int i = 0; i < (spatial_dim * (spatial_dim + 1)) / 2; i++) {
      rq += coef_S[i] * q_S[i];
    }

    return rq;
  }

  T jacobian_product(A2D::Vec<T, spatial_dim>& nrm_ref,
                     A2D::SymMat<T, spatial_dim>& S,
                     A2D::Vec<T, spatial_dim>& p_nrm_ref,
                     A2D::SymMat<T, spatial_dim>& p_S,
                     A2D::Vec<T, spatial_dim>& q_nrm_ref,
                     A2D::SymMat<T, spatial_dim>& q_S) {
    A2D::A2DObj<T> output_obj;
    A2D::A2DObj<A2D::Vec<T, spatial_dim>> Sn_obj;

    A2D::Vec<T, spatial_dim> t;
    t(0) = 1.0;
    t(1) = 0.0;

    A2D::SymMat<T, spatial_dim> s;
    s[0] = 0.45;
    s[1] = 0.213;
    s[2] = 0.336;
    s[3] = 0.503;

    // work objs
    A2D::Vec<T, spatial_dim> b_nrm_ref, jp_nrm_ref;
    A2D::SymMat<T, spatial_dim> b_S, jp_S;

    A2D::A2DObj<A2D::Vec<T, spatial_dim>&> nrm_ref_obj(nrm_ref, b_nrm_ref,
                                                       p_nrm_ref, jp_nrm_ref);
    A2D::A2DObj<A2D::SymMat<T, spatial_dim>&> S_obj(S, b_S, p_S, jp_S);

    // This works
    // auto stack = A2D::MakeStack(A2D::MatVecMult(s, nrm_ref_obj, Sn_obj),
    //                             A2D::VecDot(Sn_obj, nrm_ref_obj,
    //                             output_obj));

    auto stack = A2D::MakeStack(A2D::MatVecMult(S_obj, nrm_ref_obj, Sn_obj),
                                A2D::VecDot(Sn_obj, t, output_obj));

    output_obj.bvalue() = 1.0;
    stack.hproduct();

    T qJp = 0.0;
    for (int i = 0; i < spatial_dim; i++) {
      qJp += jp_nrm_ref[i] * q_nrm_ref[i];
    }
    for (int i = 0; i < (spatial_dim * (spatial_dim + 1)) / 2; i++) {
      qJp += jp_S[i] * q_S[i];
    }

    return qJp;
  }
};

TEST(a2d, SymMat) {
  constexpr int spatial_dim = 2;
  using T = double;
  using Tc = std::complex<double>;

  // Inputs and collectors
  A2D::Vec<T, spatial_dim> nrm_ref, q_nrm_ref, p_nrm_ref;
  A2D::SymMat<T, spatial_dim> S, q_S, p_S;

  // random initialization
  srand(42);
  for (int i = 0; i < spatial_dim; i++) {
    nrm_ref[i] = (T)rand() / RAND_MAX;
    p_nrm_ref[i] = (T)rand() / RAND_MAX;
    q_nrm_ref[i] = (T)rand() / RAND_MAX;
  }
  for (int i = 0; i < (spatial_dim * (spatial_dim + 1)) / 2; i++) {
    S[i] = (T)rand() / RAND_MAX;
    p_S[i] = (T)rand() / RAND_MAX;
    q_S[i] = (T)rand() / RAND_MAX;
  }

  SymMatTestPhysics<T, spatial_dim> phy;

  // Prepare for complex step
  SymMatTestPhysics<Tc, spatial_dim> phy_c;
  double h = 1e-30;

  A2D::Vec<Tc, spatial_dim> nrm_ref_c, p_nrm_ref_c;
  A2D::SymMat<Tc, spatial_dim> S_c, p_S_c;

  for (int i = 0; i < spatial_dim; i++) {
    nrm_ref_c[i] = Tc(nrm_ref[i], h * q_nrm_ref[i]);
    p_nrm_ref_c[i] = Tc(p_nrm_ref[i], 0.0);
  }

  for (int i = 0; i < (spatial_dim * (spatial_dim + 1)) / 2; i++) {
    S_c[i] = Tc(S[i], h * q_S[i]);
    p_S_c[i] = Tc(p_S[i], 0.0);
  }

  T rq_exact = phy.residual(nrm_ref, S, q_nrm_ref, q_S);

  T rq_cs = phy_c.energy(nrm_ref_c, S_c).imag() / h;

  std::printf("rq_exact: %30.20f\n", rq_exact);
  std::printf("rq_cs:    %30.20f\n", rq_cs);
  double relerr = (rq_exact - rq_cs) / rq_cs;
  EXPECT_NEAR(relerr, 0.0, 1e-15);

  T qJp_exact =
      phy.jacobian_product(nrm_ref, S, p_nrm_ref, p_S, q_nrm_ref, q_S);

  T qJp_cs = phy_c.residual(nrm_ref_c, S_c, p_nrm_ref_c, p_S_c).imag() / h;

  std::printf("qJp_exact: %30.20f\n", qJp_exact);
  std::printf("qJp_cs:    %30.20f\n", qJp_cs);
  relerr = (qJp_exact - qJp_cs) / qJp_cs;
  EXPECT_NEAR(relerr, 0.0, 1e-15);
}
