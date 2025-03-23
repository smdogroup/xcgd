#include "dual.hpp"
#include "quadrature_multipoly.hpp"
#include "test_commons.h"
#include "utils/misc.h"

constexpr int spatial_dim = 2;

template <typename T2, int Np_1d>
auto algoim_weight(int quad, std::vector<T2> phi_data) {
  xcgd_assert(Np_1d * Np_1d == phi_data.size(), "");
  std::vector<T2> pts, wts;
  algoim::xarray<T2, spatial_dim> phi(
      phi_data.data(), algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
  algoim::ImplicitPolyQuadrature<spatial_dim, T2> ipquad(phi);
  ipquad.integrate(algoim::AutoMixed, Np_1d,
                   [&](const algoim::uvector<T2, spatial_dim>& x, T2 w) {
                     if (algoim::bernstein::evalBernsteinPoly(phi, x) <= 0.0) {
                       for (int d = 0; d < spatial_dim; d++) {
                         pts.push_back(x(d));
                       }
                       wts.push_back(w);
                     }
                   });
  int num_quad_pts = wts.size();
  xcgd_assert(num_quad_pts * spatial_dim == pts.size(),
              "inconsistent pts.size() and wts.size()");
  xcgd_assert(quad < num_quad_pts, "not enough number of quadrature points");
  T2 w = wts[quad];
  std::vector<T2> pt = {pts[spatial_dim * quad], pts[spatial_dim * quad + 1]};
  return std::make_tuple(w, pt);
}

TEST(algoim, quad_weight_grad) {
  constexpr int Np_1d = 4;
  using T = double;
  using T2 = duals::dual<T>;

  int quad = Np_1d == 2 ? 7 : 31;
  double dh = 1e-10;

  std::vector<T> phi_bl;

  if (Np_1d == 2) {
    phi_bl = {-3.833333333333334e-01, -1.666666666666674e-02,
              -5.000000000000004e-02, 3.166666666666667e-01};
  } else if (Np_1d == 4) {
    phi_bl = {
        -3.833333333333337e-01, -2.611111111111110e-01, -1.388888888888890e-01,
        -1.666666666666650e-02, -2.722222222222227e-01, -1.499999999999999e-01,
        -2.777777777777788e-02, 9.444444444444473e-02,  -1.611111111111115e-01,
        -3.888888888888946e-02, 8.333333333333305e-02,  2.055555555555559e-01,
        -5.000000000000052e-02, 7.222222222222206e-02,  1.944444444444449e-01,
        3.166666666666678e-01};
  }

  std::vector<T2> phi1(phi_bl.size()), phi2(phi_bl.size());
  for (int i = 0; i < phi_bl.size(); i++) {
    phi1[i].rpart(phi_bl[i]);
    phi2[i].rpart(phi_bl[i]);
  }

  std::vector<T> w, grad_exact, grad_fd;

  for (int i = 0; i < phi_bl.size(); i++) {
    phi1[i].dpart(1.0);
    phi2[i].rpart(phi_bl[i] + dh);
    auto [w1, pt1] = algoim_weight<T2, Np_1d>(quad, phi1);
    auto [w2, pt2] = algoim_weight<T2, Np_1d>(quad, phi2);
    phi1[i].dpart(0.0);
    phi2[i].rpart(phi_bl[i]);

    w.push_back(w1.rpart());
    grad_exact.push_back(w1.dpart());
    grad_fd.push_back((w2.rpart() - w1.rpart()) / dh);
  }

  for (int i = 0; i < phi_bl.size(); i++) {
    T abserr = fabs(grad_exact[i] - grad_fd[i]);
    T relerr = abserr / fabs(grad_exact[i]);
    printf(
        "[Np=%d]w[%2d]:%15.5e, grad_exact[%2d]:%15.5e, grad_fd[%2d]:%15.5e, "
        "relerr:%15.5e\n",
        Np_1d, i, w[i], i, grad_exact[i], i, grad_fd[i], relerr);
  }

  // FieldToVTKNew<T, spatial_dim> quad_vtk("quad_algoim.vtk");
  // quad_vtk.add_mesh(pts);
  // quad_vtk.write_mesh();
  // quad_vtk.add_sol("wts", wts);
  // quad_vtk.write_sol("wts");
}
