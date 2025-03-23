#include "dual.hpp"
#include "quadrature_multipoly.hpp"
#include "test_commons.h"
#include "utils/misc.h"
#include "utils/vtk.h"

constexpr int spatial_dim = 2;

template <typename T, int Np_1d_bern, int Np_1d_quad>
auto algoim_weight(std::vector<T> dv) {
  xcgd_assert(dv.size() == Np_1d_bern * Np_1d_bern, "");
  int ndv = dv.size();

  std::vector<T> pts, wts;

  std::vector<T> data(Np_1d_bern * Np_1d_bern, 0.0);

  algoim::xarray<T, spatial_dim> phi(
      data.data(), algoim::uvector<int, spatial_dim>(Np_1d_bern, Np_1d_bern));

  algoim::bernstein::bernsteinInterpolate<spatial_dim>(
      [&](const algoim::uvector<T, spatial_dim>& xi) {  // xi in [0, 1]
        return xi(0) + 0.9 * xi(1) - 1.5;
      },
      phi);

  for (int i = 0; i < ndv; i++) {
    data[i] += dv[i];
  }

  algoim::ImplicitPolyQuadrature<spatial_dim, T> ipquad(phi);
  ipquad.integrate(algoim::AutoMixed, Np_1d_quad,
                   [&](const algoim::uvector<T, spatial_dim>& x, T w) {
                     if (algoim::bernstein::evalBernsteinPoly(phi, x) <= 0.0) {
                       for (int d = 0; d < spatial_dim; d++) {
                         pts.push_back(x(d));
                       }
                       wts.push_back(w);
                     }
                   });

  return std::make_tuple(pts, wts);
}

TEST(algoim, visualize) {
  int constexpr Np_1d_bern = 4;
  int constexpr Np_1d_quad = 4;
  using T = double;

  std::vector<T> dv(Np_1d_bern * Np_1d_bern, 0.0);

  auto [pts, wts] = algoim_weight<T, Np_1d_bern, Np_1d_quad>(dv);

  FieldToVTKNew<T, spatial_dim> quad_vtk("quad_algoim.vtk");
  quad_vtk.add_mesh(pts);
  quad_vtk.write_mesh();
  quad_vtk.add_sol("wts", wts);
  quad_vtk.write_sol("wts");
}

TEST(algoim, quad_weight_grad) {
  int constexpr Np_1d_bern = 4;
  int constexpr Np_1d_quad = 4;
  int j = 31;  // quad index
  using T = double;
  using T2 = duals::dual<T>;

  double dh = 1e-6;

  int ndv = Np_1d_bern * Np_1d_bern;
  std::vector<T2> dv1(ndv, 0.0), dv2(ndv, 0.0);

  std::vector<T> w, grad_exact, grad_fd;

  for (int i = 0; i < ndv; i++) {
    dv1[i].dpart(1.0);
    dv2[i].rpart(dh);

    auto [pts1, w1] = algoim_weight<T2, Np_1d_bern, Np_1d_quad>(dv1);
    auto [pts2, w2] = algoim_weight<T2, Np_1d_bern, Np_1d_quad>(dv2);
    dv1[i].dpart(0.0);
    dv2[i].rpart(0.0);

    w.push_back(w1[j].rpart());
    grad_exact.push_back(w1[j].dpart());
    grad_fd.push_back((w2[j].rpart() - w1[j].rpart()) / dh);
  }

  for (int i = 0; i < ndv; i++) {
    T abserr = fabs(grad_exact[i] - grad_fd[i]);
    T relerr = abserr / fabs(grad_exact[i]);
    printf(
        "w[%2d]:%15.5e, grad_exact[%2d]:%15.5e, grad_fd[%2d]:%15.5e, "
        "relerr:%15.5e\n",
        i, w[i], i, grad_exact[i], i, grad_fd[i], relerr);
  }
}
