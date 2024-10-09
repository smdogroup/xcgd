#include <complex>
#include <iostream>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/fe_tetrahedral.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/physics_commons.h"
#include "test_commons.h"
#include "utils/mesher.h"

TEST(elements, GD_N_Nxi_Nxixi) {
  int constexpr Np_1d = 6;
  int constexpr Nk = Np_1d * Np_1d;
  using T = std::complex<double>;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int constexpr spatial_dim = Mesh::spatial_dim;

  int constexpr nx = 5, ny = 5;
  int nxy[2] = {nx, ny};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  double h = 1e-7;
  std::vector<double> p = {0.4385123, 0.742383};
  std::vector<double> pt_old = {0.39214122, -0.24213123};  // in [-1, 1]

  std::vector<double> Nvals = {
      0.0000658649520253,  -0.0004620981006525, 0.0015485041670998,
      -0.0047721834319657, -0.0022056637364124, 0.0001508464098657,
      -0.0006974203166414, 0.0048929907904534,  -0.0163965543634773,
      0.0505309360718578,  0.0233549809745610,  -0.0015972584462122,
      -0.0118479783708548, 0.0831235449714294,  -0.2785494153529054,
      0.8584341800127379,  0.3967611823694650,  -0.0271346891851008,
      0.0011290084660559,  -0.0079209450814035, 0.0265433172060787,
      -0.0818012513569199, -0.0378078622256159, 0.0025856979861744,
      -0.0002963729903348, 0.0020793060908154,  -0.0069678151495439,
      0.0214734275310020,  0.0099248407100892,  -0.0006787646570441,
      0.0000401865672262,  -0.0002819426085606, 0.0009447978765173,
      -0.0029116801030750, -0.0013457544763187, 0.0000920367996092};

  Basis basis(mesh);

  for (int elem = 0; elem < nx * ny; elem++) {
    T xi_min[spatial_dim], xi_max[spatial_dim];
    get_computational_coordinates_limits(mesh, elem, xi_min, xi_max);
    std::vector<double> pt = {
        freal((pt_old[0] - xi_min[0]) / (xi_max[0] - xi_min[0])),
        freal((pt_old[1] - xi_min[1]) / (xi_max[1] - xi_min[1]))};
    std::vector<T> ptc(pt.size());

    ptc[0] = T(pt[0], h * p[0]);
    ptc[1] = T(pt[1], h * p[1]);

    std::vector<T> N, Nxi, Nxixi;
    basis.eval_basis_grad(elem, ptc, N, Nxi, Nxixi);

    for (int i = 0; i < Nk; i++) {
      double nxi_cs = N[i].imag() / h;
      double nxi_exact = 0.0;
      for (int j = 0; j < spatial_dim; j++) {
        nxi_exact += p[j] * Nxi[spatial_dim * i + j].real();
      }
      EXPECT_NEAR((nxi_cs - nxi_exact) / nxi_exact, 0.0, 1e-5);

      for (int j = 0; j < spatial_dim; j++) {
        double nxixi_cs = Nxi[spatial_dim * i + j].imag() / h;
        double nxixi_exact = 0.0;
        for (int k = 0; k < spatial_dim; k++) {
          nxixi_exact +=
              Nxixi[spatial_dim * spatial_dim * i + spatial_dim * j + k]
                  .real() *
              p[k];
        }
        EXPECT_NEAR((nxixi_cs - nxixi_exact) / nxixi_exact, 0.0, 1e-5);
      }
    }

    for (int i = 0; i < Nk; i++) {
      EXPECT_NEAR(Nvals[i], N[i].real(), 1e-13);
    }
  }
}
