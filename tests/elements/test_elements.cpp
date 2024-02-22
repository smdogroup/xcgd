#include <complex>
#include <iostream>
#include <vector>

#include "elements/galerkin_difference.h"
#include "test_commons.h"

TEST(ElementTest, GalerkinDiff2D) {
  using T = std::complex<double>;
  int constexpr spatial_dim = 2;

  int constexpr Np_1d = 6;
  int constexpr Nk = Np_1d * Np_1d;

  std::vector<T> N(Nk);
  std::vector<T> Nxi(spatial_dim * Nk);

  double h = 1e-7;
  std::vector<double> p = {0.4385123, 0.742383};

  std::vector<T> pt = {0.39214122, -0.24213123};
  for (int i = 0; i < spatial_dim; i++) {
    pt[i] += T(0.0, h * p[i]);
  }

  GD2DBasis<Np_1d>::eval_basis_grad(pt.data(), N.data(), Nxi.data());

  for (int i = 0; i < Nk; i++) {
    double nxi_cs = N[i].imag() / h;
    double nxi_exact = 0.0;
    for (int j = 0; j < spatial_dim; j++) {
      nxi_exact += p[j] * Nxi[spatial_dim * i + j].real();
    }
    EXPECT_NEAR((nxi_cs - nxi_exact) / nxi_exact, 0.0, 1e-7);
  }

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

  for (int i = 0; i < Nk; i++) {
    EXPECT_NEAR(Nvals[i], N[i].real(), 1e-13);
  }
}