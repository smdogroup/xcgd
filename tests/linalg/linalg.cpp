#include "utils/linalg.h"

#include <complex>
#include <vector>

#include "test_commons.h"

TEST(linalg, direct_solve) {
  constexpr static int N = 5;
  std::vector<double> A_real = {
      0.706731212977905, 0.68769964262544,  0.0403897726979,
      0.938114614228867, 0.634843691845117, 0.701326357373972,
      0.949496362380056, 0.381210102344509, 0.620529103555476,
      0.328581092257033, 0.916427131220972, 0.363038635033342,
      0.888136365460139, 0.62235765962521,  0.726312010447533,
      0.024358753608401, 0.299693362200554, 0.525309093242787,
      0.172146044070358, 0.863963137178775, 0.541962310550557,
      0.38973682118594,  0.026879405874587, 0.809411720646246,
      0.151601808178256};

  std::vector<double> b_real = {0.29826095992127, 0.570877905231764,
                                0.320630282806731, 0.853680567385518,
                                0.207552869779013};
  std::vector<double> sol_real = {-1.311832518095597, 0.437432863316586,
                                  -0.413853131069862, 0.973190293034837,
                                  2.350996112312741};

  int info = direct_solve(N, A_real.data(), b_real.data());

  EXPECT_EQ(info, 0);
  EXPECT_VEC_NEAR(N, b_real, sol_real, 1e-14);

  std::vector<std::complex<double>> A_complex = {
      std::complex<double>(0.706731212977905, 0.690134001174138),
      std::complex<double>(0.68769964262544, 0.614833834640402),
      std::complex<double>(0.0403897726979, 0.278064760541195),
      std::complex<double>(0.938114614228867, 0.072623851304389),
      std::complex<double>(0.634843691845117, 0.829601743028124),
      std::complex<double>(0.701326357373972, 0.781018361093866),
      std::complex<double>(0.949496362380056, 0.881461419804449),
      std::complex<double>(0.381210102344509, 0.200440735664526),
      std::complex<double>(0.620529103555476, 0.643074704193751),
      std::complex<double>(0.328581092257033, 0.533314158720767),
      std::complex<double>(0.916427131220972, 0.127118633103744),
      std::complex<double>(0.363038635033342, 0.278103620296407),
      std::complex<double>(0.888136365460139, 0.810775828771376),
      std::complex<double>(0.62235765962521, 0.91949654877904),
      std::complex<double>(0.726312010447533, 0.513645934801897),
      std::complex<double>(0.024358753608401, 0.558072905278365),
      std::complex<double>(0.299693362200554, 0.817575225245981),
      std::complex<double>(0.525309093242787, 0.809770391179751),
      std::complex<double>(0.172146044070358, 0.865949828572789),
      std::complex<double>(0.863963137178775, 0.356303555657246),
      std::complex<double>(0.541962310550557, 0.686761112022584),
      std::complex<double>(0.38973682118594, 0.642394983144106),
      std::complex<double>(0.026879405874587, 0.531848752206437),
      std::complex<double>(0.809411720646246, 0.950824774290089),
      std::complex<double>(0.151601808178256, 0.09059168417115)};

  std::vector<std::complex<double>> b_complex = {
      std::complex<double>(0.29826095992127, 0.277736740500722),
      std::complex<double>(0.570877905231764, 0.4696623123539),
      std::complex<double>(0.320630282806731, 0.334180456573516),
      std::complex<double>(0.853680567385518, 0.069111121927175),
      std::complex<double>(0.207552869779013, 0.012618729713515)};

  std::vector<std::complex<double>> sol_complex = {
      std::complex<double>(-0.002535482488589, 0.092668996649821),
      std::complex<double>(0.053750638058231, 0.070022280526886),
      std::complex<double>(-0.374734915990217, -0.012091173972356),
      std::complex<double>(0.512987451833535, -0.00792218093855),
      std::complex<double>(0.406159086362825, -0.707435603448901)};

  info = direct_solve(N, A_complex.data(), b_complex.data());
  EXPECT_EQ(info, 0);
  EXPECT_CPLX_VEC_NEAR(N, b_complex, sol_complex, 1e-14);
}

TEST(linalg, direct_inverse) {
  constexpr static int N = 3;
  std::vector<double> A_real = {
      0.6128483560169078, 0.3411550443931325, 0.595417123046553,
      0.2434199130278625, 0.2781149156388093, 0.3285230476819442,
      0.4622183837010163, 0.1420136808581821, 0.2677995134802622};
  std::vector<double> invA_real = {
      -2.9788216009458655, 0.7284014421126703,  5.729453000363361,
      -9.277857105765234,  11.893356698881258,  6.037924322763912,
      10.061440025683329,  -7.5642404032057895, -9.356724852012928};

  int info = direct_inverse(N, A_real.data());
  EXPECT_EQ(info, 0);
  EXPECT_VEC_NEAR(N * N, A_real, invA_real, 1e-14);

  std::vector<std::complex<double>> A_complex = {
      std::complex<double>(0.2862329040974567, 0.4965586803311326),
      std::complex<double>(0.3788716064341484, 0.5763470591857719),
      std::complex<double>(0.4662159293548004, 0.0685092269706217),
      std::complex<double>(0.1086360834430374, 0.7807212213919965),
      std::complex<double>(0.7983815346915806, 0.2644385569400267),
      std::complex<double>(0.9575083145537888, 0.479220244931871),
      std::complex<double>(0.7336319429638741, 0.494972184818358),
      std::complex<double>(0.8590125320426135, 0.2732431474400324),
      std::complex<double>(0.9156859048461716, 0.2836910065179017)};

  std::vector<std::complex<double>> invA_complex = {
      std::complex<double>(-0.6678107987709528, +0.1332422409236903),
      std::complex<double>(-1.4878523426213153, -0.6131323393520564),
      std::complex<double>(1.8424002904065686, +0.8311227971159687),
      std::complex<double>(0.2132381375706667, -1.9312741109944134),
      std::complex<double>(0.850679591253277, +0.5915611018245559),
      std::complex<double>(-0.7873118461104905, +0.1474825682472335),
      std::complex<double>(0.4115722070343439, +1.874835538643437),
      std::complex<double>(0.3557461759440042, +0.3764786950099526),
      std::complex<double>(0.3311290513160524, -1.667793795720977)};

  info = direct_inverse(N, A_complex.data());
  EXPECT_EQ(info, 0);
  EXPECT_CPLX_VEC_NEAR(N * N, A_complex, invA_complex, 1e-14);
}