#include <complex>
#include <vector>

#include "test_commons.h"
#include "utils/linalg.h"

TEST(linalg, matrix_norm_real) {
  int m = 3;
  int n = 5;
  const double A[] = {
      0.287384904701171,  0.5983721355722037, 0.9501781872479819,
      0.1881202531450925, 0.7522651012143386, 0.984570229272429,
      0.9410268009136625, 0.350102591959394,  0.755633073439118,
      0.1729823281917903, 0.5742515784711493, 0.6945572627474812,
      0.2746693030237217, 0.847368582713038,  0.4612467292953336};

  double Anrm_1 = 2.0467624663121744;
  double Anrm_inf = 3.8461854820023436;

  EXPECT_NEAR(matrix_norm('1', m, n, A), Anrm_1, 1e-30);
  EXPECT_NEAR(matrix_norm('I', m, n, A), Anrm_inf, 1e-30);
}

TEST(linalg, matrix_norm_complex) {
  int m = 3;
  int n = 5;
  std::complex<double> Ac[] = {
      std::complex<double>(0.4826960534960948, 0.0786590395418026),
      std::complex<double>(0.5620751233300275, 0.1048379112273036),
      std::complex<double>(0.8410052756317851, 0.415215344420928),
      std::complex<double>(0.4558864898368892, 0.6120566798262332),
      std::complex<double>(0.0705877512758791, 0.6843857683949574),
      std::complex<double>(0.5534554035764306, 0.8995077373445354),
      std::complex<double>(0.3105219886065855, 0.9727323125492001),
      std::complex<double>(0.9945437038355271, 0.6623007917322938),
      std::complex<double>(0.7865409022003198, 0.4917297416672572),
      std::complex<double>(0.2537214951259106, 0.2023608651100692),
      std::complex<double>(0.0782553250763756, 0.027354655893937),
      std::complex<double>(0.7552538783768123, 0.5462445167845218),
      std::complex<double>(0.9464209055467443, 0.7499974494865728),
      std::complex<double>(0.089889848639522, 0.4413784177817882),
      std::complex<double>(0.0149424614429899, 0.3484714327056916)};

  double Acnrm_1 = 3.1435840745661823;
  double Acnrm_inf = 4.202540761322937;

  EXPECT_NEAR(matrix_norm('1', m, n, Ac), Acnrm_1, 1e-30);
  EXPECT_NEAR(matrix_norm('I', m, n, Ac), Acnrm_inf, 1e-30);
}

TEST(linalg, direct_solve_real) {
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

  std::vector<double> b1_real = {0.29826095992127, 0.570877905231764,
                                 0.320630282806731, 0.853680567385518,
                                 0.207552869779013};
  std::vector<double> sol1_real = {-1.311832518095597, 0.437432863316586,
                                   -0.413853131069862, 0.973190293034837,
                                   2.350996112312741};

  direct_solve(N, A_real.data(), b1_real.data());
  EXPECT_VEC_NEAR(N, b1_real, sol1_real, 1e-14);
}

TEST(linalg, direct_solve_class_real) {
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

  std::vector<double> b1_real = {0.29826095992127, 0.570877905231764,
                                 0.320630282806731, 0.853680567385518,
                                 0.207552869779013};
  std::vector<double> b2_real = {0.1984264342268846, 0.0906610153986548,
                                 0.2536021029593606, 0.1889418067041156,
                                 0.8599787888475174};
  std::vector<double> sol1_real = {-1.311832518095597, 0.437432863316586,
                                   -0.413853131069862, 0.973190293034837,
                                   2.350996112312741};
  std::vector<double> sol2_real = {1.0991554895864208, -0.5245427118695641,
                                   0.322421773589861, 0.2822143638902147,
                                   -0.9462943639479315};

  DirectSolve<double> sol(N, A_real.data());
  sol.apply(b1_real.data());
  sol.apply(b2_real.data());
  EXPECT_VEC_NEAR(N, b1_real, sol1_real, 1e-14);
  EXPECT_VEC_NEAR(N, b2_real, sol2_real, 1e-14);
}

TEST(linalg, direct_solve_complex) {
  constexpr static int N = 5;
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

  direct_solve(N, A_complex.data(), b_complex.data());
  EXPECT_CPLX_VEC_NEAR(N, b_complex, sol_complex, 1e-14);
}

TEST(linalg, direct_solve_class_complex) {
  constexpr static int N = 5;
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

  std::vector<std::complex<double>> b1_complex = {
      std::complex<double>(0.29826095992127, 0.277736740500722),
      std::complex<double>(0.570877905231764, 0.4696623123539),
      std::complex<double>(0.320630282806731, 0.334180456573516),
      std::complex<double>(0.853680567385518, 0.069111121927175),
      std::complex<double>(0.207552869779013, 0.012618729713515)};

  std::vector<std::complex<double>> sol1_complex = {
      std::complex<double>(-0.002535482488589, 0.092668996649821),
      std::complex<double>(0.053750638058231, 0.070022280526886),
      std::complex<double>(-0.374734915990217, -0.012091173972356),
      std::complex<double>(0.512987451833535, -0.00792218093855),
      std::complex<double>(0.406159086362825, -0.707435603448901)};

  std::vector<std::complex<double>> b2_complex = {
      std::complex<double>(-0.002535482488589, 0.092668996649821),
      std::complex<double>(0.053750638058231, 0.070022280526886),
      std::complex<double>(-0.374734915990217, -0.012091173972356),
      std::complex<double>(0.512987451833535, -0.00792218093855),
      std::complex<double>(0.406159086362825, -0.707435603448901)};

  std::vector<std::complex<double>> sol2_complex = {
      std::complex<double>(0.2668873383561677, -0.467540665962378),
      std::complex<double>(-0.6341627922400568, +0.0309480793412399),
      std::complex<double>(-0.5743613193772088, -0.3165048044958452),
      std::complex<double>(0.2040956557040404, -0.0022682450116388),
      std::complex<double>(1.170972780365767, +0.2906782868605073)};

  DirectSolve<std::complex<double>> sol(N, A_complex.data());
  sol.apply(b1_complex.data());
  sol.apply(b2_complex.data());
  EXPECT_CPLX_VEC_NEAR(N, b1_complex, sol1_complex, 1e-14);
  EXPECT_CPLX_VEC_NEAR(N, b2_complex, sol2_complex, 1e-14);
}

TEST(linalg, direct_inverse_real) {
  constexpr static int N = 3;
  std::vector<double> A_real = {
      0.6128483560169078, 0.3411550443931325, 0.595417123046553,
      0.2434199130278625, 0.2781149156388093, 0.3285230476819442,
      0.4622183837010163, 0.1420136808581821, 0.2677995134802622};
  std::vector<double> invA_real = {
      -2.9788216009458655, 0.7284014421126703,  5.729453000363361,
      -9.277857105765234,  11.893356698881258,  6.037924322763912,
      10.061440025683329,  -7.5642404032057895, -9.356724852012928};

  std::vector<double> A_real_2 = A_real;
  std::vector<double> A_real_3 = A_real;
  double Arcond_1 = 0.023720066942940154;
  double Arcond_inf = 0.03398338835755088;

  direct_inverse(N, A_real.data());
  EXPECT_VEC_NEAR(N * N, A_real, invA_real, 1e-14);

  double rcond_1, rcond_inf;
  direct_inverse(N, A_real_2.data(), &rcond_1, '1');
  direct_inverse(N, A_real_3.data(), &rcond_inf, 'I');

  EXPECT_VEC_NEAR(N * N, A_real_2, invA_real, 1e-14);
  EXPECT_NEAR(rcond_1, Arcond_1, 1e-16);
  EXPECT_NEAR(rcond_inf, Arcond_inf, 1e-16);
}

TEST(linalg, direct_inverse_complex) {
  constexpr static int N = 3;

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

  direct_inverse(N, A_complex.data());
  EXPECT_CPLX_VEC_NEAR(N * N, A_complex, invA_complex, 1e-14);
}
