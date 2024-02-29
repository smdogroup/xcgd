#include <complex>
#include <iostream>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/fe_tetrahedral.h"
#include "elements/gd_vandermonde.h"
#include "physics/physics_commons.h"
#include "test_commons.h"
#include "utils/mesh.h"

TEST(ElementTest, GalerkinDiff2D) {
  int constexpr Np_1d = 6;
  int constexpr Nk = Np_1d * Np_1d;
  using T = std::complex<double>;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;

  int constexpr nx = 10, ny = 10;
  int nxy[2] = {nx, ny};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  std::vector<T> N(Nk * Basis::num_quadrature_pts);
  std::vector<T> Nxi(grid.spatial_dim * Nk * Basis::num_quadrature_pts);

  double h = 1e-7;
  std::vector<double> p = {0.4385123, 0.742383};
  std::vector<double> pt = {0.39214122, -0.24213123};

  std::vector<T> pts(Basis::num_quadrature_pts * Grid::spatial_dim);

  for (int q = 0; q < Basis::num_quadrature_pts; q++) {
    pts[Grid::spatial_dim * q] = T(pt[0], h * p[0]);
    pts[Grid::spatial_dim * q + 1] = T(pt[1], h * p[1]);
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

  Basis basis(mesh);

  for (int elem = 0; elem < nx * ny; elem++) {
    basis.eval_basis_grad(elem, pts.data(), N.data(), Nxi.data());

    for (int i = 0; i < Nk; i++) {
      double nxi_cs = N[i].imag() / h;
      double nxi_exact = 0.0;
      for (int j = 0; j < grid.spatial_dim; j++) {
        nxi_exact += p[j] * Nxi[grid.spatial_dim * i + j].real();
      }
      EXPECT_NEAR((nxi_cs - nxi_exact) / nxi_exact, 0.0, 1e-7);
    }

    for (int i = 0; i < Nk; i++) {
      EXPECT_NEAR(Nvals[i], N[i].real(), 1e-13);
    }
  }
}

template <typename T, int spatial_dim, int dof_per_node>
class Integration final : public PhysicsBase<T, spatial_dim, dof_per_node> {
 public:
  T energy(T weight, const A2D::Mat<T, spatial_dim, spatial_dim>& J,
           A2D::Vec<T, dof_per_node>& vals,
           A2D::Mat<T, dof_per_node, spatial_dim>& grad) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * vals(0);
  }
};

template <typename T, class Basis>
T hypercircle_area(typename Basis::Mesh& mesh,
                   const T pt0[Basis::spatial_dim]) {
  constexpr int dof_per_node = 1;
  using Physics = Integration<T, Basis::spatial_dim, dof_per_node>;
  using Analysis = GalerkinAnalysis<T, Basis, Physics>;
  constexpr int spatial_dim = Basis::spatial_dim;

  std::vector<double> dof(mesh.get_num_nodes() * dof_per_node, 0.0);

  // Set the circle/sphere
  for (int i = 0; i < mesh.get_num_nodes(); i++) {
    T xloc[spatial_dim];
    mesh.get_node_xloc(i, xloc);
    double r2 = 0.0;
    for (int d = 0; d < spatial_dim; d++) {
      r2 += (xloc[d] - pt0[d]) * (xloc[d] - pt0[d]);
    }
    if (r2 <= 1.0) {
      dof[i] = 1.0;
    }
  }

  Basis basis(mesh);
  Physics physics;
  Analysis analysis(basis, physics);

  return analysis.energy(dof.data());
}

TEST(IntegrationTest, Quad) {
  using Basis = QuadrilateralBasis<T>;

  int num_elements, num_nodes;
  int* element_nodes;
  double* xloc;

  int nxy[2] = {128, 128};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  create_2d_rect_quad_mesh(nxy[0], nxy[1], lxy[0], lxy[1], &num_elements,
                           &num_nodes, &element_nodes, &xloc);

  typename Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);

  std::cout << hypercircle_area<double, Basis>(mesh, pt0) << "\n";
}

TEST(IntegrationTest, GD2D) {
  constexpr int Np_1d = 4;
  using Basis = GDBasis2D<T, Np_1d>;
  using Grid = StructuredGrid2D<T>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  Grid grid(nxy, lxy);
  typename Basis::Mesh mesh(grid);

  std::cout << hypercircle_area<double, Basis>(mesh, pt0) << "\n";
}
