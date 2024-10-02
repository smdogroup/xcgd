#include <complex>
#include <iostream>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/fe_tetrahedral.h"
#include "elements/gd_vandermonde.h"
#include "physics/physics_commons.h"
#include "test_commons.h"
#include "utils/mesher.h"

#define PI 3.141592653589793

template <typename T, int spatial_dim>
class Integration final : public PhysicsBase<T, spatial_dim, 0, 1> {
 public:
  T energy(T weight, T _, const A2D::Vec<T, spatial_dim>& __,
           const A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ___) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * val;
  }
};

template <typename T, class Quadrature, class Basis>
T hypercircle_area(typename Basis::Mesh& mesh, Quadrature& quadrature,
                   Basis& basis, const T pt0[Basis::spatial_dim], double r0,
                   std::vector<double>* dof_ = nullptr) {
  using Physics = Integration<T, Basis::spatial_dim>;
  using Analysis =
      GalerkinAnalysis<T, typename Basis::Mesh, Quadrature, Basis, Physics>;
  constexpr int spatial_dim = Basis::spatial_dim;

  std::vector<double> dof(mesh.get_num_nodes(), 0.0);

  // Set the circle/sphere
  for (int i = 0; i < mesh.get_num_nodes(); i++) {
    T xloc[spatial_dim];
    mesh.get_node_xloc(i, xloc);
    double r2 = 0.0;
    for (int d = 0; d < spatial_dim; d++) {
      r2 += (xloc[d] - pt0[d]) * (xloc[d] - pt0[d]);
    }
    if (r2 <= r0 * r0) {
      dof[i] = 1.0;
    }
  }

  if (dof_) {
    *dof_ = dof;
  }

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  return analysis.energy(nullptr, dof.data());
}

TEST(elements, integration_quad2d) {
  using T = double;
  using Quadrature = QuadrilateralQuadrature<T>;
  using Basis = QuadrilateralBasis<T>;

  int num_elements, num_nodes;
  int* element_nodes;
  double* xloc;

  int nxy[2] = {128, 128};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  create_2d_rect_quad_mesh(nxy, lxy, &num_elements, &num_nodes, &element_nodes,
                           &xloc);

  typename Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Quadrature quadrature;
  Basis basis;

  double pi = hypercircle_area(mesh, quadrature, basis, pt0, r0);
  double relerr = (pi - PI) / PI;
  EXPECT_NEAR(relerr, 0.0, 1e-2);
}

TEST(elements, integration_tet3d) {
  using T = double;
  using Quadrature = TetrahedralQuadrature<T>;
  using Basis = TetrahedralBasis<T>;

  int num_elements, num_nodes;
  int* element_nodes;
  double* xloc;

  int nxyz[3] = {32, 32, 32};
  double lxyz[3] = {3.0, 3.0, 3.0};
  double pt0[3] = {1.5, 1.5, 1.5};
  double r0 = 1.0;
  create_3d_box_tet_mesh(nxyz, lxyz, &num_elements, &num_nodes, &element_nodes,
                         &xloc);

  typename Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Quadrature quadrature;
  Basis basis;

  double pi = hypercircle_area(mesh, quadrature, basis, pt0, r0) * 3.0 / 4.0;
  double relerr = (pi - PI) / PI;
  EXPECT_NEAR(relerr, 0.0, 1e-2);
}

TEST(elements, integration_gd_Np2) {
  using T = double;
  constexpr int Np_1d = 2;

  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  double pi = hypercircle_area(mesh, quadrature, basis, pt0, r0);
  double relerr = (pi - PI) / PI;
  EXPECT_NEAR(relerr, 0.0, 1e-2);
}

TEST(elements, integration_gd_Np4) {
  using T = double;
  constexpr int Np_1d = 4;

  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  double pi = hypercircle_area(mesh, quadrature, basis, pt0, r0);
  double relerr = (pi - PI) / PI;
  EXPECT_NEAR(relerr, 0.0, 1e-2);
}

TEST(elements, integration_lsf_Np2) {
  using T = double;
  constexpr int Np_1d = 2;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;
  using Physics = Integration<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [pt0, r0](double x[]) {
    return (x[0] - pt0[0]) * (x[0] - pt0[0]) +
           (x[1] - pt0[1]) * (x[1] - pt0[1]) - r0 * r0;
  });
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  std::vector<T> dof(mesh.get_num_nodes(), 1.0);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  double pi = analysis.energy(nullptr, dof.data());

  EXPECT_NEAR(pi, PI, 1e-2);
}

TEST(elements, integration_lsf_Np4) {
  using T = double;
  constexpr int Np_1d = 4;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;
  using Physics = Integration<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [pt0, r0](double x[]) {
    return (x[0] - pt0[0]) * (x[0] - pt0[0]) +
           (x[1] - pt0[1]) * (x[1] - pt0[1]) - r0 * r0;
  });
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  std::vector<T> dof(mesh.get_num_nodes(), 1.0);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  double pi = analysis.energy(nullptr, dof.data());

  EXPECT_NEAR(pi, PI, 1e-10);
}

TEST(elements, integration_surf_Np2) {
  using T = double;
  constexpr int Np_1d = 2;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;
  using Physics = Integration<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [pt0, r0](double x[]) {
    return (x[0] - pt0[0]) * (x[0] - pt0[0]) +
           (x[1] - pt0[1]) * (x[1] - pt0[1]) - r0 * r0;
  });
  Quadrature quadrature(mesh, LSFQuadType::SURFACE);
  Basis basis(mesh);

  std::vector<T> dof(mesh.get_num_nodes(), 1.0);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  double perimeter = analysis.energy(nullptr, dof.data());

  EXPECT_NEAR(perimeter, 2.0 * PI * r0, 1e-20);
}

TEST(elements, integration_surf_Np4) {
  using T = double;
  constexpr int Np_1d = 4;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;
  using Physics = Integration<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {64, 64};
  double lxy[2] = {3.0, 3.0};
  double pt0[2] = {1.5, 1.5};
  double r0 = 1.0;
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [pt0, r0](double x[]) {
    return (x[0] - pt0[0]) * (x[0] - pt0[0]) +
           (x[1] - pt0[1]) * (x[1] - pt0[1]) - r0 * r0;
  });
  Quadrature quadrature(mesh, LSFQuadType::SURFACE);
  Basis basis(mesh);

  std::vector<T> dof(mesh.get_num_nodes(), 1.0);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  double perimeter = analysis.energy(nullptr, dof.data());

  EXPECT_NEAR(perimeter, 2.0 * PI * r0, 1e-20);
}
