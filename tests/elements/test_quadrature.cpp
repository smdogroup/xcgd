#include <vector>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"
#include "test_commons.h"

template <SurfQuad surf_quad>
void test_gauss_surf_quad() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;
  using Quadrature =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, surf_quad>;
  using Basis = GDBasis2D<T, Mesh>;
  using Physics = VolumePhysics<T, Basis::spatial_dim>;  // dummy physics
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  T lxy[2] = {1.0, 1.0};
  int nxy[2] = {4, 5};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  std::vector<T> dof(mesh.get_num_nodes(), T(0.0));  // dummy dof
  auto [xloc_q, dof_q] = analysis.interpolate(dof.data());

  for (int iy = 0, cell = 0; iy < nxy[1]; iy++) {
    for (int ix = 0; ix < nxy[0]; ix++, cell++) {
      for (int q = 0; q < Np_1d; q++) {
        if constexpr (surf_quad == SurfQuad::LEFT) {
          EXPECT_NEAR(ix * lxy[0] / nxy[0], xloc_q[2 * Np_1d * cell], 1e-15);
        } else if constexpr (surf_quad == SurfQuad::LEFT) {
          EXPECT_NEAR((ix + 1.0) * lxy[0] / nxy[0], xloc_q[2 * Np_1d * cell],
                      1e-15);
        } else if constexpr (surf_quad == SurfQuad::BOTTOM) {
          EXPECT_NEAR(iy * lxy[1] / nxy[1], xloc_q[2 * Np_1d * cell + 1],
                      1e-15);
        } else if constexpr (surf_quad == SurfQuad::TOP) {
          EXPECT_NEAR((iy + 1.0) * lxy[1] / nxy[1],
                      xloc_q[2 * Np_1d * cell + 1], 1e-15);
        }
      }
    }
  }
}

TEST(quadrature, GaussSurfQuad) {
  test_gauss_surf_quad<SurfQuad::LEFT>();
  test_gauss_surf_quad<SurfQuad::RIGHT>();
  test_gauss_surf_quad<SurfQuad::BOTTOM>();
  test_gauss_surf_quad<SurfQuad::TOP>();
}

// A complex element means an element with multiple level-set lines cutting
// through it
void test_complex_element_quad(bool negate = false) {
  using T = double;
  int constexpr Np_1d = 2;

  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d, Grid>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Physics = VolumePhysics<T, Basis::spatial_dim>;  // dummy physics
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  T lxy[2] = {2.0, 1.0};
  int nxy[2] = {128, 64};

  T xy1[2] = {0.9, 0.6};
  T xy2[2] = {1.1, 0.45};

  T d = sqrt((xy1[0] - xy2[0]) * (xy1[0] - xy2[0]) +
             (xy1[1] - xy2[1]) * (xy1[1] - xy2[1]));

  T frac = 0.05;
  T r1 = (1.0 - frac) * d * 0.4;
  T r2 = (1.0 - frac) * d * 0.6;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, [xy1, xy2, r1, r2, negate](T *pt) {
    T d1 = sqrt((xy1[0] - pt[0]) * (xy1[0] - pt[0]) +
                (xy1[1] - pt[1]) * (xy1[1] - pt[1]));
    T d2 = sqrt((pt[0] - xy2[0]) * (pt[0] - xy2[0]) +
                (pt[1] - xy2[1]) * (pt[1] - xy2[1]));
    return (negate ? -1 : 1) * hard_max(std::vector<T>{r1 - d1, r2 - d2});
  });

  Quadrature quadrature(mesh);
  Basis basis(mesh);
  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  std::vector<T> dof(mesh.get_num_nodes(), T(0.0));  // dummy dof
  auto [xloc_q, dof_q] = analysis.interpolate(dof.data());

  ToVTK<T, Mesh> vtk_mesh(mesh, std::string("complex_element_") +
                                    (negate ? "negate_" : "") + "mesh.vtk");
  vtk_mesh.write_mesh();
  vtk_mesh.write_sol("lsf", mesh.get_lsf_nodes().data());

  FieldToVTKNew<T, Basis::spatial_dim> vtk_quad(
      std::string("complex_element_") + (negate ? "negate_" : "") +
      "quads.vtk");
  vtk_quad.add_mesh(xloc_q);
  vtk_quad.write_mesh();
}

TEST(quadrature, ComplexElementQuad) {
  test_complex_element_quad(false);
  test_complex_element_quad(true);
}
