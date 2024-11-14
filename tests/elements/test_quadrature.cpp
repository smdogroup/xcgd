#include <string>
#include <vector>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/volume.h"
#include "test_commons.h"
#include "utils/vtk.h"

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
  auto [xloc_q, dof_q] = analysis.interpolate_dof(dof.data());

  for (int iy = 0, cell = 0; iy < nxy[1]; iy++) {
    for (int ix = 0; ix < nxy[0]; ix++, cell++) {
      for (int q = 0; q < Np_1d; q++) {
        if constexpr (surf_quad == SurfQuad::LEFT) {
          EXPECT_NEAR(ix * lxy[0] / nxy[0], xloc_q[2 * Np_1d * cell], 1e-15);
        } else if constexpr (surf_quad == SurfQuad::LEFT) {
          EXPECT_NEAR((ix + 1.0) * lxy[0] / nxy[0], xloc_q[2 * Np_1d * cell],
                      1e-15);
        } else if constexpr (surf_quad == SurfQuad::LOWER) {
          EXPECT_NEAR(iy * lxy[1] / nxy[1], xloc_q[2 * Np_1d * cell + 1],
                      1e-15);
        } else if constexpr (surf_quad == SurfQuad::UPPER) {
          EXPECT_NEAR((iy + 1.0) * lxy[1] / nxy[1],
                      xloc_q[2 * Np_1d * cell + 1], 1e-15);
        }
      }
    }
  }
}

TEST(quadrature, GaussSurfQuad) {}
