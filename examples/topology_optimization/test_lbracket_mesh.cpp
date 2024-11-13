#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "lbracket_mesh.h"
#include "physics/volume.h"
#include "utils/vtk.h"

void test_lbracket_grid() {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = LbracketGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;

  int nx1 = 80, nx2 = 32, ny1 = 32, ny2 = 48;
  T lx1 = 1.0, ly1 = 0.6;
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);

  Mesh mesh(grid);

  ToVTK<T, Mesh> vtk(mesh, "lbracket.vtk");
  vtk.write_mesh();
}

void test_lbracket_via_lsf() {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Physics = VolumePhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {32, 32};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid, [](double x[]) {
    double region1 = x[0] - 0.4;  // <= 0
    double region2 = x[1] - 0.4;  // <= 0
    return hard_max<double>({region1, region2});
  });

  Quadrature quadrature(mesh);
  Basis basis(mesh);
  Physics physics;

  Analysis analysis(mesh, quadrature, basis, physics);

  ToVTK<T, Mesh> vtk(mesh, "lbracket_lsf.vtk");
  vtk.write_mesh();
}

int main() { test_lbracket_via_lsf(); }
