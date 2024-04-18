#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

void mesh_test() {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using LSFQuadrature = GDGaussQuadrature2D<T, Np_1d>;
  using LSFMesh = GridMesh<T, Np_1d>;
  using LSFBasis = GDBasis2D<T, LSFMesh>;

  using Filter = HelmholtzFilter<T, LSFMesh, LSFQuadrature, LSFBasis>;

  using Physics = VolumePhysics<T, Grid::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {48, 64};
  T lxy[2] = {3.0, 3.5};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  LSFMesh lsf_mesh = mesh.get_lsf_mesh();
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  int ndv = lsf_mesh.get_num_nodes();
  std::vector<T>& dvs = mesh.get_lsf_dof();
  std::vector<T> p(ndv, 0.0);

  int m = 3, n = 5;
  for (int i = 0; i < ndv; i++) {
    T xloc[Basis::spatial_dim];
    lsf_mesh.get_node_xloc(i, xloc);
    dvs[i] = cos(xloc[0] / lxy[0] * 2.0 * PI * m) *
                 cos(xloc[1] / lxy[1] * 2.0 * PI * n) -
             0.5;
  }
  mesh.update_mesh();

  double h = 1e-6;
  for (int i = 0; i < ndv; i++) {
    p[i] = (T)rand() / RAND_MAX;
  }

  T r0 = 0.2;
  LSFQuadrature lsf_quadrature(lsf_mesh);
  LSFBasis lsf_basis(lsf_mesh);
  Filter filter(r0, lsf_mesh, lsf_quadrature, lsf_basis);

  std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
  std::printf("LSF area: %20.10e\n", analysis.energy(nullptr, dummy.data()));

  for (int i = 0; i < ndv; i++) {
    dvs[i] -= h * p[i];
  }
  T a1 = analysis.energy(nullptr, dummy.data());

  for (int i = 0; i < ndv; i++) {
    dvs[i] += 2.0 * h * p[i];
  }
  T a2 = analysis.energy(nullptr, dummy.data());

  T dfdx_fd = (a2 - a1) / 2.0 / h;

  std::vector<T> dfdx(ndv, 0.0);
  analysis.LSF_volume_derivatives(dfdx.data());

  T dfdx_exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdx_exact += dfdx[i] * p[i];
  }

  std::printf("area gradient fd:    %25.15e\n", dfdx_fd);
  std::printf("area gradient exact: %25.15e\n", dfdx_exact);

  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);

  interp.to_vtk("quadratures.vtk", dummy.data());

  ToVTK<T, GridMesh<T, Np_1d>> lsf_vtk(mesh.get_lsf_mesh(), "lsf_mesh.vtk");
  lsf_vtk.write_mesh();
  lsf_vtk.write_sol("lsf", mesh.get_lsf_dof().data());

  ToVTK<T, Mesh> vtk(mesh, "cut_mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
}

int main() {
  mesh_test();
  return 0;
}