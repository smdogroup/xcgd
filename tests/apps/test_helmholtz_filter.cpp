#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"
#include "test_commons.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

template <typename T>
T scalar_function(std::vector<T>& phi, std::vector<T>& w) {
  T ret = 0.0;
  for (int i = 0; i < phi.size(); i++) {
    ret += phi[i] * w[i];
  }
  return ret;
}

TEST(apps, HelmholtzFilter) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  int constexpr spatial_dim = Grid::spatial_dim;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using LSFQuadrature = GDGaussQuadrature2D<T, Np_1d>;
  using LSFMesh = GridMesh<T, Np_1d>;
  using LSFBasis = GDBasis2D<T, LSFMesh>;

  using Filter = HelmholtzFilter<T, LSFMesh, LSFQuadrature, LSFBasis>;

  int nxy[2] = {128, 64};
  T lxy[2] = {2.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  LSFMesh lsf_mesh = mesh.get_lsf_mesh();
  LSFQuadrature lsf_quadrature(lsf_mesh);
  LSFBasis lsf_basis(lsf_mesh);

  T r0 = 0.01;
  Filter filter(r0, lsf_mesh, lsf_quadrature, lsf_basis);

  int ndv = lsf_mesh.get_num_nodes();

  std::vector<T> x(ndv, 0.0), phi(ndv, 0.0), w(ndv, 0.0), p(ndv, 0.0);

  for (int i = 0; i < ndv; i++) {
    // x[i] = (T)rand() / RAND_MAX;
    w[i] = (T)rand() / RAND_MAX;
    p[i] = (T)rand() / RAND_MAX;
  }

  // Set x
  int m = 4, n = 2;
  for (int i = 0; i < ndv; i++) {
    T xloc[spatial_dim];
    lsf_mesh.get_node_xloc(i, xloc);
    x[i] = (cos(xloc[0] / lxy[0] * 2.0 * PI * m) - 0.5) *
               (cos(xloc[1] / lxy[1] * 2.0 * PI * n) - 0.5) * 2.0 / 3.0 -
           0.5;
  }

  filter.apply(x.data(), phi.data());

  std::vector<T> dfdx(ndv, 0.0);
  filter.applyGradient(x.data(), w.data(), dfdx.data());

  double h = 1e-6, tol = 1e-6;
  for (int i = 0; i < ndv; i++) {
    x[i] -= h * p[i];
  }

  filter.apply(x.data(), phi.data());
  T s1 = scalar_function(phi, w);

  for (int i = 0; i < ndv; i++) {
    x[i] += 2.0 * h * p[i];
  }
  filter.apply(x.data(), phi.data());
  T s2 = scalar_function(phi, w);

  double dfdx_fd = (s2 - s1) / 2.0 / h;
  double dfdx_exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdx_exact += dfdx[i] * p[i];
  }

  std::printf("dfdx_fd:    %25.15e\n", dfdx_fd);
  std::printf("dfdx_exact: %25.15e\n", dfdx_exact);
  EXPECT_NEAR((dfdx_fd - dfdx_exact) / dfdx_exact, 0.0, tol);

  ToVTK<T, LSFMesh> lsf_vtk(lsf_mesh, "helmholtz.vtk");
  lsf_vtk.write_mesh();
  lsf_vtk.write_sol("x", x.data());
  lsf_vtk.write_sol("phi", phi.data());
}