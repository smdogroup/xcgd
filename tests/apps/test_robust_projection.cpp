#include "analysis.h"
#include "apps/robust_projection.h"
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

template <int Np_1d, int Np_1d_filter>
void test_helmholtz_filter() {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  int constexpr spatial_dim = Grid::spatial_dim;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {128, 64};
  T lxy[2] = {2.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  T beta = 12.3, eta = 0.56;
  int ndv = mesh.get_num_nodes();
  RobustProjection<T> proj(beta, eta, ndv);

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
    grid.get_vert_xloc(i, xloc);
    x[i] = (cos(xloc[0] / lxy[0] * 2.0 * PI * m) - 0.5) *
               (cos(xloc[1] / lxy[1] * 2.0 * PI * n) - 0.5) * 2.0 / 3.0 -
           0.5;
  }

  proj.apply(x.data(), phi.data());

  std::vector<T> dfdx(ndv, 0.0);
  proj.applyGradient(x.data(), w.data(), dfdx.data());

  double h = 1e-6, tol = 1e-6;
  for (int i = 0; i < ndv; i++) {
    x[i] -= h * p[i];
  }

  proj.apply(x.data(), phi.data());
  T s1 = scalar_function(phi, w);

  for (int i = 0; i < ndv; i++) {
    x[i] += 2.0 * h * p[i];
  }
  proj.apply(x.data(), phi.data());
  T s2 = scalar_function(phi, w);

  double dfdx_fd = (s2 - s1) / 2.0 / h;
  double dfdx_exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdx_exact += dfdx[i] * p[i];
  }

  std::printf("dfdx_fd:    %25.15e\n", dfdx_fd);
  std::printf("dfdx_exact: %25.15e\n", dfdx_exact);
  EXPECT_NEAR((dfdx_fd - dfdx_exact) / dfdx_exact, 0.0, tol);

  char name[256];
  std::snprintf(name, 256, "helmholtz_%d_%d.vtk", Np_1d, Np_1d_filter);
  ToVTK<T, Mesh> vtk(mesh, name);
  vtk.write_mesh();
  vtk.write_sol("x", x.data());
  vtk.write_sol("phi", phi.data());
}

TEST(apps, HelmholtzFilter) {
  test_helmholtz_filter<4, 4>();
  test_helmholtz_filter<4, 2>();
}
