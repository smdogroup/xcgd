#include "apps/nitsche.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/poisson.h"
#include "test_commons.h"
#include "utils/vtk.h"

template <typename T, class Mesh, class Quadrature, class Basis>
void expect_sol_near(T xmin, T xmax, T ymin, T ymax, Mesh &mesh,
                     Quadrature &quadrature, Basis &basis) {
  auto source_fun = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return T(0.0);
  };
  auto exact_fun = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return 2.0 * (1.0 + xloc[1]) /
           ((3.0 + xloc[0]) * (3.0 + xloc[0]) +
            (1.0 + xloc[1]) * (1.0 + xloc[1]));
  };

  using PoissonBulk = PoissonPhysics<T, Basis::spatial_dim, typeof(source_fun)>;
  using PoissonBCs =
      PoissonCutDirichlet<T, Basis::spatial_dim, typeof(exact_fun)>;

  using Nitsche =
      NitscheBCsApp<T, Mesh, Quadrature, Basis, PoissonBulk, PoissonBCs>;

  PoissonBulk poisson_bulk(source_fun);
  double nitsche_eta = 1e6;
  PoissonBCs poisson_bcs(nitsche_eta, exact_fun);

  Nitsche nitsche(mesh, quadrature, basis, poisson_bulk, poisson_bcs);

  int nnodes = mesh.get_num_nodes();

  std::vector<int> dof_bcs;
  std::vector<T> dof_vals;
  std::vector<T> sol_exact(nnodes, 0.0);
  double tol = 1e-6;
  for (int i = 0; i < nnodes; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    if (freal(xloc[0]) < xmin + tol or freal(xloc[1]) < ymin + tol or
        freal(xloc[0]) > freal(xmax) - tol or
        freal(xloc[1]) > freal(ymax) - tol) {
      dof_bcs.push_back(i);
      dof_vals.push_back(2.0 * (1.0 + xloc[1]) /
                         ((3.0 + xloc[0]) * (3.0 + xloc[0]) +
                          (1.0 + xloc[1]) * (1.0 + xloc[1])));
    }
    sol_exact[i] = exact_fun(xloc);
  }

  std::vector<T> sol = nitsche.solve();

  ToVTK<T, Mesh> vtk(mesh, "nitsche_poisson.vtk");
  vtk.write_mesh();
  vtk.write_sol("u", sol.data());

  T sol_l2 = 0.0, exact_l2 = 0.0;
  for (int i = 0; i < nnodes; i++) {
    sol_l2 += sol[i] * sol[i];
    exact_l2 += sol_exact[i] * sol_exact[i];
  }
  sol_l2 = sqrt(sol_l2);
  exact_l2 = sqrt(exact_l2);

  EXPECT_NEAR(sol_l2 / exact_l2 - 1.0, 0.0, 1e-4);
}

template <int Np_1d>
void test_poisson_app_nitsche() {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  int nxy[2] = {32, 32};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};
  Grid grid(nxy, lxy, xy0);
  Mesh mesh(grid, [](T *x) { return x[0] * x[0] + x[1] * x[1] - 0.8 * 0.8; });
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  expect_sol_near(-1.0, 1.0, -1.0, 1.0, mesh, quadrature, basis);
}
TEST(apps, PoissonNitsche) { test_poisson_app_nitsche<4>(); }
