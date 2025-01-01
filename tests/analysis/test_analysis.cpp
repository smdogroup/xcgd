#include <complex>
#include <functional>
#include <memory>
#include <numeric>
#include <string>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/fe_tetrahedral.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/cut_bcs.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "physics/stress.h"
#include "physics/volume.h"
#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/mesher.h"

using T = double;

template <class Mesh>
void save_mesh(const Mesh& mesh, std::string vtk_path) {
  ToVTK<T, typeof(mesh.get_lsf_mesh())> vtk(mesh.get_lsf_mesh(), vtk_path);
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_dof().data());
}

template <int Np_1d, class Physics>
T LSF_jacobian_adjoint_product_fd_check(const Physics& physics,
                                        double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {18, 12};
  T lxy[2] = {3.0, 2.0};
  T pt0[2] = {3.2, -0.5};
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [pt0](T x[]) {
    return 1.0 - (x[0] - pt0[0]) * (x[0] - pt0[0]) / 3.5 / 3.5 -
           (x[1] - pt0[1]) * (x[1] - pt0[1]) / 2.0 / 2.0;  // <= 0
  });
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  Analysis analysis(mesh, quadrature, basis, physics);

  int ndof = mesh.get_num_nodes() * Physics::dof_per_node;
  int ndv = grid.get_num_verts();

  std::vector<T> dof(ndof, T(0.0)), psi(ndof, T(0.0)), res1(ndof, T(0.0)),
      res2(ndof, T(0.0));
  std::vector<T> dfdphi(ndv, T(0.0)), p(ndv, T(0.0));

  srand(42);
  for (int i = 0; i < ndv; i++) {
    p[i] = (double)rand() / RAND_MAX;
  }

  for (int i = 0; i < ndof; i++) {
    dof[i] = (double)rand() / RAND_MAX;
    psi[i] = (double)rand() / RAND_MAX;
  }

  analysis.LSF_jacobian_adjoint_product(dof.data(), psi.data(), dfdphi.data());
  analysis.residual(nullptr, dof.data(), res1.data());

  save_mesh(mesh, "LSF_jacobian_adjoint_product_fd_check_fd1_Np_" +
                      std::to_string(Np_1d) + ".vtk");

  auto& phi = mesh.get_lsf_dof();
  if (phi.size() != ndv) throw std::runtime_error("dimension mismatch");
  for (int i = 0; i < ndv; i++) {
    phi[i] += dh * p[i];
  }
  mesh.update_mesh();

  save_mesh(mesh, "LSF_jacobian_adjoint_product_fd_check_fd2_Np_" +
                      std::to_string(Np_1d) + ".vtk");

  analysis.residual(nullptr, dof.data(), res2.data());

  T fd = 0.0, exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    exact += dfdphi[i] * p[i];
  }

  for (int i = 0; i < ndof; i++) {
    fd += psi[i] * (res2[i] - res1[i]) / dh;
  }

  T relerr = fabs(fd - exact) / fabs(exact);

  std::printf(
      "Np_1d: %d, dh: %.5e, FD: %30.20e, Actual: %30.20e, Rel err: %20.10e\n",
      Np_1d, dh, fd, exact, relerr);

  // Debug
  auto [xloc_q, val_q] = analysis.interpolate(
      std::vector<T>(mesh.get_num_nodes() * Physics::dof_per_node, 0.0).data());
  FieldToVTKNew<T, Basis::spatial_dim> field_vtk(
      "quad_Np_" + std::to_string(Np_1d) + ".vtk");
  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();

  return relerr;
}

template <int Np_1d, class Physics>
T LSF_energy_derivatives_fd_check(const Physics& physics, double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {18, 12};
  T lxy[2] = {3.0, 2.0};
  T pt0[2] = {3.2, -0.5};
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [pt0](T x[]) {
    return 1.0 - (x[0] - pt0[0]) * (x[0] - pt0[0]) / 3.5 / 3.5 -
           (x[1] - pt0[1]) * (x[1] - pt0[1]) / 2.0 / 2.0;  // <= 0
  });
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  Analysis analysis(mesh, quadrature, basis, physics);

  int ndof = mesh.get_num_nodes() * Physics::dof_per_node;
  int ndv = grid.get_num_verts();

  std::vector<T> dof(ndof, T(0.0));
  std::vector<T> dfdphi(ndv, T(0.0)), p(ndv, T(0.0));

  srand(42);
  for (int i = 0; i < ndv; i++) {
    p[i] = (double)rand() / RAND_MAX;
  }

  for (int i = 0; i < ndof; i++) {
    dof[i] = (double)rand() / RAND_MAX;
  }

  T e1 = analysis.energy(nullptr, dof.data());
  analysis.LSF_energy_derivatives(dof.data(), dfdphi.data());

  // Update mesh
  auto& phi = mesh.get_lsf_dof();
  if (phi.size() != ndv) throw std::runtime_error("dimension mismatch");
  for (int i = 0; i < ndv; i++) {
    phi[i] += dh * p[i];
  }
  mesh.update_mesh();

  T e2 = analysis.energy(nullptr, dof.data());

  T fd = (e2 - e1) / dh;
  T exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    exact += dfdphi[i] * p[i];
  }

  T relerr = fabs(fd - exact) / fabs(exact);

  std::printf(
      "Np_1d: %d, dh: %.5e, FD: %30.20e, Actual: %30.20e, Rel err: %20.10e\n",
      Np_1d, dh, fd, exact, relerr);

  return relerr;
}

template <int Np_1d, class Physics>
void test_LSF_jacobian_adjoint_product(const Physics& physics) {
  T relerr_min = 1.0;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    T ret = LSF_jacobian_adjoint_product_fd_check<Np_1d, Physics>(physics, dh);
    if (ret < relerr_min) relerr_min = ret;
  }

  EXPECT_LE(relerr_min, 1e-6);
}

template <int Np_1d, class Physics>
void test_LSF_energy_derivatives(const Physics& physics) {
  T relerr_min = 1.0;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    T ret = LSF_energy_derivatives_fd_check<Np_1d, Physics>(physics, dh);
    if (ret < relerr_min) relerr_min = ret;
  }

  EXPECT_LE(relerr_min, 1e-6);
}

TEST(analysis, AdjJacProductPoisson) {
  auto source_func = [](const A2D::Vec<T, 2> xloc) {
    // return -1.2 * xloc(0) + 3.4 * xloc(1);
    return 0.0;
  };
  using Physics = PoissonPhysics<T, 2, typeof(source_func)>;
  Physics physics(source_func);
  test_LSF_jacobian_adjoint_product<2>(physics);
  test_LSF_jacobian_adjoint_product<4>(physics);
  test_LSF_jacobian_adjoint_product<6>(physics);
}

TEST(analysis, AdjJacProductElasticity) {
  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    // ret(0) = -1.2 * xloc(0);
    // ret(1) = 3.4 * xloc(1);
    return ret;
  };
  using Physics = LinearElasticity<T, 2, typeof(int_func)>;
  T E = 10.0, nu = 0.3;
  Physics physics(E, nu, int_func);
  test_LSF_jacobian_adjoint_product<2>(physics);
  test_LSF_jacobian_adjoint_product<4>(physics);
  test_LSF_jacobian_adjoint_product<6>(physics);
}

TEST(analysis, EnergyPartialStressKS) {
  using Physics = LinearElasticity2DVonMisesStressAggregation<T>;

  double ksrho = 12.34;
  T E = 10.0, nu = 0.3;
  Physics physics(ksrho, E, nu);

  test_LSF_energy_derivatives<2>(physics);
}
