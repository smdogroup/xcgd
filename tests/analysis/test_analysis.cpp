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
#include "interface_analysis.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "physics/stress.h"
#include "physics/volume.h"
#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/mesher.h"
#include "utils/misc.h"

using T = double;

template <class Mesh>
void save_mesh(const Mesh& mesh, std::string vtk_path) {
  ToVTK<T, typeof(mesh.get_lsf_mesh())> vtk(mesh.get_lsf_mesh(), vtk_path);
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_dof().data());
}

template <int Np_1d, class Physics>
T LSF_jacobian_adjoint_product_fd_check(Physics& physics, double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {13, 9};
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
      "LSF_jap_quad_Np_" + std::to_string(Np_1d) + ".vtk");
  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();

  return relerr;
}

template <int Np_1d, class PhysicsBulk, class PhysicsInterface>
T two_sided_LSF_jacobian_adjoint_product_fd_check(
    PhysicsBulk& physics_bulk_master, PhysicsBulk& physics_bulk_slave,
    PhysicsInterface& physics_interface, double dh = 1e-6) {
  static_assert(PhysicsBulk::dof_per_node == PhysicsInterface::dof_per_node,
                "");
  constexpr int dof_per_node = PhysicsBulk::dof_per_node;

  using Grid = StructuredGrid2D<T>;
  using Mesh = FiniteCellMesh<T, Np_1d>;
  using BulkQuadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;
  using InterfaceQuadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE, Grid, Np_1d, Mesh>;

  using Basis = GDBasis2D<T, Mesh>;

  using BulkAnalysis =
      GalerkinAnalysis<T, Mesh, BulkQuadrature, Basis, PhysicsBulk>;

  using InterfaceAnalysis =
      InterfaceGalerkinAnalysis<T, Mesh, InterfaceQuadrature, Basis,
                                PhysicsInterface>;

  int nxy[2] = {13, 9};
  T lxy[2] = {3.0, 2.0};
  T pt0[2] = {3.2, -0.5};
  Grid grid(nxy, lxy);

  auto lsf_master = [pt0](T x[]) {
    return 1.0 - (x[0] - pt0[0]) * (x[0] - pt0[0]) / 3.5 / 3.5 -
           (x[1] - pt0[1]) * (x[1] - pt0[1]) / 2.0 / 2.0;  // <= 0
  };
  auto lsf_slave = [lsf_master](T x[]) { return -lsf_master(x); };

  Mesh mesh_master(grid, lsf_master);
  Mesh mesh_slave(grid, lsf_slave);

  Basis basis_master(mesh_master);
  Basis basis_slave(mesh_slave);

  BulkQuadrature quadrature_master(mesh_master);
  BulkQuadrature quadrature_slave(mesh_slave);
  InterfaceQuadrature quadrature_interface(
      mesh_master);  // Note, we use master mesh here

  BulkAnalysis analysis_master(mesh_master, quadrature_master, basis_master,
                               physics_bulk_master);
  BulkAnalysis analysis_slave(mesh_slave, quadrature_slave, basis_slave,
                              physics_bulk_slave);
  InterfaceAnalysis analysis_interface(
      mesh_master, mesh_slave, quadrature_interface,
      basis_master /*NOTE: we use basis_master here*/, physics_interface);

  int ndof =
      dof_per_node * (mesh_master.get_num_nodes() + mesh_slave.get_num_nodes());
  int node_offset = mesh_master.get_num_nodes();
  int ndv = grid.get_num_verts();

  std::vector<T> dof(ndof, T(0.0)), psi(ndof, T(0.0)), psi_neg(ndof, T(0.0)),
      res1(ndof, T(0.0)), res2(ndof, T(0.0));
  std::vector<T> dfdphi(ndv, T(0.0)), p(ndv, T(0.0));

  srand(42);
  for (int i = 0; i < ndv; i++) {
    p[i] = (double)rand() / RAND_MAX;
  }

  for (int i = 0; i < ndof; i++) {
    dof[i] = (double)rand() / RAND_MAX;
    psi[i] = (double)rand() / RAND_MAX;
    psi_neg[i] = -psi[i];
  }

  analysis_master.LSF_jacobian_adjoint_product(dof.data(), psi.data(),
                                               dfdphi.data());
  analysis_slave.LSF_jacobian_adjoint_product(dof.data(), psi_neg.data(),
                                              dfdphi.data(), node_offset);

  analysis_master.residual(nullptr, dof.data(), res1.data());
  analysis_slave.residual(nullptr, dof.data(), res1.data(), node_offset);

  save_mesh(mesh_master,
            "two_sided_master_LSF_jacobian_adjoint_product_fd_check_fd1_Np_" +
                std::to_string(Np_1d) + "_h_" + std::to_string(dh) + ".vtk");
  save_mesh(mesh_slave,
            "two_sided_slave_LSF_jacobian_adjoint_product_fd_check_fd1_Np_" +
                std::to_string(Np_1d) + "_h_" + std::to_string(dh) + ".vtk");

  auto& phi_master = mesh_master.get_lsf_dof();
  auto& phi_slave = mesh_slave.get_lsf_dof();

  if (phi_master.size() != ndv) throw std::runtime_error("dimension mismatch");
  if (phi_slave.size() != ndv) throw std::runtime_error("dimension mismatch");

  for (int i = 0; i < ndv; i++) {
    phi_master[i] += dh * p[i];
    phi_slave[i] -= dh * p[i];
  }
  mesh_master.update_mesh();
  mesh_slave.update_mesh();

  save_mesh(mesh_master,
            "two_sided_master_LSF_jacobian_adjoint_product_fd_check_fd2_Np_" +
                std::to_string(Np_1d) + "_h_" + std::to_string(dh) + ".vtk");
  save_mesh(mesh_slave,
            "two_sided_slave_LSF_jacobian_adjoint_product_fd_check_fd2_Np_" +
                std::to_string(Np_1d) + "_h_" + std::to_string(dh) + ".vtk");

  analysis_master.residual(nullptr, dof.data(), res2.data());
  analysis_slave.residual(nullptr, dof.data(), res2.data(), node_offset);

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
  // auto [xloc_q, val_q] = analysis.interpolate(
  //     std::vector<T>(mesh.get_num_nodes() * Physics::dof_per_node,
  //     0.0).data());
  // FieldToVTKNew<T, Basis::spatial_dim> field_vtk(
  //     "LSF_jap_quad_Np_" + std::to_string(Np_1d) + ".vtk");
  // field_vtk.add_mesh(xloc_q);
  // field_vtk.write_mesh();

  return relerr;
}

template <int Np_1d>
T LSF_energy_derivatives_fd_check(double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Physics = LinearElasticity2DVonMisesStressAggregation<T>;
  using Stress = LinearElasticity2DVonMisesStress<T>;

  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using StressAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Stress>;

  double ksrho = 1.0;
  T E = 10.0, nu = 0.3;
  T yield_stress = 100.0;
  Physics physics(ksrho, E, nu, yield_stress);
  Stress stress(E, nu);

  int nxy[2] = {13, 9};
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
  StressAnalysis stress_analysis(mesh, quadrature, basis, stress);

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

  auto [xloc_q, stress_q] = stress_analysis.interpolate_energy(dof.data());
  T max_stress_val = *std::max_element(stress_q.begin(), stress_q.end());
  physics.set_max_stress_ratio(max_stress_val / yield_stress);

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
void test_LSF_jacobian_adjoint_product(Physics& physics, double tol = 1e-6) {
  T relerr_min = 1.0;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    T ret = LSF_jacobian_adjoint_product_fd_check<Np_1d, Physics>(physics, dh);
    if (ret < relerr_min) relerr_min = ret;
  }

  EXPECT_LE(relerr_min, tol);
}

template <int Np_1d, class PhysicsBulk, class PhysicsInterface>
void test_two_sided_LSF_jacobian_adjoint_product(
    PhysicsBulk& physics_master, PhysicsBulk& physics_slave,
    PhysicsInterface& physics_interface, double tol = 1e-6) {
  T relerr_min = 1.0;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    T ret = two_sided_LSF_jacobian_adjoint_product_fd_check<Np_1d, PhysicsBulk,
                                                            PhysicsInterface>(
        physics_master, physics_slave, physics_interface, dh);
    if (ret < relerr_min) relerr_min = ret;
  }

  EXPECT_LE(relerr_min, tol);
}

template <int Np_1d>
void test_LSF_energy_derivatives(double tol = 1e-6) {
  T relerr_min = 1.0;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    T ret = LSF_energy_derivatives_fd_check<Np_1d>(dh);
    if (ret < relerr_min) relerr_min = ret;
  }

  EXPECT_LE(relerr_min, tol);
}

TEST(analysis, AdjJacProductPoisson) {
  auto source_func = [](const A2D::Vec<T, 2> xloc) {
    // return -1.2 * xloc(0) + 3.4 * xloc(1);
    return 0.0;
  };
  using Physics = PoissonPhysics<T, 2, typeof(source_func)>;
  Physics physics(source_func);
  test_LSF_jacobian_adjoint_product<2>(physics, 1e-5);
  test_LSF_jacobian_adjoint_product<4>(physics, 1e-5);
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
  test_LSF_jacobian_adjoint_product<2>(physics, 1e-5);
  test_LSF_jacobian_adjoint_product<4>(physics, 1e-5);
}

TEST(analysis, AdjJacProductElasticityInterface) {
  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    return ret;
  };

  int constexpr spatial_dim = 2;
  using PhysicsBulk = LinearElasticity<T, spatial_dim, typeof(int_func)>;
  using PhysicsInterface = LinearElasticityInterface<T, spatial_dim>;

  T E1 = 20.0, nu1 = 0.3;
  T E2 = 12.0, nu2 = 0.4;
  PhysicsBulk physics_master(E1, nu1, int_func);
  PhysicsBulk physics_slave(E2, nu2, int_func);

  double eta = 12.345;
  PhysicsInterface physics_interface(eta, E1, nu1, E2, nu2);

  test_two_sided_LSF_jacobian_adjoint_product<2>(physics_master, physics_slave,
                                                 physics_interface, 1e-5);
}

TEST(analysis, EnergyPartialStressKS) {
  test_LSF_energy_derivatives<2>(1e-5);
  test_LSF_energy_derivatives<4>(1e-5);
}
