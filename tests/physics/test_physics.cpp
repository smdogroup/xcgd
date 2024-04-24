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
#include "physics/grad_penalization.h"
#include "physics/helmholtz.h"
#include "physics/linear_elasticity.h"
#include "physics/neohookean.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/mesher.h"

using T = std::complex<double>;

template <class Physics, class Mesh, class Quadrature, class Basis>
void test_physics(std::tuple<Mesh *, Quadrature *, Basis *> tuple,
                  Physics &physics, double h = 1e-30, double tol = 1e-14,
                  bool check_res_only = false) {
  Mesh *mesh = std::get<0>(tuple);
  Quadrature *quadrature = std::get<1>(tuple);
  Basis *basis = std::get<2>(tuple);

  int num_nodes = mesh->get_num_nodes();
  int num_elements = mesh->get_num_elements();

  // Set the number of degrees of freeom
  int ndof = Physics::dof_per_node * num_nodes;

  // Allocate space for the degrees of freeom
  T *dof = new T[ndof];
  T *res = new T[ndof];
  T *Jp = new T[ndof];
  T *Jp_axpy = new T[ndof];
  T *direction = new T[ndof];
  double *p = new double[ndof];
  for (int i = 0; i < ndof; i++) {
    direction[i] = (double)rand() / RAND_MAX;
    p[i] = (double)rand() / RAND_MAX;
    dof[i] = 0.01 * rand() / RAND_MAX;
    dof[i] += T(0.0, h * direction[i].real());
    res[i] = 0.0;
    Jp[i] = 0.0;
    Jp_axpy[i] = 0.0;
  }

  // Allocate space for the residual
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  Analysis analysis(*mesh, *quadrature, *basis, physics);

  T energy = analysis.energy(nullptr, dof);
  analysis.residual(nullptr, dof, res);
  analysis.jacobian_product(nullptr, dof, direction, Jp);

  double dres_cs = energy.imag() / h;
  double dres_exact = 0.0;
  double dres_relerr = 0.0;
  double dJp_cs = 0.0;
  double dJp_exact = 0.0;
  double dJp_relerr = 0.0;
  for (int i = 0; i < ndof; i++) {
    dres_exact += res[i].real() * direction[i].real();
    dJp_cs += p[i] * res[i].imag() / h;
    dJp_exact += Jp[i].real() * p[i];
  }

  dres_relerr = (dres_exact - dres_cs) / dres_cs;
  if (dres_exact == 0 and dres_cs == 0) dres_relerr = 0.0;

  dJp_relerr = (dJp_exact - dJp_cs) / dJp_cs;
  if (dJp_exact == 0 and dJp_cs == 0) dJp_relerr = 0.0;

  std::printf("\nDerivatives check for the residual\n");
  std::printf("complex step derivatives: %25.15e\n", dres_cs);
  std::printf("exact derivatives:        %25.15e\n", dres_exact);
  std::printf("relative error:           %25.15e\n", dres_relerr);
  EXPECT_NEAR(dres_relerr, 0.0, tol);

  if (check_res_only) return;

  std::printf("\nDerivatives check for the Jacobian-vector product\n");
  std::printf("complex step derivatives: %25.15e\n", dJp_cs);
  std::printf("exact derivatives:        %25.15e\n", dJp_exact);
  std::printf("relative error:           %25.15e\n", dJp_relerr);
  EXPECT_NEAR(dJp_relerr, 0.0, tol);

  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivityFunctor(
      num_nodes, num_elements, Basis::nodes_per_element,
      [mesh](int elem, int *nodes) { mesh->get_elem_dof_nodes(elem, nodes); },
      &rowp, &cols);
  int nnz = rowp[num_nodes];
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  BSRMat *jac_bsr = new BSRMat(num_nodes, nnz, rowp, cols);
  analysis.jacobian(nullptr, dof, jac_bsr);
  jac_bsr->axpy(direction, Jp_axpy);

  double Jp_l1 = 0.0;
  double Jp_axpy_l1 = 0.0;
  for (int i = 0; i < ndof; i++) {
    Jp_l1 += Jp[i].real() * p[i];
    Jp_axpy_l1 += Jp_axpy[i].real() * p[i];
  }
  double Jp_relerr = (Jp_l1 - Jp_axpy_l1) / Jp_l1;
  if (Jp_l1 == 0 and Jp_axpy_l1 == 0) Jp_relerr = 0.0;

  std::printf("\nDerivatives check for the Jacobian matrix\n");
  std::printf("Jac-vec product:          %25.15e\n", Jp_l1);
  std::printf("Jac-vec product by axpy:  %25.15e\n", Jp_axpy_l1);
  std::printf("relative error:           %25.15e\n", Jp_relerr);
  EXPECT_NEAR(Jp_relerr, 0.0, tol);
}

std::tuple<QuadrilateralBasis<T>::Mesh *, QuadrilateralQuadrature<T> *,
           QuadrilateralBasis<T> *>
create_quad_basis() {
  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  int nxy[2] = {1, 1};
  T lxy[2] = {1.3, 2.4};
  create_2d_rect_quad_mesh(nxy, lxy, &num_elements, &num_nodes, &element_nodes,
                           &xloc);
  using Quadrature = QuadrilateralQuadrature<T>;
  using Basis = QuadrilateralBasis<T>;
  using Mesh = typename Basis::Mesh;
  Mesh *mesh = new Mesh(num_elements, num_nodes, element_nodes, xloc);
  return {mesh, new Quadrature, new Basis};
}

std::tuple<TetrahedralBasis<T>::Mesh *, TetrahedralQuadrature<T> *,
           TetrahedralBasis<T> *>
create_tet_basis() {
  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  int nxyz[3] = {2, 2, 2};
  T lxyz[3] = {1.1, 1.2, 1.3};
  create_3d_box_tet_mesh(nxyz, lxyz, &num_elements, &num_nodes, &element_nodes,
                         &xloc);
  using Quadrature = TetrahedralQuadrature<T>;
  using Basis = TetrahedralBasis<T>;
  using Mesh = typename Basis::Mesh;
  Mesh *mesh = new Mesh(num_elements, num_nodes, element_nodes, xloc);
  return {mesh, new Quadrature, new Basis};
}

template <int Np_1d = 4>
std::tuple<GridMesh<T, Np_1d> *, GDGaussQuadrature2D<T, Np_1d> *,
           GDBasis2D<T, GridMesh<T, Np_1d>> *>
create_gd_basis() {
  int constexpr nx = 5, ny = 7;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {nx, ny};
  T lxy[2] = {1.0, 1.4};
  Grid *grid = new Grid(nxy, lxy);
  Mesh *mesh = new Mesh(*grid);
  return {mesh, new Quadrature(*mesh), new Basis(*mesh)};
}

template <class Quadrature, class Basis>
void test_neohookean(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-30, double tol = 1e-14) {
  T C1 = 0.01;
  T D1 = 0.5;
  NeohookeanPhysics<T, Basis::spatial_dim> physics(C1, D1);
  test_physics(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_elasticity(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-30, double tol = 1e-14) {
  T E = 30.0, nu = 0.3;
  LinearElasticity<T, Basis::spatial_dim> physics(E, nu);
  test_physics(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_poisson(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-30, double tol = 1e-14) {
  using Physics = PoissonPhysics<T, Basis::spatial_dim>;
  Physics physics;
  test_physics(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_helmholtz(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-30, double tol = 1e-14) {
  using Physics = HelmholtzPhysics<T, Basis::spatial_dim>;
  T r0 = 1.2;
  Physics physics(r0);
  test_physics(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_grad_penalization(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-30, double tol = 1e-14) {
  using Physics = GradPenalization<T, Basis::spatial_dim>;
  Physics physics(1.23);
  test_physics(tuple, physics, h, tol, true);
}

TEST(physics, NeohookeanQuad) { test_neohookean(create_quad_basis()); }
TEST(physics, NeohookeanTet) { test_neohookean(create_tet_basis()); }
TEST(physics, NeohookeanGD) { test_neohookean(create_gd_basis(), 1e-8, 1e-6); }

TEST(physics, PoissonQuad) { test_poisson(create_quad_basis()); }
TEST(physics, PoissonTet) { test_poisson(create_tet_basis()); }
TEST(physics, PoissonGD) { test_poisson(create_gd_basis(), 1e-8, 1e-6); }

TEST(physics, HelmholtzQuad) {
  test_helmholtz(create_quad_basis(), 1e-30, 1e-13);
}
TEST(physics, HelmholtzTet) {
  test_helmholtz(create_tet_basis(), 1e-30, 1e-13);
}
TEST(physics, HelmholtzGD) { test_helmholtz(create_gd_basis(), 1e-8, 1e-6); }

TEST(physics, LinearElasticityQuad) { test_elasticity(create_quad_basis()); }
TEST(physics, LinearElasticityTet) { test_elasticity(create_tet_basis()); }
TEST(physics, LinearElasticityGD) {
  test_elasticity(create_gd_basis(), 1e-8, 1e-8);
}

TEST(physics, GradPenalizationQuad) {
  test_grad_penalization(create_quad_basis());
}
TEST(physics, GradPenalizationTet) {
  test_grad_penalization(create_tet_basis());
}
TEST(physics, GradPenalizationGD) {
  test_grad_penalization(create_gd_basis(), 1e-8, 1e-8);
}