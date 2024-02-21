#include <complex>
#include <functional>
#include <numeric>
#include <string>

#include "analysis.h"
#include "elements/quadrilateral.h"
#include "elements/tetrahedral.h"
#include "physics/neohookean.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/mesh.h"

template <typename T, int spatial_dim, class Physics, class Basis,
          class Quadrature>
void test_physics(int num_elements, int num_nodes, int *element_nodes, T *xloc,
                  Physics &physics) {
  // Set the number of degrees of freeom
  int ndof = Physics::dof_per_node * num_nodes;

  // Allocate space for the degrees of freeom
  T *dof = new T[ndof];
  T *res = new T[ndof];
  T *Jp = new T[ndof];
  T *Jp_axpy = new T[ndof];
  T *direction = new T[ndof];
  double *p = new double[ndof];
  double h = 1e-30;
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
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;
  T energy = Analysis::energy(physics, num_elements, element_nodes, xloc, dof);
  Analysis::residual(physics, num_elements, element_nodes, xloc, dof, res);
  Analysis::jacobian_product(physics, num_elements, element_nodes, xloc, dof,
                             direction, Jp);

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
  dJp_relerr = (dJp_exact - dJp_cs) / dJp_cs;

  std::printf("\nDerivatives check for the residual\n");
  std::printf("complex step derivatives: %25.15e\n", dres_cs);
  std::printf("exact derivatives:        %25.15e\n", dres_exact);
  std::printf("relative error:           %25.15e\n", dres_relerr);

  std::printf("\nDerivatives check for the Jacobian-vector product\n");
  std::printf("complex step derivatives: %25.15e\n", dJp_cs);
  std::printf("exact derivatives:        %25.15e\n", dJp_exact);
  std::printf("relative error:           %25.15e\n", dJp_relerr);

  EXPECT_NEAR(dres_relerr, 0.0, 1e-14);
  EXPECT_NEAR(dJp_relerr, 0.0, 1e-14);

  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivity(num_nodes, num_elements,
                                   Basis::nodes_per_element, element_nodes,
                                   &rowp, &cols);
  int nnz = rowp[num_nodes];
  using BSRMat =
      SparseUtils::BSRMat<T, Physics::dof_per_node, Physics::dof_per_node>;
  BSRMat *jac_bsr = new BSRMat(num_nodes, num_nodes, nnz, rowp, cols);
  Analysis::jacobian(physics, num_elements, element_nodes, xloc, dof, jac_bsr);
  jac_bsr->axpy(direction, Jp_axpy);

  double Jp_l1 = 0.0;
  double Jp_axpy_l1 = 0.0;
  for (int i = 0; i < ndof; i++) {
    Jp_l1 += Jp[i].real() * p[i];
    Jp_axpy_l1 += Jp_axpy[i].real() * p[i];
  }
  double Jp_relerr = (Jp_l1 - Jp_axpy_l1) / Jp_l1;
  std::printf("\nDerivatives check for the Jacobian matrix\n");
  std::printf("Jac-vec product:          %25.15e\n", Jp_l1);
  std::printf("Jac-vec product by axpy:  %25.15e\n", Jp_axpy_l1);
  std::printf("relative error:           %25.15e\n", Jp_relerr);
  EXPECT_NEAR(Jp_relerr, 0.0, 1e-14);
}

TEST(Neohookean, Quad) {
  using T = std::complex<double>;
  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  int nx = 1, ny = 1;
  T lx = 1.0, ly = 1.0;
  create_2d_rect_quad_mesh(nx, ny, lx, ly, &num_elements, &num_nodes,
                           &element_nodes, &xloc);

  constexpr static int spatial_dim = 2;
  using Physics = NeohookeanPhysics<T, spatial_dim>;
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);
  test_physics<T, spatial_dim, Physics, QuadrilateralBasis,
               QuadrilateralQuadrature>(num_elements, num_nodes, element_nodes,
                                        xloc, physics);
}

TEST(Neohookean, Tet) {
  using T = std::complex<double>;
  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  create_single_element_mesh(&num_elements, &num_nodes, &element_nodes, &xloc);

  constexpr static int spatial_dim = 3;
  using Physics = NeohookeanPhysics<T, spatial_dim>;
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);
  test_physics<T, spatial_dim, Physics, TetrahedralBasis,
               TetrahedralQuadrature>(num_elements, num_nodes, element_nodes,
                                      xloc, physics);
}

TEST(Poisson, Quad) {
  using T = std::complex<double>;
  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  int nx = 1, ny = 1;
  T lx = 1.0, ly = 1.0;
  create_2d_rect_quad_mesh(nx, ny, lx, ly, &num_elements, &num_nodes,
                           &element_nodes, &xloc);

  constexpr static int spatial_dim = 2;
  using Physics = PoissonPhysics<T, spatial_dim>;
  Physics physics;
  test_physics<T, spatial_dim, Physics, QuadrilateralBasis,
               QuadrilateralQuadrature>(num_elements, num_nodes, element_nodes,
                                        xloc, physics);
}

TEST(Poisson, Tet) {
  using T = std::complex<double>;
  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  create_single_element_mesh(&num_elements, &num_nodes, &element_nodes, &xloc);

  constexpr static int spatial_dim = 3;
  using Physics = PoissonPhysics<T, spatial_dim>;
  Physics physics;
  test_physics<T, spatial_dim, Physics, TetrahedralBasis,
               TetrahedralQuadrature>(num_elements, num_nodes, element_nodes,
                                      xloc, physics);
}