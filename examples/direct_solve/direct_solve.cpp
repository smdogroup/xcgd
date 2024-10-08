#include <iostream>
#include <string>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/fe_tetrahedral.h"
#include "physics/neohookean.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/linalg.h"
#include "utils/mesher.h"
#include "utils/timer.h"
#include "utils/vtk.h"

int main(int argc, char *argv[]) {
  using T = double;

  using Quadrature = TetrahedralQuadrature<T>;
  using Basis = TetrahedralBasis<T>;
  using Physics = NeohookeanPhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Basis::Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;
  int ndof_bcs, *dof_bcs;

  // Create the simple mesh
  create_single_element_mesh(&num_elements, &num_nodes, &element_nodes, &xloc,
                             &ndof_bcs, &dof_bcs);

  Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Quadrature quadrature;

  ToVTK<T, typename Basis::Mesh> vtk(mesh);

  vtk.write_mesh();

  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivity(num_nodes, num_elements,
                                   Basis::max_nnodes_per_element, element_nodes,
                                   &rowp, &cols);

  int nnz = rowp[num_nodes];
  BSRMat *jac_bsr = new BSRMat(num_nodes, nnz, rowp, cols);
  CSCMat *jac_csc = nullptr;

  // Set the number of degrees of freedom
  int ndof = Basis::spatial_dim * num_nodes;

  // Allocate space for the degrees of freedom
  T *dof = new T[ndof];
  T *p = new T[ndof];
  T *Jp = new T[ndof];

  for (int i = 0; i < ndof; i++) {
    dof[i] = 0.01 * rand() / RAND_MAX;
    p[i] = 0.0;
    Jp[i] = 0.0;
  }
  p[0] = 1.0;

  // Allocate the physics
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);

  StopWatch watch;
  Basis basis;
  Analysis analysis(mesh, quadrature, basis, physics);
  analysis.jacobian(nullptr, dof, jac_bsr);
  double t1 = watch.lap();
  std::printf("Jacobian assembly time: %.3e s\n", t1);

  jac_bsr->write_mtx("jac_bsr_no_bcs.mtx");

  jac_bsr->zero_rows(ndof_bcs, dof_bcs);
  jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
  jac_csc->zero_columns(ndof_bcs, dof_bcs);

  jac_bsr->write_mtx("jac_bsr.mtx");
  jac_csc->write_mtx("jac_csc.mtx");

  // Create rhs
  std::vector<T> b(jac_csc->nrows);
  for (int i = 0; i < jac_csc->nrows; i++) {
    b[i] = 0.0;
  }
  for (int i = 0; i < jac_csc->nrows; i++) {
    for (int jp = jac_csc->colp[i]; jp < jac_csc->colp[i + 1]; jp++) {
      b[jac_csc->rows[jp]] += jac_csc->vals[jp];
    }
  }

  double t4 = watch.lap();
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T> *chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  double t5 = watch.lap();
  std::printf("Setup/order/setvalue time: %12.5e s\n", t5 - t4);
  chol->factor();
  double t6 = watch.lap();
  std::printf("Factor time:               %12.5e s\n", t6 - t5);
  chol->solve(b.data());
  double t7 = watch.lap();
  std::printf("Solve time:                %12.5e s\n", t7 - t6);
  T err = 0.0;
  for (int i = 0; i < jac_csc->nrows; i++) {
    err += (1.0 - b[i]) * (1.0 - b[i]);
  }
  std::printf("||x - e||: %25.15e\n", sqrt(err));
  return 0;
}
