#include <string>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/mesher.h"
#include "utils/vtk.h"

template <typename T, class Mesh, class Quadrature, class Basis>
void solve_poisson(T *lxy, Mesh &mesh, Quadrature &quadrature, Basis &basis,
                   std::string name) {
  auto source_fun = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return T(0.0);
  };
  using Physics = PoissonPhysics<T, Basis::spatial_dim, typeof(source_fun)>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  Physics physics(source_fun);
  Analysis analysis(mesh, quadrature, basis, physics);

  int ndof = Basis::spatial_dim * mesh.get_num_nodes();

  // Set up Jacobian matrix
  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivityFunctor(
      mesh.get_num_nodes(), mesh.get_num_elements(),
      mesh.max_nnodes_per_element,
      [&mesh](int elem, int *nodes) -> int {
        return mesh.get_elem_dof_nodes(elem, nodes);
      },
      &rowp, &cols);

  int nnz = rowp[mesh.get_num_nodes()];
  BSRMat *jac_bsr = new BSRMat(mesh.get_num_nodes(), nnz, rowp, cols);

  // Compute Jacobian matrix
  std::vector<T> dof(ndof, 0.0);
  analysis.jacobian(nullptr, dof.data(), jac_bsr);

  // Store right hand size to sol
  std::vector<T> sol(ndof);
  analysis.residual(nullptr, dof.data(), sol.data());
  for (int i = 0; i < sol.size(); i++) {
    sol[i] *= -1.0;
  }

  // Set up bcs
  std::vector<int> dof_bcs;
  double tol = 1e-6;
  for (int i = 0; i < mesh.get_num_nodes(); i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    if (freal(xloc[0]) < freal(tol) or freal(xloc[1]) < freal(tol) or
        freal(xloc[0]) > freal(lxy[0] - tol) or
        freal(xloc[1]) > freal(lxy[1] - tol)) {
      dof_bcs.push_back(i);
    }
  }

  jac_bsr->write_mtx("K0_" + name + ".mtx");

  // Apply bcs to Jacobian matrix
  jac_bsr->zero_rows(dof_bcs.size(), dof_bcs.data());
  CSCMat *jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
  jac_csc->zero_columns(dof_bcs.size(), dof_bcs.data());

  jac_csc->write_mtx("K1_" + name + ".mtx");

  // Apply bcs to right hand size
  for (int dof : dof_bcs) {
    sol[dof] = 0.0;
  }
  std::vector<T> rhs = sol;

  // solve
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T> *chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  chol->factor();
  chol->solve(sol.data());

  // Check error
  // res = Ku - rhs
  std::vector<T> Ku(sol.size());
  jac_bsr->axpy(sol.data(), Ku.data());
  T err = 0.0;
  for (int i = 0; i < Ku.size(); i++) {
    err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
  }
  std::printf("||Ku - f||: %25.15e\n", sqrt(err));

  // Write to vtk
  ToVTK<T, typename Basis::Mesh> vtk(mesh, name + ".vtk");
  vtk.write_mesh();
  vtk.write_sol("u", sol.data());
}

void solve_poisson_fem() {
  using T = double;
  using Quadrature = QuadrilateralQuadrature<T>;
  using Basis = QuadrilateralBasis<T>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;
  int nxy[2] = {64, 64};
  T lxy[2] = {1.0, 1.0};
  create_2d_rect_quad_mesh(nxy, lxy, &num_elements, &num_nodes, &element_nodes,
                           &xloc);

  Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Quadrature quadrature;
  Basis basis;

  solve_poisson(lxy, mesh, quadrature, basis, "fe");
}

void solve_poisson_gd() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  int nxy[2] = {32, 32};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  solve_poisson(lxy, mesh, quadrature, basis, "gd");
}

int main(int argc, char *argv[]) {
  solve_poisson_fem();
  solve_poisson_gd();

  return 0;
}
