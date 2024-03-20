#include "physics/linear_elasticity.h"

#include <string>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/mesh.h"
#include "utils/vtk.h"

template <typename T, class Quadrature, class Basis>
void solve_linear_elasticity(T E, T nu, Basis &basis, std::string name) {
  using Physics = LinearElasticity<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  Quadrature quadrature(basis.mesh);
  Physics physics(E, nu);
  Analysis analysis(quadrature, basis, physics);

  int ndof = Physics::dof_per_node * basis.mesh.get_num_nodes();

  // Set up Jacobian matrix
  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivityFunctor(
      basis.mesh.get_num_nodes(), basis.mesh.get_num_elements(),
      basis.mesh.nodes_per_element,
      [&basis](int elem, int *nodes) {
        basis.mesh.get_elem_dof_nodes(elem, nodes);
      },
      &rowp, &cols);

  int nnz = rowp[basis.mesh.get_num_nodes()];
  BSRMat *jac_bsr = new BSRMat(basis.mesh.get_num_nodes(), nnz, rowp, cols);

  // Compute Jacobian matrix
  std::vector<T> dof(ndof, 0.0);
  analysis.jacobian(nullptr, dof.data(), jac_bsr);

  // Store right hand size to sol
  std::vector<T> sol(ndof, 0.0);

  // Set boundary conditions
  std::vector<int> bc_nodes = basis.mesh.get_left_boundary_nodes();
  std::vector<int> bc_dof;
  bc_dof.reserve(Physics::spatial_dim * bc_nodes.size());
  for (int node : bc_nodes) {
    for (int d = 0; d < Physics::spatial_dim; d++) {
      bc_dof.push_back(Physics::spatial_dim * node + d);
    }
  }

  // Apply bcs
  jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
  CSCMat *jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
  jac_csc->zero_columns(bc_dof.size(), bc_dof.data());
  jac_csc->write_mtx("K_" + name + ".mtx");
  return;  // TODO: finish from here

  // std::vector<T> rhs = sol;

  // // solve
  // SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  // SparseUtils::SparseCholesky<T> *chol =
  //     new SparseUtils::SparseCholesky<T>(jac_csc);
  // chol->factor();
  // chol->solve(sol.data());

  // // Check error
  // // res = Ku - rhs
  // std::vector<T> Ku(sol.size());
  // jac_bsr->axpy(sol.data(), Ku.data());
  // T err = 0.0;
  // for (int i = 0; i < Ku.size(); i++) {
  //   err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
  // }
  // std::printf("||Ku - f||: %25.15e\n", sqrt(err));

  // // Write to vtk
  // ToVTK<T, typename Basis::Mesh> vtk(basis.mesh, name + ".vtk");
  // vtk.write_mesh();
  // vtk.write_sol("x", x.data());
  // vtk.write_sol("u", sol.data());
  // vtk.write_sol("rhs", rhs.data());
}

void solve_linear_elasticity_gd() {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;
  int nxy[2] = {4, 4};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Basis::Mesh mesh(grid);
  Basis basis(mesh);

  T E = 30.0, nu = 0.3;
  solve_linear_elasticity<T, Quadrature, Basis>(E, nu, basis, "gd");
}

int main() { solve_linear_elasticity_gd(); }
