#include "physics/linear_elasticity.h"

#include <string>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/mesher.h"
#include "utils/vtk.h"

template <typename T, class Mesh, class Quadrature, class Basis>
void solve_linear_elasticity(T E, T nu, Mesh &mesh, Quadrature &quadrature,
                             Basis &basis, std::string name) {
  using Physics = LinearElasticity<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  Physics physics(E, nu);
  Analysis analysis(mesh, quadrature, basis, physics);

  int ndof = Physics::dof_per_node * mesh.get_num_nodes();

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

  // Set boundary conditions
  std::vector<int> bc_nodes = mesh.get_left_boundary_nodes();
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

  // Set rhs
  std::vector<int> force_nodes = mesh.get_right_boundary_nodes();
  std::vector<T> rhs(ndof, 0.0);

  for (int node : force_nodes) {
    rhs[Physics::spatial_dim * node + 1] = -1.0;  // unit nodal force on -y
  }

  for (int i : bc_dof) {
    rhs[i] = 0.0;
  }

  std::vector<T> sol = rhs;

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
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
  vtk.write_vec("rhs", rhs.data());
  vtk.write_vec("sol", sol.data());

  // Export quadrature points to vtk
  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interpolator(mesh, quadrature, basis);
  interpolator.to_vtk(name + "_pts.vtk", sol.data());
}

template <int spatial_dim>
class Circle {
 public:
  Circle(double *center, double radius, bool flip = false) {
    for (int d = 0; d < spatial_dim; d++) {
      x0[d] = center[d];
    }
    r = radius;
    if (flip) {
      sign = -1.0;
    }
  }

  template <typename T>
  T operator()(const T *x) const {
    return sign * ((x[0] - x0[0]) * (x[0] - x0[0]) +
                   (x[1] - x0[1]) * (x[1] - x0[1]) - r * r);
  }

 private:
  double x0[spatial_dim];
  double r;
  double sign = 1.0;
};

void solve_linear_elasticity_gd() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using LSF = Circle<Grid::spatial_dim>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  int nxy[2] = {96, 64};
  T lxy[2] = {1.5, 1.0};

  double center[2] = {0.75, 0.5};
  double r = 0.3;

  LSF lsf(center, r, true);

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  T E = 30.0, nu = 0.3;
  solve_linear_elasticity(E, nu, mesh, quadrature, basis, "gd");
}

int main() { solve_linear_elasticity_gd(); }
