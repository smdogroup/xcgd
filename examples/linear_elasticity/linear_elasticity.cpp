#include "physics/linear_elasticity.h"

#include <array>
#include <string>
#include <vector>

#include "a2dcore.h"
#include "analysis.h"
#include "apps/static_elastic.h"
#include "elements/element_commons.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/mesher.h"
#include "utils/vtk.h"

template <typename T, class Mesh, class Quadrature, class Basis>
void solve_linear_elasticity(T E, T nu, Mesh &mesh, Quadrature &quadrature,
                             Basis &basis, std::string name) {
  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{};
  };
  using Physics = LinearElasticity<T, Basis::spatial_dim, typeof(int_func)>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  Physics physics(E, nu, int_func);
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

void solve_linear_elasticity_cut_gd() {
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
  solve_linear_elasticity(E, nu, mesh, quadrature, basis, "cut_gd");
}

void solve_linear_elasticity_gd() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  int nxy[2] = {96, 16};
  T lxy[2] = {6.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  T E = 30.0, nu = 0.3;
  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{
        std::array<T, Basis::spatial_dim>{0.0, -1.0}.data()};
  };
  StaticElastic elastic(E, nu, mesh, quadrature, basis, int_func);

  // Set boundary conditions
  std::vector<int> bc_nodes = mesh.get_left_boundary_nodes();
  std::vector<int> bc_dof;
  bc_dof.reserve(Basis::spatial_dim * bc_nodes.size());
  for (int node : bc_nodes) {
    for (int d = 0; d < Basis::spatial_dim; d++) {
      bc_dof.push_back(Basis::spatial_dim * node + d);
    }
  }

  std::vector<T> sol =
      elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), T(0.0)));

  ToVTK<T, Mesh> vtk(mesh, "gravity.vtk");
  vtk.write_mesh();
  vtk.write_vec("u", sol.data());
}

void solve_linear_elasticity_nitsche() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using QuadratureBulk = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER>;
  using QuadratureBCs = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{
        std::array<T, Basis::spatial_dim>{0.0, -1.0}.data()};
  };

  using PhysicsBulk = LinearElasticity<T, Basis::spatial_dim, typeof(int_func)>;
  auto bc_fun = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return A2D::Vec<T, PhysicsBulk::dof_per_node>{};
  };

  using PhysicsBCs =
      LinearElasticityCutDirichlet<T, Basis::spatial_dim,
                                   PhysicsBulk::dof_per_node, typeof(bc_fun)>;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PhysicsBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PhysicsBCs>;

  using BSRMat = GalerkinBSRMat<T, PhysicsBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int nxy[2] = {32, 32};
  T lxy[2] = {1.0, 1.0};
  T R = 0.49;
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [R](T *x) {
    return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) -
           R * R;  // <= 0
  });

  Basis basis(mesh);
  QuadratureBulk quadrature_bulk(mesh);
  QuadratureBCs quadrature_bcs(mesh);

  PhysicsBulk physics_bulk(30.0, 0.3, int_func);
  PhysicsBCs physics_bcs(1e6, bc_fun);

  AnalysisBulk analysis_bulk(mesh, quadrature_bulk, basis, physics_bulk);
  AnalysisBCs analysis_bcs(mesh, quadrature_bcs, basis, physics_bcs);

  int ndof = PhysicsBulk::dof_per_node * mesh.get_num_nodes();

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
  std::vector<T> zeros(ndof, 0.0);
  analysis_bulk.jacobian(nullptr, zeros.data(), jac_bsr);
  analysis_bcs.jacobian(nullptr, zeros.data(), jac_bsr, false);
  CSCMat *jac_csc = SparseUtils::bsr_to_csc(jac_bsr);

  // Set up the right hand side
  std::vector<T> rhs(ndof, 0.0);
  analysis_bulk.residual(nullptr, zeros.data(), rhs.data());
  analysis_bcs.residual(nullptr, zeros.data(), rhs.data());
  for (int i = 0; i < ndof; i++) {
    rhs[i] *= -1.0;
  }

  // Solve
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T> *chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  chol->factor();
  std::vector<T> sol = rhs;
  chol->solve(sol.data());

  ToVTK<T, Mesh> vtk(mesh, "nitsche_gravity.vtk");
  vtk.write_mesh();
  vtk.write_vec("u", sol.data());
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
}

int main() {
  // solve_linear_elasticity_cut_gd();
  // solve_linear_elasticity_gd();
  solve_linear_elasticity_nitsche();
}
