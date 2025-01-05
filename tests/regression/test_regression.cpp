#include <map>
#include <stdexcept>
#include <type_traits>

#include "apps/poisson_app.h"
#include "apps/static_elastic.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/json.h"
#include "utils/loggers.h"

#define PI 3.14159265358979323846

template <typename T, int spatial_dim>
class L2normBulk final : public PhysicsBase<T, spatial_dim, 0, 1> {
 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ____) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * val * val;
  }
};

template <typename T, typename Mesh>
void write_vtk(const Mesh& mesh, const char* vtkname, const std::vector<T>& sol,
               const std::vector<int>& bc_dof, const std::vector<int>& load_dof,
               const std::vector<T>& load_vals) {
  assert(sol.size() == mesh.get_num_nodes() * Mesh::spatial_dim);

  ToVTK<T, Mesh> vtk(mesh, vtkname);
  vtk.write_mesh();

  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  std::vector<T> bcs(sol.size(), 0.0);
  std::vector<T> loads(sol.size(), 0.0);

  for (int b : bc_dof) bcs[b] = 1.0;
  for (int i = 0; i < load_dof.size(); i++) {
    loads[load_dof[i]] = load_vals[i];
  }
  vtk.write_vec("displacement", sol.data());
  vtk.write_vec("bcs", bcs.data());
  vtk.write_vec("loads", loads.data());

  // Write degenerate stencils
  auto degenerate_stencils = DegenerateStencilLogger::get_stencils();
  for (auto e : degenerate_stencils) {
    int elem = e.first;
    std::vector<int> nodes = e.second;
    std::vector<T> dof(mesh.get_num_nodes(), 0.0);
    for (int n : nodes) {
      dof[n] = 1.0;
    }
    char name[256];
    std::snprintf(name, 256, "degenerate_stencil_elem_%05d", elem);
    vtk.write_sol(name, dof.data());
  }

  std::vector<T> elem_indices(mesh.get_num_elements());
  for (int i = 0; i < mesh.get_num_elements(); i++) {
    elem_indices[i] = T(i);
  }
  vtk.write_cell_sol("elem_indices", elem_indices.data());

  // Write condition numbers of Vandermonde matrices
  std::vector<double> conds(mesh.get_num_elements());
  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    conds[elem] = VandermondeCondLogger::get_conds().at(elem);
  }
  vtk.write_cell_sol("cond", conds.data());
}

template <typename T, int Np_1d>
void test_regression_static(json j) {
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{};
  };
  using StaticElastic =
      StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_func)>;

  DegenerateStencilLogger::enable();
  VandermondeCondLogger::enable();

  int nxy[2] = {j["nx"], j["ny"]};
  T lxy[2] = {j["lx"], j["ly"]};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  StaticElastic elastic(j["E"], j["nu"], mesh, quadrature, basis, int_func);

  std::vector<T> lsf_dof = j["lsf_dof"];
  EXPECT_EQ(lsf_dof.size(), mesh.get_lsf_dof().size());
  mesh.get_lsf_dof() = lsf_dof;
  mesh.update_mesh();

  std::vector<int> bc_dof(j["bc_dof"]);
  std::vector<int> load_dof(j["load_dof"]);
  std::vector<T> load_vals(j["load_vals"]);
  EXPECT_EQ(load_dof.size(), load_vals.size());

  std::vector<T> sol =
      elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), T(0.0)));

  char vtkname[256];
  std::snprintf(vtkname, 256, "regression_Np_1d_%d.vtk", Np_1d);
  write_vtk<T>(mesh, vtkname, sol, bc_dof, load_dof, load_vals);

  EXPECT_EQ(sol.size(), 2 * mesh.get_num_nodes());
  EXPECT_VEC_NEAR(sol.size(), sol, j["u"], 1e-10);
}

// This test is no longer valid, we skip it via DISABLED_ prefix
TEST(regression, DISABLED_static) {
  using T = double;
  std::string json_path = "./data_static.json";

  json j = read_json(json_path);
  switch (int(j["Np_1d"])) {
    case 2:
      test_regression_static<T, 2>(j);
      break;
    case 4:
      test_regression_static<T, 4>(j);
      break;
    default:
      char msg[256];
      std::snprintf(msg, 256,
                    "Np_1d = %d loaded from json is not precompiled by the "
                    "test code, you may want to manually add it in the source "
                    "code and compile the test again",
                    int(j["Np_1d"]));
      throw std::runtime_error(msg);
      break;
  }
}

// Define the exact solution and source term using method of manufactured
// solutions
template <typename T, typename Vec>
T exact_solution(Vec xloc) {
  return sin(xloc[0] * 1.9 * PI) * sin(xloc[1] * 1.9 * PI);
}
template <typename T, typename Vec>
T exact_source(Vec xloc) {
  return -2.0 * 1.9 * 1.9 * PI * PI * sin(xloc[0] * 1.9 * PI) *
         sin(xloc[1] * 1.9 * PI);
}

template <int Np_1d>
void verify_regular_poisson(double tol = 1e-20) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto source_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return exact_source<T>(xloc);
  };

  using Poisson = PoissonApp<T, Mesh, Quadrature, Basis, typeof(source_fun)>;

  int nx_ny[2] = {16, 16};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nx_ny, lxy);
  Mesh mesh(grid);

  Quadrature quadrature(mesh);
  Basis basis(mesh);

  Poisson poisson(mesh, quadrature, basis, source_fun);

  int nnodes = mesh.get_num_nodes();

  // Set boundary conditions for the regular problem
  std::vector<int> dof_bcs;
  std::vector<T> dof_vals;
  double epsilon = 1e-6, xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;
  for (int i = 0; i < nnodes; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    if (freal(xloc[0]) < xmin + epsilon or freal(xloc[1]) < ymin + epsilon or
        freal(xloc[0]) > freal(xmax) - epsilon or
        freal(xloc[1]) > freal(ymax) - epsilon) {
      dof_bcs.push_back(i);
      dof_vals.push_back(exact_solution<T>(xloc));
    }
  }

  // Solve
  std::vector<T> sol = poisson.solve(dof_bcs, dof_vals);

  // Get exact solution
  std::vector<T> sol_exact(nnodes, 0.0);
  for (int i = 0; i < nnodes; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    sol_exact[i] = exact_solution<T>(xloc);
  }

  // Compute the L2 norm of the solution field (not vector)
  std::vector<T> diff(sol.size());
  for (int i = 0; i < sol.size(); i++) {
    diff[i] = (sol[i] - sol_exact[i]);
  }

  GalerkinAnalysis<T, Mesh, Quadrature, Basis,
                   L2normBulk<T, Basis::spatial_dim>>
      integrator(mesh, quadrature, basis, {});

  std::vector<T> ones(sol.size(), 1.0);

  T err_l2norm = integrator.energy(nullptr, diff.data());
  T l2norm = integrator.energy(nullptr, sol_exact.data());

  EXPECT_NEAR(err_l2norm / l2norm, 0.0, tol);

  ToVTK<T, Mesh> tovtk(mesh,
                       "regular_poisson_Np_" + std::to_string(Np_1d) + ".vtk");
  tovtk.write_mesh();
  tovtk.write_sol("u", sol.data());
  tovtk.write_sol("u_exact", sol_exact.data());
}

template <int Np_1d>
void verify_nitsche_poisson(double tol = 1e-20, double nitsche_eta = 1e8) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using QuadratureBulk = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER>;
  using QuadratureBCs = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto source_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return exact_source<T>(xloc);
  };
  auto bc_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return exact_solution<T>(xloc);
  };

  using PoissonBulk = PoissonPhysics<T, Basis::spatial_dim, typeof(source_fun)>;
  using PoissonBCs = PoissonCutDirichlet<T, Basis::spatial_dim, typeof(bc_fun)>;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PoissonBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PoissonBCs>;

  using BSRMat = GalerkinBSRMat<T, PoissonBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int nx_ny[2] = {16, 16};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nx_ny, lxy);
  Mesh mesh(grid, [](T x[]) {
    return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) -
           0.49 * 0.49;
  });

  QuadratureBulk quadrature_bulk(mesh);
  QuadratureBCs quadrature_bcs(mesh);
  Basis basis(mesh);

  PoissonBulk poisson_bulk(source_fun);
  PoissonBCs poisson_bcs(nitsche_eta, bc_fun);

  AnalysisBulk analysis_bulk(mesh, quadrature_bulk, basis, poisson_bulk);
  AnalysisBCs analysis_bcs(mesh, quadrature_bcs, basis, poisson_bcs);

  int ndof = mesh.get_num_nodes();

  // Set up the Jacobian matrix for Poisson's problem with Nitsche's boundary
  // conditions
  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivityFunctor(
      mesh.get_num_nodes(), mesh.get_num_elements(),
      mesh.max_nnodes_per_element,
      [&mesh](int elem, int* nodes) -> int {
        return mesh.get_elem_dof_nodes(elem, nodes);
      },
      &rowp, &cols);
  int nnz = rowp[mesh.get_num_nodes()];
  BSRMat* jac_bsr = new BSRMat(ndof, nnz, rowp, cols);
  std::vector<T> zeros(ndof, 0.0);
  analysis_bulk.jacobian(nullptr, zeros.data(), jac_bsr);
  analysis_bcs.jacobian(nullptr, zeros.data(), jac_bsr,
                        false);  // Add bcs contribution
  CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);

  // Set up the right hand side
  std::vector<T> rhs(ndof, 0.0);

  analysis_bulk.residual(nullptr, zeros.data(), rhs.data());
  analysis_bcs.residual(nullptr, zeros.data(), rhs.data());
  for (int i = 0; i < ndof; i++) {
    rhs[i] *= -1.0;
  }

  // Solve
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T>* chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  chol->factor();
  std::vector<T> sol = rhs;
  chol->solve(sol.data());

  // Get exact solution
  std::vector<T> sol_exact(ndof, 0.0);
  for (int i = 0; i < ndof; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    sol_exact[i] = exact_solution<T>(xloc);
  }

  // Compute the L2 norm of the solution field (not vector)
  std::vector<T> diff(sol.size());
  for (int i = 0; i < sol.size(); i++) {
    diff[i] = (sol[i] - sol_exact[i]);
  }

  GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis,
                   L2normBulk<T, Basis::spatial_dim>>
      integrator(mesh, quadrature_bulk, basis, {});

  std::vector<T> ones(sol.size(), 1.0);

  T err_l2norm = integrator.energy(nullptr, diff.data());
  T l2norm = integrator.energy(nullptr, sol_exact.data());

  EXPECT_NEAR(err_l2norm / l2norm, 0.0, tol);

  ToVTK<T, Mesh> tovtk(mesh,
                       "nitsche_poisson_Np_" + std::to_string(Np_1d) + ".vtk");
  tovtk.write_mesh();
  tovtk.write_sol("u", sol.data());
  tovtk.write_sol("u_exact", sol_exact.data());
}

TEST(regression, PoissonRegular) {
  verify_regular_poisson<2>(1e-3);
  verify_regular_poisson<4>(1e-6);
  verify_regular_poisson<6>(1e-9);
}

TEST(regression, PoissonNitsche) {
  verify_nitsche_poisson<2>(1e-2);
  verify_nitsche_poisson<4>(1e-4);
  verify_nitsche_poisson<6>(1e-8);
}
