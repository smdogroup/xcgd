#include <algorithm>
#include <cassert>
#include <exception>
#include <set>
#include <string>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/cut_bcs.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/vtk.h"

#define PI 3.14159265358979323846

double hard_max(std::vector<double> vals) {
  return *std::max_element(vals.begin(), vals.end());
}

double ks_max(std::vector<double> vals, double ksrho = 50.0) {
  double umax = hard_max(vals);
  std::vector<double> eta(vals.size());
  std::transform(vals.begin(), vals.end(), eta.begin(),
                 [umax, ksrho](double x) { return exp(ksrho * (x - umax)); });
  return log(std::accumulate(eta.begin(), eta.end(), 0.0)) / ksrho + umax;
}

template <typename T, class Mesh>
void write_vtk(std::string vtkpath, const Mesh& mesh, const std::vector<T>& sol,
               const std::vector<T>& exact_sol, bool save_stencils) {
  ToVTK<T, Mesh> vtk(mesh, vtkpath);
  vtk.write_mesh();

  std::vector<double> nstencils(mesh.get_num_elements(),
                                Mesh::Np_1d * Mesh::Np_1d);
  auto degenerate_stencils = DegenerateStencilLogger::get_stencils();
  for (auto e : degenerate_stencils) {
    int elem = e.first;
    nstencils[elem] = e.second.size();
  }
  vtk.write_cell_sol("nstencils", nstencils.data());
  vtk.write_sol("u", sol.data());
  vtk.write_sol("u_exact", exact_sol.data());
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  if (save_stencils) {
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
  }
}

// Get the l2 error of the numerical Poisson solution
template <int Np_1d>
void solve_poisson_problem(std::string prefix, int nxy, bool save_stencils,
                           double nitsche_eta) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using QuadratureBulk = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER>;
  using QuadratureBCs = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto source_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    T r2 = xloc[0] * xloc[0] + xloc[1] * xloc[1];
    T r = sqrt(r2);
    return -(4.0 - r2) * sin(r) - 5.0 * r * cos(r);
  };

  auto bc_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    T r2 = xloc[0] * xloc[0] + xloc[1] * xloc[1];
    T r = sqrt(r2);
    return r2 * sin(r);
  };

  using PoissonBulk = PoissonPhysics<T, Basis::spatial_dim, typeof(source_fun)>;
  using PoissonBCs = CutDirichlet<T, Basis::spatial_dim, typeof(bc_fun)>;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PoissonBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PoissonBCs>;

  using BSRMat = GalerkinBSRMat<T, PoissonBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  DegenerateStencilLogger::enable();

  int nx_ny[2] = {nxy, nxy};
  double R = 2.0 * PI;
  double lxy[2] = {3.0 * R, 3.0 * R};
  double xy0[2] = {-1.5 * R, -1.5 * R};
  double angle = PI / 6.0;
  Grid grid(nx_ny, lxy, xy0);
  Mesh mesh(grid, [R, angle](double x[]) {
    double region1 = x[0] * x[0] + x[1] * x[1] - R * R;          // <= 0
    double region2 = -x[1];                                      // <= 0
    double region3 = x[1] - tan(angle) * x[0];                   // <= 0
    double region4 = -x[0] * x[0] - x[1] * x[1] + 0.25 * R * R;  // <= 0
    return region1;
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
  jac_bsr->write_mtx(std::filesystem::path(prefix) /
                     std::filesystem::path("poisson_jac.mtx"));
  analysis_bcs.jacobian(nullptr, zeros.data(), jac_bsr,
                        false);  // Add bcs contribution
  jac_bsr->write_mtx(std::filesystem::path(prefix) /
                     std::filesystem::path("poisson_jac_with_nitsche.mtx"));
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
    T r2 = xloc[0] * xloc[0] + xloc[1] * xloc[1];
    T r = sqrt(r2);
    sol_exact[i] = r2 * sin(r);
  }

  json j = {
      {"sol", sol}, {"sol_exact", sol_exact}, {"lsf", mesh.get_lsf_nodes()}};

  char json_name[256];
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);
  write_vtk<T>(
      std::filesystem::path(prefix) / std::filesystem::path("poisson.vtk"),
      mesh, sol, sol_exact, save_stencils);
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--save-degenerate-stencils", 0);
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--nxy", 64);
  p.add_argument<double>("--nitsche_eta", 1e6);
  p.add_argument<std::string>("--prefix", {});
  p.parse_args(argc, argv);

  bool save_stencils = p.get<int>("save-degenerate-stencils");

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int nxy = p.get<int>("nxy");
  double nitsche_eta = p.get<double>("nitsche_eta");

  switch (Np_1d) {
    case 2:
      solve_poisson_problem<2>(prefix, nxy, save_stencils, nitsche_eta);
      break;
    case 4:
      solve_poisson_problem<4>(prefix, nxy, save_stencils, nitsche_eta);
      break;
    case 6:
      solve_poisson_problem<6>(prefix, nxy, save_stencils, nitsche_eta);
      break;
    case 8:
      solve_poisson_problem<8>(prefix, nxy, save_stencils, nitsche_eta);
      break;
    case 10:
      solve_poisson_problem<10>(prefix, nxy, save_stencils, nitsche_eta);
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
