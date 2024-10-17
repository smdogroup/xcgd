#include <cassert>
#include <exception>
#include <set>
#include <string>

#include "apps/poisson_app.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/vtk.h"

template <typename T, class Mesh>
void write_vtk(std::string vtkpath, const Mesh& mesh, const std::vector<T>& sol,
               bool save_stencils) {
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
void solve_poisson_problem(std::string prefix, int nxy, int Np_bc,
                           bool save_stencils) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto source_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    T r2 = xloc[0] * xloc[0] + xloc[1] * xloc[1];
    return r2 * r2;
  };
  using Poisson = PoissonApp<T, Mesh, Quadrature, Basis, typeof(source_fun)>;

  DegenerateStencilLogger::enable();

  int nx_ny[2] = {nxy, nxy};
  T lxy[2] = {3.0, 3.0};
  T xy0[2] = {-1.5, -1.5};
  T R = 1.0;
  Grid grid(nx_ny, lxy, xy0);
  Mesh mesh(grid,
            [R](double x[]) { return x[0] * x[0] + x[1] * x[1] - R * R; });
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  Poisson poisson(mesh, quadrature, basis, source_fun);

  int nnodes = mesh.get_num_nodes();

  std::vector<T> sol_exact(nnodes, 0.0);
  for (int i = 0; i < nnodes; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);

    T r = sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);
    sol_exact[i] = (r * r * r * r * r * r - R * R * R * R * R * R) / 42.0;
  }

  std::set<int> dof_bcs{};
  for (int i = 0; i < mesh.get_num_elements(); i++) {
    // Find all boundary elements
    // TODO: finish from here
  }

  std::vector<T> dof_vals{};

  std::vector<T> sol = poisson.solve(dof_bcs, dof_vals);

  json j = {{"sol", sol}, {"sol_exact", sol_exact}};

  char json_name[256];
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);
  write_vtk<T>(
      std::filesystem::path(prefix) / std::filesystem::path("poisson.vtk"),
      mesh, sol, save_stencils);
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--save-degenerate-stencils", 0);
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--Np_bc", 2);
  p.add_argument<int>("--nxy", 64);
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
  int Np_bc = p.get<int>("Np_bc");
  int nxy = p.get<int>("nxy");

  switch (Np_1d) {
    case 2:
      solve_poisson_problem<2>(prefix, nxy, Np_bc, save_stencils);
      break;
    case 4:
      solve_poisson_problem<4>(prefix, nxy, Np_bc, save_stencils);
      break;
    case 6:
      solve_poisson_problem<6>(prefix, nxy, Np_bc, save_stencils);
      break;
    case 8:
      solve_poisson_problem<8>(prefix, nxy, Np_bc, save_stencils);
      break;
    case 10:
      solve_poisson_problem<10>(prefix, nxy, Np_bc, save_stencils);
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
