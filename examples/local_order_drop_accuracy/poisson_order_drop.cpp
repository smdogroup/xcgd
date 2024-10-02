#define POISSON_ORDER_DROP_EXPERIMENT

#include <exception>
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
void write_vtk(std::string vtkpath, const Mesh& mesh,
               const std::vector<T>& sol) {
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

// Get the l2 error of the numerical Poisson solution
template <int Np_1d>
void solve_poisson_problem(int nxy, std::string prefix) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Poisson = PoissonApp<T, Mesh, Quadrature, Basis>;

  DegenerateStencilLogger::enable();

  int nx_ny[2] = {nxy, nxy};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};
  Grid grid(nx_ny, lxy, xy0);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  Poisson poisson(mesh, quadrature, basis, 0.0);

  int nnodes = mesh.get_num_nodes();

  std::vector<int> dof_bcs;
  std::vector<T> dof_vals;
  std::vector<T> sol_exact(nnodes, 0.0);
  double tol = 1e-6, xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;
  for (int i = 0; i < nnodes; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    if (freal(xloc[0]) < xmin + tol or freal(xloc[1]) < ymin + tol or
        freal(xloc[0]) > freal(xmax) - tol or
        freal(xloc[1]) > freal(ymax) - tol) {
      dof_bcs.push_back(i);
      dof_vals.push_back(2.0 * (1.0 + xloc[1]) /
                         ((3.0 + xloc[0]) * (3.0 + xloc[0]) +
                          (1.0 + xloc[1]) * (1.0 + xloc[1])));
    }
    sol_exact[i] =
        2.0 * (1.0 + xloc[1]) /
        ((3.0 + xloc[0]) * (3.0 + xloc[0]) + (1.0 + xloc[1]) * (1.0 + xloc[1]));
  }

  std::vector<T> sol = poisson.solve(dof_bcs, dof_vals);
  json j = {{"sol", sol}, {"sol_exact", sol_exact}};

  char json_name[256];
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);
  write_vtk<T>(
      std::filesystem::path(prefix) / std::filesystem::path("poisson.vtk"),
      mesh, sol);
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 4);
  p.add_argument<int>("--nxy", 6);
  p.add_argument<std::string>("--prefix", {});
  p.parse_args(argc, argv);

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int nxy = p.get<int>("nxy");

  switch (Np_1d) {
    case 2:
      solve_poisson_problem<2>(nxy, prefix);
      break;
    case 4:
      solve_poisson_problem<4>(nxy, prefix);
      break;
    case 6:
      solve_poisson_problem<6>(nxy, prefix);
      break;
    case 8:
      solve_poisson_problem<8>(nxy, prefix);
      break;
    case 10:
      solve_poisson_problem<10>(nxy, prefix);
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
