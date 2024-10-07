#include <cassert>
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

template <typename T, int Np_1d>
class GridMeshDropOrder : public GridMesh<T, Np_1d> {
 private:
  using MeshBase = GDMeshBase<T, Np_1d>;

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::max_nnodes_per_element;
  using MeshBase::spatial_dim;
  using typename MeshBase::Grid;
  GridMeshDropOrder(const Grid& grid, int Np_bc)
      : GridMesh<T, Np_1d>(grid), Np_bc(Np_bc) {
    assert(Np_bc <= Np_1d);
    assert(Np_bc >= 2);
  }

  int get_elem_dof_nodes(
      int elem, int* nodes,
      std::vector<std::vector<bool>>* pstencil = nullptr) const {
    if (pstencil) {
      pstencil->clear();
      pstencil->resize(Np_1d);
      for (int I = 0; I < Np_1d; I++) {
        (*pstencil)[I] = std::vector<bool>(Np_1d, false);
        for (int J = 0; J < Np_1d; J++) {
          (*pstencil)[I][J] = false;
        }
      }
    }

    int tnodes[Np_1d * Np_1d];
    this->get_cell_ground_stencil(elem, tnodes);

    int eij[2] = {-1, -1};
    this->grid.get_cell_coords(elem, eij);
    const int* nxy = this->grid.get_nxy();

    int nnodes = 0;
    for (int iy = 0; iy < Np_1d; iy++) {
      for (int ix = 0; ix < Np_1d; ix++) {
        int idx = ix + Np_1d * iy;

        // lower-left corner element
        if (eij[0] == 0 and eij[1] == 0) {
          if (ix >= Np_bc or iy >= Np_bc) continue;
        }
        // upper-left corner element
        else if (eij[0] == 0 and eij[1] == nxy[1] - 1) {
          if (ix >= Np_bc or iy < Np_1d - Np_bc) continue;
        }
        // lower-right corner element
        else if (eij[0] == nxy[0] - 1 and eij[1] == 0) {
          if (ix < Np_1d - Np_bc or iy >= Np_bc) continue;
        }
        // upper-right corner element
        else if (eij[0] == nxy[0] - 1 and eij[1] == nxy[1] - 1) {
          if (ix < Np_1d - Np_bc or iy < Np_1d - Np_bc) continue;
        }
        // left boundary elements
        else if (eij[0] == 0) {
          if (ix >= Np_bc) continue;
        }
        // right bonudary elements
        else if (eij[0] == nxy[0] - 1) {
          if (ix < Np_1d - Np_bc) continue;
        }
        // lower boundary elements
        else if (eij[1] == 0) {
          if (iy >= Np_bc) continue;
        }
        // upper boundary elements
        else if (eij[1] == nxy[0] - 1) {
          if (iy < Np_1d - Np_bc) continue;
        }

        nodes[nnodes] = tnodes[idx];
        if (pstencil) {
          (*pstencil)[ix][iy] = true;
        }
        nnodes++;
      }
    }

    if (nnodes != max_nnodes_per_element) {
      DegenerateStencilLogger::add(elem, nnodes, nodes);
    }

    return nnodes;
  }

 private:
  int Np_bc;
};

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
void solve_poisson_problem(std::string prefix, int nxy, int Np_bc) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMeshDropOrder<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Poisson = PoissonApp<T, Mesh, Quadrature, Basis>;

  int nx_ny[2] = {nxy, nxy};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};
  Grid grid(nx_ny, lxy, xy0);
  Mesh mesh(grid, Np_bc);
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
  p.add_argument<int>("--save-degenerate-stencils", 1);
  p.add_argument<int>("--Np_1d", 4);
  p.add_argument<int>("--Np_bc", 4);
  p.add_argument<int>("--nxy", 6);
  p.add_argument<std::string>("--prefix", {});
  p.parse_args(argc, argv);

  if (p.get<int>("save-degenerate-stencils")) {
    DegenerateStencilLogger::enable();
  }

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
      solve_poisson_problem<2>(prefix, nxy, Np_bc);
      break;
    case 4:
      solve_poisson_problem<4>(prefix, nxy, Np_bc);
      break;
    case 6:
      solve_poisson_problem<6>(prefix, nxy, Np_bc);
      break;
    case 8:
      solve_poisson_problem<8>(prefix, nxy, Np_bc);
      break;
    case 10:
      solve_poisson_problem<10>(prefix, nxy, Np_bc);
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
