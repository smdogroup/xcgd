#include <string>
#include <vector>

#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/vtk.h"

template <typename T, typename Mesh>
void write_vtk(const Mesh& mesh, const char* vtkname,
               const std::vector<T>& sol) {
  assert(sol.size() == mesh.get_num_nodes() * Mesh::spatial_dim);

  ToVTK<T, Mesh> vtk(mesh, vtkname);
  vtk.write_mesh();

  std::vector<T> elem_indices(mesh.get_num_elements());
  for (int i = 0; i < mesh.get_num_elements(); i++) {
    elem_indices[i] = T(i);
  }
  vtk.write_cell_sol("elem_indices", elem_indices.data());
  vtk.write_vec("displacement", sol.data());
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> dof(mesh.get_num_nodes(), 0.0);
    int nodes[Mesh::max_nnodes_per_element];
    int nnodes = mesh.get_elem_dof_nodes(elem, nodes);
    for (int i = 0; i < nnodes; i++) {
      dof[nodes[i]] = 1.0;
    }
    char name[256];
    std::snprintf(name, 256, "stencil_elem_%05d", elem);
    vtk.write_sol(name, dof.data());
  }
}

template <int Np_1d>
void solve_wedge_problem() {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{};
  };
  using StaticElastic =
      StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_func)>;

  constexpr int nx = 5, ny = 5;
  int nxy[2] = {nx, ny};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  auto lsf = [](const T* x) { return 0.9 * x[0] + x[1] - 0.9; };
  Mesh mesh(grid, lsf);
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  StaticElastic elastic(70, 0.3, mesh, quadrature, basis, int_func);

  std::vector<int> bc_dof;
  std::vector<int> load_dof;
  auto vert_nodes = mesh.get_vert_nodes();
  for (int j = 0; j < ny; j++) {
    for (int d = 0; d < Mesh::spatial_dim; d++) {
      bc_dof.push_back(
          Mesh::spatial_dim * vert_nodes.at(grid.get_coords_vert(0, j)) + d);
    }
  }
  load_dof.push_back(2 * vert_nodes.at(grid.get_coords_vert(nx - 1, 0)));
  load_dof.push_back(2 * vert_nodes.at(grid.get_coords_vert(nx - 1, 0)) + 1);
  std::vector<T> load_vals = {0.0, -1.0};

  std::vector<T> sol = elastic.solve(bc_dof, load_dof, load_vals);

  char vtkname[256];
  std::snprintf(vtkname, 256, "wedge_mesh_Np_1d_%d.vtk", Np_1d);
  write_vtk<T>(mesh, vtkname, sol);
}

TEST(stencil, wedge_Np2) { solve_wedge_problem<2>(); }
TEST(stencil, wedge_Np4) { solve_wedge_problem<4>(); }
