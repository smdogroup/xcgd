#include <vector>

#include "elements/gd_mesh.h"
#include "test_commons.h"
#include "utils/vtk.h"

template <int Np_1d>
void create_wedge_mesh() {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  auto lsf = [](const T* x) { return 0.9 * x[0] + x[1] - 0.9; };
  Mesh mesh(grid, lsf);

  char vtkname[256];
  std::snprintf(vtkname, 256, "wedge_mesh_Np_1d_%d.vtk", Np_1d);
  ToVTK<T, Mesh> vtk(mesh, vtkname);
  vtk.write_mesh();

  std::vector<T> elem_indices(mesh.get_num_elements());
  for (int i = 0; i < mesh.get_num_elements(); i++) {
    elem_indices[i] = T(i);
  }
  vtk.write_cell_sol("elem_indices", elem_indices.data());

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

TEST(stencil, wedge_Np2) { create_wedge_mesh<2>(); }
TEST(stencil, wedge_Np4) { create_wedge_mesh<4>(); }
