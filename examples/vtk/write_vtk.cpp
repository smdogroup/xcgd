#include <vector>

#include "elements/fe_tetrahedral.h"
#include "elements/gd_vandermonde.h"
#include "utils/mesher.h"
#include "utils/vtk.h"

void write_tet_to_vtk() {
  int nxyz[3] = {5, 6, 7};
  double lxyz[3] = {1.0, 1.1, 1.2};
  int num_elements, num_nodes, *element_nodes;
  double *xloc;

  create_3d_box_tet_mesh(nxyz, lxyz, &num_elements, &num_nodes, &element_nodes,
                         &xloc);

  using Basis = TetrahedralBasis<double>;
  using Mesh = Basis::Mesh;

  Mesh mesh(num_elements, num_nodes, element_nodes, xloc);

  std::vector<double> dof(mesh.get_num_nodes());
  for (int i = 0; i < dof.size(); i++) {
    dof[i] = (double)rand() / RAND_MAX;
  }

  ToVTK<double, Mesh> vtk(mesh, "mesh_tet.vtk");
  vtk.write_mesh();
  vtk.write_sol("sol", dof.data());
}

void write_gd_mesh_to_vtk() {
  using T = double;
  constexpr int Np_1d = 2;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Grid = StructuredGrid2D<T>;

  int nxy[2] = {10, 10};
  T lxy[2] = {1.0, 1.5};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  ToVTK<double, Mesh> vtk(mesh, "mesh_gd.vtk");
  vtk.write_mesh();
}

int main() {
  write_tet_to_vtk();
  write_gd_mesh_to_vtk();
  return 0;
}