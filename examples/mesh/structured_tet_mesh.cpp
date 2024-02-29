#include "utils/mesh.h"
#include "utils/vtk.h"

int main() {
  int nxyz[3] = {5, 6, 7};
  double lxyz[3] = {1.0, 1.1, 1.2};
  int num_elements, num_nodes, *element_nodes;
  double *xloc;

  create_3d_box_tet_mesh(nxyz, lxyz, &num_elements, &num_nodes, &element_nodes,
                         &xloc);

  ToVTK<double> vtk(3, num_nodes, num_elements, 10, element_nodes, xloc);
  vtk.write_mesh();
  return 0;
}