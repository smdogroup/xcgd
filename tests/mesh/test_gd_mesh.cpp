#include <vector>

#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/vtk.h"

TEST(mesh, GDMeshStructured) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;

  int nxy[2] = {4, 3};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  std::vector<int> nodes1 = {0,  1,  2,  3,  5,  6,  7,  8,
                             10, 11, 12, 13, 15, 16, 17, 18};
  std::vector<int> nodes2 = {1,  2,  3,  4,  6,  7,  8,  9,
                             11, 12, 13, 14, 16, 17, 18, 19};

  for (int elem = 0; elem < 12; elem++) {
    int nodes[mesh.nodes_per_element];
    mesh.get_elem_dof_nodes(elem, nodes);
    int eij[2];
    grid.get_cell_coords(elem, eij);
    if (eij[0] < 2) {
      EXPECT_VEC_EQ(nodes1.size(), nodes, nodes1);
    } else {
      EXPECT_VEC_EQ(nodes2.size(), nodes, nodes2);
    }
  }
}

// TEST(GDMeshTest, LSF) {
//   constexpr int Np_1d = 4;
//   using T = double;
//   using Grid = StructuredGrid2D<T>;
//   using Mesh = GDMesh2D<T, Np_1d>;

//   int nxy[2] = {20, 20};
//   T lxy[2] = {1.0, 1.0};

//   auto lsf = [](T xy[2]) {
//     T x = xy[0];
//     T y = xy[1];
//     return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - 0.25;
//   };

//   Grid grid(nxy, lxy);
//   Mesh mesh(grid, lsf);

//   std::vector<T> active_lsf_verts(grid.get_num_verts(), 0.0);
//   std::vector<T> active_dof_verts(grid.get_num_verts(), 0.0);

//   for (int i = 0; i < grid.get_num_verts(); i++) {
//     if (mesh.active_lsf_verts[i]) {
//       active_lsf_verts[i] = 1.0;
//     }
//   }

//   for (int vert : mesh.active_dof_nodes) {
//     active_dof_verts[vert] = 1.0;
//   }

//   ToVTK<T, Mesh> vtk(mesh, "mesh_gd.vtk");

//   vtk.write_mesh();
//   vtk.write_sol("active_lsf_nodes", active_lsf_verts.data());
//   vtk.write_sol("active_dof_nodes", active_dof_verts.data());
// }
