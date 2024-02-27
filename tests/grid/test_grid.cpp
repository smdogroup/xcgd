#include <vector>

#include "elements/gd_vandermonde.h"
#include "test_commons.h"

TEST(GridTest, GalerkinDiff2D) {
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
    grid.get_elem_coords(elem, eij);
    if (eij[0] < 2) {
      EXPECT_VEC_EQ(nodes1.size(), nodes, nodes1);
    } else {
      EXPECT_VEC_EQ(nodes2.size(), nodes, nodes2);
    }
  }
}