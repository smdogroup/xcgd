#include <vector>

#include "elements/gd_vandermonde.h"
#include "quadrature_general.hpp"
#include "test_commons.h"
#include "utils/vtk.h"

TEST(mesh, GDMeshStructured) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d>;

  int nxy[2] = {4, 3};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  std::vector<int> nodes1 = {0,  1,  2,  3,  5,  6,  7,  8,
                             10, 11, 12, 13, 15, 16, 17, 18};
  std::vector<int> nodes2 = {1,  2,  3,  4,  6,  7,  8,  9,
                             11, 12, 13, 14, 16, 17, 18, 19};

  for (int elem = 0; elem < 12; elem++) {
    int nodes[mesh.max_nnodes_per_element];
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

template <typename T>
class Circle {
 public:
  Circle(T* center, T radius, bool flip = false) {
    x0[0] = center[0];
    x0[1] = center[1];
    r = radius;
    if (flip) {
      sign = -1.0;
    }
  }

  T operator()(const T* x) const {
    return sign * ((x[0] - x0[0]) * (x[0] - x0[0]) +
                   (x[1] - x0[1]) * (x[1] - x0[1]) - r * r);
  }

 private:
  T x0[2];
  T r;
  double sign = 1.0;
};

void generate_lsf_mesh(bool flip = false) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;

  int nxy[2] = {21, 21};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};

  T center[2] = {0.0, 0.0};
  T r = 0.5;

  Circle lsf(center, r, flip);

  Grid grid(nxy, lxy, xy0);
  Mesh mesh(grid, lsf);

  char vtkname[256];
  std::snprintf(vtkname, 256, "mesh_gd%s.vtk", flip ? "_flip" : "");
  ToVTK<T, Mesh> vtk(mesh, vtkname);
  vtk.write_mesh();

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> dof(mesh.get_num_nodes(), 0.0);
    int nodes[Mesh::max_nnodes_per_element];
    mesh.get_elem_dof_nodes(elem, nodes);
    for (int i = 0; i < Mesh::max_nnodes_per_element; i++) {
      dof[nodes[i]] = 1.0;
    }
    char name[256];
    std::snprintf(name, 256, "elem_%05d", elem);
    vtk.write_sol(name, dof.data());
  }
}

TEST(mesh, LSFPositive) { generate_lsf_mesh(true); }
TEST(mesh, LSFNegative) { generate_lsf_mesh(false); }
