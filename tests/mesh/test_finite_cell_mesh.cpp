#include <gtest/gtest.h>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "elements/lbracket_mesh.h"
#include "physics/volume.h"
#include "test_commons.h"
#include "utils/json.h"
#include "utils/vtk.h"

void test_finite_cell_mesh_np_2() {
  int constexpr Np_1d = 2;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;

  int constexpr spatial_dim = Mesh::spatial_dim;

  json j = read_json("lsf_dof.json");
  int nxy[2] = {j["nxy"], j["nxy"]};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  mesh.get_lsf_dof() = std::vector<double>(j["lsf_dof"]);
  mesh.update_mesh();

  using FCMesh = FiniteCellMesh<T, Np_1d, Grid>;

  FCMesh fc_mesh(grid);
  fc_mesh.get_lsf_dof() = std::vector<double>(j["lsf_dof"]);
  fc_mesh.update_mesh();

  EXPECT_EQ(mesh.spatial_dim, fc_mesh.spatial_dim);
  EXPECT_EQ(mesh.get_num_nodes(), fc_mesh.get_num_nodes());
  EXPECT_EQ(mesh.get_num_elements(), fc_mesh.get_num_elements());

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    EXPECT_EQ(mesh.get_elem_cell(elem), fc_mesh.get_elem_cell(elem));
  }

  for (auto& [cell, elem] : mesh.get_cell_elems()) {
    EXPECT_EQ(fc_mesh.get_cell_elems().at(cell), elem);
  }

  for (auto& [cell, elem] : fc_mesh.get_cell_elems()) {
    EXPECT_EQ(mesh.get_cell_elems().at(cell), elem);
  }

  // for (int n = 0; n < mesh.get_num_nodes(); n++) {
  //   EXPECT_EQ(mesh.get_node_vert(n), fc_mesh.get_node_vert(n));
  // }

  // for (auto& [vert, node] : mesh.get_vert_nodes()) {
  //   EXPECT_EQ(fc_mesh.get_vert_nodes().at(vert), node);
  // }

  // for (auto& [vert, node] : fc_mesh.get_vert_nodes()) {
  //   EXPECT_EQ(mesh.get_vert_nodes().at(vert), node);
  // }

  // for (int n = 0; n < mesh.get_num_nodes(); n++) {
  //   T xloc1[spatial_dim], xloc2[spatial_dim];
  //   mesh.get_node_xloc(n, xloc1);
  //   fc_mesh.get_node_xloc(n, xloc2);
  //   for (int d = 0; d < spatial_dim; d++) {
  //     EXPECT_DOUBLE_EQ(xloc1[d], xloc2[d]);
  //   }
  // }

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
  }
}

TEST(finite_cell_mesh, Np2) { test_finite_cell_mesh_np_2(); }
