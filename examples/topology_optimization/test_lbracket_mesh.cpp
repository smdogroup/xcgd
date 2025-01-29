#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "elements/lbracket_mesh.h"
#include "physics/volume.h"
#include "test_commons.h"
#include "utils/vtk.h"

TEST(lbracket_mesh, lbracket_grid) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = LbracketGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;

  int nx1 = 10, nx2 = 4, ny1 = 4, ny2 = 6;
  T lx1 = 1.0, ly1 = 0.4;
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);

  Mesh mesh(grid);

  ToVTK<T, Mesh> vtk(mesh, "lbracket_grid.vtk");
  vtk.write_mesh();

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> dof(mesh.get_num_nodes(), 0.0);
    int nodes[Mesh::max_nnodes_per_element];
    int nnodes = mesh.get_elem_dof_nodes(elem, nodes);
    for (int i = 0; i < nnodes; i++) {
      dof[nodes[i]] = 1.0;
    }
    char name[256];
    std::snprintf(name, 256, "elem_%05d", elem);
    vtk.write_sol(name, dof.data());
  }
}

TEST(lbracket_mesh, lbracket_via_lsf) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Physics = VolumePhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {32, 32};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid, [](double x[]) {
    double region1 = x[0] - 0.4;  // <= 0
    double region2 = x[1] - 0.4;  // <= 0
    return hard_min<double>({region1, region2});
  });

  Quadrature quadrature(mesh);
  Basis basis(mesh);
  Physics physics;

  Analysis analysis(mesh, quadrature, basis, physics);

  ToVTK<T, Mesh> vtk(mesh, "lbracket_lsf.vtk");
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  auto [xloc_q, dof_q] = analysis.interpolate(mesh.get_lsf_nodes().data());

  FieldToVTKNew<T, Basis::spatial_dim> field_vtk("lbracket_lsf_quad.vtk");
  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();
}

TEST(lbracket_mesh, LbracketGrid2D) {
  // Set lbracket grid
  using T = double;
  using Grid = LbracketGrid2D<T>;
  int nx1 = 5, nx2 = 3, ny1 = 3, ny2 = 2;
  T lx1 = 1.0, ly1 = 0.48;
  T hx = lx1 / nx1;
  T hy = ly1 / ny1;
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);

  // basics
  int num_verts = 32;
  int num_cells = 21;
  EXPECT_EQ(num_verts, grid.get_num_verts());
  EXPECT_EQ(num_cells, grid.get_num_cells());

  // Vert coordinates -> vert
  std::map<std::pair<int, int>, int> coords_vert;
  int vert = 0;
  for (int iy = 0; iy <= 3; iy++) {
    for (int ix = 0; ix <= 5; ix++, vert++) {
      coords_vert[{ix, iy}] = vert;
      EXPECT_EQ(vert, grid.get_coords_vert(ix, iy));
      EXPECT_EQ(vert, grid.get_coords_vert(std::vector<int>{ix, iy}.data()));
    }
  }
  for (int iy = 4; iy <= 5; iy++) {
    for (int ix = 0; ix <= 3; ix++, vert++) {
      coords_vert[{ix, iy}] = vert;
      EXPECT_EQ(vert, grid.get_coords_vert(ix, iy));
      EXPECT_EQ(vert, grid.get_coords_vert(std::vector<int>{ix, iy}.data()));
    }
  }

  // Vert -> coordinates
  for (auto it = coords_vert.begin(); it != coords_vert.end(); it++) {
    int ix = (it->first).first;
    int iy = (it->first).second;
    int vert = it->second;
    int ixy[2];
    grid.get_vert_coords(vert, ixy);
    EXPECT_EQ(ix, ixy[0]);
    EXPECT_EQ(iy, ixy[1]);
  }

  // Cell coordinates -> cell
  std::map<std::pair<int, int>, int> coords_cell;
  int cell = 0;
  for (int ey = 0; ey < 3; ey++) {
    for (int ex = 0; ex < 5; ex++, cell++) {
      coords_cell[{ex, ey}] = cell;
      EXPECT_EQ(cell, grid.get_coords_cell(ex, ey));
      EXPECT_EQ(cell, grid.get_coords_cell(std::vector<int>{ex, ey}.data()));
    }
  }
  for (int ey = 3; ey < 5; ey++) {
    for (int ex = 0; ex < 3; ex++, cell++) {
      coords_cell[{ex, ey}] = cell;
      EXPECT_EQ(cell, grid.get_coords_cell(ex, ey));
      EXPECT_EQ(cell, grid.get_coords_cell(std::vector<int>{ex, ey}.data()));
    }
  }

  // Cell -> coordinates
  for (auto it = coords_cell.begin(); it != coords_cell.end(); it++) {
    int ex = (it->first).first;
    int ey = (it->first).second;
    int cell = it->second;
    int exy[2];
    grid.get_cell_coords(cell, exy);
    EXPECT_EQ(ex, exy[0]) << "cell: " << std::to_string(cell);
    EXPECT_EQ(ey, exy[1]) << "cell: " << std::to_string(cell);
  }

  // Cell -> verts
  cell = 0, vert = 0;
  for (int ey = 0; ey < 3; ey++) {
    for (int ex = 0; ex < 5; ex++, cell++, vert++) {
      int verts[Grid::nverts_per_cell];
      verts[0] = vert;
      verts[1] = vert + 1;
      verts[2] = vert + nx1 + 2;
      verts[3] = vert + nx1 + 1;

      int verts_grid[Grid::nverts_per_cell];
      grid.get_cell_verts(cell, verts_grid);
      EXPECT_VEC_EQ(Grid::nverts_per_cell, verts_grid, verts);

      if (ex == 4) vert++;
    }
  }

  for (int ey = 3; ey < 5; ey++) {
    for (int ex = 0; ex < 3; ex++, cell++) {
      int verts[Grid::nverts_per_cell];
      if (ey == 3) {
        verts[0] = ex + 18;
        verts[1] = ex + 19;
        verts[2] = ex + 25;
        verts[3] = ex + 24;
      } else {
        verts[0] = ex + 24;
        verts[1] = ex + 25;
        verts[2] = ex + 29;
        verts[3] = ex + 28;
      }

      int verts_grid[Grid::nverts_per_cell];
      grid.get_cell_verts(cell, verts_grid);

      for (int i = 0; i < Grid::nverts_per_cell; i++) {
        EXPECT_EQ(verts_grid[i], verts[i])
            << "[" << std::to_string(i) << "]"
            << "ex: " << std::to_string(ex) << ", ey: " << std::to_string(ey)
            << ", expect vert: " << std::to_string(verts[i])
            << ", actual vert: " << std::to_string(verts_grid[i]);
      }

      if (ex == 2) vert++;
    }
  }

  // Cell -> xloc
  for (auto it = coords_cell.begin(); it != coords_cell.end(); it++) {
    int ex = (it->first).first;
    int ey = (it->first).second;
    int cell = it->second;

    T xloc_expect[8], xloc[8];

    grid.get_cell_verts_xloc(cell, xloc);

    xloc_expect[0] = ex * hx;
    xloc_expect[1] = ey * hy;

    xloc_expect[2] = (ex + 1.0) * hx;
    xloc_expect[3] = ey * hy;

    xloc_expect[4] = (ex + 1.0) * hx;
    xloc_expect[5] = (ey + 1.0) * hy;

    xloc_expect[6] = ex * hx;
    xloc_expect[7] = (ey + 1.0) * hy;

    for (int i = 0; i < 8; i++) {
      EXPECT_DOUBLE_EQ(xloc_expect[i], xloc[i]);
    }
  }

  // Cell vert ranges
  for (auto it = coords_cell.begin(); it != coords_cell.end(); it++) {
    int ex = (it->first).first;
    int ey = (it->first).second;
    int cell = it->second;

    T xloc_min[2], xloc_max[2];
    T xloc_min_expect[2], xloc_max_expect[2];

    grid.get_cell_vert_ranges(cell, xloc_min, xloc_max);

    xloc_min_expect[0] = ex * hx;
    xloc_min_expect[1] = ey * hy;

    xloc_max_expect[0] = (ex + 1.0) * hx;
    xloc_max_expect[1] = (ey + 1.0) * hy;

    for (int i = 0; i < 2; i++) {
      EXPECT_DOUBLE_EQ(xloc_min_expect[i], xloc_min[i]);
      EXPECT_DOUBLE_EQ(xloc_max_expect[i], xloc_max[i]);
    }
  }

  // Cell ground stencil
  constexpr int Np_1d = 2;

  std::vector<std::array<int, Grid::nverts_per_cell>> nodes_expect = {
      {0, 1, 6, 7},     {1, 2, 7, 8},     {2, 3, 8, 9},     {3, 4, 9, 10},
      {4, 5, 10, 11},   {6, 7, 12, 13},   {7, 8, 13, 14},   {8, 9, 14, 15},
      {9, 10, 15, 16},  {10, 11, 16, 17}, {12, 13, 18, 19}, {13, 14, 19, 20},
      {14, 15, 20, 21}, {15, 16, 21, 22}, {16, 17, 22, 23}, {18, 19, 24, 25},
      {19, 20, 25, 26}, {20, 21, 26, 27}, {24, 25, 28, 29}, {25, 26, 29, 30},
      {26, 27, 30, 31}};

  for (int cell = 0; cell < num_cells; cell++) {
    int verts[Grid::nverts_per_cell];
    grid.get_cell_ground_stencil<Np_1d>(cell, verts);

    for (int i = 0; i < Grid::nverts_per_cell; i++) {
      EXPECT_EQ(nodes_expect[cell][i], verts[i]);
    }
  }
}
