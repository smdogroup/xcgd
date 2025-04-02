#include <limits>
#include <numeric>
#include <vector>

#include "elements/gd_mesh.h"
#include "elements/lbracket_mesh.h"
#include "test_commons.h"
#include "utils/json.h"
#include "utils/vtk.h"

TEST(gd_mesh, GDMeshStructured) {
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

template <int Np_1d>
void generate_lsf_mesh(bool flip = false) {
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
  std::snprintf(vtkname, 256, "mesh_gd%s_Np%d.vtk", flip ? "_flip" : "", Np_1d);
  ToVTK<T, Mesh> vtk(mesh, vtkname);
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

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

  std::vector<T> cut_elem(mesh.get_num_elements(), 0.0),
      regular_stencil_elem(mesh.get_num_elements(), 0.0);
  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    if (mesh.is_cut_elem(elem)) {
      cut_elem[elem] = 1.0;
    }
    if (mesh.is_regular_stencil_elem(elem)) {
      regular_stencil_elem[elem] = 1.0;
    }
  }
  vtk.write_cell_sol("is_elem_cut", cut_elem.data());
  vtk.write_cell_sol("regular_stencil_elem", regular_stencil_elem.data());

  std::set<int> cut_elem_cells = {
      112, 113, 114, 115, 116, 117, 118, 132, 133, 139, 140, 152, 153, 161,
      162, 173, 183, 194, 204, 215, 225, 236, 246, 257, 267, 278, 279, 287,
      288, 300, 301, 307, 308, 322, 323, 324, 325, 326, 327, 328};

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    if (cut_elem_cells.count(mesh.get_elem_cell(elem))) {
      EXPECT_EQ(mesh.is_cut_elem(elem), true);
    } else {
      EXPECT_EQ(mesh.is_cut_elem(elem), false);
    }
  }

  std::set<int> regular_stencil_elems;
  if (flip) {
    regular_stencil_elems = {
        22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
        36,  37,  38,  39,  40,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  64,  65,  66,  67,
        68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
        82,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99,  100, 101, 102, 103, 106, 107, 108, 109, 110, 111, 119, 120,
        121, 122, 123, 124, 127, 128, 129, 130, 131, 136, 137, 138, 139, 140,
        143, 144, 145, 146, 151, 152, 153, 154, 157, 158, 159, 160, 163, 164,
        165, 166, 169, 170, 171, 172, 175, 176, 177, 178, 181, 182, 183, 184,
        187, 188, 189, 190, 193, 194, 195, 196, 199, 200, 201, 202, 205, 206,
        207, 208, 211, 212, 213, 214, 217, 218, 219, 220, 225, 226, 227, 228,
        231, 232, 233, 234, 235, 240, 241, 242, 243, 244, 247, 248, 249, 250,
        251, 252, 260, 261, 262, 263, 264, 265, 268, 269, 270, 271, 272, 273,
        274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289,
        290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303,
        304, 305, 306, 307, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
        320, 321, 322, 323, 324, 325, 326, 327, 328, 331, 332, 333, 334, 335,
        336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349};
  } else {
    regular_stencil_elems = {
        9,  10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 84, 85, 86, 87, 88, 89, 90, 95, 96, 97, 98, 99};
  }

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    if (cut_elem_cells.count(mesh.get_elem_cell(elem))) {
      EXPECT_EQ(mesh.is_cut_elem(elem), true);
    } else {
      EXPECT_EQ(mesh.is_cut_elem(elem), false);
    }

    if (regular_stencil_elems.count(elem)) {
      EXPECT_EQ(mesh.is_regular_stencil_elem(elem), true);
    } else {
      EXPECT_EQ(mesh.is_regular_stencil_elem(elem), false);
    }
  }
}

TEST(gd_mesh, LSFPositiveNp4) { generate_lsf_mesh<4>(true); }
TEST(gd_mesh, LSFNegativeNp4) { generate_lsf_mesh<4>(false); }

template <class GDMesh>
void test_gdmesh_get_node_patch_elems(const GDMesh& mesh) {
  std::map<int, std::set<int>> node_elems_actual;

  // Populate node_elems
  for (int e = 0; e < mesh.get_num_elements(); e++) {
    constexpr int nnodes = GDMesh::corner_nodes_per_element;
    int nodes[nnodes];
    mesh.get_elem_corner_nodes(e, nodes);
    for (int i = 0; i < nnodes; i++) {
      int n = nodes[i];
      node_elems_actual[n].insert(e);
    }
  }

  for (int n = 0; n < mesh.get_num_nodes(); n++) {
    std::set<int> patch_elems = mesh.get_node_patch_elems(n);
    EXPECT_EQ(patch_elems, node_elems_actual.at(n));
  }
}

template <typename T, class GDMesh>
void test_gdmesh_get_node_patch_elems_node_ranges(const GDMesh& mesh) {
  for (int n = 0; n < mesh.get_num_nodes(); n++) {
    T xloc_min_actual[GDMesh::spatial_dim] = {std::numeric_limits<T>::max(),
                                              std::numeric_limits<T>::max()};
    T xloc_max_actual[GDMesh::spatial_dim] = {std::numeric_limits<T>::min(),
                                              std::numeric_limits<T>::min()};
    // Loop over all patch nodes
    auto patch_elems = mesh.get_node_patch_elems(n);
    for (int e : patch_elems) {
      T t1[2], t2[2];
      mesh.get_elem_corner_node_ranges(e, t1, t2);
      for (int d = 0; d < GDMesh::spatial_dim; d++) {
        xloc_min_actual[d] = std::min(xloc_min_actual[d], t1[d]);
        xloc_max_actual[d] = std::max(xloc_max_actual[d], t2[d]);
      }
    }
    T xloc_min[GDMesh::spatial_dim];
    T xloc_max[GDMesh::spatial_dim];
    mesh.get_node_patch_elems_node_ranges(n, xloc_min, xloc_max);
    for (int d = 0; d < GDMesh::spatial_dim; d++) {
      EXPECT_NEAR(xloc_min[d], xloc_min_actual[d], 1e-12);
      EXPECT_NEAR(xloc_max[d], xloc_max_actual[d], 1e-12);
    }
  }
}

TEST(gd_mesh, GridMesh_elem_patch_Np2) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;
  int nxy[2] = {10, 6};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  test_gdmesh_get_node_patch_elems(mesh);
  test_gdmesh_get_node_patch_elems_node_ranges<T>(mesh);
}

TEST(gd_mesh, GridMesh_elem_patch_Np4) {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;
  int nxy[2] = {10, 6};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  test_gdmesh_get_node_patch_elems(mesh);
  test_gdmesh_get_node_patch_elems_node_ranges<T>(mesh);
}

TEST(gd_mesh, LbracketGridMesh_elem_patch_Np2) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = LbracketGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;
  int nx1 = 10, nx2 = 4, ny1 = 4, ny2 = 6;
  T lx1 = 1.0, ly1 = 0.4;
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);
  Mesh mesh(grid);
  test_gdmesh_get_node_patch_elems(mesh);
  test_gdmesh_get_node_patch_elems_node_ranges<T>(mesh);
}

TEST(gd_mesh, LbracketGridMesh_elem_patch_Np4) {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = LbracketGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d, Grid>;
  int nx1 = 10, nx2 = 4, ny1 = 4, ny2 = 6;
  T lx1 = 1.0, ly1 = 0.4;
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);
  Mesh mesh(grid);
  test_gdmesh_get_node_patch_elems(mesh);
  test_gdmesh_get_node_patch_elems_node_ranges<T>(mesh);
}

template <typename T, class Grid, int Np_1d>
void test_elem_patch_cut_mesh() {
  using Mesh = CutMesh<T, Np_1d>;

  json j = read_json("lsf_dof.json");

  int nxy[2] = {j["nxy"], j["nxy"]};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  mesh.get_lsf_dof() = std::vector<double>(j["lsf_dof"]);
  mesh.update_mesh();

  test_gdmesh_get_node_patch_elems(mesh);
  test_gdmesh_get_node_patch_elems_node_ranges<T>(mesh);

  for (auto& v : mesh.get_lsf_dof()) {
    v *= -1.0;
  }
  mesh.update_mesh();

  test_gdmesh_get_node_patch_elems(mesh);
  test_gdmesh_get_node_patch_elems_node_ranges<T>(mesh);
}

TEST(gd_mesh, CutMesh_elem_patch_Np2) {
  test_elem_patch_cut_mesh<double, StructuredGrid2D<double>, 2>();
}

TEST(gd_mesh, CutMesh_elem_patch_Np4) {
  test_elem_patch_cut_mesh<double, StructuredGrid2D<double>, 4>();
}

TEST(mesh, verts_to_pterms_1) {
  /*
   * 6 │  0    0    x    x
   *   │
   * 5 │  0    x    0    x
   *   │
   * 4 │  x    x    x    0
   *   │
   * 3 │  0    x    x    0
   *   └───────────────────
   *      5    6    7    8
   * */

  std::vector<std::pair<int, int>> verts = {
      {6, 3}, {7, 3}, {6, 5}, {8, 5}, {7, 6}, {8, 6}, {5, 4}, {6, 4}, {7, 4}};

  std::vector<std::pair<int, int>> pterms_v = {
      {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {3, 0}};
  std::vector<std::pair<int, int>> pterms_h = {
      {0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}};

  EXPECT_EQ(pterms_v, verts_to_pterms(verts, true));
  EXPECT_EQ(pterms_h, verts_to_pterms(verts, false));
}

TEST(mesh, verts_to_pterms_2) {
  /*
   *                              x
   * 6 │  0    0    x    x
   *   │
   * 5 │  0    x    0                      x
   *   │
   * 4 │  x    x    x    0
   *   │
   * 3 │  0    x    x    0
   *   └───────────────────
   *      5    6    7    8
   * */

  std::vector<std::pair<int, int>> verts = {{6, 3}, {7, 3}, {6, 5}, {20, 5},
                                            {7, 6}, {8, 6}, {5, 4}, {6, 4},
                                            {7, 4}, {18, 7}};

  std::vector<std::pair<int, int>> pterms_v = {{0, 0}, {0, 1}, {0, 2}, {1, 0},
                                               {1, 1}, {1, 2}, {2, 0}, {3, 0},
                                               {4, 0}, {5, 0}};
  std::vector<std::pair<int, int>> pterms_h = {{0, 0}, {1, 0}, {2, 0}, {0, 1},
                                               {1, 1}, {0, 2}, {1, 2}, {0, 3},
                                               {1, 3}, {0, 4}};

  EXPECT_EQ(pterms_v, verts_to_pterms(verts, true));
  EXPECT_EQ(pterms_h, verts_to_pterms(verts, false));
}
