#ifndef XCGD_GD_MESH_H
#define XCGD_GD_MESH_H

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "elements/element_commons.h"
#include "quadrature_general.hpp"
#include "utils/misc.h"

// The structured ground grid
template <typename T>
class StructuredGrid2D final {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int nverts_per_cell = 4;

  /**
   * @param nxy_ numbers of cells in x, y directions
   * @param lxy_ lengths of the grid in x, y directions
   * @param xy0_ origin (x, y) of the grid
   */
  StructuredGrid2D(const int* nxy_, const T* lxy_, const T* xy0_ = nullptr) {
    for (int d = 0; d < spatial_dim; d++) {
      nxy[d] = nxy_[d];
      lxy[d] = lxy_[d];
      if (xy0_) {
        xy0[d] = xy0_[d];
      } else {
        xy0[d] = 0.0;
      }
      h[d] = lxy[d] / T(nxy[d]);
    }
  }

  inline int get_num_verts() const {
    const int* nxy = get_nxy();
    int nnodes = 1;
    for (int d = 0; d < spatial_dim; d++) {
      nnodes *= nxy[d] + 1;
    }
    return nnodes;
  }

  inline int get_num_cells() const {
    const int* nxy = get_nxy();
    int nelems = 1;
    for (int d = 0; d < spatial_dim; d++) {
      nelems *= nxy[d];
    }
    return nelems;
  }

  // coordinates -> vert
  inline int get_coords_vert(int ni, int nj) const {
    return ni + (nxy[0] + 1) * nj;
  }
  inline int get_coords_vert(const int* nij) const {
    return nij[0] + (nxy[0] + 1) * nij[1];
  }

  // vert -> coordinates
  inline void get_vert_coords(int vert, int* nij) const {
    nij[0] = vert % (nxy[0] + 1);
    nij[1] = vert / (nxy[0] + 1);
  }

  // coordinates -> cell
  inline int get_coords_cell(int ei, int ej) const { return ei + nxy[0] * ej; }
  inline int get_coords_cell(const int* eij) const {
    return eij[0] + nxy[0] * eij[1];
  }

  // cell -> coordinates
  inline void get_cell_coords(int cell, int* eij) const {
    eij[0] = cell % nxy[0];
    eij[1] = cell / nxy[0];
  }

  // cell -> verts
  void get_cell_verts(int cell, int* verts) const {
    if (verts) {
      int nij[spatial_dim] = {cell % nxy[0], cell / nxy[0]};
      verts[0] = get_coords_vert(nij);   //  3-------2
      verts[1] = verts[0] + 1;           //  |       |
      verts[2] = verts[1] + nxy[0] + 1;  //  |       |
      verts[3] = verts[2] - 1;           //  0-------1
    }
  }

  // cell -> xloc for all verts
  void get_cell_verts_xloc(int cell, T* xloc) const {
    if (xloc) {
      int nij[spatial_dim] = {cell % nxy[0], cell / nxy[0]};

      xloc[0] = xy0[0] + h[0] * nij[0];
      xloc[1] = xy0[1] + h[1] * nij[1];

      xloc[2] = xy0[0] + h[0] * (nij[0] + 1);
      xloc[3] = xy0[1] + h[1] * nij[1];

      xloc[4] = xy0[0] + h[0] * (nij[0] + 1);
      xloc[5] = xy0[1] + h[1] * (nij[1] + 1);

      xloc[6] = xy0[0] + h[0] * nij[0];
      xloc[7] = xy0[1] + h[1] * (nij[1] + 1);
    }
  }

  void get_cell_vert_ranges(int cell, T* xloc_min, T* xloc_max) const {
    int verts[4];
    get_cell_verts(cell, verts);

    get_vert_xloc(verts[0], xloc_min);
    get_vert_xloc(verts[2], xloc_max);
  }

  void get_vert_xloc(int vert, T* xloc) const {
    if (xloc) {
      int nij[spatial_dim];
      get_vert_coords(vert, nij);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xy0[d] + h[d] * T(nij[d]);
      }
    }
  }

  // Get the xloc of the centroid of the cell
  void get_cell_xloc(int cell, T* xloc) const {
    if (xloc) {
      int nij[spatial_dim];
      get_cell_coords(cell, nij);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xy0[d] + h[d] * (T(nij[d]) + 0.5);
      }
    }
  }

  const int* get_nxy() const { return nxy; };
  const T* get_lxy() const { return lxy; };
  const T* get_xy0() const { return xy0; };
  const T* get_h() const { return h; };

 private:
  int nxy[spatial_dim];
  T lxy[spatial_dim];
  T xy0[spatial_dim];
  T h[spatial_dim];
};

/**
 * @brief The Galerkin difference mesh defined on a structured
 * grid, i.e. no cuts by a level set function
 *
 * Note: This class is light-weight, as the mesh data is computed on-the-fly.
 */
template <typename T, int Np_1d>
class GridMesh final : public GDMeshBase<T, Np_1d> {
 private:
  using MeshBase = GDMeshBase<T, Np_1d>;

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::nodes_per_element;
  using MeshBase::spatial_dim;
  using typename MeshBase::Grid;

  /**
   * @brief Construct GD Mesh given grid only
   */
  GridMesh(const Grid& grid)
      : MeshBase(grid),
        num_nodes(grid.get_num_verts()),
        num_elements(grid.get_num_cells()) {}

  inline int get_num_nodes() const { return num_nodes; }
  inline int get_num_elements() const { return num_elements; }

  inline void get_node_xloc(int node, T* xloc) const {
    this->grid.get_vert_xloc(node, xloc);
  }

  /**
   * @brief For a GD element, get all dof nodes
   *
   * @param elem element index
   * @param nodes dof node indices, length: nodes_per_element
   */
  void get_elem_dof_nodes(int elem, int* nodes) const {
    this->get_cell_ground_stencil(elem, nodes);
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    this->grid.get_cell_verts(elem, nodes);
  }

  inline void get_elem_vert_ranges(int elem, T* xloc_min, T* xloc_max) const {
    this->grid.get_cell_vert_ranges(elem, xloc_min, xloc_max);
  }

  std::vector<int> get_left_boundary_nodes() const {
    std::vector<int> nodes;
    const int* nxy = this->grid.get_nxy();
    for (int j = 0; j < nxy[1] + 1; j++) {
      int coords[2] = {0, j};
      int node = this->grid.get_coords_vert(coords);
      nodes.push_back(node);
    }
    return nodes;
  }

  std::vector<int> get_right_boundary_nodes() const {
    std::vector<int> nodes;
    const int* nxy = this->grid.get_nxy();
    for (int j = 0; j < nxy[1] + 1; j++) {
      int coords[2] = {nxy[0], j};
      int node = this->grid.get_coords_vert(coords);
      nodes.push_back(node);
    }
    return nodes;
  }

 private:
  int num_nodes = -1;
  int num_elements = -1;
};

/**
 * @brief The Galerkin difference mesh defined on a structured
 * grid with cuts defined by a level set function
 */
template <typename T, int Np_1d>
class CutMesh final : public GDMeshBase<T, Np_1d> {
 private:
  using MeshBase = GDMeshBase<T, Np_1d>;
  using LSFMesh = GridMesh<T, Np_1d>;

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::nodes_per_element;
  using MeshBase::spatial_dim;
  using typename MeshBase::Grid;
  static constexpr bool is_cut_mesh = true;

  /**
   * @brief Construct GD Mesh given grid and LSF function
   *
   * @tparam Func functor type
   * @param grid GD grid
   * @param lsf initial level set function that determines the analysis domain,
   * lsf(T* xyz) where xyz contains x,y,(z) coordinates.
   *
   * Note: Within the analysis domain, lsf <= 0
   */
  template <class Func>
  CutMesh(const Grid& grid, const Func& lsf)
      : MeshBase(grid), lsf_mesh(grid), lsf_dof(lsf_mesh.get_num_nodes()) {
    for (int i = 0; i < lsf_dof.size(); i++) {
      T xloc[spatial_dim];
      lsf_mesh.get_node_xloc(i, xloc);
      lsf_dof[i] = lsf(xloc);
    }
    update_mesh();
  }

  CutMesh(const Grid& grid)
      : MeshBase(grid),
        lsf_mesh(grid),
        lsf_dof(lsf_mesh.get_num_nodes(), -1.0) {
    update_mesh();
  }

  int get_num_nodes() const { return num_nodes; }
  int get_num_elements() const { return num_elements; }

  inline void get_node_xloc(int node, T* xloc) const {
    this->grid.get_vert_xloc(node_verts.at(node), xloc);
  }

  /**
   * @brief For a GD element, get all dof nodes
   *
   * @param elem element index
   * @param nodes dof node indices, length: nodes_per_element
   */
  void get_elem_dof_nodes(int elem, int* nodes) const {
    int cell = elem_cells.at(elem);
    this->get_cell_ground_stencil(cell, nodes);
    adjust_stencil(cell, nodes);
    for (int i = 0; i < nodes_per_element; i++) {
      nodes[i] = vert_nodes.at(nodes[i]);
    }
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    this->grid.get_cell_verts(elem_cells.at(elem), nodes);
    for (int i = 0; i < corner_nodes_per_element; i++) {
      nodes[i] = vert_nodes.at(nodes[i]);
    }
  }

  inline void get_elem_vert_ranges(int elem, T* xloc_min, T* xloc_max) const {
    this->grid.get_cell_vert_ranges(elem_cells.at(elem), xloc_min, xloc_max);
  }

  /* Helper function */
  std::vector<int> get_left_boundary_nodes(double tol = 1e-10) const {
    const T* xy0 = this->grid.get_xy0();
    std::vector<int> nodes;
    for (int i = 0; i < num_nodes; i++) {
      T xloc[spatial_dim];
      get_node_xloc(i, xloc);
      if (xloc[0] - xy0[0] < tol) {
        nodes.push_back(i);
      }
    }
    return nodes;
  }

  /* Helper function */
  std::vector<int> get_right_boundary_nodes(double tol = 1e-10) const {
    const T* xy0 = this->grid.get_xy0();
    const T* lxy = this->grid.get_lxy();
    std::vector<int> nodes;
    for (int i = 0; i < num_nodes; i++) {
      T xloc[spatial_dim];
      get_node_xloc(i, xloc);
      if (xy0[0] + lxy[0] - xloc[0] < tol) {
        nodes.push_back(i);
      }
    }
    return nodes;
  }

  std::vector<T> get_lsf_nodes() const {
    std::vector<T> lsf_nodes(get_num_nodes());
    for (auto kv : node_verts) {
      // lsf_nodes[node] = lsf_dof[vert]
      lsf_nodes[kv.first] = lsf_dof[kv.second];
    }
    return lsf_nodes;
  }

  std::vector<T> get_lsf_nodes(const std::vector<T>& lsf_dof) const {
    std::vector<T> lsf_nodes(get_num_nodes());
    for (auto kv : node_verts) {
      // lsf_nodes[node] = lsf_dof[vert]
      lsf_nodes[kv.first] = lsf_dof[kv.second];
    }
    return lsf_nodes;
  }

  const LSFMesh& get_lsf_mesh() const { return lsf_mesh; }

  inline const std::vector<T>& get_lsf_dof() const { return lsf_dof; }
  inline std::vector<T>& get_lsf_dof() { return lsf_dof; }

  inline int get_elem_cell(int elem) const { return elem_cells[elem]; }

  // Update the mesh when the lsf_dof is updated
  void update_mesh() {
    node_verts.clear();
    vert_nodes.clear();
    elem_cells.clear();
    dir_cells.clear();

    // LSF values are always associated with the ground grid verts, unlike the
    // dof values which might only be associated with part of the ground grid
    // verts (i.e. nodes)
    int nverts = this->grid.get_num_verts();

    // Given lsf dof values, obtain active lsf vertices
    std::vector<bool> active_lsf_verts(nverts, false);
    for (int i = 0; i < nverts; i++) {
      if (freal(lsf_dof[i]) <= freal(T(0.0))) {
        active_lsf_verts[i] = true;
      }
    }

    // Active cell is a cell with at least one active lsf vert
    int ncells = this->grid.get_num_cells();
    std::vector<bool> active_cells(ncells, false);

    // Unlike LSF values, dof are associated with nodes, which is a subset of
    // verts. Here we determine which verts are dof nodes
    for (int c = 0; c < ncells; c++) {
      if (active_cells[c]) continue;
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(c, verts);
      for (int i = 0; i < Grid::nverts_per_cell; i++) {
        if (active_lsf_verts[verts[i]]) {
          active_cells[c] = true;
          break;
        }
      }
    }

    // Create active dof nodes and the mapping to verts
    int node = 0;
    for (int c = 0; c < ncells; c++) {
      if (!active_cells[c]) continue;
      elem_cells.push_back(c);
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(c, verts);
      for (int i = 0; i < Grid::nverts_per_cell; i++) {
        if (vert_nodes.count(verts[i]) == 0) {
          vert_nodes[verts[i]] = node;
          node_verts[node] = verts[i];
          node++;
        }
      }
    }

    // For each cell, get the push direction for the outlying ground stencil
    // vertices
    dir_cells = std::vector<int>(ncells, -1);  // val | direction
                                               // 0   | +x
                                               // 1   | -x
                                               // 2   | +y
                                               // 3   | -y
                                               // 4   | +z
                                               // 5   | -z

    for (int c = 0; c < ncells; c++) {
      std::array<T, spatial_dim> grad = interp_lsf_grad(c);
      double tmp = 0.0;
      int dim = -1;
      for (int d = 0; d < spatial_dim; d++) {
        if (fabs(grad[d]) > tmp) {
          tmp = fabs(grad[d]);
          dim = d;
        }
      }
      dir_cells[c] = 2 * dim + (freal(grad[dim]) < 0.0 ? 0 : 1);
    }

    int num_nodes_old = num_nodes;
    int num_elems_old = num_elements;

    num_nodes = node_verts.size();
    num_elements = elem_cells.size();

#ifdef XCGD_DEBUG_MODE
    std::printf("[Debug]Updating mesh, nnodes: %d -> %d, nelems: %d -> %d\n",
                num_nodes_old, num_nodes, num_elems_old, num_elements);
#endif
  }

  // Helper function
  bool is_irregular_stencil(int elem) {
    int cell = elem_cells.at(elem);
    int verts[nodes_per_element];
    this->get_cell_ground_stencil(cell, verts);
    for (int index = 0; index < nodes_per_element; index++) {
      if (vert_nodes.count(verts[index]) == 0) {
        return true;
      }
    }
    return false;
  }

  inline const std::unordered_map<int, int>& get_vert_nodes() const {
    return vert_nodes;
  }

 private:
  // Given the lsf dof, interpolate the gradient of the lsf at the centroid
  // a cell using bilinear quad element
  std::array<T, spatial_dim> interp_lsf_grad(int cell) {
    int verts[Grid::nverts_per_cell];
    this->grid.get_cell_verts(cell, verts);

    const T* h = this->grid.get_h();
    std::array<T, spatial_dim> grad;

    grad[0] = 0.5 / h[0] *
              (-lsf_dof[verts[0]] + lsf_dof[verts[1]] + lsf_dof[verts[2]] -
               lsf_dof[verts[3]]);
    grad[1] = 0.5 / h[1] *
              (-lsf_dof[verts[0]] - lsf_dof[verts[1]] + lsf_dof[verts[2]] +
               lsf_dof[verts[3]]);
    return grad;
  }

  /**
   * @brief Adjust the stencil by pushing the stencil verts that are outside the
   * LSF boundary inward such that all nodes are active nodes
   */
  void adjust_stencil(int cell, int* verts) const {
    // Get push direction
    int dir = dir_cells[cell];
    int dim = dir / spatial_dim;
    int sign = dir % spatial_dim == 0 ? 1 : -1;

    // Adjust nodes
    for (int index = 0; index < nodes_per_element; index++) {
      int vert = verts[index];
      if (vert_nodes.count(vert) == 0) {
        int vert_coords[spatial_dim] = {-1, -1};
        this->grid.get_vert_coords(vert, vert_coords);
        vert_coords[dim] += sign * Np_1d;
        verts[index] = this->grid.get_coords_vert(vert_coords);
      }
    }

// check  if the stencil is a valid stencil
#ifdef XCGD_DEBUG_MODE
    for (int index = 0; index < nodes_per_element; index++) {
      int vert = verts[index];
      if (vert_nodes.count(vert) == 0) {
        int coords[spatial_dim];
        this->grid.get_vert_coords(vert, coords);
        std::printf(
            "[Debug] vert %d (%d, %d) is not an active node for cell: %d\n",
            vert, coords[0], coords[1], cell);
      }
    }
#endif
  }

  LSFMesh lsf_mesh;

  int num_nodes = -1;
  int num_elements = -1;

  // level set function values at vertices of the ground grid
  std::vector<T> lsf_dof;

  // indices of vertices that are dof nodes, i.e. vertices that have active
  // degrees of freedom
  std::unordered_map<int, int> node_verts;  // node -> vert
  std::unordered_map<int, int> vert_nodes;  // vert -> node

  // indices of cells that are dof elements, i.e. cells that have active degrees
  // of freedom
  std::vector<int> elem_cells;  // elem -> cell

  // push direction for each cell
  std::vector<int> dir_cells;
};

/**
 * @brief Helper function: get the limit of computational coordinates (xi, eta,
 * zeta) given a stencil.
 *
 * Note: Defined in the mesh object, for each element there are a set of dof
 * nodes (stencil nodes) a set of vertices that defines the physical boundary of
 * an element. For numerical stability for the Vandermonde bases, the dof nodes
 * need to be mapped onto a [-1, 1]^d hyperrectangle, where d is the spatial
 * dimension (2 or 3). Based on this bound, this function evaluates the bounds
 * for the computational coordinates (xi, eta, zeta). For linear element, this
 * is trivially (-1, 1) too, because all verts are nodes. For higher order
 * elements, such limits are narrower than [-1, 1].
 *
 * @tparam GDMesh a GDMesh type
 * @param mesh mesh object
 * @param elem element index
 * @param xi_min output, lower bounds of computational coordinates
 * @param xi_max output, upper bounds of computational coordinates
 * @return T area ratio
 */
template <typename T, class Mesh>
T get_computational_coordinates_limits(const Mesh& mesh, int elem, T* xi_min,
                                       T* xi_max) {
  static_assert(Mesh::is_gd_mesh, "function only works with a GD mesh");
  int constexpr spatial_dim = Mesh::spatial_dim;
  T xy_min[spatial_dim], xy_max[spatial_dim];
  T uv_min[spatial_dim], uv_max[spatial_dim];
  mesh.get_elem_node_ranges(elem, xy_min, xy_max);
  mesh.get_elem_vert_ranges(elem, uv_min, uv_max);

  T hx = (uv_max[0] - uv_min[0]) / (xy_max[0] - xy_min[0]);
  T hy = (uv_max[1] - uv_min[1]) / (xy_max[1] - xy_min[1]);
  T wcoef = 4.0 * hx * hy;

  T cx = (2.0 * uv_min[0] - xy_min[0] - xy_max[0]) / (xy_max[0] - xy_min[0]);
  T dx = 2.0 * hx;
  T cy = (2.0 * uv_min[1] - xy_min[1] - xy_max[1]) / (xy_max[1] - xy_min[1]);
  T dy = 2.0 * hy;

  xi_min[0] = cx;
  xi_min[1] = cy;
  xi_max[0] = cx + dx;
  xi_max[1] = cy + dy;

  return wcoef;
}

#endif  // XCGD_GD_MESH_H