#ifndef XCGD_GD_MESH_H
#define XCGD_GD_MESH_H

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "elements/element_commons.h"
#include "elements/vandermonde_evaluator.h"
#include "quadrature_multipoly.hpp"
#include "utils/loggers.h"
#include "utils/misc.h"

/**
 * @brief The structured ground grid
 *
 * The numbering of the grid vertices and cells is as follows
 *
 *      12--- 13--- 14--- 15--- 16--- 17
 *      |     |     |     |     |     |
 *      |  5  |  6  |  7  |  8  |  9  |
 *      6 --- 7 --- 8 --- 9 --- 10--- 11
 *      |     |     |     |     |     |
 *      |  0  |  1  |  2  |  3  |  4  |
 *      0 --- 1 --- 2 --- 3 --- 4 --- 5
 *
 * For convenience, we can also use index coordinates to locate cell and vertex,
 * for example:
 *
 *  vertex 8 has index coordinates (2, 1)
 *  cell 5 has index coordinates (0, 1)
 *
 *  Note that index coordiantes are NOT (row, col), instead, they are like
 *  Cartisian coordiantes but are integers
 *
 */
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
    int nnodes = 1;
    for (int d = 0; d < spatial_dim; d++) {
      nnodes *= nxy[d] + 1;
    }
    return nnodes;
  }

  inline int get_num_cells() const {
    int nelems = 1;
    for (int d = 0; d < spatial_dim; d++) {
      nelems *= nxy[d];
    }
    return nelems;
  }

  inline bool is_valid_vert(int ix, int iy) const {
    return (ix >= 0 and ix <= nxy[0] and iy >= 0 and iy <= nxy[1]);
  }

  inline bool is_valid_cell(int ex, int ey) const {
    return (ex >= 0 and ex < nxy[0] and ey >= 0 and ey < nxy[1]);
  }

  // coordinates -> vert
  inline int get_coords_vert(int ix, int iy) const {
    return ix + (nxy[0] + 1) * iy;
  }
  inline int get_coords_vert(const int* ixy) const {
    return ixy[0] + (nxy[0] + 1) * ixy[1];
  }

  // vert -> coordinates
  inline void get_vert_coords(int vert, int* ixy) const {
    ixy[0] = vert % (nxy[0] + 1);
    ixy[1] = vert / (nxy[0] + 1);
  }

  // coordinates -> cell
  inline int get_coords_cell(int ex, int ey) const { return ex + nxy[0] * ey; }
  inline int get_coords_cell(const int* exy) const {
    return exy[0] + nxy[0] * exy[1];
  }

  // cell -> coordinates
  inline void get_cell_coords(int cell, int* exy) const {
    exy[0] = cell % nxy[0];
    exy[1] = cell / nxy[0];
  }

  // cell -> verts
  void get_cell_verts(int cell, int* verts) const {
    if (verts) {
      int ixy[spatial_dim] = {cell % nxy[0], cell / nxy[0]};
      verts[0] = get_coords_vert(ixy);   //  3-------2
      verts[1] = verts[0] + 1;           //  |       |
      verts[2] = verts[1] + nxy[0] + 1;  //  |       |
      verts[3] = verts[2] - 1;           //  0-------1
    }
  }

  // cell -> xloc for all verts
  void get_cell_verts_xloc(int cell, T* xloc) const {
    if (xloc) {
      int ixy[spatial_dim] = {cell % nxy[0], cell / nxy[0]};

      xloc[0] = xy0[0] + h[0] * ixy[0];
      xloc[1] = xy0[1] + h[1] * ixy[1];

      xloc[2] = xy0[0] + h[0] * (ixy[0] + 1);
      xloc[3] = xy0[1] + h[1] * ixy[1];

      xloc[4] = xy0[0] + h[0] * (ixy[0] + 1);
      xloc[5] = xy0[1] + h[1] * (ixy[1] + 1);

      xloc[6] = xy0[0] + h[0] * ixy[0];
      xloc[7] = xy0[1] + h[1] * (ixy[1] + 1);
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
      int ixy[spatial_dim];
      get_vert_coords(vert, ixy);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xy0[d] + h[d] * T(ixy[d]);
      }
    }
  }

  // Get the xloc of the centroid of the cell
  void get_cell_xloc(int cell, T* xloc) const {
    if (xloc) {
      int ixy[spatial_dim];
      get_cell_coords(cell, ixy);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xy0[d] + h[d] * (T(ixy[d]) + 0.5);
      }
    }
  }

  const T* get_lxy() const { return lxy; };
  const T* get_xy0() const { return xy0; };
  const T* get_h() const { return h; };

  /**
   * @brief For a GD element, get the stencil associated to the ground grid,
   * regardless the optional boundary defined by the level set function
   *
   * @param cell [in] cell index
   * @param verts [out] stencil vertices
   * @return bool whether the stensil is regular (i.e. not a boundary stencil)
   */
  template <int Np_1d>
  bool get_cell_ground_stencil(int cell, int* verts) const {
    bool is_stencil_regular = true;
    constexpr int q = Np_1d / 2;
    int exy[spatial_dim];
    get_cell_coords(cell, exy);
    for (int d = 0; d < spatial_dim; d++) {
      if (exy[d] < q - 1) {
        is_stencil_regular = false;
        exy[d] = q - 1;
      } else if (exy[d] > nxy[d] - q) {
        exy[d] = nxy[d] - q;
        is_stencil_regular = false;
      }
    }

    int index = 0;
    for (int j = 0; j < Np_1d; j++) {
      for (int i = 0; i < Np_1d; i++, index++) {
        verts[index] = get_coords_vert(exy[0] - q + 1 + i, exy[1] - q + 1 + j);
      }
    }
    return is_stencil_regular;
  };

  template <int Np_1d>
  void check_grid_compatibility() const {
    for (int d = 0; d < spatial_dim; d++) {
      if (nxy[d] < Np_1d - 1) {
        char msg[256];
        std::snprintf(
            msg, 256,
            "too few elements (%d) for Np_1d (%d) along %d-th dimension",
            nxy[d], Np_1d, d);
        throw std::runtime_error(msg);
      }
    }
  }

  const int* get_nxy() const { return nxy; };

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
template <typename T, int Np_1d, class Grid_ = StructuredGrid2D<T>>
class GridMesh : public GDMeshBase<T, Np_1d, Grid_> {
 private:
  using MeshBase = GDMeshBase<T, Np_1d, Grid_>;

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::max_nnodes_per_element;
  using MeshBase::spatial_dim;
  using typename MeshBase::Grid;

  /**
   * @brief Construct GD Mesh given grid only
   */
  GridMesh(const Grid& grid)
      : MeshBase(grid),
        num_nodes(grid.get_num_verts()),
        num_elements(grid.get_num_cells()) {
    // Identify all the elements with regular stencils
    for (int i = 0; i < num_elements; i++) {
      int _[max_nnodes_per_element];
      if (this->grid.template get_cell_ground_stencil<Np_1d>(i, _)) {
        regular_stencil_elems.insert(i);
      }
    }
  }

  inline int get_num_nodes() const { return num_nodes; }
  inline int get_num_elements() const { return num_elements; }

  inline void get_node_xloc(int node, T* xloc) const {
    this->grid.get_vert_xloc(node, xloc);
  }

  /**
   * @brief For a GD element, get all dof nodes
   *
   * @param elem element index
   * @param nodes dof node indices, length: max_nnodes_per_element
   */
  int get_elem_dof_nodes(int elem, int* nodes) const {
    this->grid.template get_cell_ground_stencil<Np_1d>(elem, nodes);
    return max_nnodes_per_element;
  }

  std::set<int> get_node_patch_elems(int node) const {
    std::set<int> ret;
    int ixy[spatial_dim] = {-1, -1};
    this->grid.get_vert_coords(node, ixy);

    int ex = -1, ey = -1;

    // lower-left patch, if any
    ex = ixy[0] - 1, ey = ixy[1] - 1;
    if (this->grid.is_valid_cell(ex, ey)) {
      ret.insert(this->grid.get_coords_cell(ex, ey));
    }

    // lower-right patch, if any
    ex = ixy[0], ey = ixy[1] - 1;
    if (this->grid.is_valid_cell(ex, ey)) {
      ret.insert(this->grid.get_coords_cell(ex, ey));
    }

    // upper-left patch, if any
    ex = ixy[0] - 1, ey = ixy[1];
    if (this->grid.is_valid_cell(ex, ey)) {
      ret.insert(this->grid.get_coords_cell(ex, ey));
    }

    // upper-right patch, if any
    ex = ixy[0], ey = ixy[1];
    if (this->grid.is_valid_cell(ex, ey)) {
      ret.insert(this->grid.get_coords_cell(ex, ey));
    }

    return ret;
  }

  std::vector<std::vector<bool>> get_elem_pstencil(int elem) const {
    std::vector<std::vector<bool>> pstencil(Np_1d);
    for (int I = 0; I < Np_1d; I++) {
      pstencil[I] = std::vector<bool>(Np_1d, false);
      for (int J = 0; J < Np_1d; J++) {
        pstencil[I][J] = true;
      }
    }
    return pstencil;
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    this->grid.get_cell_verts(elem, nodes);
  }

  inline void get_elem_corner_node_ranges(int elem, T* xloc_min,
                                          T* xloc_max) const {
    this->grid.get_cell_vert_ranges(elem, xloc_min, xloc_max);
  }

  inline bool is_regular_stencil_elem(int elem) const {
    return static_cast<bool>(regular_stencil_elems.count(elem));
  }
  const inline std::set<int>& get_regular_stencil_elems() const {
    return regular_stencil_elems;
  }

  inline int get_node_vert(int node) const { return node; }

  // Caution: this is a dummy function that does not return meaningful value
  inline int get_elem_dir(int elem) const { return 0; }

 private:
  int num_nodes = -1;
  int num_elements = -1;

  // Whether the element has the regular stencil
  // elements far from the boundaries usually have regular stencils
  std::set<int> regular_stencil_elems;
};

/**
 * - AllowOutsideLSF: when creating element -> dof node mapping, allow nodes to
 * reside outside the boundary defined by the level-set function, which usually
 * happens at the cut cells, default option
 * - StrictlyInsideLSF: typically for cut elements, don't allow vertices outside
 * the boundary defined by the level-set function to be used as dof nodes
 */
enum class NodeStrategy { AllowOutsideLSF, StrictlyInsideLSF };

/**
 * When determining DOF nodes for elements near the cut boundary,
 * we often need to modify from the ground stencil because some of the vertices
 * in the ground stensil is too deep outside the cut boundary. This enum
 * marks the two different strategies of such modification:
 *
 * - PerElement: we determine a push direction based on gradient of the LSF
 * function at the element centroid, then use that direction for all nodes
 * associated to this element. Main drawback of this approach is that we do not
 * maintain consistency across elements, meaning that a same active vertex could
 * be pushed to different location for differnet elements near it.
 *
 * - PerNode: in contrast to PerElement, we determine a push direction for each
 * vertex, so the push modification is always consistent across elements.
 * */
// enum class PushStrategy { PerElement, PerNode };

/**
 * @brief The Galerkin difference mesh defined on a structured
 * grid with cuts defined by a level set function
 */
template <typename T, int Np_1d, class Grid_ = StructuredGrid2D<T>>
class CutMesh final : public GDMeshBase<T, Np_1d, Grid_> {
 private:
  constexpr static NodeStrategy node_strategy = NodeStrategy::AllowOutsideLSF;
  using MeshBase = GDMeshBase<T, Np_1d, Grid_>;
  using LSFMesh = GridMesh<T, Np_1d, Grid_>;

  template <class T1, class T2>
  using Map =
      std::unordered_map<T1, T2>;  // TODO(fyc): std::map and
                                   // std::unordered_map, which is better?

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::max_nnodes_per_element;
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
   * @param nodes dof node indices, length: max_nnodes_per_element
   *
   * @return number of nodes associated to this element
   */
  int get_elem_dof_nodes_deprecated(
      int elem, int* nodes,
      std::vector<std::vector<bool>>* pstencil = nullptr) const {
    if (pstencil) {
      pstencil->clear();
      pstencil->resize(Np_1d);
      for (int i = 0; i < Np_1d; i++) {
        (*pstencil)[i] = std::vector<bool>(Np_1d, false);
      }
    }

    int nnodes = 0;
    int cell = elem_cells.at(elem);
    this->grid.template get_cell_ground_stencil<Np_1d>(cell, nodes);
    adjust_stencil(cell, nodes);

    for (int i = 0; i < max_nnodes_per_element; i++) {
      try {
        nodes[nnodes] = vert_nodes.at(nodes[i]);
        if (pstencil) {
          int I = i % Np_1d;
          int J = i / Np_1d;
          (*pstencil)[I][J] = true;
        }
      } catch (const std::out_of_range& e) {
        // throw StencilConstructionFailed(elem);
        continue;
      }
      nnodes++;
    }
    if (nnodes != max_nnodes_per_element) {
      DegenerateStencilLogger::add(elem, nnodes, nodes);
    }
    return nnodes;
  }

  /**
   * @brief For a GD element, get all dof nodes
   *
   * @param elem element index
   * @return nodes dof node indices, length: nnodes
   */
  int get_elem_dof_nodes(int elem, int* nodes) const {
    const auto& nodes_v = elem_nodes.at(elem);
    int nnodes = nodes_v.size();
    for (int i = 0; i < nnodes; i++) {
      nodes[i] = nodes_v[i];
    }
    return nnodes;
  }

  std::set<int> get_node_patch_elems(int node) const {
    return node_patch_elems.at(node);
  }

  const Map<int, std::vector<int>>& get_elem_nodes() const {
    return elem_nodes;
  }
  const std::vector<int>& get_nodes(int elem) const {
    return elem_nodes.at(elem);
  }

  // Similar to get_elem_dof_nodes, but use grid indices (i.e. cell, vert)
  // instead so we can facilitate mesh patching
  int get_cell_dof_verts(int cell, int* verts) const {
    int nodes[max_nnodes_per_element];
    int nnodes = get_elem_dof_nodes(cell_elems.at(cell), nodes);
    for (int i = 0; i < nnodes; i++) {
      verts[i] = get_node_vert(nodes[i]);
    }
    return nnodes;
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    this->grid.get_cell_verts(elem_cells.at(elem), nodes);
    for (int i = 0; i < corner_nodes_per_element; i++) {
      nodes[i] = vert_nodes.at(nodes[i]);
    }
  }

  inline void get_elem_corner_node_ranges(int elem, T* xloc_min,
                                          T* xloc_max) const {
    this->grid.get_cell_vert_ranges(elem_cells.at(elem), xloc_min, xloc_max);
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

  inline int get_elem_cell(int elem) const { return elem_cells.at(elem); }
  inline int get_node_vert(int node) const { return node_verts.at(node); }

  inline const std::map<int, int>& get_cell_elems() const { return cell_elems; }

  // Update the mesh as well as the element -> node mapping
  inline void update_mesh() {
    // First, we update on which vertex we have a dof node
    update_mesh_dof_nodes();

    // Then, we update the element -> node mapping
    // update_mesh_elem_node_mapping_spiral();
    update_elem_nodes_per_element();

    // if (push_strategy == PushStrategy::PerElement) {
    //   update_elem_nodes_per_element();
    // } else if (push_strategy == PushStrategy::PerNode) {
    //   update_elem_nodes_per_node();
    // } else {
    //   throw std::runtime_error("Unknown push_strategy");
    // }
  }

  inline const Map<int, int>& get_vert_nodes() const { return vert_nodes; }

  inline bool is_cut_elem(int elem) const {
    return static_cast<bool>(cut_elems.count(elem));
  }
  inline bool is_regular_stencil_elem(int elem) const {
    return static_cast<bool>(regular_stencil_elems.count(elem));
  }

  const inline std::set<int>& get_cut_elems() const { return cut_elems; }
  const inline std::set<int>& get_regular_stencil_elems() const {
    return regular_stencil_elems;
  }

  inline int get_elem_dir(int elem) const {
    return cell_dirs[elem_cells[elem]];
  }

 private:
  void populate_cut_elems_deprecated() {
    cut_elems.clear();

    for (int i = 0; i < num_elements; i++) {
      int verts[Grid::nverts_per_cell];
      T lsf_vals[Grid::nverts_per_cell];
      this->grid.get_cell_verts(get_elem_cell(i), verts);
      for (int j = 0; j < Grid::nverts_per_cell; j++) {
        lsf_vals[j] = lsf_dof[verts[j]];
      }

      std::sort(lsf_vals, lsf_vals + Grid::nverts_per_cell);
      if (lsf_vals[0] * lsf_vals[Grid::nverts_per_cell - 1] <= 0.0) {
        cut_elems.insert(i);
      }
    }
  }

  auto get_cut_cells_active_cells() {
    std::set<int> active_cells;
    std::set<int> cut_cells;

    int ncells = this->grid.get_num_cells();
    for (int cell = 0; cell < ncells; cell++) {
      VandermondeEvaluator<T, LSFMesh> eval(lsf_mesh, cell);

      // Get element LSF dofs
      T element_lsf[max_nnodes_per_element];
      constexpr int lsf_dim = 1;
      get_element_vars<T, lsf_dim, LSFMesh, max_nnodes_per_element,
                       spatial_dim>(lsf_mesh, cell, lsf_dof.data(),
                                    element_lsf);

      T data[Np_1d * Np_1d];
      algoim::xarray<T, spatial_dim> phi(
          data, algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
      get_phi_vals(eval, element_lsf, phi);

      T max_val = *std::max_element(data, data + Np_1d * Np_1d);
      T min_val = *std::min_element(data, data + Np_1d * Np_1d);

      bool is_interface_cell = (max_val * min_val < 0.0);
      if (is_interface_cell) {
        cut_cells.insert(cell);
      }

      bool is_active_cell = min_val < 0.0;
      if (is_active_cell) {
        active_cells.insert(cell);
      }
    }

    return std::make_tuple(cut_cells, active_cells);
  }

  // Update the dof nodes when the level-set function is updated, specifically,
  // the following variables and mappings are updated:
  //
  // Variables:
  //   - num nodes
  //
  // Mappings:
  //   - node -> vertex, vertex -> node
  //   - element -> cell, cell -> element
  //   - push direction of each cell
  //   - if element has regular stencil
  //   - node -> path elements
  //   - if element is cut element
  void update_mesh_dof_nodes() {
    elem_cells.clear();
    node_verts.clear();
    vert_nodes.clear();
    cell_elems.clear();
    cut_elems.clear();
    regular_stencil_elems.clear();
    node_patch_elems.clear();

    int ncells = this->grid.get_num_cells();

    // Get active cells and cut cells
    auto [cut_cells, active_cells] = get_cut_cells_active_cells();

    // Determine elements and dof nodes, and the mapping between elem <-> cell
    // and node <-> vert
    int node = 0;
    for (int c = 0; c < ncells; c++) {
      if (not active_cells.count(c)) continue;
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

    int num_nodes_old = num_nodes;
    int num_elems_old = num_elements;

    num_nodes = node_verts.size();
    num_elements = elem_cells.size();

    // Populate cell_elems
    for (int e = 0; e < num_elements; e++) {
      cell_elems[elem_cells[e]] = e;
    }

    // Populate cut elements
    for (int c = 0; c < ncells; c++) {
      if (cut_cells.count(c)) {
        cut_elems.insert(cell_elems.at(c));
      }
    }

    // Identify all the elements with regular stencils
    for (int i = 0; i < num_elements; i++) {
      int verts[max_nnodes_per_element];
      if (this->grid.template get_cell_ground_stencil<Np_1d>(get_elem_cell(i),
                                                             verts)) {
        bool is_regular_stencil = true;
        for (int j = 0; j < max_nnodes_per_element; j++) {
          if (vert_nodes.count(verts[j]) == 0) {
            is_regular_stencil = false;
            break;
          }
        }
        if (is_regular_stencil) {
          regular_stencil_elems.insert(i);
        }
      }
    }

    // Populate node -> element patches
    for (int e = 0; e < num_elements; e++) {
      int nodes[corner_nodes_per_element];
      get_elem_corner_nodes(e, nodes);
      for (int i = 0; i < corner_nodes_per_element; i++) {
        int n = nodes[i];
        node_patch_elems[n].insert(e);
      }
    }

#ifdef XCGD_DEBUG_MODE
    std::printf("[Debug]Updated mesh, nnodes: %d -> %d, nelems: %d -> %d\n",
                num_nodes_old, num_nodes, num_elems_old, num_elements);
#endif
  }

  void update_mesh_elem_node_mapping_spiral() {
    elem_nodes.clear();

    for (int elem = 0; elem < num_elements; elem++) {
      // Initialize elem -> nodes
      std::vector<int>& nodes = elem_nodes[elem];
      nodes.reserve(max_nnodes_per_element);

      /*
       * Next, we populate the nodes associated to the element in a spiral
       * following the order of R0 -> D0 -> L0 -> U0 -> R1 -> ..., where the
       * four letters represent right, down, left and upper set of nodes (spiral
       * legs), and the numbers represent the level of the spiral legs
       *
       *             ┌── U0 ───┐
       *
       *    ┌   x →  x .. x .. x   ..  x   ┐
       *    │   ↑                          │
       *    L0  x    x----x    x   ┐   x   │
       *    │   ↑    |elem|    ↓   │       │
       *    └   x    x----x    x   R0  x   │
       *        ↑              ↓   │       R1
       *        x  ← x  ← x  ← x   ┘   x   │
       *                                   │
       *        └─── D0 ──┘            x   │
       *                                   │
       *                   ..  x   x   x   ┘
       * */

      // First, we add the corner nodes, which are guaranteed to be the dof
      // nodes
      int cnodes[corner_nodes_per_element];
      get_elem_corner_nodes(elem, cnodes);
      for (int i = 0; i < corner_nodes_per_element; i++) {
        nodes.push_back(cnodes[i]);
      }

      /*
       * Next, we follow the spiral and attempt to populate nodes on the spiral
       * legs. This is when we might encounter a grid vertex on the spiral
       * pattern that is not a valid node due to one of the following two
       * reasons:
       *   1. the vertex is not an active node
       *   2. the vertex is outside the grid at all (for example, for a high
       *      degree element residing on the edge of the grid)
       * */

      // Np_1d = 4 -> levels = 0
      // Np_1d = 6 -> levels = 0, 1
      // Np_1d = 8 -> levels = 0, 1, 2
      // ...
      for (int level = 0; level < Np_1d / 2 - 1; level++) {
        // Prepare
        int exy[2] = {-1, -1};
        this->grid.get_cell_coords(elem_cells[elem], exy);
        int nnodes_per_leg = 2 * level + 3;

        // Work on vertices on each leg
        // clang-format off
        add_leg<Leg::RIGHT>(exy[0] + 2 + level, exy[1] + 1 + level, nnodes_per_leg, nodes);
        add_leg<Leg::DOWN>( exy[0] + 1 + level, exy[1] - 1 - level, nnodes_per_leg, nodes);
        add_leg<Leg::LEFT>( exy[0] - 1 - level, exy[1]     - level, nnodes_per_leg, nodes);
        add_leg<Leg::UP>(   exy[0]     - level, exy[1] + 2 + level, nnodes_per_leg, nodes);
        // clang-format on
      }
    }
  }

  // val | direction
  // 0   | +x
  // 1   | -x
  // 2   | +y
  // 3   | -y
  // 4   | +z
  // 5   | -z
  int get_push_dir(std::array<T, spatial_dim> grad) {
    double tmp = std::numeric_limits<double>::lowest();
    int dim = -1;
    for (int d = 0; d < spatial_dim; d++) {
      if (fabs(grad[d]) > tmp) {
        tmp = fabs(grad[d]);
        dim = d;
      }
    }
    return 2 * dim + (freal(grad[dim]) < 0.0 ? 0 : 1);
  }

  std::pair<int, int> parse_push_dir(int dir) {
    int dim = dir / spatial_dim;
    int sign = dir % spatial_dim == 0 ? 1 : -1;
    return {dim, sign};
  }

  void update_elem_nodes_per_element() {
    elem_nodes.clear();
    cell_dirs.clear();

    int ncells = this->grid.get_num_cells();

    // For each cell, get the push direction for the outlying ground stencil
    // vertices
    cell_dirs = std::vector<int>(ncells, -1);

    for (int c = 0; c < ncells; c++) {
      cell_dirs[c] = get_push_dir(interp_lsf_grad_at_cell(c));
    }

    for (int elem = 0; elem < num_elements; elem++) {
      // Initialize elem -> nodes
      std::vector<int>& nodes = elem_nodes[elem];
      nodes.reserve(max_nnodes_per_element);

      // Get ground stencils
      int verts[max_nnodes_per_element];
      int cell = elem_cells.at(elem);
      this->grid.template get_cell_ground_stencil<Np_1d>(cell, verts);

      // Get push direction
      auto [dim, sign] = parse_push_dir(cell_dirs[cell]);

      // Add nodes
      for (int i = 0; i < max_nnodes_per_element; i++) {
        int ixy[2] = {-1, -1};
        this->grid.get_vert_coords(verts[i], ixy);

        if (not vert_is_valid_dof_node(ixy[0], ixy[1])) {
          ixy[dim] += sign * Np_1d;

          if (not vert_is_valid_dof_node(ixy[0], ixy[1])) {
            continue;
          }
        }

        nodes.push_back(vert_nodes.at(this->grid.get_coords_vert(ixy)));
      }
    }
  }

  void update_elem_nodes_per_node() {
    elem_nodes.clear();
    cell_dirs.clear();

    // First, we loop over all elements, and we collect the verts that all
    // element ground stencil touches
    std::set<int> touched_verts;
    for (int e = 0; e < num_elements; e++) {
      int verts[max_nnodes_per_element];
      int cell = elem_cells.at(e);
      this->grid.template get_cell_ground_stencil<Np_1d>(cell, verts);
      for (int i = 0; i < max_nnodes_per_element; i++) {
        touched_verts.insert(verts[i]);
      }
    }

    // Next, we create the push rule by populating a 1-1 mapping from touched
    // verts to active verts. For touched verts that are active verts, the
    // active verts they map to is likely themselves. The map is only
    // non-trivial for those inactive touched verts. The push direction is
    // determined based on the largest component of the LSF gradient evaluated
    // at the touched verts.
    std::map<int, int> touched_to_active_vert_mapping;
    for (int touched_vert : touched_verts) {
      /* If touched vert is an active vert */
      if (vert_nodes.count(touched_vert)) {
        touched_to_active_vert_mapping[touched_vert] = touched_vert;
      } else {
        auto [dim, sign] =
            parse_push_dir(interp_lsf_grad_at_vert(touched_vert));

        int ixy[spatial_dim] = {-1, -1};
        this->grid.get_vert_coords(touched_vert, ixy);
        ixy[dim] += sign * Np_1d;

        int candidate_vert = this->grid.get_coords_vert(ixy);

        /* If candidate vert is an active vert */
        if (vert_nodes.count(candidate_vert)) {
          touched_to_active_vert_mapping[touched_vert] = candidate_vert;
        } else {
          touched_to_active_vert_mapping[touched_vert] =
              -1;  // does not have a map, hence the stencil is degenerate
        }
      }
    }

    // Next, we populate the dof mapping as well as elem dir
    for (int elem = 0; elem < num_elements; elem++) {
      // Initialize elem -> nodes
      std::vector<int>& nodes = elem_nodes[elem];
      nodes.reserve(max_nnodes_per_element);

      // Get ground stencil
      int touched_verts[max_nnodes_per_element];
      int cell = elem_cells.at(elem);
      this->grid.template get_cell_ground_stencil<Np_1d>(cell, touched_verts);

      // Add nodes
      for (int i = 0; i < max_nnodes_per_element; i++) {
        int target_vert = touched_to_active_vert_mapping[touched_verts[i]];
        if (target_vert > 0) {
          nodes.push_back(vert_nodes.at(target_vert));
        }
      }
    }
  }

  // Given the lsf dof, interpolate the gradient of the lsf at the centroid
  // a cell using bilinear quad element
  std::array<T, spatial_dim> interp_lsf_grad_at_cell(int cell) {
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

  std::array<T, spatial_dim> interp_lsf_grad_at_vert(int vert) {
    int ixy[spatial_dim];
    this->grid.get_vert_coords(vert, ixy);
    const int* nxy = this->grid.get_nxy();

    std::array<int, Grid::nverts_per_cell> verts;

    if (ixy[0] < nxy[0] and ixy[1] < nxy[1]) {
      verts[0] = this->grid.get_coords_vert(ixy[0], ixy[1]);
      verts[1] = this->grid.get_coords_vert(ixy[0] + 1, ixy[1]);
      verts[2] = this->grid.get_coords_vert(ixy[0] + 1, ixy[1] + 1);
      verts[3] = this->grid.get_coords_vert(ixy[0], ixy[1] + 1);
    } else if (ixy[0] == nxy[0] and ixy[1] < nxy[1]) {
      verts[0] = this->grid.get_coords_vert(ixy[0] - 1, ixy[1]);
      verts[1] = this->grid.get_coords_vert(ixy[0], ixy[1]);
      verts[2] = this->grid.get_coords_vert(ixy[0], ixy[1] + 1);
      verts[3] = this->grid.get_coords_vert(ixy[0] - 1, ixy[1] + 1);
    } else if (ixy[0] < nxy[0] and ixy[1] == nxy[1]) {
      verts[0] = this->grid.get_coords_vert(ixy[0], ixy[1] - 1);
      verts[1] = this->grid.get_coords_vert(ixy[0] + 1, ixy[1] - 1);
      verts[2] = this->grid.get_coords_vert(ixy[0] + 1, ixy[1]);
      verts[3] = this->grid.get_coords_vert(ixy[0], ixy[1]);
    } else if (ixy[0] == nxy[0] and ixy[1] == nxy[1]) {
      verts[0] = this->grid.get_coords_vert(ixy[0] - 1, ixy[1] - 1);
      verts[1] = this->grid.get_coords_vert(ixy[0], ixy[1] - 1);
      verts[2] = this->grid.get_coords_vert(ixy[0], ixy[1]);
      verts[3] = this->grid.get_coords_vert(ixy[0] - 1, ixy[1]);
    } else {
      throw std::runtime_error("unreachable");
    }

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
    int dir = cell_dirs[cell];
    int dim = dir / spatial_dim;
    int sign = dir % spatial_dim == 0 ? 1 : -1;

    // Adjust nodes
    for (int index = 0; index < max_nnodes_per_element; index++) {
      int vert = verts[index];
      if (vert_nodes.count(vert) == 0) {
        int vert_coords[spatial_dim] = {-1, -1};
        this->grid.get_vert_coords(vert, vert_coords);
        vert_coords[dim] += sign * Np_1d;
        verts[index] = this->grid.get_coords_vert(vert_coords);
      }
    }

    // check  if the stencil is a valid stencil
    /*
    #ifdef XCGD_DEBUG_MODE
        for (int index = 0; index < max_nnodes_per_element; index++) {
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
    */
  }

  bool vert_is_valid_dof_node(int ix, int iy) {
    // If the vertex is in the cut mesh
    if constexpr (node_strategy == NodeStrategy::AllowOutsideLSF) {
      if (not this->grid.is_valid_vert(ix, iy)) {
        return false;
      }
      return vert_nodes.count(this->grid.get_coords_vert(ix, iy)) != 0;

    }

    // If the vertex is in the cut mesh and within the region defined by the
    // level set function, i.e. phi <= 0
    else {  // node_strategy == NodeStrategy::StrictlyInsideLSF
      return (lsf_dof[this->grid.get_coords_vert(ix, iy)] <= 0.0);
    }
  };

  enum class Leg { LEFT, RIGHT, UP, DOWN };

  /**
   * @brief Attempt to add verts on a spiral leg as dof nodes
   *
   * @param ix, iy coordinartes for the beginning vert of the spiral leg
   */
  template <Leg leg>
  void add_leg(int ix, int iy, int nnodes_per_leg, std::vector<int>& nodes) {
    for (int i = 0; i < nnodes_per_leg; i++) {
      // Case 2 candidate:
      int ix_2 = ix, iy_2 = iy;
      if constexpr (leg == Leg::RIGHT) {
        ix_2 -= Np_1d;
      } else if constexpr (leg == Leg::LEFT) {
        ix_2 += Np_1d;
      } else if constexpr (leg == Leg::DOWN) {
        iy_2 += Np_1d;
      } else {  // leg == Leg::UP
        iy_2 -= Np_1d;
      }

      // Case 3 candidate:
      int ix_3 = ix, iy_3 = iy;
      if constexpr (leg == Leg::RIGHT) {
        iy_3 += Np_1d;
      } else if constexpr (leg == Leg::LEFT) {
        iy_3 -= Np_1d;
      } else if constexpr (leg == Leg::DOWN) {
        ix_3 += Np_1d;
      } else {  // leg == Leg::UP
        ix_3 -= Np_1d;
      }

      // Case 1: stencil vertex hit, nice and easy!
      if (vert_is_valid_dof_node(ix, iy)) {
        nodes.push_back(vert_nodes[this->grid.get_coords_vert(ix, iy)]);

      }
      // Case 2: stencil vert miss, but the symmetric vert on the other side of
      // the stencil hit
      else if (vert_is_valid_dof_node(ix_2, iy_2)) {
        nodes.push_back(vert_nodes[this->grid.get_coords_vert(ix_2, iy_2)]);
      }
      // Case 3: Case 2 still miss, but this vert is a corner vert, hence we
      // have another symmetric vert to try
      else if (i == nnodes_per_leg - 1 and vert_is_valid_dof_node(ix_3, iy_3)) {
        nodes.push_back(vert_nodes[this->grid.get_coords_vert(ix_3, iy_3)]);
      }

      // Move to the next candidate vertex
      if constexpr (leg == Leg::RIGHT) {
        iy--;  // move down for next candidate
      } else if constexpr (leg == Leg::LEFT) {
        iy++;  // move up for next candidate
      } else if constexpr (leg == Leg::DOWN) {
        ix--;   // move left for next candidate
      } else {  // leg == Leg::UP
        ix++;   // move right for next candidate
      }
    }
  }

  LSFMesh lsf_mesh;

  int num_nodes = -1;
  int num_elements = -1;

  // level set function values at vertices of the ground grid
  std::vector<T> lsf_dof;

  // PushStrategy push_strategy;

  /* Below is information about the element-to-node topology, at each update to
   * the mesh, the dof nodes associated to an element may change from verts to
   * verts, as well as the elements themselves might correspond to different
   * cells.
   *
   * Recall that cells and verts are defined on the ground grid, and elements
   * and nodes are defined on the dynamic mesh itself */

  Map<int, std::vector<int>> elem_nodes;

  // indices of vertices that are dof nodes, i.e. vertices that have active
  // degrees of freedom
  Map<int, int> node_verts;  // node -> vert
  Map<int, int> vert_nodes;  // vert -> node

  // indices of cells that are dof elements, i.e. cells that have active degrees
  // of freedom
  std::vector<int> elem_cells;    // elem -> cell
  std::map<int, int> cell_elems;  // cell-> elem

  // push direction for each cell
  std::vector<int> cell_dirs;

  // Whether the element is cut element or interior element
  std::set<int> cut_elems;

  // Whether the element has the regular stencil
  // elements far from the cut or grid boundaries usually have regular stencils
  std::set<int> regular_stencil_elems;

  // node -> element patches
  std::map<int, std::set<int>> node_patch_elems;
};

/*
 * This class implements a finite cell mesh based on Galerkin difference
 * approach. The finite cell mesh comprises an underground grid and a level-set
 * function that defines an analysis domain implicitly. Unlike the CutMesh,
 * finite cell mesh has degree of freedom outside the analysis domain.
 *
 * Ref:
 *   - Schillinger, D., Ruess, M. The Finite Cell Method: A Review in the
 * Context of Higher-Order Structural Analysis of CAD and Image-Based Geometric
 * Models. Arch Computat Methods Eng 22, 391–455 (2015).
 * https://doi.org/10.1007/s11831-014-9115-y
 * */
template <typename T, int Np_1d, class Grid_ = StructuredGrid2D<T>>
class FiniteCellMesh final : public GDMeshBase<T, Np_1d, Grid_> {
 private:
  using MeshBase = GDMeshBase<T, Np_1d, Grid_>;
  using LSFMesh = GridMesh<T, Np_1d, Grid_>;

  template <class T1, class T2>
  using Map =
      std::unordered_map<T1, T2>;  // TODO(fyc): std::map and
                                   // std::unordered_map, which is better?

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::max_nnodes_per_element;
  using MeshBase::spatial_dim;
  using typename MeshBase::Grid;
  static constexpr bool is_cut_mesh = true;
  static constexpr bool is_finite_cell_mesh = true;

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
  FiniteCellMesh(const Grid& grid, const Func& lsf)
      : MeshBase(grid), lsf_mesh(grid), lsf_dof(lsf_mesh.get_num_nodes()) {
    for (int i = 0; i < lsf_dof.size(); i++) {
      T xloc[spatial_dim];
      lsf_mesh.get_node_xloc(i, xloc);
      lsf_dof[i] = lsf(xloc);
    }
    update_mesh();
  }

  FiniteCellMesh(const Grid& grid)
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
   * @return nodes dof node indices, length: nnodes
   */
  int get_elem_dof_nodes(int elem, int* nodes) const {
    const auto& nodes_v = elem_nodes.at(elem);
    int nnodes = nodes_v.size();
    for (int i = 0; i < nnodes; i++) {
      nodes[i] = nodes_v[i];
    }
    return nnodes;
  }

  std::set<int> get_node_patch_elems(int node) const {
    return node_patch_elems.at(node);
  }

  const Map<int, std::vector<int>>& get_elem_nodes() const {
    return elem_nodes;
  }
  const std::vector<int>& get_nodes(int elem) const {
    return elem_nodes.at(elem);
  }

  // Similar to get_elem_dof_nodes, but use grid indices (i.e. cell, vert)
  // instead so we can facilitate mesh patching
  int get_cell_dof_verts(int cell, int* verts) const {
    int nodes[max_nnodes_per_element];
    int nnodes = get_elem_dof_nodes(cell_elems.at(cell), nodes);
    for (int i = 0; i < nnodes; i++) {
      verts[i] = get_node_vert(nodes[i]);
    }
    return nnodes;
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    this->grid.get_cell_verts(elem_cells.at(elem), nodes);
    for (int i = 0; i < corner_nodes_per_element; i++) {
      nodes[i] = vert_nodes.at(nodes[i]);
    }
  }

  inline void get_elem_corner_node_ranges(int elem, T* xloc_min,
                                          T* xloc_max) const {
    this->grid.get_cell_vert_ranges(elem_cells.at(elem), xloc_min, xloc_max);
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

  inline int get_elem_cell(int elem) const { return elem_cells.at(elem); }
  inline int get_node_vert(int node) const { return node_verts.at(node); }

  inline const std::map<int, int>& get_cell_elems() const { return cell_elems; }

  // Update the mesh as well as the element -> node mapping
  inline void update_mesh() {
    elem_nodes.clear();
    node_verts.clear();
    vert_nodes.clear();
    elem_cells.clear();
    cell_elems.clear();
    cut_elems.clear();
    regular_stencil_elems.clear();
    node_patch_elems.clear();

    int ncells = this->grid.get_num_cells();

    // Populate elem_cells and cut_elems
    for (int cell = 0; cell < ncells; cell++) {
      // Get Bernstein coefficients
      VandermondeEvaluator<T, LSFMesh> eval(lsf_mesh, cell);
      T element_lsf[max_nnodes_per_element];
      constexpr int lsf_dim = 1;
      get_element_vars<T, lsf_dim, LSFMesh, max_nnodes_per_element,
                       spatial_dim>(lsf_mesh, cell, lsf_dof.data(),
                                    element_lsf);
      int constexpr data_len = Np_1d * Np_1d;
      T data[data_len];
      algoim::xarray<T, spatial_dim> phi(
          data, algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
      get_phi_vals(eval, element_lsf, phi);

      // Sort Bernstein coefficients in ascending order
      std::sort(data, data + data_len);
      if (data[0] < 0.0) {  // cell is an element
        elem_cells.push_back(cell);
        if (data[data_len - 1] > 0.0) {  // cell is a cut element
          cut_elems.insert(elem_cells.size() - 1);
        }
      }
    }

    // Get number of elements and populate cell_elems
    num_elements = elem_cells.size();
    for (int e = 0; e < num_elements; e++) {
      cell_elems[elem_cells[e]] = e;
    }

    // Populate vert_nodes, node_verts, elem_nodes and regular_stencil_elems
    int node = 0;
    for (int e = 0; e < num_elements; e++) {
      // Get ground stencils
      int verts[max_nnodes_per_element];
      int cell = elem_cells.at(e);
      bool is_regular_stencil =
          this->grid.template get_cell_ground_stencil<Np_1d>(cell, verts);
      if (is_regular_stencil) {
        regular_stencil_elems.insert(e);
      }
      for (int i = 0; i < max_nnodes_per_element; i++) {
        int vert = verts[i];
        if (not vert_nodes.count(vert)) {
          vert_nodes[vert] = node;
          node_verts[node] = vert;
          node++;
        }
      }
    }

    // Populate elem_nodes
    for (int e = 0; e < num_elements; e++) {
      // Get ground stencils
      int verts[max_nnodes_per_element];
      int cell = elem_cells.at(e);
      this->grid.template get_cell_ground_stencil<Np_1d>(cell, verts);
      for (int i = 0; i < max_nnodes_per_element; i++) {
        elem_nodes[e].push_back(vert_nodes.at(verts[i]));
      }
    }

    // Get number of nodes
    num_nodes = vert_nodes.size();

    // Populate node_patch_elems
    for (int e = 0; e < num_elements; e++) {
      int nodes[corner_nodes_per_element];
      get_elem_corner_nodes(e, nodes);
      for (int i = 0; i < corner_nodes_per_element; i++) {
        int n = nodes[i];
        node_patch_elems[n].insert(e);
      }
    }
  }

  inline const Map<int, int>& get_vert_nodes() const { return vert_nodes; }

  inline bool is_cut_elem(int elem) const {
    return static_cast<bool>(cut_elems.count(elem));
  }
  inline bool is_regular_stencil_elem(int elem) const {
    return static_cast<bool>(regular_stencil_elems.count(elem));
  }

  const inline std::set<int>& get_cut_elems() const { return cut_elems; }
  const inline std::set<int>& get_regular_stencil_elems() const {
    return regular_stencil_elems;
  }

  // Caution: this is a dummy function that does not return meaningful value
  inline int get_elem_dir(int elem) const { return 0; }

 private:
  LSFMesh lsf_mesh;

  int num_nodes = -1;
  int num_elements = -1;

  // level set function values at vertices of the ground grid
  std::vector<T> lsf_dof;

  /* Below is information about the element-to-node topology, at each update to
   * the mesh, the dof nodes associated to an element may change from verts to
   * verts, as well as the elements themselves might correspond to different
   * cells.
   *
   * Recall that cells and verts are defined on the ground grid, and elements
   * and nodes are defined on the dynamic mesh itself */

  Map<int, std::vector<int>> elem_nodes;

  // indices of vertices that are dof nodes, i.e. vertices that have active
  // degrees of freedom
  Map<int, int> node_verts;  // node -> vert
  Map<int, int> vert_nodes;  // vert -> node

  // indices of cells that are dof elements, i.e. cells that have active degrees
  // of freedom
  std::vector<int> elem_cells;    // elem -> cell
  std::map<int, int> cell_elems;  // cell-> elem

  // Whether the element is cut element or interior element
  std::set<int> cut_elems;

  // Whether the element has the regular stencil
  // elements far from the cut or grid boundaries usually have regular stencils
  std::set<int> regular_stencil_elems;

  // node -> element patches
  std::map<int, std::set<int>> node_patch_elems;
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
 */
template <typename T, class Mesh>
void get_computational_coordinates_limits(const Mesh& mesh, int elem, T* xi_min,
                                          T* xi_max) {
  static_assert(Mesh::is_gd_mesh, "function only works with a GD mesh");
  int constexpr spatial_dim = Mesh::spatial_dim;
  T xy_min[spatial_dim], xy_max[spatial_dim];
  T uv_min[spatial_dim], uv_max[spatial_dim];
  mesh.get_elem_node_ranges(elem, xy_min, xy_max);
  mesh.get_elem_corner_node_ranges(elem, uv_min, uv_max);

  T hx = (uv_max[0] - uv_min[0]) / (xy_max[0] - xy_min[0]);
  T hy = (uv_max[1] - uv_min[1]) / (xy_max[1] - xy_min[1]);

  T cx = (2.0 * uv_min[0] - xy_min[0] - xy_max[0]) / (xy_max[0] - xy_min[0]);
  T dx = 2.0 * hx;
  T cy = (2.0 * uv_min[1] - xy_min[1] - xy_max[1]) / (xy_max[1] - xy_min[1]);
  T dy = 2.0 * hy;

  xi_min[0] = cx;
  xi_min[1] = cy;
  xi_max[0] = cx + dx;
  xi_max[1] = cy + dy;
}

#endif  // XCGD_GD_MESH_H
