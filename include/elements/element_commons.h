#ifndef XCGD_ELEMENT_COMMONS_H
#define XCGD_ELEMENT_COMMONS_H

#include <set>
#include <vector>

#include "a2dcore.h"
#include "utils/exceptions.h"
#include "utils/misc.h"

/**
 * @brief The abstract base class for a Galerkin (finite element or Galerkin
 * difference) mesh
 */
template <typename T, int spatial_dim_, int max_nnodes_per_element_,
          int corner_nodes_per_element_>
class MeshBase {
 public:
  static constexpr int spatial_dim = spatial_dim_;
  static constexpr int max_nnodes_per_element = max_nnodes_per_element_;
  static constexpr int corner_nodes_per_element = corner_nodes_per_element_;
  static constexpr bool is_gd_mesh = false;

  virtual int get_num_nodes() const = 0;
  virtual int get_num_elements() const = 0;
  virtual void get_node_xloc(int node, T* xloc) const = 0;
  virtual int get_elem_dof_nodes(int elem, int* nodes) const = 0;
  virtual void get_elem_corner_nodes(int elem, int* nodes) const = 0;

  // Given a dof node, find all element patches for the node, useful for stress
  // patch recover
  virtual std::set<int> get_node_patch_elems(int node) const = 0;
};

template <typename T>
class StructuredGrid2D;

/**
 * @brief The abstract base class for a 2D Galerkin difference mesh
 */
template <typename T, int Np_1d_, class Grid_ = StructuredGrid2D<T>>
class GDMeshBase : public MeshBase<T, 2, Np_1d_ * Np_1d_, 4> {
 private:
  using MeshBase_ = MeshBase<T, 2, Np_1d_ * Np_1d_, 4>;
  static_assert(Np_1d_ % 2 == 0);

 public:
  using Grid = Grid_;
  using MeshBase_::corner_nodes_per_element;
  using MeshBase_::max_nnodes_per_element;
  using MeshBase_::spatial_dim;
  static constexpr bool is_gd_mesh = true;
  static constexpr bool is_cut_mesh = false;
  static constexpr int Np_1d = Np_1d_;

  GDMeshBase(const Grid& grid) : grid(grid) {
    grid.template check_grid_compatibility<Np_1d>();
  }

  inline const Grid& get_grid() const { return grid; }

  virtual void get_elem_corner_node_ranges(int elem, T* xloc_min,
                                           T* xloc_max) const = 0;

  /**
   * @brief Get the bounding box of dof nodes
   */
  void get_elem_node_ranges(int elem, T* xloc_min, T* xloc_max) const {
    int nodes[max_nnodes_per_element];
    int nnodes = this->get_elem_dof_nodes(elem, nodes);

    // [x0, ..., xN, y0, ..., yN, ...]
    std::vector<T> coords(spatial_dim * nnodes);

    for (int i = 0; i < nnodes; i++) {
      T xloc[spatial_dim];
      this->get_node_xloc(nodes[i], xloc);
      for (int d = 0; d < spatial_dim; d++) {
        coords[i + d * nnodes] = xloc[d];
      }
    }

    for (int d = 0; d < spatial_dim; d++) {
      xloc_min[d] =
          *std::min_element(&coords[d * nnodes], &coords[d * nnodes] + nnodes,
                            [](T& a, T& b) { return freal(a) < freal(b); });

      xloc_max[d] =
          *std::max_element(&coords[d * nnodes], &coords[d * nnodes] + nnodes,
                            [](T& a, T& b) { return freal(a) < freal(b); });
    }
  }

  /**
   * @brief Get the bounding box of all nodes that belong to the patch elements
   * for a given patch assembly node
   */
  void get_node_patch_elems_node_ranges(int node, T* xloc_min,
                                        T* xloc_max) const {
    std::set<int> patch_elems = this->get_node_patch_elems(node);
    std::set<int> patch_nodes;

    for (int e : patch_elems) {
      int nodes[corner_nodes_per_element];
      this->get_elem_corner_nodes(e, nodes);
      for (int i = 0; i < corner_nodes_per_element; i++) {
        patch_nodes.insert(nodes[i]);
      }
    }

    int nnodes = patch_nodes.size();

    // [x0, ..., xN, y0, ..., yN, ...]
    std::vector<T> coords(spatial_dim * nnodes);

    int index = 0;
    for (int n : patch_nodes) {
      T xloc[spatial_dim];
      this->get_node_xloc(n, xloc);
      for (int d = 0; d < spatial_dim; d++) {
        coords[index + d * nnodes] = xloc[d];
      }
      index++;
    }

    for (int d = 0; d < spatial_dim; d++) {
      xloc_min[d] =
          *std::min_element(&coords[d * nnodes], &coords[d * nnodes] + nnodes,
                            [](T& a, T& b) { return freal(a) < freal(b); });

      xloc_max[d] =
          *std::max_element(&coords[d * nnodes], &coords[d * nnodes] + nnodes,
                            [](T& a, T& b) { return freal(a) < freal(b); });
    }
  }

  std::vector<int> get_left_boundary_nodes(double tol = 1e-10) const {
    const T* xy0 = this->grid.get_xy0();
    std::vector<int> nodes;
    for (int i = 0; i < this->get_num_nodes(); i++) {
      T xloc[spatial_dim];
      this->get_node_xloc(i, xloc);
      if (xloc[0] - xy0[0] < tol) {
        nodes.push_back(i);
      }
    }
    return nodes;
  }

  std::vector<int> get_lower_boundary_nodes(double tol = 1e-10) const {
    const T* xy0 = this->grid.get_xy0();
    std::vector<int> nodes;
    for (int i = 0; i < this->get_num_nodes(); i++) {
      T xloc[spatial_dim];
      this->get_node_xloc(i, xloc);
      if (xloc[1] - xy0[1] < tol) {
        nodes.push_back(i);
      }
    }
    return nodes;
  }

  std::vector<int> get_right_boundary_nodes(double tol = 1e-10) const {
    const T* xy0 = this->grid.get_xy0();
    const T* lxy = this->grid.get_lxy();
    std::vector<int> nodes;
    for (int i = 0; i < this->get_num_nodes(); i++) {
      T xloc[spatial_dim];
      this->get_node_xloc(i, xloc);
      if (xy0[0] + lxy[0] - xloc[0] < tol) {
        nodes.push_back(i);
      }
    }
    return nodes;
  }

  std::vector<int> get_upper_boundary_nodes(double tol = 1e-10) const {
    const T* xy0 = this->grid.get_xy0();
    const T* lxy = this->grid.get_lxy();
    std::vector<int> nodes;
    for (int i = 0; i < this->get_num_nodes(); i++) {
      T xloc[spatial_dim];
      this->get_node_xloc(i, xloc);
      if (xy0[1] + lxy[1] - xloc[1] < tol) {
        nodes.push_back(i);
      }
    }
    return nodes;
  }

 protected:
  const Grid& grid;
};

enum class QuadPtType { INNER, SURFACE };

// A quadrature class needs to implement the following method
template <typename T, QuadPtType quad_type_ = QuadPtType::INNER>
class QuadratureBase {
 public:
  static constexpr QuadPtType quad_type = quad_type_;
  virtual int get_quadrature_pts(int elem, std::vector<T>& pts,
                                 std::vector<T>& wts,
                                 std::vector<T>& optional_wns) const = 0;
};

/**
 * @brief The base class for a Galerkin (finite element or Galerkin
 * difference) basis
 *
 * Note:
 *  This class serves as a "template" for implementation, it is not an abstract
 *  base class, so don't use the pointer/reference of this type to invoke
 *  overloading methods in derived classes
 */
template <typename T, class Mesh_>
class BasisBase {
 public:
  using Mesh = Mesh_;
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int max_nnodes_per_element = Mesh::max_nnodes_per_element;
  static constexpr bool is_gd_basis = false;

  virtual void eval_basis_grad(int elem, const std::vector<T>& pts,
                               std::vector<T>& N,
                               std::vector<T>& Nxi) const = 0;
};

#endif  // XCGD_ELEMENT_COMMONS_H
