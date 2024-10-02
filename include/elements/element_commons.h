#ifndef XCGD_ELEMENT_COMMONS_H
#define XCGD_ELEMENT_COMMONS_H

#include <vector>

#include "a2dcore.h"
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
  virtual int get_elem_dof_nodes(
      int elem, int* nodes,
      std::vector<std::vector<bool>>* pstencil = nullptr) const = 0;
  virtual void get_elem_corner_nodes(int elem, int* nodes) const = 0;
};

template <typename T>
class StructuredGrid2D;

/**
 * @brief The abstract base class for a 2D Galerkin difference mesh
 */
template <typename T, int Np_1d_>
class GDMeshBase : public MeshBase<T, 2, Np_1d_ * Np_1d_, 4> {
 private:
  using MeshBase_ = MeshBase<T, 2, Np_1d_ * Np_1d_, 4>;
  static_assert(Np_1d_ % 2 == 0);

 public:
  using Grid = StructuredGrid2D<T>;
  using MeshBase_::corner_nodes_per_element;
  using MeshBase_::max_nnodes_per_element;
  using MeshBase_::spatial_dim;
  static constexpr bool is_gd_mesh = true;
  static constexpr bool is_cut_mesh = false;
  static constexpr int Np_1d = Np_1d_;

  GDMeshBase(const Grid& grid) : grid(grid) { check_grid_compatibility(grid); }

  inline const Grid& get_grid() const { return grid; }

  virtual void get_elem_vert_ranges(int elem, T* xloc_min,
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

 protected:
  void check_grid_compatibility(const Grid& grid) const {
    const int* nxy = grid.get_nxy();
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

  /**
   * @brief For a GD element, get the stencil associated to the ground grid,
   * regardless the optional boundary defined by the level set function
   */
  void get_cell_ground_stencil(int cell, int* verts) const {
    constexpr int q = Np_1d / 2;
    int eij[spatial_dim];
    grid.get_cell_coords(cell, eij);
    const int* nxy = grid.get_nxy();
    for (int d = 0; d < spatial_dim; d++) {
      if (eij[d] < q - 1) {
        eij[d] = q - 1;
      } else if (eij[d] > nxy[d] - q) {
        eij[d] = nxy[d] - q;
      }
    }

    int index = 0;
    for (int j = 0; j < Np_1d; j++) {
      for (int i = 0; i < Np_1d; i++, index++) {
        verts[index] =
            grid.get_coords_vert(eij[0] - q + 1 + i, eij[1] - q + 1 + j);
      }
    }
  };

  const Grid& grid;
};

// A quadrature class needs to implement the following method
template <typename T>
class QuadratureBase {
 public:
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
