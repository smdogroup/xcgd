#ifndef XCGD_ELEMENT_COMMONS_H
#define XCGD_ELEMENT_COMMONS_H

#include <vector>

#include "a2dcore.h"

/**
 * @brief The abstract base class for a Galerkin (finite element or Galerkin
 * difference) mesh
 */
template <typename T, int spatial_dim_, int nodes_per_element_,
          int corner_nodes_per_element_>
class MeshBase {
 public:
  static constexpr int spatial_dim = spatial_dim_;
  static constexpr int nodes_per_element = nodes_per_element_;
  static constexpr int corner_nodes_per_element = corner_nodes_per_element_;

  virtual int get_num_nodes() const = 0;
  virtual int get_num_elements() const = 0;
  virtual void get_node_xloc(int node, T* xloc) const = 0;
  virtual void get_elem_dof_nodes(int elem, int* nodes) const = 0;
  virtual void get_elem_corner_nodes(int elem, int* nodes) const = 0;
};

// A quadrature class needs to implement the following method
template <typename T>
class QuadratureBase {
 public:
  virtual int get_quadrature_pts(int elem, std::vector<T>& pts,
                                 std::vector<T>& wts) const = 0;
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
  static constexpr int nodes_per_element = Mesh::nodes_per_element;
  static constexpr bool is_gd_basis = false;

  virtual void eval_basis_grad(int elem, const std::vector<T>& pts,
                               std::vector<T>& N,
                               std::vector<T>& Nxi) const = 0;
};

#endif  // XCGD_ELEMENT_COMMONS_H