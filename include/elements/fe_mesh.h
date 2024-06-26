#ifndef XCGD_FE_MESH_H
#define XCGD_FE_MESH_H

#include "element_commons.h"

template <typename T, int spatial_dim, int nodes_per_element_>
class FEMesh final
    : public MeshBase<T, spatial_dim, nodes_per_element_, nodes_per_element_> {
 private:
  using MeshBase =
      MeshBase<T, spatial_dim, nodes_per_element_, nodes_per_element_>;

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::nodes_per_element;
  using MeshBase::spatial_dim;

  FEMesh(int num_elements, int num_nodes, int* element_nodes, T* xloc)
      : num_elements(num_elements),
        num_nodes(num_nodes),
        element_nodes(element_nodes),
        xloc(xloc) {}

  inline int get_num_nodes() const { return num_nodes; }
  inline int get_num_elements() const { return num_elements; }
  inline void get_node_xloc(int node, T* xloc_) const {
    for (int d = 0; d < spatial_dim; d++) {
      xloc_[d] = xloc[spatial_dim * node + d];
    }
  }

  inline void get_elem_dof_nodes(int elem, int* nodes) const {
    for (int i = 0; i < nodes_per_element; i++) {
      nodes[i] = element_nodes[elem * nodes_per_element + i];
    }
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    get_elem_dof_nodes(elem, nodes);
  }

 private:
  int num_elements, num_nodes;
  int* element_nodes;
  T* xloc;
};

#endif  // XCGD_FE_MESH_H