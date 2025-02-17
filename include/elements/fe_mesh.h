#ifndef XCGD_FE_MESH_H
#define XCGD_FE_MESH_H

#include <map>
#include <set>

#include "element_commons.h"

template <typename T, int spatial_dim_, int max_nnodes_per_element_>
class FEMesh final : public MeshBase<T, spatial_dim_, max_nnodes_per_element_,
                                     max_nnodes_per_element_> {
 private:
  using MeshBase_ = MeshBase<T, spatial_dim_, max_nnodes_per_element_,
                             max_nnodes_per_element_>;

 public:
  using MeshBase_::corner_nodes_per_element;
  using MeshBase_::max_nnodes_per_element;
  using MeshBase_::spatial_dim;

  FEMesh(int num_elements, int num_nodes, int* element_nodes, T* xloc)
      : num_elements(num_elements),
        num_nodes(num_nodes),
        element_nodes(element_nodes),
        xloc(xloc) {
    for (int e = 0; e < num_elements; e++) {
      for (int i = 0; i < max_nnodes_per_element; i++) {
        int n = element_nodes[e * max_nnodes_per_element + i];
        node_patch_elems[n].insert(e);
      }
    }
  }

  inline int get_num_nodes() const { return num_nodes; }
  inline int get_num_elements() const { return num_elements; }
  inline void get_node_xloc(int node, T* xloc_) const {
    for (int d = 0; d < spatial_dim; d++) {
      xloc_[d] = xloc[spatial_dim * node + d];
    }
  }

  inline int get_elem_dof_nodes(int elem, int* nodes) const {
    for (int i = 0; i < max_nnodes_per_element; i++) {
      nodes[i] = element_nodes[elem * max_nnodes_per_element + i];
    }
    return max_nnodes_per_element;
  }

  // Note: for high-order internal nodes, this only returns the single element
  // that the internal node belongs to
  std::set<int> get_node_patch_elems(int node) const {
    return node_patch_elems.at(node);
  }

  inline void get_elem_corner_nodes(int elem, int* nodes) const {
    get_elem_dof_nodes(elem, nodes);
  }

 private:
  int num_elements, num_nodes;
  int* element_nodes;
  T* xloc;
  std::map<int, std::set<int>> node_patch_elems;  // node -> element patches
};

#endif  // XCGD_FE_MESH_H
