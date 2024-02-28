#ifndef XCGD_GD_COMMONS_H
#define XCGD_GD_COMMONS_H

// The structured ground grid
template <typename T>
class StructuredGrid2D final {
 public:
  static constexpr int spatial_dim = 2;
  StructuredGrid2D(const int* nxy_, const T* lxy_) {
    for (int d = 0; d < spatial_dim; d++) {
      nxy[d] = nxy_[d];
      lxy[d] = lxy_[d];
    }
  }

  // Compute element/node coordinates <-> element/node index
  inline int get_coords_node(int ni, int nj) const {
    return ni + (nxy[0] + 1) * nj;
  }
  inline int get_coords_node(const int* nij) const {
    return nij[0] + (nxy[0] + 1) * nij[1];
  }
  inline void get_node_coords(int node, int* nij) const {
    nij[0] = node % (nxy[0] + 1);
    nij[1] = node / (nxy[0] + 1);
  }
  inline int get_coords_elem(int ei, int ej) const { return ei + nxy[0] * ej; }
  inline int get_coords_elem(const int* eij) const {
    return eij[0] + nxy[0] * eij[1];
  }
  inline void get_elem_coords(int elem, int* eij) const {
    eij[0] = elem % nxy[0];
    eij[1] = elem / nxy[0];
  }

  int get_elem_nodes(int elem, int* nodes) const {
    if (nodes) {
      int nij[spatial_dim] = {elem % nxy[0], elem / nxy[1]};
      nodes[0] = get_coords_node(nij);
      nodes[1] = nodes[0] + 1;
      nodes[2] = nodes[0] + nxy[0] + 1;
      nodes[3] = nodes[2] + 1;
    }
    return 4;
  }

  void get_node_xloc(int node, T* xloc) const {
    if (xloc) {
      int nij[spatial_dim];
      get_node_coords(node, nij);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = lxy[d] * T(nij[d]) / T(nxy[d]);
      }
    }
  }

  const int* get_nxy() const { return nxy; };
  const T* get_lxy() const { return lxy; };

 private:
  int nxy[spatial_dim];
  T lxy[spatial_dim];
};

template <typename T, int Np_1d>
class GDMesh2D final : public MeshBase<T, 2, Np_1d * Np_1d> {
 private:
  using MeshBase = MeshBase<T, 2, Np_1d * Np_1d>;
  static_assert(Np_1d % 2 == 0);
  using Grid = StructuredGrid2D<T>;

 public:
  using MeshBase::nodes_per_element;
  using MeshBase::spatial_dim;

  GDMesh2D(const Grid& grid) : grid(grid) {
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

  int get_num_nodes() const {
    const int* nxy = grid.get_nxy();
    int nnodes = 1;
    for (int d = 0; d < spatial_dim; d++) {
      nnodes *= nxy[d] + 1;
    }
    return nnodes;
  }

  int get_num_elements() const {
    const int* nxy = grid.get_nxy();
    int nelems = 1;
    for (int d = 0; d < spatial_dim; d++) {
      nelems *= nxy[d];
    }
    return nelems;
  }

  inline void get_node_xloc(int node, T* xloc) const {
    grid.get_node_xloc(node, xloc);
  }

  /**
   * @brief Get the stencil nodes given a gd element (i.e. a grid cell)
   *
   * @param elem element index
   * @param nodes dof node indices, length: nodes_per_element
   */
  void get_elem_dof_nodes(int elem, int* nodes) const {
    constexpr int q = Np_1d / 2;
    int eij[spatial_dim];
    grid.get_elem_coords(elem, eij);
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
        nodes[index] =
            grid.get_coords_node(eij[0] - q + 1 + i, eij[1] - q + 1 + j);
      }
    }
  };

 private:
  const Grid& grid;
};

#endif  // XCGD_GD_COMMONS_H