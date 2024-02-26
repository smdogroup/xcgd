#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

#include "utils/linalg.h"
#include "utils/misc.h"

// template <typename T>
// class Grid {
//   // Get node indices given an element index, return number of nodes
//   virtual int get_elem_nodes(int elem, int* nodes) const = 0;

//   // Get nodal coordinates given a nodal index
//   virtual void get_node_xloc(int node, T* xloc) const = 0;
// };

// // The unstructured ground grid
// template <typename T, int spatial_dim>
// class UnstructuredGrid final : public Grid<T> {
//  public:
//   UnstructuredGrid(int num_elements, int num_nodes, int nnodes_per_element,
//                    const int* element_nodes, const T* xloc)
//       : num_elements(num_elements),
//         num_nodes(num_nodes),
//         nnodes_per_element(nnodes_per_element),
//         element_nodes(element_nodes),
//         xloc(xloc) {}

//   int get_elem_nodes(int elem, int* nodes) const {
//     if (nodes) {
//       for (int n = 0; n < nnodes_per_element; n++) {
//         nodes[n] = element_nodes[nnodes_per_element * elem + n];
//       }
//     }
//     return nnodes_per_element;
//   }

//   void get_node_xloc(int node, T* xloc) const {
//     if (xloc) {
//       for (int d = 0; d < spatial_dim; d++) {
//         xloc[d] = xloc[spatial_dim * node + d];
//       }
//     }
//   }

//  private:
//   int num_elements, num_nodes, nnodes_per_element;
//   const int* element_nodes;
//   const T* xloc;
// };

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

template <typename T, int spatial_dim_, int nodes_per_element_>
class MeshBase {
 public:
  static constexpr int spatial_dim = spatial_dim_;
  static constexpr int nodes_per_element = nodes_per_element_;

  virtual int get_num_elements() const = 0;
  virtual void get_node_xloc(int node, T* xloc) const = 0;
  virtual void get_elem_dof_nodes(int elem, int* nodes) const = 0;
};

template <typename T, int spatial_dim, int nodes_per_element_>
class FEMesh final : public MeshBase<T, spatial_dim, nodes_per_element_> {
 private:
  using MeshBase = MeshBase<T, spatial_dim, nodes_per_element_>;

 public:
  using MeshBase::nodes_per_element;
  using MeshBase::spatial_dim;

  FEMesh(int num_elements, int num_nodes, int* element_nodes, T* xloc)
      : num_elements(num_elements),
        num_nodes(num_nodes),
        element_nodes(element_nodes),
        xloc(xloc) {}

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

 private:
  int num_elements, num_nodes;
  int* element_nodes;
  T* xloc;
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

  GDMesh2D(Grid& grid) : grid(grid) {
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
  Grid& grid;
};

template <typename T, class Mesh_>
class BasisBase {
 public:
  using Mesh = Mesh_;
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int nodes_per_element = Mesh::nodes_per_element;

  BasisBase(Mesh& mesh) : mesh(mesh) {}

  virtual void eval_basis_grad(int elem, const T* pt, T* N, T* Nxi) = 0;

  inline int get_num_elements() const { return mesh.get_num_elements(); };
  inline void get_node_xloc(int node, T* xloc) const {
    mesh.get_node_xloc(node, xloc);
  };
  inline void get_elem_dof_nodes(int elem, int* nodes) const {
    mesh.get_elem_dof_nodes(elem, nodes);
  };

 protected:
  Mesh& mesh;
};

/**
 * Galerkin difference basis given a set of stencil nodes
 *
 * @tparam Np_1d number of nodes along one dimension, number of stencil nodes
 *               should be Np_1d^2, Np_1d >= 2, Np_1d should be even
 */
template <typename T, int Np_1d>
class GDBasis2D final : public BasisBase<T, GDMesh2D<T, Np_1d>> {
 private:
  using BasisBase = BasisBase<T, GDMesh2D<T, Np_1d>>;

 public:
  using BasisBase::nodes_per_element;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;

 private:
  static constexpr int Np = Mesh::nodes_per_element;
  static constexpr int Nk = Mesh::nodes_per_element;

 public:
  GDBasis2D(Mesh& mesh) : BasisBase(mesh) {}

  // TODO: make pt contain all quadrature points
  void eval_basis_grad(int elem, const T* pt, T* N, T* Nxi) {
    T Ck[Nk * Np];
    std::vector<T> xpows(Np_1d);
    std::vector<T> ypows(Np_1d);

    int nodes[Nk];
    this->mesh.get_elem_dof_nodes(elem, nodes);

    std::vector<double> xloc_min(spatial_dim,
                                 std::numeric_limits<double>::max());
    std::vector<double> xloc_max(spatial_dim,
                                 std::numeric_limits<double>::min());
    for (int i = 0; i < Nk; i++) {
      T xloc[spatial_dim];
      this->mesh.get_node_xloc(nodes[i], xloc);
      for (int d = 0; d < spatial_dim; d++) {
        xloc_min[d] = std::min(xloc_min[d], freal(xloc[d]));
        xloc_max[d] = std::max(xloc_max[d], freal(xloc[d]));
      }
    }

    for (int i = 0; i < Nk; i++) {
      T xloc[spatial_dim];
      this->mesh.get_node_xloc(nodes[i], xloc);

      T x = -1.0 + 2.0 * (xloc[0] - xloc_min[0]) / (xloc_max[0] - xloc_min[0]);
      T y = -1.0 + 2.0 * (xloc[1] - xloc_min[1]) / (xloc_max[1] - xloc_min[1]);

      for (int ii = 0; ii < Np_1d; ii++) {
        xpows[ii] = pow(x, ii);
        ypows[ii] = pow(y, ii);
      }

      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          Ck[i + Np * idx] = xpows[j] * ypows[k];  // (i, idx) entry
        }
      }
    }

    direct_inverse(Nk, Ck);

    std::vector<T> dxpows(Np_1d);
    std::vector<T> dypows(Np_1d);

    T x = pt[0];
    T y = pt[1];

    for (int ii = 0; ii < Np_1d; ii++) {
      xpows[ii] = pow(x, ii);
      ypows[ii] = pow(y, ii);
      dxpows[ii] = T(ii) * pow(x, ii - 1);
      dypows[ii] = T(ii) * pow(y, ii - 1);
    }

    for (int i = 0; i < Nk; i++) {
      if (N) {
        N[i] = 0.0;
      }
      if (Nxi) {
        Nxi[spatial_dim * i] = 0.0;
        Nxi[spatial_dim * i + 1] = 0.0;
      }

      for (int j = 0; j < Np_1d; j++) {
        for (int k = 0; k < Np_1d; k++) {
          int idx = j * Np_1d + k;
          if (N) {
            N[i] += Ck[idx + Nk * i] * xpows[j] * ypows[k];
          }
          if (Nxi) {
            Nxi[spatial_dim * i] += Ck[idx + Nk * i] * dxpows[j] * ypows[k];
            Nxi[spatial_dim * i + 1] += Ck[idx + Nk * i] * xpows[j] * dypows[k];
          }
        }
      }
    }
  }
};

template <int Np_1d>
class GDQuadrature2D {
 public:
  static constexpr int num_quadrature_pts = Np_1d * Np_1d;

  template <typename T>
  static T get_quadrature_pt(int k, T pt[]) {
    pt[0] = 0.1;
    pt[1] = 0.1;
    return 1.0;
  }
};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
