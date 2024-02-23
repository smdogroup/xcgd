#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "utils/linalg.h"

template <typename T>
class Grid {
  // Get node indices given an element index, return number of nodes
  virtual int get_elem_nodes(int elem, int* nodes) const = 0;

  // Get nodal coordinates given a nodal index
  virtual void get_node_xloc(int node, T* xloc) const = 0;
};

// The unstructured ground grid
template <typename T, int spatial_dim>
class UnstructuredGrid final : public Grid<T> {
 public:
  UnstructuredGrid(int num_elements, int num_nodes, int nnodes_per_element,
                   const int* element_nodes, const T* xloc)
      : num_elements(num_elements),
        num_nodes(num_nodes),
        nnodes_per_element(nnodes_per_element),
        element_nodes(element_nodes),
        xloc(xloc) {}

  int get_elem_nodes(int elem, int* nodes) const {
    if (nodes) {
      for (int n = 0; n < nnodes_per_element; n++) {
        nodes[n] = element_nodes[nnodes_per_element * elem + n];
      }
    }
    return nnodes_per_element;
  }

  void get_node_xloc(int node, T* xloc) const {
    if (xloc) {
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xloc[spatial_dim * node + d];
      }
    }
  }

 private:
  int num_elements, num_nodes, nnodes_per_element;
  const int* element_nodes;
  const T* xloc;
};

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
class GDMesh2D {
 public:
  static_assert(Np_1d % 2 == 0);
  static constexpr int nodes_per_element = Np_1d * Np_1d;

  GDMesh2D(StructuredGrid2D<T>& grid) : grid(grid) {
    const int* nxy = grid.get_nxy();
    for (int d = 0; d < grid.spatial_dim; d++) {
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

  // Get the stencil nodes given a gd element (i.e. a grid cell)
  void get_elem_dof_nodes(int elem, int* nodes) {
    constexpr int q = Np_1d / 2;
    int eij[grid.spatial_dim];
    grid.get_elem_coords(elem, eij);
    const int* nxy = grid.get_nxy();
    for (int d = 0; d < grid.spatial_dim; d++) {
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
  StructuredGrid2D<T>& grid;
};

/**
 * Galerkin difference basis given a set of stencil nodes
 *
 * @tparam Np_1d number of nodes along one dimension, number of stencil nodes
 *               should be Np_1d^2, Np_1d >= 2, Np_1d should be even
 */
template <typename T, int Np_1d>
class GDBasis2D {
 public:
  static_assert(Np_1d % 2 == 0);
  static constexpr int spatial_dim = 2;
  static constexpr int nodes_per_element = Np_1d * Np_1d;

  GDBasis2D(GDMesh2D<T, Np_1d>& mesh) : mesh(mesh) {}

  // TODO: make pt contain all quadrature points
  static void eval_basis_grad(int elem, const T* pt, T* N, T* Nxi) {
    static constexpr int Np = nodes_per_element;
    static constexpr int Nk = nodes_per_element;

    T Ck[Nk * Np];
    std::vector<T> xpows(Np_1d);
    std::vector<T> ypows(Np_1d);

    for (int i = 0; i < Nk; i++) {
      T x = 2.0 * (i % Np_1d) / (Np_1d - 1) - 1.0;
      T y = 2.0 * (i / Np_1d) / (Np_1d - 1) - 1.0;

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

 private:
  GDMesh2D<T, Np_1d>& mesh;
};

template <int q>
class GD2DQuadrature {};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
