#pragma once

#include <numeric>
#include <vector>

#include "element_utils.h"
#include "quadrature_multipoly.hpp"
#include "utils/linalg.h"
#include "utils/loggers.h"

// This class implements a functor that evaluate basis values and basis
// gradients given a set of computational coordinates
template <typename T, class Mesh>
class VandermondeEvaluatorDeprecated {
 private:
  static_assert(Mesh::is_gd_mesh, "VandermondeEvaluator requires a GD Mesh");
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int Np_1d = Mesh::Np_1d;
  static constexpr int Np = Mesh::max_nnodes_per_element;
  static constexpr int Nk = Mesh::max_nnodes_per_element;

 public:
  VandermondeEvaluatorDeprecated(const Mesh& mesh, int elem) : Ck(Nk * Np) {
    int nodes[Nk];
    std::vector<T> xpows(Np_1d), ypows(Np_1d);

    mesh.get_elem_dof_nodes(elem, nodes);

    T xloc_min[spatial_dim], xloc_max[spatial_dim];
    mesh.get_elem_node_ranges(elem, xloc_min, xloc_max);

    for (int i = 0; i < Nk; i++) {
      T xloc[spatial_dim];
      mesh.get_node_xloc(nodes[i], xloc);

      // make x, y in [-1, 1]
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
    direct_inverse(Nk, Ck.data());
  }

  // Evaluate the shape function and derivatives given a quadrature point
  template <typename T2>
  void operator()(const T2* pt, T2* N, T2* Nxi,
                  T2* Nxixi = (T2*)nullptr) const {
    std::vector<T2> xpows(Np_1d), ypows(Np_1d), dxpows(Np_1d), dypows(Np_1d),
        dx2pows(Np_1d), dy2pows(Np_1d);

    for (int ii = 0; ii < Np_1d; ii++) {
      xpows[ii] = pow(pt[0], ii);
      ypows[ii] = pow(pt[1], ii);
      dxpows[ii] = ii > 0 ? T(ii) * pow(pt[0], ii - 1) : T(0.0);
      dypows[ii] = ii > 0 ? T(ii) * pow(pt[1], ii - 1) : T(0.0);
      dx2pows[ii] = ii > 1 ? T(ii) * T(ii - 1) * pow(pt[0], ii - 2) : T(0.0);
      dy2pows[ii] = ii > 1 ? T(ii) * T(ii - 1) * pow(pt[1], ii - 2) : T(0.0);
    }

    for (int i = 0; i < Nk; i++) {
      if (N) {
        N[i] = 0.0;
      }
      if (Nxi) {
        Nxi[spatial_dim * i] = 0.0;
        Nxi[spatial_dim * i + 1] = 0.0;
      }
      if (Nxixi) {
        Nxixi[spatial_dim * spatial_dim * i] = 0.0;
        Nxixi[spatial_dim * spatial_dim * i + 1] = 0.0;
        Nxixi[spatial_dim * spatial_dim * i + 2] = 0.0;
        Nxixi[spatial_dim * spatial_dim * i + 3] = 0.0;
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
          if (Nxixi) {
            Nxixi[spatial_dim * spatial_dim * i] +=
                Ck[idx + Nk * i] * dx2pows[j] * ypows[k];
            Nxixi[spatial_dim * spatial_dim * i + 1] +=
                Ck[idx + Nk * i] * dxpows[j] * dypows[k];
            Nxixi[spatial_dim * spatial_dim * i + 2] +=
                Ck[idx + Nk * i] * dxpows[j] * dypows[k];
            Nxixi[spatial_dim * spatial_dim * i + 3] +=
                Ck[idx + Nk * i] * xpows[j] * dy2pows[k];
          }
        }
      }
    }
  }

 private:
  std::vector<T> Ck;
};

// An adaptive Vandermonde evaluator. Adaptive meaning we might locally drop
// order when there's not enough nodes for certain elements. Number of nodes per
// element is no longer constant in this case.
template <typename T, class Mesh>
class VandermondeEvaluator {
 private:
  static_assert(Mesh::is_gd_mesh, "VandermondeEvaluator requires a GD Mesh");
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int Np_1d = Mesh::Np_1d;
  static constexpr int p = Np_1d - 1;

 public:
  /**
   * @param mesh mesh object
   * @param elem element index
   * @param reorder_nodes if true, reorder the nodes by the vertex index
   */
  VandermondeEvaluator(const Mesh& mesh, int elem, bool reorder_nodes = false)
      : mesh(mesh), reorder_nodes(reorder_nodes) {
    int nodes[Np_1d * Np_1d];
    nnodes = mesh.get_elem_dof_nodes(elem, nodes);

    std::vector<int> perm, _;
    if (reorder_nodes) {
      construct_permutation(nodes, perm, _);
    }

    Ck.resize(nnodes * nnodes);

    std::vector<std::pair<int, int>> verts(nnodes, {-1, -1});
    for (int i = 0; i < verts.size(); i++) {
      int ixy[2] = {-1, -1};
      mesh.get_grid().get_vert_coords(mesh.get_node_vert(nodes[i]), ixy);
      verts[i] = {ixy[0], ixy[1]};
    }

    int dir = mesh.get_elem_dir(elem);
    int dim = dir / spatial_dim;
    pterms = verts_to_pterms(verts, dim == 1);

    std::vector<T> xpows(Np_1d), ypows(Np_1d);

    T xloc_min[spatial_dim], xloc_max[spatial_dim], xi_max[spatial_dim];
    mesh.get_elem_node_ranges(elem, xloc_min, xloc_max);
    get_computational_coordinates_limits(mesh, elem, xi_min, xi_max);

    for (int d = 0; d < spatial_dim; d++) {
      xi_h[d] = xi_max[d] - xi_min[d];
    }

    for (int i = 0; i < nnodes; i++) {
      T xloc[spatial_dim];

      if (perm.size()) {
        mesh.get_node_xloc(nodes[perm[i]], xloc);
      } else {
        mesh.get_node_xloc(nodes[i], xloc);
      }

      // make x, y in the Vandermonde frame, i.e. [-1, 1]^d
      T x = -1.0 + 2.0 * (xloc[0] - xloc_min[0]) / (xloc_max[0] - xloc_min[0]);
      T y = -1.0 + 2.0 * (xloc[1] - xloc_min[1]) / (xloc_max[1] - xloc_min[1]);

      for (int ii = 0; ii < Np_1d; ii++) {
        xpows[ii] = pow(x, ii);
        ypows[ii] = pow(y, ii);
      }

      for (int col = 0; col < nnodes; col++) {
        auto indices = pterms[col];
        Ck[i + nnodes * col] = xpows[indices.first] * ypows[indices.second];
      }
    }

    double cond;
    direct_inverse(nnodes, Ck.data(), &cond, '1');
    cond = 1.0 / cond;

    VandermondeCondLogger::add(elem, cond);
  }

  /**
   * @brief Evaluate the shape function and derivatives given a quadrature point
   *
   * @tparam T2 numeric type
   * @param elem [in] element index, only used if reorder_nodes is specified
   * @param pt [in] quadrature points in the reference frame, i.e. [0, 1]^d
   * @param N [out] shape function evaluations for each dof
   * @param Nxi [out] shape function derivatives for each dof
   * @param Nxixi [out] shape function Hessians for each dof, optional
   */
  template <typename T2>
  void operator()(int elem, const T2* pt, T2* N, T2* Nxi,
                  T2* Nxixi = (T2*)nullptr) const {
    static constexpr int max_nnodes_per_element = Mesh::max_nnodes_per_element;

    std::vector<int> _, iperm;
    if (reorder_nodes) {
      int nodes[Np_1d * Np_1d];
      int nnodes_this = mesh.get_elem_dof_nodes(elem, nodes);
      if (nnodes != nnodes_this) {
        throw std::runtime_error(
            "Attempting to reuse the VandermondeEvaluator on an element that "
            "is "
            "different from the element that constructed the evaluator, but "
            "the "
            "number of nodes does not match (get " +
            std::to_string(nnodes_this) + ", expect " + std::to_string(nnodes) +
            ")");
      }
      construct_permutation(nodes, _, iperm);
    }

    std::vector<T2> xpows(Np_1d), ypows(Np_1d), dxpows(Np_1d), dypows(Np_1d),
        dx2pows(Np_1d), dy2pows(Np_1d);

    T2 xi = pt[0] * xi_h[0] + xi_min[0];
    T2 eta = pt[1] * xi_h[1] + xi_min[1];
    for (int ii = 0; ii < Np_1d; ii++) {
      xpows[ii] = pow(xi, ii);
      ypows[ii] = pow(eta, ii);
      dxpows[ii] = ii > 0 ? T(ii) * xi_h[0] * pow(xi, ii - 1) : T(0.0);
      dypows[ii] = ii > 0 ? T(ii) * xi_h[1] * pow(eta, ii - 1) : T(0.0);
      dx2pows[ii] =
          ii > 1 ? T(ii) * T(ii - 1) * xi_h[0] * xi_h[0] * pow(xi, ii - 2)
                 : T(0.0);
      dy2pows[ii] =
          ii > 1 ? T(ii) * T(ii - 1) * xi_h[1] * xi_h[1] * pow(eta, ii - 2)
                 : T(0.0);
    }

    if (N) {
      std::fill(N, N + max_nnodes_per_element, T2(0.0));
    }
    if (Nxi) {
      std::fill(Nxi, Nxi + spatial_dim * max_nnodes_per_element, T2(0.0));
    }
    if (Nxixi) {
      std::fill(Nxixi,
                Nxixi + spatial_dim * spatial_dim * max_nnodes_per_element,
                T2(0.0));
    }

    for (int i = 0; i < nnodes; i++) {
      for (int row = 0; row < nnodes; row++) {
        auto [j, k] = pterms[row];
        int index = iperm.size() ? row + nnodes * iperm[i] : row + nnodes * i;
        if (N) {
          // N = C^T v
          // Ni = C[j, i] v[j]
          N[i] += Ck[index] * xpows[j] * ypows[k];  // (row, i) entry
        }
        if (Nxi) {
          Nxi[spatial_dim * i] += Ck[index] * dxpows[j] * ypows[k];
          Nxi[spatial_dim * i + 1] += Ck[index] * xpows[j] * dypows[k];
        }
        if (Nxixi) {
          Nxixi[spatial_dim * spatial_dim * i] +=
              Ck[index] * dx2pows[j] * ypows[k];
          Nxixi[spatial_dim * spatial_dim * i + 1] +=
              Ck[index] * dxpows[j] * dypows[k];
          Nxixi[spatial_dim * spatial_dim * i + 2] +=
              Ck[index] * dxpows[j] * dypows[k];
          Nxixi[spatial_dim * spatial_dim * i + 3] +=
              Ck[index] * xpows[j] * dy2pows[k];
        }
      }
    }
  }

 private:
  /**
   * @brief Construct the permutation of nodes so Vandermonde matrix V and its
   *  inverse matrix C using the permuated ordering of nodes. This is useful
   *  when we share one instance of VandermondeEvaluator for multiple elements
   *  that share the same stencil pattern but with potentially different local
   *  ordering of the stencil nodes.
   *
   * @param nodes [in] nodes associated to the element
   * @param perm [out] j = perm[i]: i-th node externally is j-th node internally
   * @param iperm [out] i = iperm[j]: j-th node internally is i-th node
   * extrnally
   */
  void construct_permutation(const int* nodes, std::vector<int>& perm,
                             std::vector<int>& iperm) const {
    perm.resize(nnodes);
    iperm.resize(nnodes);
    std::iota(perm.begin(), perm.end(),
              0);  // set values0 , 1, 2, ...
    std::sort(perm.begin(), perm.end(), [this, &nodes](int p1, int p2) {
      return this->mesh.get_node_vert(nodes[p1]) <
             this->mesh.get_node_vert(nodes[p2]);
    });

    // iperm is the inverted permutation that maps back from internal ordering
    // to external ordering of nodes
    for (int i = 0; i < nnodes; i++) {
      iperm[perm[i]] = i;
    }
  }

  const Mesh& mesh;

  int nnodes;
  std::vector<T> Ck;
  std::vector<std::pair<int, int>> pterms;
  T xi_min[spatial_dim], xi_h[spatial_dim];

  bool reorder_nodes = false;
};
template <typename T, typename T2, class Mesh>
void get_phi_vals(const VandermondeEvaluator<T, Mesh>& eval,
                  const T2 element_dof[],
                  algoim::xarray<T2, Mesh::spatial_dim>& phi) {
  algoim::bernstein::bernsteinInterpolate<Mesh::spatial_dim>(
      [&](const algoim::uvector<T2, Mesh::spatial_dim>& xi) {  // xi in [0, 1]
        T2 N[Mesh::max_nnodes_per_element];
        // xi in [xi_min, xi_max]
        eval(-1, xi.data(), N, (T2*)nullptr);
        T2 val;
        interp_val_grad<T2, Mesh::spatial_dim, Mesh::max_nnodes_per_element>(
            element_dof, N, nullptr, &val, nullptr);
        return val;
      },
      phi);
}
