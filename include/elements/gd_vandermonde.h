#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include "dual.hpp"
#include "element_commons.h"
#include "element_utils.h"
#include "gaussquad.hpp"
#include "gd_mesh.h"
#include "quadrature_multipoly.hpp"
#include "utils/linalg.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/testing.h"

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

enum class SurfQuad { LEFT, RIGHT, BOTTOM, TOP, NA };

template <typename T, int Np_1d, QuadPtType quad_type = QuadPtType::INNER,
          SurfQuad surf_quad = SurfQuad::NA, class Mesh = GridMesh<T, Np_1d>>
class GDGaussQuadrature2D final : public QuadratureBase<T, quad_type> {
 private:
  // algoim limit, see gaussquad.hpp
  static_assert(Np_1d <= algoim::GaussQuad::p_max);  // algoim limit
  static_assert((quad_type == QuadPtType::SURFACE) xor
                    (surf_quad == SurfQuad::NA),
                "quad_type and surf_quad are not compatible");

 public:
  GDGaussQuadrature2D(const Mesh& mesh, const std::set<int> elements = {})
      : mesh(mesh), elements(elements) {
    for (int i = 0; i < Np_1d; i++) {
      pts_1d[i] = algoim::GaussQuad::x(Np_1d, i);  // in (0, 1)
      wts_1d[i] = algoim::GaussQuad::w(Np_1d, i);
    }
  }

  /**
   * @brief Get the quadrature points and weights
   *
   * @param elem element index
   * @param pts concatenation of [ξ, η] for each quadrature point, size:
   * num_quad * spatial_dim
   * @param wts quadrature weights, size: num_quad
   * @param ns if is surface quadrature, stores the outer normal vector in
   * reference frame
   * @return int num_quad
   */
  int get_quadrature_pts(int elem, std::vector<T>& pts, std::vector<T>& wts,
                         std::vector<T>& ns) const {
    if (elements.size() and !elements.count(elem)) {
      return 0;
    }
    int constexpr spatial_dim = Mesh::spatial_dim;

    if constexpr (quad_type == QuadPtType::INNER) {
      static constexpr int num_quad_pts = Np_1d * Np_1d;
      pts.resize(spatial_dim * num_quad_pts);
      wts.resize(num_quad_pts);
      for (int q = 0; q < num_quad_pts; q++) {  // q = i * Np_1d + j
        int i = q / Np_1d;
        int j = q % Np_1d;
        pts[q * spatial_dim] = pts_1d[i];
        pts[q * spatial_dim + 1] = pts_1d[j];
        wts[q] = wts_1d[i] * wts_1d[j];
      }
      return num_quad_pts;
    } else {  // QuadPtType::SURFACE
      static constexpr int num_quad_pts = Np_1d;
      pts.resize(spatial_dim * num_quad_pts);
      ns.resize(spatial_dim * num_quad_pts);
      wts.resize(num_quad_pts);
      for (int q = 0; q < num_quad_pts; q++) {
        wts[q] = wts_1d[q];
        if constexpr (surf_quad == SurfQuad::LEFT) {
          pts[q * spatial_dim] = T(0.0);
          pts[q * spatial_dim + 1] = pts_1d[q];
          ns[q * spatial_dim] = T(-1.0);
          ns[q * spatial_dim + 1] = T(0.0);
        } else if constexpr (surf_quad == SurfQuad::RIGHT) {
          pts[q * spatial_dim] = T(1.0);
          pts[q * spatial_dim + 1] = pts_1d[q];
          ns[q * spatial_dim] = T(1.0);
          ns[q * spatial_dim + 1] = T(0.0);
        } else if constexpr (surf_quad == SurfQuad::BOTTOM) {
          pts[q * spatial_dim] = pts_1d[q];
          pts[q * spatial_dim + 1] = T(0.0);
          ns[q * spatial_dim] = T(0.0);
          ns[q * spatial_dim + 1] = T(-1.0);
        } else if constexpr (surf_quad == SurfQuad::TOP) {
          pts[q * spatial_dim] = pts_1d[q];
          pts[q * spatial_dim + 1] = T(1.0);
          ns[q * spatial_dim] = T(0.0);
          ns[q * spatial_dim + 1] = T(1.0);
        }
      }
      return num_quad_pts;
    }
  }

 private:
  const Mesh& mesh;
  std::set<int> elements;
  std::array<T, Np_1d> pts_1d, wts_1d;
};

// Forward declaration
template <typename T, class Mesh>
class GDBasis2D;

template <typename T, int Np_1d, QuadPtType quad_type = QuadPtType::INNER,
          class Grid = StructuredGrid2D<T>>
class GDLSFQuadrature2D final : public QuadratureBase<T, quad_type> {
 private:
  // algoim limit, see gaussquad.hpp
  static_assert(Np_1d <= algoim::GaussQuad::p_max);  // algoim limit
  using GridMesh_ = GridMesh<T, Np_1d, Grid>;
  using CutMesh_ = CutMesh<T, Np_1d, Grid>;
  using Basis = GDBasis2D<T, CutMesh_>;

  constexpr static int spatial_dim = Basis::spatial_dim;
  constexpr static int max_nnodes_per_element = Basis::max_nnodes_per_element;

 public:
  GDLSFQuadrature2D(const CutMesh_& mesh)
      : mesh(mesh), lsf_mesh(mesh.get_lsf_mesh()) {}

  /**
   * @brief Get the quadrature points and weights
   *
   * @param elem element index
   * @param pts concatenation of [ξ, η] for each quadrature point, size:
   * num_quad * spatial_dim
   * @param wts quadrature weights, size: num_quad
   * @param ns if is surface quadrature, stores the normal vector in reference
   * frame
   * @return int num_quad
   */
  int get_quadrature_pts(int elem, std::vector<T>& pts, std::vector<T>& wts,
                         std::vector<T>& ns) const {
    // this is the element index in lsf mesh
    int cell = mesh.get_elem_cell(elem);

    // Create the functor that evaluates the interpolation given an arbitrary
    // point within the computational coordinates
    VandermondeEvaluator<T, GridMesh_> eval(lsf_mesh, cell);

    // Get element LSF dofs
    const std::vector<T>& lsf_dof = mesh.get_lsf_dof();
    T element_lsf[max_nnodes_per_element];
    constexpr int lsf_dim = 1;
    get_element_vars<T, lsf_dim, GridMesh_, Basis>(lsf_mesh, cell,
                                                   lsf_dof.data(), element_lsf);

    // Get quadrature points and weights
    getQuadrature(element_lsf, eval, pts, wts, ns);

    return wts.size();
  }

  /**
   * @brief Get the quadrature points and weights and derivatives w.r.t.
   * element LSF dof
   *
   * @param elem element index
   * @param pts concatenation of [ξ, η] for each quadrature point, size:
   * num_quad * spatial_dim
   * @param wts quadrature weights, size: num_quad
   * @param pts_grad concatenation of [∂ξ/∂φ0, ∂η/∂φ0, ∂ξ/∂φ1, ∂η/∂φ1, ...] for
   * each quadrature point, size: num_quad * spatial_dim *
   * max_nnodes_per_element
   * @param wts_grad concatenation of [∂w/∂φ0, ∂w/∂φ1, ...] for each quadrature
   * point, size: num_quad * max_nnodes_per_element
   */
  int get_quadrature_pts_grad(int elem, std::vector<T>& pts,
                              std::vector<T>& wts, std::vector<T>& ns,
                              std::vector<T>& pts_grad,
                              std::vector<T>& wts_grad) const {
    // this is the element index in lsf mesh
    int cell = mesh.get_elem_cell(elem);

    // Create the functor that evaluates the interpolation given an arbitrary
    // point within the computational coordinates
    VandermondeEvaluator<T, GridMesh_> eval(lsf_mesh, cell);

    // Get element LSF dofs
    const std::vector<T>& lsf_dof = mesh.get_lsf_dof();
    T element_lsf[max_nnodes_per_element];
    constexpr int lsf_dim = 1;
    get_element_vars<T, lsf_dim, GridMesh_, Basis>(lsf_mesh, cell,
                                                   lsf_dof.data(), element_lsf);

    // Get quadrature points and weights
    getQuadrature(element_lsf, eval, pts, wts, ns);

    int num_quad_pts = wts.size();

    // Get quadrature gradients
    duals::dual<T> element_lsf_d[max_nnodes_per_element];
    for (int i = 0; i < max_nnodes_per_element; i++) {
      element_lsf_d[i].rpart(element_lsf[i]);
      element_lsf_d[i].dpart(0.0);
    }

    pts_grad.clear();
    wts_grad.clear();
    // wns_grad.clear();
    pts_grad.resize(num_quad_pts * spatial_dim * max_nnodes_per_element);
    wts_grad.resize(num_quad_pts * max_nnodes_per_element);
    // wns_grad.resize(num_quad_pts * spatial_dim * max_nnodes_per_element);

    for (int i = 0; i < max_nnodes_per_element; i++) {
      element_lsf_d[i].dpart(1.0);
      std::vector<T> dpts, dwts, dwns;
      getQuadrature(element_lsf_d, eval, dpts, dwts, dwns);
      element_lsf_d[i].dpart(0.0);

      if (dwts.size() != num_quad_pts) {
        char msg[256];
        std::snprintf(
            msg, 256,
            "number of quadrature points for ∂pt/∂φ_%d is inconsistent. Got "
            "%ld, expect %d.",
            i, dwts.size(), num_quad_pts);
        throw std::runtime_error(msg);
      }

      for (int q = 0; q < num_quad_pts; q++) {
        int index = q * max_nnodes_per_element + i;
        wts_grad[index] = dwts[q];
        for (int d = 0; d < spatial_dim; d++) {
          pts_grad[index * spatial_dim + d] = dpts[q * spatial_dim + d];
          // wns_grad[index * spatial_dim + d] = dwns[q * spatial_dim + d];
        }
      }
    }

    return num_quad_pts;
  }

 private:
  template <typename T2>
  void get_phi_vals(const VandermondeEvaluator<T, GridMesh_>& eval,
                    const T2 element_dof[],
                    algoim::xarray<T2, spatial_dim>& phi) const {
    algoim::bernstein::bernsteinInterpolate<spatial_dim>(
        [&](const algoim::uvector<T2, spatial_dim>& xi) {  // xi in [0, 1]
          T2 N[max_nnodes_per_element];
          // xi in [xi_min, xi_max]
          eval(-1, xi.data(), N, (T2*)nullptr);
          T2 val;
          interp_val_grad<T2, Basis>(element_dof, N, nullptr, &val, nullptr);
          return val;
        },
        phi);
  }

  template <typename T2>
  void getQuadrature(const T2 element_lsf[],
                     const VandermondeEvaluator<T, GridMesh_>& eval,
                     std::vector<T>& pts, std::vector<T>& wts,
                     std::vector<T>& ns) const {
    constexpr bool is_dual = is_specialization<T2, duals::dual>::value;

    // Obtain the Bernstein polynomial representation of the level-set
    // function
    T2 data[Np_1d * Np_1d];
    algoim::xarray<T2, spatial_dim> phi(
        data, algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
    get_phi_vals(eval, element_lsf, phi);

    pts.clear();
    wts.clear();
    ns.clear();

    algoim::ImplicitPolyQuadrature<spatial_dim, T2> ipquad(phi);
    if constexpr (quad_type == QuadPtType::INNER) {
      ipquad.integrate(
          algoim::AutoMixed, Np_1d,
          [&](const algoim::uvector<T2, spatial_dim>& x, T2 w) {
            if (algoim::bernstein::evalBernsteinPoly(phi, x) <= 0.0) {
              for (int d = 0; d < spatial_dim; d++) {
                if constexpr (is_dual) {
                  pts.push_back(x(d).dpart());
                } else {
                  pts.push_back(x(d));
                }
              }
              if constexpr (is_dual) {
                wts.push_back(w.dpart());
              } else {
                wts.push_back(w);
              }
            }
          });
    } else {  // quad_type == QuadPtType::SURFACE
      ipquad.integrate_surf(algoim::AutoMixed, Np_1d,
                            [&](const algoim::uvector<T2, spatial_dim>& x, T2 w,
                                const algoim::uvector<T2, spatial_dim>& _) {
                              // Evaluate the gradient on the quadrature point
                              // We assume that ipquad.phi.count() == 1 here
                              algoim::uvector<T2, spatial_dim> g =
                                  algoim::bernstein::evalBernsteinPolyGradient(
                                      ipquad.phi.poly(0), x);

                              // Normalize g
                              T2 nrm = T2(0.0);
                              for (int d = 0; d < spatial_dim; d++) {
                                nrm += g(d) * g(d);
                              }
                              nrm = sqrt(nrm);
                              for (int d = 0; d < spatial_dim; d++) {
                                g(d) = g(d) / nrm;
                              }

                              for (int d = 0; d < spatial_dim; d++) {
                                if constexpr (is_dual) {
                                  pts.push_back(x(d).dpart());
                                  ns.push_back(g(d).dpart());
                                } else {
                                  pts.push_back(x(d));
                                  ns.push_back(g(d));
                                }
                              }
                              if constexpr (is_dual) {
                                wts.push_back(w.dpart());
                              } else {
                                wts.push_back(w);
                              }
                            });
    }
  }

  // Mesh for physical dof. Dof nodes is a subset of grid verts due to
  // LSF-cut.
  const CutMesh_& mesh;

  // Mesh for the LSF dof. All grid verts are dof nodes.
  const GridMesh_& lsf_mesh;
};

/**
 * Galerkin difference basis given a set of stencil nodes
 *
 * @tparam Np_1d number of nodes along one dimension, number of stencil nodes
 *               should be Np_1d^2, Np_1d >= 2, Np_1d should be even
 */
template <typename T, class Mesh_>
class GDBasis2D final : public BasisBase<T, Mesh_> {
 private:
  static_assert(Mesh_::is_gd_mesh, "This basis requires a GD Mesh");
  using BasisBase_ = BasisBase<T, Mesh_>;

 public:
  static constexpr bool is_gd_basis = true;
  using BasisBase_::max_nnodes_per_element;
  using BasisBase_::spatial_dim;
  using Mesh = Mesh_;

 public:
  GDBasis2D(Mesh& mesh)
      : mesh(mesh),
        regular_eval(std::make_shared<VandermondeEvaluator<T, Mesh>>(
            mesh, *(mesh.get_regular_stencil_elems().begin()), true)) {}

  /**
   * @brief Given all quadrature points, evaluate the shape function values,
   * gradients and Hessians w.r.t. computational coordinates
   *
   * @param elem element index
   * @param pts collection of quadrature points
   * @param N shape function values
   * @param Nxi shape function gradients, concatenation of (∇_xi N_q, ∇_eta N_q)
   * @param Nxixi shape function Hessians, concatenation of (∇_xi_xi N_q,
   * ∇_xi_eta N_q, ∇_eta_xi N_q, ∇_eta_eta N_q)
   */
  void eval_basis_grad(int elem, const std::vector<T>& pts, std::vector<T>& N,
                       std::vector<T>& Nxi) const {
    int num_quad_pts = pts.size() / spatial_dim;
    N.resize(max_nnodes_per_element * num_quad_pts);
    Nxi.resize(max_nnodes_per_element * num_quad_pts * spatial_dim);

    std::shared_ptr<VandermondeEvaluator<T, Mesh>> eval;
    if (mesh.is_regular_stencil_elem(elem)) {
      eval = regular_eval;
      VandermondeCondLogger::add(
          elem, VandermondeCondLogger::get(
                    *(mesh.get_regular_stencil_elems().begin())));
    } else {
      eval = std::make_shared<VandermondeEvaluator<T, Mesh>>(mesh, elem);
    }

    for (int q = 0; q < num_quad_pts; q++) {
      int offset_n = q * max_nnodes_per_element;
      int offset_nxi = q * max_nnodes_per_element * spatial_dim;
      (*eval)(elem, &pts[spatial_dim * q], N.data() + offset_n,
              Nxi.data() + offset_nxi);
    }
  }
  void eval_basis_grad(int elem, const std::vector<T>& pts, std::vector<T>& N,
                       std::vector<T>& Nxi, std::vector<T>& Nxixi) const {
    int num_quad_pts = pts.size() / spatial_dim;
    N.resize(max_nnodes_per_element * num_quad_pts);
    Nxi.resize(max_nnodes_per_element * num_quad_pts * spatial_dim);
    Nxixi.resize(max_nnodes_per_element * num_quad_pts * spatial_dim *
                 spatial_dim);

    std::shared_ptr<VandermondeEvaluator<T, Mesh>> eval;
    if (mesh.is_regular_stencil_elem(elem)) {
      eval = regular_eval;
      VandermondeCondLogger::add(
          elem, VandermondeCondLogger::get(
                    *(mesh.get_regular_stencil_elems().begin())));
    } else {
      eval = std::make_shared<VandermondeEvaluator<T, Mesh>>(mesh, elem);
    }

    for (int q = 0; q < num_quad_pts; q++) {
      int offset_n = q * max_nnodes_per_element;
      int offset_nxi = q * max_nnodes_per_element * spatial_dim;
      int offset_nxixi = q * max_nnodes_per_element * spatial_dim * spatial_dim;
      (*eval)(elem, &pts[spatial_dim * q], N.data() + offset_n,
              Nxi.data() + offset_nxi, Nxixi.data() + offset_nxixi);
    }
  }

 private:
  const Mesh& mesh;
  std::shared_ptr<VandermondeEvaluator<T, Mesh>>
      regular_eval;  // evaluator for regular stencil
};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
