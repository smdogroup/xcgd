#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

#include "element_commons.h"
#include "element_utils.h"
#include "gaussquad.hpp"
#include "gd_mesh.h"
#include "quadrature_multipoly.hpp"
#include "utils/linalg.h"
#include "utils/misc.h"

template <typename T, int Np_1d>
class GDGaussQuadrature2D final : public QuadratureBase<T> {
 private:
  static constexpr int num_quad_pts = Np_1d * Np_1d;
  using Mesh = GDMesh2D<T, Np_1d>;

 public:
  GDGaussQuadrature2D(const Mesh& mesh) : mesh(mesh) {
    for (int i = 0; i < Np_1d; i++) {
      pts_1d[i] = algoim::GaussQuad::x(Np_1d, i);  // in [0, 1]
      wts_1d[i] = algoim::GaussQuad::w(Np_1d, i);
    }
  }

  int get_quadrature_pts(int elem, std::vector<T>& pts,
                         std::vector<T>& wts) const {
    int constexpr spatial_dim = Mesh::spatial_dim;
    pts.resize(spatial_dim * num_quad_pts);
    wts.resize(num_quad_pts);
    T xymin[spatial_dim], xymax[spatial_dim];
    T wt = get_computational_coordinates_limits(mesh, elem, xymin, xymax);
    T cx = xymin[0];
    T cy = xymin[1];
    T dx = xymax[0] - cx;
    T dy = xymax[1] - cy;
    for (int q = 0; q < num_quad_pts; q++) {  // q = i * Np_1d + j
      int i = q / Np_1d;
      int j = q % Np_1d;
      pts[q * spatial_dim] = cx + dx * pts_1d[i];
      pts[q * spatial_dim + 1] = cy + dy * pts_1d[j];
      wts[q] = wt * wts_1d[i] * wts_1d[j];
    }

    return num_quad_pts;
  }

 private:
  const Mesh& mesh;
  std::array<T, Np_1d> pts_1d, wts_1d;
};

// Forward declaration
template <typename T, int Np_1d>
class GDBasis2D;

template <typename T, int Np_1d>
class GDLSFQuadrature2D final : public QuadratureBase<T> {
 private:
  using Mesh = GDMesh2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;

 public:
  GDLSFQuadrature2D(const Mesh& mesh, const Basis& basis)
      : mesh(mesh),
        basis(basis),
        lsf_mesh(mesh.get_grid()),
        lsf_basis(lsf_mesh) {}

  int get_quadrature_pts(int elem, std::vector<T>& pts,
                         std::vector<T>& wts) const {
    int constexpr spatial_dim = Mesh::spatial_dim;
    int constexpr nodes_per_element = Basis::nodes_per_element;

    // Get element LSF dofs
    const std::vector<T>& lsf_dof = mesh.get_lsf_dof();
    T element_dof[nodes_per_element];
    get_element_vars<T, 1, Basis>(lsf_mesh, elem, lsf_dof.data(), element_dof);

    // Get bounds of the hyperrectangle
    algoim::uvector<T, spatial_dim> xmin, xmax;
    get_computational_coordinates_limits(lsf_mesh, elem, xmin.data(),
                                         xmax.data());

    // Create the functor that evaluates the interpolation given an arbitrary
    // point within the computational coordinates
    typename Basis::Evaluator eval(lsf_mesh, elem);

    // Obtain the Bernstein polynomial representation of the level-set
    // function
    T data[Np_1d * Np_1d];
    algoim::xarray<T, spatial_dim> phi(
        data, algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
    algoim::bernstein::bernsteinInterpolate<spatial_dim>(
        [&](const algoim::uvector<T, spatial_dim>& x) {
          T N[nodes_per_element];
          // T Nxi[nodes_per_element * spatial_dim];
          eval(x.data(), N, nullptr);
          return 0.0;
        },
        phi);

    int num_quad_pts = 0;
    // int num_quad_pts = quad.nodes.size();
    // pts.resize(spatial_dim * num_quad_pts);
    // wts.resize(num_quad_pts);

    // for (int q = 0; q < num_quad_pts; q++) {
    //   wts[q] = quad.nodes[q].w;
    //   for (int d = 0; d < spatial_dim; d++) {
    //     pts[spatial_dim * q + d] = quad.nodes[q].x(d);
    //   }
    // }

    return num_quad_pts;
  }

 private:
  // Mesh and basis for physical dof. Dof nodes is a subset of grid verts due to
  // LSF-cut.
  const Mesh& mesh;
  const Basis& basis;

  // Mesh and basis for the LSF dof. All grid verts are dof nodes.
  Mesh lsf_mesh;
  Basis lsf_basis;
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
  // algoim limit, see gaussquad.hpp
  static_assert(Np_1d <= algoim::GaussQuad::p_max);  // algoim limit
  using BasisBase = BasisBase<T, GDMesh2D<T, Np_1d>>;

 public:
  using BasisBase::nodes_per_element;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;

 private:
  static constexpr int Np = Mesh::nodes_per_element;
  static constexpr int Nk = Mesh::nodes_per_element;

 public:
  GDBasis2D(Mesh& mesh) : mesh(mesh) {}

  void eval_basis_grad(int elem, const std::vector<T>& pts, std::vector<T>& N,
                       std::vector<T>& Nxi) const {
    int num_quad_pts = pts.size() / spatial_dim;
    N.resize(nodes_per_element * num_quad_pts);
    Nxi.resize(nodes_per_element * num_quad_pts * spatial_dim);

    Evaluator eval(mesh, elem);

    for (int q = 0; q < num_quad_pts; q++) {
      int offset_n = q * nodes_per_element;
      int offset_nxi = q * nodes_per_element * spatial_dim;
      eval(&pts[spatial_dim * q], N.data() + offset_n, Nxi.data() + offset_nxi);
    }
  }

  class Evaluator {
   private:
    static constexpr int spatial_dim = Mesh::spatial_dim;

   public:
    Evaluator(const Mesh& mesh, int elem) {
      int nodes[Nk];
      std::vector<T> xpows(Np_1d), ypows(Np_1d);

      mesh.get_elem_dof_nodes(elem, nodes);

      T xloc_min[spatial_dim], xloc_max[spatial_dim];
      mesh.get_elem_node_ranges(elem, xloc_min, xloc_max);

      for (int i = 0; i < Nk; i++) {
        T xloc[spatial_dim];
        mesh.get_node_xloc(nodes[i], xloc);

        // make x, y in [-1, 1]
        T x =
            -1.0 + 2.0 * (xloc[0] - xloc_min[0]) / (xloc_max[0] - xloc_min[0]);
        T y =
            -1.0 + 2.0 * (xloc[1] - xloc_min[1]) / (xloc_max[1] - xloc_min[1]);

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
    }

    // Evaluate the shape function and derivatives given a quadrature point
    void operator()(const T* pt, T* N, T* Nxi) const {
      std::vector<T> xpows(Np_1d), ypows(Np_1d), dxpows(Np_1d), dypows(Np_1d);

      for (int ii = 0; ii < Np_1d; ii++) {
        xpows[ii] = pow(pt[0], ii);
        ypows[ii] = pow(pt[1], ii);
        dxpows[ii] = T(ii) * pow(pt[0], ii - 1);
        dypows[ii] = T(ii) * pow(pt[1], ii - 1);
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
              Nxi[spatial_dim * i + 1] +=
                  Ck[idx + Nk * i] * xpows[j] * dypows[k];
            }
          }
        }
      }
    }

   private:
    T Ck[Nk * Np];
  };

  const Mesh& mesh;
};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
