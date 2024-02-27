#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

#include "commons.h"
#include "gaussquad.hpp"
#include "gd_commons.h"
#include "utils/linalg.h"
#include "utils/misc.h"

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

      // x, y in [-1, 1]
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

template <typename T, int Np_1d>
class GDQuadrature2D final : public QuadratureBase<T, 2, Np_1d * Np_1d> {
 private:
  using QuadratureBase = QuadratureBase<T, 2, Np_1d * Np_1d>;

 public:
  using QuadratureBase::num_quadrature_pts;
  using QuadratureBase::spatial_dim;

  static T get_quadrature_pt(int k, T pt[]) {
    int i = k / Np_1d;
    int j = k % Np_1d;
    pt[0] = algoim::GaussQuad::x(Np_1d, i) * 2.0 - 1.0;
    pt[1] = algoim::GaussQuad::x(Np_1d, j) * 2.0 - 1.0;
    return 4.0 * algoim::GaussQuad::w(Np_1d, i) *
           algoim::GaussQuad::w(Np_1d, j);
  }

  static void get_quadrature_pts(T pts[], T wts[]) {
    for (int q = 0; q < num_quadrature_pts; q++) {  // q = i * Np_1d + j
      int i = q / Np_1d;
      int j = q % Np_1d;
      for (int d = 0; d < spatial_dim; d++) {
        pts[q * spatial_dim + d];  // TODO
      }
    }
  }
};

#endif  // XCGD_GALERKIN_DIFFERENCE_H
