#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

#include "dual.hpp"
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
      pts_1d[i] = algoim::GaussQuad::x(Np_1d, i);  // in (0, 1)
      wts_1d[i] = algoim::GaussQuad::w(Np_1d, i);
    }
  }

  int get_quadrature_pts(int elem, std::vector<T>& pts,
                         std::vector<T>& wts) const {
    int constexpr spatial_dim = Mesh::spatial_dim;
    pts.resize(spatial_dim * num_quad_pts);
    wts.resize(num_quad_pts);
    T xi_min[spatial_dim], xi_max[spatial_dim];
    T wt = get_computational_coordinates_limits(mesh, elem, xi_min, xi_max);
    T cx = xi_min[0];
    T cy = xi_min[1];
    T dx = xi_max[0] - cx;
    T dy = xi_max[1] - cy;
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
  GDLSFQuadrature2D(const Mesh& mesh, const Mesh& lsf_mesh)
      : mesh(mesh), lsf_mesh(lsf_mesh) {}

  int get_quadrature_pts(int elem, std::vector<T>& pts,
                         std::vector<T>& wts) const {
    // this is the element index in lsf mesh
    int cell = mesh.get_elem_cell(elem);

    // Get bounds of the hyperrectangle
    algoim::uvector<T, Mesh::spatial_dim> xi_min, xi_max;
    get_computational_coordinates_limits(lsf_mesh, cell, xi_min.data(),
                                         xi_max.data());

    // Create the functor that evaluates the interpolation given an arbitrary
    // point within the computational coordinates
    typename Basis::Evaluator eval(lsf_mesh, cell);

    // Get element LSF dofs
    const std::vector<T>& lsf_dof = mesh.get_lsf_dof();
    T element_lsf[Basis::nodes_per_element];
    get_element_vars<T, 1, Basis>(lsf_mesh, cell, lsf_dof.data(), element_lsf);

    // Get quadrature based onf element lsf values
    getQuadrature(cell, xi_min, xi_max, element_lsf, eval, pts, wts);

    return wts.size();
  }

  /**
   * @brief Get derivatives of quadrature points and weights with respect to
   * element LSF dof
   *
   * @param elem element index
   * @param pts_grad pts_grad[i] = ∂ξ/∂phi_i
   * @param wts_grad wts_grad[i] = ∂w/∂phi_i
   * @return int number of quadrature points
   */
  int get_quadrature_grad(int elem, std::vector<std::vector<T>>& pts_grad,
                          std::vector<std::vector<T>>& wts_grad) const {
    constexpr int spatial_dim = Mesh::spatial_dim;
    constexpr int nodes_per_element = Basis::nodes_per_element;

    // this is the element index in lsf mesh
    int cell = mesh.get_elem_cell(elem);

    // Get bounds of the hyperrectangle
    algoim::uvector<T, spatial_dim> xi_min, xi_max;
    get_computational_coordinates_limits(lsf_mesh, cell, xi_min.data(),
                                         xi_max.data());

    // Create the functor that evaluates the interpolation given an arbitrary
    // point within the computational coordinates
    typename Basis::Evaluator eval(lsf_mesh, cell);

    // Get element LSF dofs
    const std::vector<T>& lsf_dof = mesh.get_lsf_dof();
    T element_lsf[nodes_per_element];
    get_element_vars<T, 1, Basis>(lsf_mesh, cell, lsf_dof.data(), element_lsf);

    duals::dual<T> element_lsf_d[nodes_per_element];
    for (int i = 0; i < nodes_per_element; i++) {
      element_lsf_d[i].rpart(element_lsf[i]);
      element_lsf_d[i].dpart(0.0);
    }

    pts_grad.clear();
    wts_grad.clear();
    pts_grad.resize(nodes_per_element);
    wts_grad.resize(nodes_per_element);

    for (int i = 0; i < nodes_per_element; i++) {
      element_lsf_d[i].dpart(1.0);
      getQuadrature(cell, xi_min, xi_max, element_lsf_d, eval, pts_grad[i],
                    wts_grad[i]);
      element_lsf_d[i].dpart(0.0);
    }

    return wts_grad[0].size();
  }

  const Mesh& get_lsf_mesh() const { return lsf_mesh; };

 private:
  template <typename T2>
  void get_phi_vals(int cell, const typename Basis::Evaluator& eval,
                    const algoim::uvector<T, Mesh::spatial_dim>& xi_min,
                    const algoim::uvector<T, Mesh::spatial_dim>& xi_max,
                    const T2 element_dof[],
                    algoim::xarray<T2, Mesh::spatial_dim>& phi) const {
    algoim::bernstein::bernsteinInterpolate<Mesh::spatial_dim>(
        [&](const algoim::uvector<T2, Mesh::spatial_dim>& x) {  // x in [0, 1]
          T2 N[Basis::nodes_per_element];
          // xi in [xi_min, xi_max]
          algoim::uvector<T2, Mesh::spatial_dim> xi =
              xi_min + x * (xi_max - xi_min);
          eval(xi.data(), N, (T2*)nullptr);
          T2 val;
          interp_val_grad<T2, Basis>(cell, element_dof, N, nullptr, &val,
                                     nullptr);
          return val;
        },
        phi);
  }

  template <typename T2>
  void getQuadrature(int cell,
                     const algoim::uvector<T, Mesh::spatial_dim>& xi_min,
                     const algoim::uvector<T, Mesh::spatial_dim>& xi_max,
                     const T2 element_lsf[],
                     const typename Basis::Evaluator& eval, std::vector<T>& pts,
                     std::vector<T>& wts) const {
    constexpr bool is_dual = is_specialization<T2, duals::dual>::value;

    // Obtain the Bernstein polynomial representation of the level-set
    // function
    T2 data[Np_1d * Np_1d];
    algoim::xarray<T2, Mesh::spatial_dim> phi(
        data, algoim::uvector<int, Mesh::spatial_dim>(Np_1d, Np_1d));
    get_phi_vals(cell, eval, xi_min, xi_max, element_lsf, phi);

    pts.clear();
    wts.clear();

    algoim::ImplicitPolyQuadrature<Mesh::spatial_dim, T2> ipquad(phi);
    ipquad.integrate(
        algoim::AutoMixed, Np_1d,
        [&](const algoim::uvector<T2, Mesh::spatial_dim>& x, T2 w) {
          if (algoim::bernstein::evalBernsteinPoly(phi, x) <= 0.0) {
            for (int d = 0; d < Mesh::spatial_dim; d++) {
              if constexpr (is_dual) {
                pts.push_back(
                    (xi_min(d) + x(d) * (xi_max(d) - xi_min(d))).dpart());
              } else {
                pts.push_back(xi_min(d) + x(d) * (xi_max(d) - xi_min(d)));
              }
            }
            if constexpr (is_dual) {
              wts.push_back(w.dpart());
            } else {
              wts.push_back(w);
            }
          }
        });
  }

  // Mesh for physical dof. Dof nodes is a subset of grid verts due to
  // LSF-cut.
  const Mesh& mesh;

  // Mesh for the LSF dof. All grid verts are dof nodes.
  const Mesh& lsf_mesh;
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
  static constexpr bool is_gd_basis = true;
  using BasisBase::nodes_per_element;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;

 private:
  static constexpr int Np = Mesh::nodes_per_element;
  static constexpr int Nk = Mesh::nodes_per_element;

 public:
  GDBasis2D(Mesh& mesh) : mesh(mesh) {}

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
    N.resize(nodes_per_element * num_quad_pts);
    Nxi.resize(nodes_per_element * num_quad_pts * spatial_dim);

    Evaluator eval(mesh, elem);

    for (int q = 0; q < num_quad_pts; q++) {
      int offset_n = q * nodes_per_element;
      int offset_nxi = q * nodes_per_element * spatial_dim;
      eval(&pts[spatial_dim * q], N.data() + offset_n, Nxi.data() + offset_nxi);
    }
  }
  void eval_basis_grad(int elem, const std::vector<T>& pts, std::vector<T>& N,
                       std::vector<T>& Nxi, std::vector<T>& Nxixi) const {
    int num_quad_pts = pts.size() / spatial_dim;
    N.resize(nodes_per_element * num_quad_pts);
    Nxi.resize(nodes_per_element * num_quad_pts * spatial_dim);
    Nxixi.resize(nodes_per_element * num_quad_pts * spatial_dim * spatial_dim);

    Evaluator eval(mesh, elem);

    for (int q = 0; q < num_quad_pts; q++) {
      int offset_n = q * nodes_per_element;
      int offset_nxi = q * nodes_per_element * spatial_dim;
      int offset_nxixi = q * nodes_per_element * spatial_dim * spatial_dim;
      eval(&pts[spatial_dim * q], N.data() + offset_n, Nxi.data() + offset_nxi,
           Nxixi.data() + offset_nxixi);
    }
  }

  // This class implements a functor that evaluate basis values and basis
  // gradients given a set of computational coordinates
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
    template <typename T2>
    void operator()(const T2* pt, T2* N, T2* Nxi,
                    T2* Nxixi = (T2*)nullptr) const {
      std::vector<T2> xpows(Np_1d), ypows(Np_1d), dxpows(Np_1d), dypows(Np_1d),
          dx2pows(Np_1d), dy2pows(Np_1d);

      for (int ii = 0; ii < Np_1d; ii++) {
        xpows[ii] = pow(pt[0], ii);
        ypows[ii] = pow(pt[1], ii);
        dxpows[ii] = T(ii) * pow(pt[0], ii - 1);
        dypows[ii] = T(ii) * pow(pt[1], ii - 1);
        dx2pows[ii] = T(ii) * T(ii - 1) * pow(pt[0], ii - 2);
        dy2pows[ii] = T(ii) * T(ii - 1) * pow(pt[1], ii - 2);
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
          Nxixi[spatial_dim * i] = 0.0;
          Nxixi[spatial_dim * i + 1] = 0.0;
          Nxixi[spatial_dim * i + 2] = 0.0;
          Nxixi[spatial_dim * i + 3] = 0.0;
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
            if (Nxixi) {
              Nxixi[spatial_dim * i] +=
                  Ck[idx + Nk * i] * dx2pows[j] * ypows[k];
              Nxixi[spatial_dim * i + 1] +=
                  Ck[idx + Nk * i] * dxpows[j] * dypows[k];
              Nxixi[spatial_dim * i + 2] +=
                  Ck[idx + Nk * i] * dxpows[j] * dypows[k];
              Nxixi[spatial_dim * i + 3] +=
                  Ck[idx + Nk * i] * xpows[j] * dy2pows[k];
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
