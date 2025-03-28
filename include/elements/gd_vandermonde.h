#ifndef XCGD_GALERKIN_DIFFERENCE_H
#define XCGD_GALERKIN_DIFFERENCE_H

#include <array>
#include <cmath>
#include <set>
#include <vector>

#include "dual.hpp"
#include "element_commons.h"
#include "element_utils.h"
#include "gaussquad.hpp"
#include "gd_mesh.h"
#include "quadrature_multipoly.hpp"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "vandermonde_evaluator.h"

enum class SurfQuad { LEFT, RIGHT, BOTTOM, TOP, NA };

template <typename T, int Np_1d, QuadPtType quad_type = QuadPtType::INNER,
          SurfQuad surf_quad = SurfQuad::NA, class Mesh = GridMesh<T, Np_1d>,
          int nquad_1d = Np_1d>
class GDGaussQuadrature2D final : public QuadratureBase<T, quad_type> {
 private:
  // algoim limit, see gaussquad.hpp
  static_assert(Np_1d <= algoim::GaussQuad::p_max);     // algoim limit
  static_assert(nquad_1d <= algoim::GaussQuad::p_max);  // algoim limit
  static_assert((quad_type == QuadPtType::SURFACE) xor
                    (surf_quad == SurfQuad::NA),
                "quad_type and surf_quad are not compatible");

 public:
  GDGaussQuadrature2D(Mesh& mesh, const std::set<int> elements = {})
      : mesh(mesh), elements(elements) {
    for (int i = 0; i < Np_1d; i++) {
      pts_1d[i] = algoim::GaussQuad::x(nquad_1d, i);  // in (0, 1)
      wts_1d[i] = algoim::GaussQuad::w(nquad_1d, i);
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
      constexpr int num_quad_pts = nquad_1d * nquad_1d;
      pts.resize(spatial_dim * num_quad_pts);
      wts.resize(num_quad_pts);
      for (int q = 0; q < num_quad_pts; q++) {  // q = i * nquad_1d + j
        int i = q / nquad_1d;
        int j = q % nquad_1d;
        pts[q * spatial_dim] = pts_1d[i];
        pts[q * spatial_dim + 1] = pts_1d[j];
        wts[q] = wts_1d[i] * wts_1d[j];
      }
      return num_quad_pts;
    } else {  // QuadPtType::SURFACE
      constexpr int num_quad_pts = nquad_1d;
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
  Mesh& mesh;
  std::set<int> elements;
  std::array<T, nquad_1d> pts_1d, wts_1d;
};

// Forward declaration
template <typename T, class Mesh>
class GDBasis2D;

template <typename T, int Np_1d, QuadPtType quad_type = QuadPtType::INNER,
          class Grid = StructuredGrid2D<T>, int nquad_1d = Np_1d,
          class CutMesh_ = CutMesh<T, Np_1d, Grid>>
class GDLSFQuadrature2D final : public QuadratureBase<T, quad_type> {
 public:
  using BulkQuad =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, nquad_1d, CutMesh_>;
  using InterfaceQuad = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE, Grid,
                                          nquad_1d, CutMesh_>;

 private:
  // algoim limit, see gaussquad.hpp
  static_assert(Np_1d <= algoim::GaussQuad::p_max);     // algoim limit
  static_assert(nquad_1d <= algoim::GaussQuad::p_max);  // algoim limit
  using GridMesh_ = GridMesh<T, Np_1d, Grid>;
  using Basis = GDBasis2D<T, CutMesh_>;

  constexpr static int spatial_dim = Basis::spatial_dim;
  constexpr static int max_nnodes_per_element = Basis::max_nnodes_per_element;

 public:
  GDLSFQuadrature2D(CutMesh_& mesh, const std::set<int> elements = {})
      : mesh(mesh), lsf_mesh(mesh.get_lsf_mesh()), elements(elements) {}

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
    if (elements.size() and !elements.count(elem)) {
      return 0;
    }

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
  inline int get_quadrature_pts_grad(int elem, std::vector<T>& pts,
                                     std::vector<T>& wts, std::vector<T>& ns,
                                     std::vector<T>& pts_grad,
                                     std::vector<T>& wts_grad) const {
    std::vector<T> dummy;
    return get_quadrature_pts_grad_impl<false>(elem, pts, wts, ns, pts_grad,
                                               wts_grad, dummy);
  }
  inline int get_quadrature_pts_grad(int elem, std::vector<T>& pts,
                                     std::vector<T>& wts, std::vector<T>& ns,
                                     std::vector<T>& pts_grad,
                                     std::vector<T>& wts_grad,
                                     std::vector<T>& wns_grad) const {
    return get_quadrature_pts_grad_impl<true>(elem, pts, wts, ns, pts_grad,
                                              wts_grad, wns_grad);
  }

 private:
  // Somehow this gradient is not always precise, we use finite difference
  // instead
  template <bool compute_ns_grad>
  int get_quadrature_pts_grad_impl_deprecated(int elem, std::vector<T>& pts,
                                              std::vector<T>& wts,
                                              std::vector<T>& ns,
                                              std::vector<T>& pts_grad,
                                              std::vector<T>& wts_grad,
                                              std::vector<T>& wns_grad) const {
    if (elements.size() and !elements.count(elem)) {
      return 0;
    }

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
    pts_grad.resize(num_quad_pts * spatial_dim * max_nnodes_per_element);
    wts_grad.resize(num_quad_pts * max_nnodes_per_element);

    if constexpr (compute_ns_grad) {
      wns_grad.clear();
      wns_grad.resize(num_quad_pts * spatial_dim * max_nnodes_per_element);
    }

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
        wts_grad[index] = dwts[q];  // ∂w/∂phi
        for (int d = 0; d < spatial_dim; d++) {
          pts_grad[index * spatial_dim + d] = dpts[q * spatial_dim + d];
          if constexpr (compute_ns_grad) {
            wns_grad[index * spatial_dim + d] = dwns[q * spatial_dim + d];
          }
        }
      }
    }

    return num_quad_pts;
  }

  template <bool compute_ns_grad>
  int get_quadrature_pts_grad_impl(int elem, std::vector<T>& pts,
                                   std::vector<T>& wts, std::vector<T>& ns,
                                   std::vector<T>& pts_grad,
                                   std::vector<T>& wts_grad,
                                   std::vector<T>& ns_grad) const {
    if (elements.size() and !elements.count(elem)) {
      return 0;
    }

    // this is the element index in lsf mesh
    int cell = mesh.get_elem_cell(elem);

    // Create the functor that evaluates the interpolation given an arbitrary
    // point within the computational coordinates
    VandermondeEvaluator<T, GridMesh_> eval(lsf_mesh, cell);

    // Get element LSF dofs
    const std::vector<T>& lsf_dof = mesh.get_lsf_dof();
    std::vector<T> element_lsf(max_nnodes_per_element, 0.0);
    constexpr int lsf_dim = 1;
    get_element_vars<T, lsf_dim, GridMesh_, Basis>(
        lsf_mesh, cell, lsf_dof.data(), element_lsf.data());

    // Get quadrature points and weights
    getQuadrature(element_lsf.data(), eval, pts, wts, ns);

    int num_quad_pts = wts.size();

    // Get quadrature gradients
    pts_grad.clear();
    wts_grad.clear();
    pts_grad.resize(num_quad_pts * spatial_dim * max_nnodes_per_element);
    wts_grad.resize(num_quad_pts * max_nnodes_per_element);

    if constexpr (compute_ns_grad) {
      ns_grad.clear();
      ns_grad.resize(num_quad_pts * spatial_dim * max_nnodes_per_element);
    }

    // Get the step size
    T diff = hard_max(element_lsf) - hard_min(element_lsf);
    T dh = 1e-8 * diff;
    if (fabs(diff) < 1e-14) {
      dh = 1e-8;
    }

    for (int i = 0; i < max_nnodes_per_element; i++) {
      element_lsf[i] += dh;
      std::vector<T> pts1, wts1, ns1;
      getQuadrature(element_lsf.data(), eval, pts1, wts1, ns1);
      element_lsf[i] -= dh;

      if (wts1.size() != num_quad_pts) {
        char msg[256];
        std::snprintf(
            msg, 256,
            "number of quadrature points for ∂pt/∂φ_%d is inconsistent. Got "
            "%ld, expect %d.",
            i, wts1.size(), num_quad_pts);
        throw std::runtime_error(msg);
      }

      for (int q = 0; q < num_quad_pts; q++) {
        int index = q * max_nnodes_per_element + i;
        wts_grad[index] = (wts1[q] - wts[q]) / dh;  // ∂w/∂phi
        for (int d = 0; d < spatial_dim; d++) {
          pts_grad[index * spatial_dim + d] =
              (pts1[q * spatial_dim + d] - pts[q * spatial_dim + d]) / dh;
          if constexpr (compute_ns_grad) {
            ns_grad[index * spatial_dim + d] =
                (ns1[q * spatial_dim + d] - ns[q * spatial_dim + d]) / dh;
          }
        }
      }
    }

    return num_quad_pts;
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
          algoim::AutoMixed, nquad_1d,
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
      ipquad.integrate_surf(algoim::AutoMixed, nquad_1d,
                            [&](const algoim::uvector<T2, spatial_dim>& x, T2 w,
                                const algoim::uvector<T2, spatial_dim>& _) {
                              // Evaluate the gradient on the quadrature point
                              // We assume that ipquad.phi.count() == 1 here
                              algoim::uvector<T2, spatial_dim> g =
                                  algoim::bernstein::evalBernsteinPolyGradient(
                                      ipquad.phi.poly(0), x);

                              T2 nrm2 = T2(0.0);
                              for (int d = 0; d < spatial_dim; d++) {
                                nrm2 += g(d) * g(d);
                              }
                              T2 nrm = sqrt(nrm2);

                              // Normalize g
                              algoim::uvector<T2, spatial_dim> gn;
                              for (int d = 0; d < spatial_dim; d++) {
                                gn(d) = g(d) / nrm;
                              }

                              for (int d = 0; d < spatial_dim; d++) {
                                if constexpr (is_dual) {
                                  pts.push_back(x(d).dpart());
                                  ns.push_back(gn(d).dpart());
                                } else {
                                  pts.push_back(x(d));
                                  ns.push_back(gn(d));
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
  CutMesh_& mesh;

  // Mesh for the LSF dof. All grid verts are dof nodes.
  const GridMesh_& lsf_mesh;

  std::set<int> elements;
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
  GDBasis2D(Mesh& mesh) : mesh(mesh) {
    if (not mesh.get_regular_stencil_elems().empty()) {
      regular_eval = std::make_shared<VandermondeEvaluator<T, Mesh>>(
          mesh, *(mesh.get_regular_stencil_elems().begin()), true);
    }
  }

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
