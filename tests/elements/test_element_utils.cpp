#include <array>
#include <complex>
#include <numeric>
#include <stdexcept>

#include "elements/element_utils.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "test_commons.h"
#include "utils/exceptions.h"

using T = double;

template <class Mesh, class Quadrature, class Basis, class Physics>
struct ElementAjpTester {
  static int constexpr spatial_dim = Basis::spatial_dim;
  static int constexpr max_nnodes_per_element = Basis::max_nnodes_per_element;
  static int constexpr num_quad_pts = max_nnodes_per_element;
  static int constexpr dof_per_node = Physics::dof_per_node;
  static int constexpr ndof_per_element = max_nnodes_per_element * dof_per_node;

  ElementAjpTester(const Mesh& mesh, const Quadrature& quadrature,
                   const Basis& basis, const Physics& physics)
      : mesh(mesh),
        quadrature(quadrature),
        basis(basis),
        physics(physics),
        element_xloc(max_nnodes_per_element * spatial_dim),
        element_psi(ndof_per_element),
        phip(max_nnodes_per_element),
        i(0),
        j(0) {
    for (T& e : element_xloc) e = (T)rand() / RAND_MAX;
    for (T& e : element_psi) e = (T)rand() / RAND_MAX;
    for (T& e : phip) e = (T)rand() / RAND_MAX;
  }

  T eval_energy(std::vector<T> element_dof, std::vector<T> quad_pts,
                std::vector<T> quad_wts) {
    // Sanity check
    if (element_dof.size() != ndof_per_element or
        quad_pts.size() != num_quad_pts * spatial_dim or
        quad_wts.size() != num_quad_pts) {
      throw std::runtime_error("some input has invalid size");
    }

    // Basis functions
    std::vector<T> N, Nxi;
    basis.eval_basis_grad(i, quad_pts, N, Nxi);

    int offset_n = j * max_nnodes_per_element;
    int offset_nxi = j * max_nnodes_per_element * spatial_dim;

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    A2D::Vec<T, spatial_dim> xloc, nrm_ref;
    A2D::Mat<T, spatial_dim, spatial_dim> J;
    interp_val_grad<T, Basis, spatial_dim>(element_xloc.data(), &N[offset_n],
                                           &Nxi[offset_nxi], &xloc, &J);

    // Evaluate the derivative of the dof in the computational coordinates
    typename Physics::dof_t vals{};
    typename Physics::grad_t grad{}, grad_ref{};
    interp_val_grad<T, Basis>(element_dof.data(), &N[offset_n],
                              &Nxi[offset_nxi], &vals, &grad_ref);

    // Transform gradient from ref coordinates to physical coordinates
    transform(J, grad_ref, grad);

    return physics.energy(quad_wts[j], 0.0, xloc, nrm_ref, J, vals, grad);
  }

  std::vector<T> eval_residual(std::vector<T> element_dof,
                               std::vector<T> quad_pts,
                               std::vector<T> quad_wts) {
    // Sanity check
    if (element_dof.size() != ndof_per_element or
        quad_pts.size() != num_quad_pts * spatial_dim or
        quad_wts.size() != num_quad_pts) {
      throw std::runtime_error("some input has invalid size");
    }

    // Basis functions
    std::vector<T> N, Nxi;
    basis.eval_basis_grad(i, quad_pts, N, Nxi);

    int offset_n = j * max_nnodes_per_element;
    int offset_nxi = j * max_nnodes_per_element * spatial_dim;

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    A2D::Vec<T, spatial_dim> xloc, nrm_ref;
    A2D::Mat<T, spatial_dim, spatial_dim> J;
    interp_val_grad<T, Basis, spatial_dim>(element_xloc.data(), &N[offset_n],
                                           &Nxi[offset_nxi], &xloc, &J);

    // Evaluate the derivative of the dof in the computational coordinates
    typename Physics::dof_t vals{};
    typename Physics::grad_t grad{}, grad_ref{};
    interp_val_grad<T, Basis>(element_dof.data(), &N[offset_n],
                              &Nxi[offset_nxi], &vals, &grad_ref);

    // Transform gradient from ref coordinates to physical coordinates
    transform(J, grad_ref, grad);

    // Evaluate the residuals at the quadrature points
    typename Physics::dof_t coef_vals{};
    typename Physics::grad_t coef_grad{}, coef_grad_ref{};
    physics.residual(quad_wts[j], 0.0, xloc, nrm_ref, J, vals, grad, coef_vals,
                     coef_grad);

    // Transform gradient from physical coordinates back to ref coordinates
    rtransform(J, coef_grad, coef_grad_ref);

    // Add the contributions to the element residual
    std::vector<T> element_res(ndof_per_element);
    add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals, coef_grad_ref,
                       element_res.data());

    return element_res;
  }

  std::pair<std::vector<T>, std::vector<T>> eval_N_Nxi(
      std::vector<T> quad_pts) {
    if (quad_pts.size() != num_quad_pts * spatial_dim) {
      throw std::runtime_error("some input has invalid size");
    }

    // Basis functions
    std::vector<T> N, Nxi;
    basis.eval_basis_grad(i, quad_pts, N, Nxi);

    int offset_n = j * max_nnodes_per_element;
    int offset_nxi = j * max_nnodes_per_element * spatial_dim;

    return {std::vector<T>(N.begin() + offset_n,
                           N.begin() + offset_n + max_nnodes_per_element),
            std::vector<T>(Nxi.begin() + offset_nxi,
                           Nxi.begin() + offset_nxi +
                               spatial_dim * max_nnodes_per_element)};
  }

  std::pair<typename Physics::dof_t, typename Physics::grad_t> eval_coef_grad(
      std::vector<T> element_dof, std::vector<T> quad_pts,
      std::vector<T> quad_wts) {
    // Sanity check
    if (element_dof.size() != ndof_per_element or
        quad_pts.size() != num_quad_pts * spatial_dim or
        quad_wts.size() != num_quad_pts) {
      throw std::runtime_error("some input has invalid size");
    }

    // Basis functions
    auto [N, Nxi] = eval_N_Nxi(quad_pts);

    // Evaluate the derivative of the spatial dof in the computational
    // coordinates
    A2D::Vec<T, spatial_dim> xloc, nrm_ref;
    A2D::Mat<T, spatial_dim, spatial_dim> J;
    interp_val_grad<T, Basis, spatial_dim>(element_xloc.data(), N.data(),
                                           Nxi.data(), &xloc, &J);

    // Evaluate the derivative of the dof in the computational coordinates
    typename Physics::dof_t vals{};
    typename Physics::grad_t grad{}, grad_ref{};
    interp_val_grad<T, Basis>(element_dof.data(), N.data(), Nxi.data(), &vals,
                              &grad_ref);

    // Transform gradient from ref coordinates to physical coordinates
    transform(J, grad_ref, grad);

    // Evaluate the residuals at the quadrature points
    typename Physics::dof_t coef_vals{};  // ∂e/∂u, (dof_per_node,)
    typename Physics::grad_t coef_grad{},
        coef_grad_ref{};  // ∂e/∂∇u, (dof_per_node, spatial_dim)
    physics.residual(quad_wts[j], 0.0, xloc, nrm_ref, J, vals, grad, coef_vals,
                     coef_grad);

    // Transform gradient from physical coordinates back to ref coordinates
    rtransform(J, coef_grad, coef_grad_ref);

    return {coef_vals, coef_grad_ref};
  }

  T assemble_residual_psi(std::vector<T> element_dof,
                          std::vector<T> element_psi, std::vector<T> quad_pts,
                          std::vector<T> quad_wts) {
    auto [N, Nxi] = eval_N_Nxi(quad_pts);
    auto [coef_vals, coef_grad_ref] =
        eval_coef_grad(element_dof, quad_pts, quad_wts);

    T ret = 0.0;
    for (int i = 0; i < dof_per_node; i++) {
      for (int m = 0; m < max_nnodes_per_element; m++) {
        ret += N[m] * element_psi[m * dof_per_node + i] * coef_vals(i);
        for (int d = 0; d < spatial_dim; d++) {
          ret += Nxi[spatial_dim * m + d] * coef_grad_ref(i, d) *
                 element_psi[m * dof_per_node + i];
        }
      }
    }
    return ret;
  }

  template <bool skip_ajp1 = false, bool skip_ajp3 = false>
  T eval_ajp_fd(std::vector<T> element_dof, std::vector<T> quad_pts,
                std::vector<T> quad_pts_grad, std::vector<T> quad_wts,
                std::vector<T> quad_wts_grad, T h = 1e-5) {
    // Sanity check
    if (element_dof.size() != ndof_per_element or
        quad_pts.size() != num_quad_pts * spatial_dim or
        quad_pts_grad.size() !=
            num_quad_pts * spatial_dim * max_nnodes_per_element or
        quad_wts.size() != num_quad_pts or
        quad_wts_grad.size() != num_quad_pts * max_nnodes_per_element) {
      throw std::runtime_error("some input has invalid size");
    }

    std::vector<T> quad_wts_p(num_quad_pts, 0.0);
    for (int ii = 0; ii < num_quad_pts; ii++) {
      int offset = ii * max_nnodes_per_element;
      for (int jj = 0; jj < max_nnodes_per_element; jj++) {
        quad_wts_p[ii] +=
            quad_wts_grad[offset + jj] /*∂w(ii)/∂phi(jj)*/ * phip[jj];
      }
    }

    std::vector<T> quad_pts_p(num_quad_pts * spatial_dim, 0.0);
    for (int ii = 0; ii < num_quad_pts; ii++) {
      int offset = ii * max_nnodes_per_element * spatial_dim;
      for (int jj = 0; jj < max_nnodes_per_element; jj++) {
        for (int d = 0; d < spatial_dim; d++) {
          quad_pts_p[ii * spatial_dim + d] +=
              quad_pts_grad[offset + spatial_dim * jj +
                            d] /*∂xi(ii, d)/∂phi(jj)*/
              * phip[jj];
        }
      }
    }

    std::vector<T> quad_wts_perturb = quad_wts;
    for (int i = 0; i < quad_wts.size(); i++) {
      quad_wts_perturb[i] += h * quad_wts_p[i];
    }

    std::vector<T> quad_pts_perturb = quad_pts;
    for (int i = 0; i < quad_pts.size(); i++) {
      quad_pts_perturb[i] += h * quad_pts_p[i];
    }

    auto R_psi_func = [&element_dof, this](std::vector<T>& quad_pts,
                                           std::vector<T>& quad_wts) -> T {
      return assemble_residual_psi(element_dof, this->element_psi, quad_pts,
                                   quad_wts);
    };

    T ret = 0.0;
    if constexpr (not skip_ajp1) {
      ret += (R_psi_func(quad_pts, quad_wts_perturb) -
              R_psi_func(quad_pts, quad_wts)) /
             h;
    }
    if constexpr (not skip_ajp3) {
      ret += (R_psi_func(quad_pts_perturb, quad_wts) -
              R_psi_func(quad_pts, quad_wts)) /
             h;

      // auto [N, Nxi] = eval_N_Nxi(quad_pts);
      //
      // auto [coef_vals_perturb, coef_grad_ref_perturb] =
      //     eval_coef_grad(element_dof, quad_pts_perturb, quad_wts);
      //
      // auto [coef_vals, coef_grad_ref] =
      //     eval_coef_grad(element_dof, quad_pts, quad_wts);
      //
      // for (int i = 0; i < dof_per_node; i++) {
      //   for (int m = 0; m < max_nnodes_per_element; m++) {
      //     ret += N[m] * element_psi[m * dof_per_node + i] *
      //            (coef_vals_perturb(i) - coef_vals(i)) / h;
      //     for (int d = 0; d < spatial_dim; d++) {
      //       ret += Nxi[spatial_dim * m + d] * coef_grad_ref(i, d) *
      //              element_psi[m * dof_per_node + i];
      //     }
      //   }
      // }
    }
    return ret;
  }

  // Not used once eval_ajp_fd is implemented and verified
  T eval_ajp1_fd(std::vector<T> element_dof, std::vector<T> quad_pts,
                 std::vector<T> quad_wts, std::vector<T> quad_wts_grad,
                 T h = 1e-5) {
    // Sanity check
    if (element_dof.size() != ndof_per_element or
        quad_pts.size() != num_quad_pts * spatial_dim or
        quad_wts.size() != num_quad_pts or
        quad_wts_grad.size() != num_quad_pts * max_nnodes_per_element) {
      throw std::runtime_error("some input has invalid size");
    }

    if (element_dof.size() != ndof_per_element) {
      throw std::runtime_error("inconsistent element_dof size");
    }

    std::vector<T> element_dof_perturb(ndof_per_element);
    for (int i = 0; i < ndof_per_element; i++) {
      element_dof_perturb[i] = element_dof[i] + h * element_psi[i];
    }

    int offset_wts = j * max_nnodes_per_element;
    T dedu_psi = (eval_energy(element_dof_perturb, quad_pts, quad_wts) -
                  eval_energy(element_dof, quad_pts, quad_wts)) /
                 h;
    return dedu_psi / quad_wts[j] *
           std::inner_product(phip.begin(), phip.end(),
                              quad_wts_grad.begin() + offset_wts, 0.0);
  }

  // Evaluate ajoint-Jacobian product inner-producted by phip:
  template <bool skip_ajp1, bool skip_ajp3>
  T eval_ajp(std::vector<T> element_dof, std::vector<T> quad_pts,
             std::vector<T> quad_pts_grad, std::vector<T> quad_wts,
             std::vector<T> quad_wts_grad) {
    // Sanity check
    if (element_dof.size() != ndof_per_element or
        quad_pts.size() != num_quad_pts * spatial_dim or
        quad_pts_grad.size() !=
            num_quad_pts * spatial_dim * max_nnodes_per_element or
        quad_wts.size() != num_quad_pts or
        quad_wts_grad.size() != num_quad_pts * max_nnodes_per_element) {
      throw std::runtime_error("some input has invalid size");
    }

    // Basis functions
    std::vector<T> N, Nxi, Nxixi;
    basis.eval_basis_grad(i, quad_pts, N, Nxi, Nxixi);

    int offset_n = j * max_nnodes_per_element;
    int offset_nxi = j * max_nnodes_per_element * spatial_dim;
    int offset_nxixi = j * max_nnodes_per_element * spatial_dim * spatial_dim;

    A2D::Vec<T, spatial_dim> xloc, nrm_ref;
    A2D::Mat<T, spatial_dim, spatial_dim> J;
    interp_val_grad<T, Basis, spatial_dim>(element_xloc.data(), &N[offset_n],
                                           &Nxi[offset_nxi], &xloc, &J);

    // Evaluate the derivative of the dof in the computational coordinates
    typename Physics::dof_t uq{}, psiq{};           // uq, psiq
    typename Physics::grad_t ugrad{}, ugrad_ref{};  // (∇_x)uq, (∇_ξ)uq
    typename Physics::grad_t pgrad{}, pgrad_ref{};  // (∇_x)psiq, (∇_ξ)psiq
    typename Physics::hess_t uhess_ref{};           //(∇2_ξ)uq
    typename Physics::hess_t phess_ref{};           //(∇2_ξ)psiq

    // Interpolate the quantities at the quadrature point
    interp_val_grad<T, Basis>(element_dof.data(), &N[offset_n],
                              &Nxi[offset_nxi], &uq, &ugrad_ref);
    interp_val_grad<T, Basis>(element_psi.data(), &N[offset_n],
                              &Nxi[offset_nxi], &psiq, &pgrad_ref);
    interp_hess<T, Basis>(element_dof.data(), &Nxixi[offset_nxixi], uhess_ref);
    interp_hess<T, Basis>(element_psi.data(), &Nxixi[offset_nxixi], phess_ref);

    transform(J, ugrad_ref, ugrad);
    transform(J, pgrad_ref, pgrad);

    typename Physics::dof_t coef_uq{};      // ∂e/∂uq
    typename Physics::grad_t coef_ugrad{};  // ∂e/∂(∇_x)uq
    typename Physics::dof_t jp_uq{};        // ∂2e/∂uq2 * psiq
    typename Physics::grad_t jp_ugrad{};    // ∂2e/∂(∇_x)uq2 * (∇_x)psiq

    T detJ;
    A2D::MatDet(J, detJ);

    physics.residual(1.0 / detJ, 0.0, xloc, nrm_ref, J, uq, ugrad, coef_uq,
                     coef_ugrad);
    physics.jacobian_product(1.0 / detJ, 0.0, xloc, nrm_ref, J, uq, ugrad, psiq,
                             pgrad, jp_uq, jp_ugrad);

    typename Physics::grad_t coef_ugrad_ref{};  // ∂e/∂(∇_ξ)uq
    typename Physics::grad_t jp_ugrad_ref{};  // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq

    // Transform gradient from physical coordinates back to ref
    // coordinates
    rtransform(J, coef_ugrad, coef_ugrad_ref);
    rtransform(J, jp_ugrad, jp_ugrad_ref);

    int offset_wts = j * max_nnodes_per_element;
    int offset_pts = j * max_nnodes_per_element * spatial_dim;

    std::vector<T> element_dfdphi(max_nnodes_per_element, 0.0);
    if constexpr (dof_per_node == 1) {
      add_jac_adj_product<T, Basis, skip_ajp1, skip_ajp3>(
          quad_wts[j], detJ, &quad_wts_grad[offset_wts],
          &quad_pts_grad[offset_pts], psiq, ugrad_ref, pgrad_ref, uhess_ref,
          phess_ref, coef_uq, coef_ugrad_ref, jp_uq, jp_ugrad_ref,
          element_dfdphi.data());
    } else {
      add_jac_adj_product<T, Basis, dof_per_node, skip_ajp1, skip_ajp3, false,
                          false, false, false>(
          quad_wts[j], detJ, &quad_wts_grad[offset_wts],
          &quad_pts_grad[offset_pts], psiq, ugrad_ref, pgrad_ref, uhess_ref,
          phess_ref, coef_uq, coef_ugrad_ref, jp_uq, jp_ugrad_ref,
          element_dfdphi.data());
    }

    if (element_dfdphi.size() != phip.size()) {
      throw std::runtime_error("element_dfdphi and phip have different sizes");
    }
    return std::inner_product(element_dfdphi.begin(), element_dfdphi.end(),
                              phip.begin(), 0.0);
  }

  const Mesh& mesh;
  const Quadrature& quadrature;
  const Basis& basis;
  const Physics& physics;

  std::vector<T> element_xloc;
  std::vector<T> element_dof;
  std::vector<T> element_psi;
  std::vector<T> phip;

  int i, j;  // i is element index, j is quadrature index
};

enum class WHICH { Ajp1, Ajp3, Ajp };

template <int Np_1d, WHICH which, class Physics>
void test_adj_jac_product(const Physics& physics) {
  srand(42);

  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {12, 12};
  T lxy[2] = {12.0, 12.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);
  using Tester = ElementAjpTester<Mesh, Quadrature, Basis, Physics>;
  Tester tester(mesh, quadrature, basis, physics);

  std::vector<T> element_dof(Tester::ndof_per_element),
      quad_pts(Tester::num_quad_pts * Tester::spatial_dim),
      quad_pts_grad(Tester::num_quad_pts * Tester::spatial_dim *
                    Tester::max_nnodes_per_element),
      quad_wts(Tester::num_quad_pts),
      quad_wts_grad(Tester::num_quad_pts * Tester::max_nnodes_per_element);

  for (T& dof : element_dof) dof = (T)rand() / RAND_MAX;
  for (T& e : quad_pts) e = (T)rand() / RAND_MAX;
  for (T& e : quad_pts_grad) e = (T)rand() / RAND_MAX;
  for (T& e : quad_wts) e = (T)rand() / RAND_MAX;
  for (T& e : quad_wts_grad) e = (T)rand() / RAND_MAX;

  T relerr_min = 1.0;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    T ajp_fd = 0.0, ajp_exact = 0.0;

    if constexpr (which == WHICH::Ajp) {
      ajp_fd = tester.template eval_ajp_fd<false, false>(
          element_dof, quad_pts, quad_pts_grad, quad_wts, quad_wts_grad, dh);
      ajp_exact = tester.template eval_ajp<false, false>(
          element_dof, quad_pts, quad_pts_grad, quad_wts, quad_wts_grad);
    } else if constexpr (which == WHICH::Ajp1) {
      ajp_fd = tester.template eval_ajp_fd<false, true>(
          element_dof, quad_pts, quad_pts_grad, quad_wts, quad_wts_grad, dh);
      ajp_exact = tester.template eval_ajp<false, true>(
          element_dof, quad_pts, quad_pts_grad, quad_wts, quad_wts_grad);
    } else if constexpr (which == WHICH::Ajp3) {
      ajp_fd = tester.template eval_ajp_fd<true, false>(
          element_dof, quad_pts, quad_pts_grad, quad_wts, quad_wts_grad, dh);
      ajp_exact = tester.template eval_ajp<true, false>(
          element_dof, quad_pts, quad_pts_grad, quad_wts, quad_wts_grad);
    } else {
      throw NotImplemented("");
    }

    T relerr = fabs(ajp_fd - ajp_exact) / fabs(ajp_exact);
    if (relerr < relerr_min) relerr_min = relerr;
    std::printf(
        "Np_1d: %d, dh: %.5e, FD: %30.20e, Actual: %30.20e, Rel err: %20.10e\n",
        Np_1d, dh, ajp_fd, ajp_exact, relerr);
  }

  EXPECT_LE(relerr_min, 1e-6);
}

TEST(element_utils, ExplicitResidual) {
  using T = double;
  int constexpr Np_1d = 2;

  srand(42);

  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {12, 12};
  T lxy[2] = {3.0, 2.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);
  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    ret(0) = -1.2 * xloc(0);
    ret(1) = 3.4 * xloc(1);
    return ret;
  };
  using Physics = LinearElasticity<T, 2, typeof(int_func)>;
  T E = 10.0, nu = 0.3;
  Physics physics(E, nu, int_func);
  using Tester = ElementAjpTester<Mesh, Quadrature, Basis, Physics>;
  Tester tester(mesh, quadrature, basis, physics);

  std::vector<T> element_dof(Tester::ndof_per_element);
  std::vector<T> element_psi(Tester::ndof_per_element);

  std::vector<T> quad_pts(Tester::num_quad_pts * Tester::spatial_dim);
  std::vector<T> quad_wts(Tester::num_quad_pts);

  for (T& dof : element_dof) dof = (T)rand() / RAND_MAX;
  for (T& psi : element_psi) psi = (T)rand() / RAND_MAX;
  for (T& e : quad_pts) e = (T)rand() / RAND_MAX;
  for (T& e : quad_wts) e = (T)rand() / RAND_MAX;

  std::vector<T> element_res =
      tester.eval_residual(element_dof, quad_pts, quad_wts);
  T rpsi1 = std::inner_product(element_res.begin(), element_res.end(),
                               element_psi.begin(), 0.0);

  T rpsi2 = tester.assemble_residual_psi(element_dof, element_psi, quad_pts,
                                         quad_wts);

  std::printf("implicit: %30.15e, explicit: %30.15e, relerr: %30.15e\n", rpsi1,
              rpsi2, fabs(rpsi1 - rpsi2) / fabs(rpsi1));
}

// TEST(element_utils, TestAjp1Poisson) {
//   auto source_func = [](const A2D::Vec<T, 2> xloc) {
//     return -1.2 * xloc(0) + 3.4 * xloc(1);
//   };
//   using Physics = PoissonPhysics<T, 2, typeof(source_func)>;
//   Physics physics(source_func);
//   test_adj_jac_product<2, WHICH::Ajp1>(physics);
//   test_adj_jac_product<4, WHICH::Ajp1>(physics);
//   test_adj_jac_product<6, WHICH::Ajp1>(physics);
//   test_adj_jac_product<8, WHICH::Ajp1>(physics);
// }

TEST(element_utils, TestAjpElasticity) {
  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    // ret(0) = -1.2 * xloc(0);
    // ret(1) = 3.4 * xloc(1);
    return ret;
  };
  using Physics = LinearElasticity<T, 2, typeof(int_func)>;
  T E = 10.0, nu = 0.3;
  Physics physics(E, nu, int_func);

  test_adj_jac_product<2, WHICH::Ajp>(physics);
  // test_adj_jac_product<4, WHICH::Ajp>(physics);
  // test_adj_jac_product<6, WHICH::Ajp>(physics);
  // test_adj_jac_product<8, WHICH::Ajp>(physics);
}

TEST(element_utils, TestAjp1Elasticity) {
  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    // ret(0) = -1.2 * xloc(0);
    // ret(1) = 3.4 * xloc(1);
    return ret;
  };
  using Physics = LinearElasticity<T, 2, typeof(int_func)>;
  T E = 10.0, nu = 0.3;
  Physics physics(E, nu, int_func);

  test_adj_jac_product<2, WHICH::Ajp1>(physics);
  // test_adj_jac_product<4, WHICH::Ajp1>(physics);
  // test_adj_jac_product<6, WHICH::Ajp1>(physics);
  // test_adj_jac_product<8, WHICH::Ajp1>(physics);
}

TEST(element_utils, TestAjp3Elasticity) {
  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    // ret(0) = -1.2 * xloc(0);
    // ret(1) = 3.4 * xloc(1);
    return ret;
  };
  using Physics = LinearElasticity<T, 2, typeof(int_func)>;
  T E = 10.0, nu = 0.3;
  Physics physics(E, nu, int_func);

  test_adj_jac_product<2, WHICH::Ajp3>(physics);
  // test_adj_jac_product<4, WHICH::Ajp1>(physics);
  // test_adj_jac_product<6, WHICH::Ajp1>(physics);
  // test_adj_jac_product<8, WHICH::Ajp1>(physics);
}
