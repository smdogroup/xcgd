#include "analysis.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"

template <typename T, class Mesh, class Quadrature, class Basis, class Physics>
class SingleQuadAnalysis {
 public:
  static constexpr int spatial_dim = Basis::spatial_dim;
  static constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  // Static data taken from the physics
  static constexpr int dof_per_node = Physics::dof_per_node;
  static constexpr int data_per_node = Physics::data_per_node;

  static constexpr int max_dof_per_element =
      dof_per_node * max_nnodes_per_element;

  // Constructor for regular analysis
  SingleQuadAnalysis(const Mesh& mesh, const Quadrature& quadrature,
                     const Basis& basis, const Physics& physics)
      : mesh(mesh), quadrature(quadrature), basis(basis), physics(physics) {}

  // TODO: revert
  auto residual(int elem, int quad, const T x[], const T dof[],
                std::vector<T> debug_psi = {}) const {
    T xq = 0.0;
    std::vector<T> element_x = std::vector<T>(max_nnodes_per_element);

    std::vector<T>
        debug_rTp_q;  // inner product of element residual and element psi
    std::vector<T> debug_dedu_psi_q;  // inner product of ded∇uq * ∇psiq
    std::vector<T> debug_xloc_q;

    // for (int i = 0; i < mesh.get_num_elements(); i++)
    {
      int i = elem;
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes;
      nnodes = mesh.get_elem_dof_nodes(i, nodes);

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element degrees of freedom
      T element_dof[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);

      std::vector<T> debug_element_psi(max_dof_per_element, 0.0);
      if (not debug_psi.empty()) {
        get_element_vars<T, dof_per_node, Basis>(
            nnodes, nodes, debug_psi.data(), debug_element_psi.data());
      }

      // Get element design variable if needed
      if (x) {
        get_element_vars<T, 1, Basis>(nnodes, nodes, x, element_x.data());
      }

      // Create the element residual
      T element_res[max_dof_per_element];
      for (int j = 0; j < max_dof_per_element; j++) {
        element_res[j] = 0.0;
      }

      std::vector<T> pts, wts, ns;
      int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(i, pts, N, Nxi);

      // for (int j = 0; j < num_quad_pts; j++)
      {
        int j = quad;
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;

        // Evaluate the derivative of the spatial dof in the computational
        // coordinates
        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t vals{};
        typename Physics::grad_t grad{}, grad_ref{};
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(vals),
            get_ptr(grad_ref));
        if (x) {
          interp_val_grad<T, spatial_dim, max_nnodes_per_element,
                          data_per_node>(element_x.data(), &N[offset_n],
                                         nullptr, get_ptr(xq), nullptr);
        }

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Transform gradient from ref coordinates to physical coordinates
        transform(J, grad_ref, grad);

        // Evaluate the residuals at the quadrature points
        typename Physics::dof_t coef_vals{};
        typename Physics::grad_t coef_grad{}, coef_grad_ref{};
        physics.residual(wts[j], xq, xloc, nrm_ref, J, vals, grad, coef_vals,
                         coef_grad);

        // Transform gradient from physical coordinates back to ref coordinates
        rtransform(J, coef_grad, coef_grad_ref);

        // Add the contributions to the element residual
        add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals,
                           coef_grad_ref, element_res);

        // TODO: delete
        if (not debug_psi.empty()) {
          std::vector<T> q_res(max_dof_per_element, 0.0);
          add_grad<T, Basis>(&N[offset_n], &Nxi[offset_nxi], coef_vals,
                             coef_grad_ref, q_res.data());
          T rTp = 0.0;
          for (int ii = 0; ii < max_dof_per_element; ii++) {
            rTp += debug_element_psi[ii] * q_res[ii];
          }
          debug_rTp_q.push_back(rTp);
          for (int d = 0; d < spatial_dim; d++) {
            debug_xloc_q.push_back(xloc(d));
          }

          typename Physics::dof_t psiq{};        // uq, psiq
          typename Physics::grad_t pgrad_ref{};  // (∇_x)psiq, (∇_ξ)psiq
          interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
              debug_element_psi.data(), &N[offset_n], &Nxi[offset_nxi],
              get_ptr(psiq), get_ptr(pgrad_ref));

          T dedu_psi = 0.0;
          for (int jj = 0; jj < Physics::grad_t::ncomp; jj++) {
            dedu_psi += coef_grad_ref[jj] * pgrad_ref[jj];
          }
          debug_dedu_psi_q.push_back(dedu_psi);
        }
      }
    }

    return std::make_tuple(debug_xloc_q, debug_rTp_q);
  }

  auto LSF_jacobian_adjoint_product(int elem, int quad, const T dof[],
                                    const T psi[],
                                    std::vector<T> debug_p = {}) const {
    static_assert(Basis::is_gd_basis, "This method only works with GD Basis");
    static_assert(Mesh::is_cut_mesh,
                  "This method requires a level-set-cut mesh");

    std::vector<T> debug_xloc_q;
    std::vector<T> debug_dajp_q;

    // for (int i = 0; i < mesh.get_num_elements(); i++)
    {
      int i = elem;
      // Get nodes associated to this element
      int nodes[Mesh::max_nnodes_per_element];
      int nnodes = mesh.get_elem_dof_nodes(i, nodes);

      // Get the element node locations
      T element_xloc[spatial_dim * max_nnodes_per_element];
      get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

      // Get the element states and adjoints
      T element_dof[max_dof_per_element], element_psi[max_dof_per_element];
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, dof, element_dof);
      get_element_vars<T, dof_per_node, Basis>(nnodes, nodes, psi, element_psi);

      // Create the element dfdphi
      std::vector<T> element_dfdphi(max_nnodes_per_element, 0.0);

      std::vector<T> pts, wts, ns, pts_grad, wts_grad;
      int num_quad_pts = quadrature.get_quadrature_pts_grad(i, pts, wts, ns,
                                                            pts_grad, wts_grad);

      std::vector<T> N, Nxi, Nxixi;
      basis.eval_basis_grad(i, pts, N, Nxi, Nxixi);

      // for (int j = 0; j < num_quad_pts; j++)
      {
        int j = quad;
        int offset_n = j * max_nnodes_per_element;
        int offset_nxi = j * max_nnodes_per_element * spatial_dim;
        int offset_nxixi =
            j * max_nnodes_per_element * spatial_dim * spatial_dim;

        A2D::Vec<T, spatial_dim> xloc, nrm_ref;
        A2D::Mat<T, spatial_dim, spatial_dim> J;
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, spatial_dim>(
            element_xloc, &N[offset_n], &Nxi[offset_nxi], get_ptr(xloc),
            get_ptr(J));

        if constexpr (Quadrature::quad_type == QuadPtType::SURFACE) {
          for (int d = 0; d < spatial_dim; d++) {
            nrm_ref[d] = ns[spatial_dim * j + d];
          }
        }

        // Evaluate the derivative of the dof in the computational coordinates
        typename Physics::dof_t uq{}, psiq{};           // uq, psiq
        typename Physics::grad_t ugrad{}, ugrad_ref{};  // (∇_x)uq, (∇_ξ)uq
        typename Physics::grad_t pgrad{}, pgrad_ref{};  // (∇_x)psiq, (∇_ξ)psiq
        typename Physics::hess_t uhess_ref{};           //(∇2_ξ)uq
        typename Physics::hess_t phess_ref{};           //(∇2_ξ)psiq

        // Interpolate the quantities at the quadrature point
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &N[offset_n], &Nxi[offset_nxi], get_ptr(uq),
            get_ptr(ugrad_ref));
        interp_val_grad<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi, &N[offset_n], &Nxi[offset_nxi], get_ptr(psiq),
            get_ptr(pgrad_ref));

        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_dof, &Nxixi[offset_nxixi], get_ptr(uhess_ref));
        interp_hess<T, spatial_dim, max_nnodes_per_element, dof_per_node>(
            element_psi, &Nxixi[offset_nxixi], get_ptr(phess_ref));

        transform(J, ugrad_ref, ugrad);
        transform(J, pgrad_ref, pgrad);

        typename Physics::dof_t coef_uq{};      // ∂e/∂uq
        typename Physics::grad_t coef_ugrad{};  // ∂e/∂(∇_x)uq
        typename Physics::dof_t jp_uq{};        // ∂2e/∂uq2 * psiq
        typename Physics::grad_t jp_ugrad{};    // ∂2e/∂(∇_x)uq2 * (∇_x)psiq

        physics.residual(wts[j], 0.0, xloc, nrm_ref, J, uq, ugrad, coef_uq,
                         coef_ugrad);
        physics.jacobian_product(wts[j], 0.0, xloc, nrm_ref, J, uq, ugrad, psiq,
                                 pgrad, jp_uq, jp_ugrad);

        typename Physics::grad_t coef_ugrad_ref{};  // ∂e/∂(∇_ξ)uq
        typename Physics::grad_t jp_ugrad_ref{};    // ∂2e/∂(∇_ξ)uq2 * (∇_ξ)psiq

        // Transform gradient from physical coordinates back to ref
        // coordinates
        rtransform(J, coef_ugrad, coef_ugrad_ref);
        rtransform(J, jp_ugrad, jp_ugrad_ref);

        int offset_wts = j * max_nnodes_per_element;
        int offset_pts = j * max_nnodes_per_element * spatial_dim;

        add_jac_adj_product_bulk<T, Basis>(
            wts[j], &wts_grad[offset_wts], &pts_grad[offset_pts], psiq,
            ugrad_ref, pgrad_ref, uhess_ref, phess_ref, coef_uq, coef_ugrad_ref,
            jp_uq, jp_ugrad_ref, element_dfdphi.data());

        // TODO delete
        if (not debug_p.empty()) {
          std::vector<T> q_dfdphi(max_nnodes_per_element, 0.0);
          add_jac_adj_product_bulk<T, Basis>(
              wts[j], &wts_grad[offset_wts], &pts_grad[offset_pts], psiq,
              ugrad_ref, pgrad_ref, uhess_ref, phess_ref, coef_uq,
              coef_ugrad_ref, jp_uq, jp_ugrad_ref, q_dfdphi.data());

          std::vector<T> element_p(max_nnodes_per_element, 0.0);
          const auto& lsf_mesh = mesh.get_lsf_mesh();
          int c = mesh.get_elem_cell(i);
          get_element_dfdphi<T, decltype(lsf_mesh), Basis>(
              lsf_mesh, c, debug_p.data(), element_p.data());

          T dajp = 0.0;
          for (int ii = 0; ii < max_nnodes_per_element; ii++) {
            dajp += element_p[ii] * q_dfdphi[ii];
          }

          debug_dajp_q.push_back(dajp);
          for (int d = 0; d < spatial_dim; d++) {
            debug_xloc_q.push_back(xloc(d));
          }
        }
      }
    }

    return std::make_tuple(debug_xloc_q, debug_dajp_q);
  }

 private:
  const Mesh& mesh;
  const Quadrature& quadrature;
  const Basis& basis;
  const Physics& physics;
};

template <int Np_1d>
void finite_difference_check(double dh = 1e-6) {
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh(grid, [](T x[]) {
    T n = 0.201;  // bad
    return (x[0] - 0.5) + n * (x[1] - 0.5);
  });

  Basis basis(mesh);
  Quadrature quadrature(mesh);

  auto int_func = [](const A2D::Vec<T, 2> xloc) {
    A2D::Vec<T, 2> ret;
    return ret;
  };

  using Physics = LinearElasticity<T, Basis::spatial_dim, typeof(int_func)>;

  T E = 20.0, nu = 0.3;
  Physics physics(E, nu, int_func);

  using Analysis = SingleQuadAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  Analysis analysis(mesh, quadrature, basis, physics);

  constexpr int dof_per_node = Physics::dof_per_node;

  int ndof = dof_per_node * mesh.get_num_nodes();
  int ndv = grid.get_num_verts();

  std::vector<T> dof(ndof, T(0.0)), psi(ndof, T(0.0)), res1(ndof, T(0.0)),
      res2(ndof, T(0.0));
  std::vector<T> dfdphi(ndv, T(0.0)), p(ndv, T(0.0));

  srand(42);
  for (int i = 0; i < ndv; i++) {
    p[i] = (double)rand() / RAND_MAX;
  }

  for (int i = 0; i < ndof; i++) {
    dof[i] = (double)rand() / RAND_MAX;
    psi[i] = (double)rand() / RAND_MAX;
  }

  int ix = 2, iy = 0;
  int cell = grid.get_coords_cell(ix, iy);
  int elem = mesh.get_cell_elems().at(cell);
  int quad = 0;

  // Exact
  auto [debug_xloc_q, debug_dajp_q] = analysis.LSF_jacobian_adjoint_product(
      elem, quad, dof.data(), psi.data(), p);

  // FD
  auto [debug_xloc_q_1, debug_rTp_q_1] =
      analysis.residual(elem, quad, nullptr, dof.data(), psi);

  auto& phi = mesh.get_lsf_dof();
  for (int i = 0; i < ndv; i++) {
    phi[i] += dh * p[i];
  }
  mesh.update_mesh();

  auto [debug_xloc_q_2, debug_rTp_q_2] =
      analysis.residual(elem, quad, nullptr, dof.data(), psi);

  {
    if (debug_rTp_q_1.size() != debug_rTp_q_2.size() or
        debug_rTp_q_1.size() != debug_dajp_q.size()) {
      std::printf("number of quad pts changes through FD, skipping...\n");
    } else {
      int num_quads = debug_rTp_q_1.size();

      for (int j = 0; j < num_quads; j++) {
        T fd_q = (debug_rTp_q_2[j] - debug_rTp_q_1[j]) / dh;
        T exact_q = debug_dajp_q[j];
        T relerr_q = fabs(fd_q - exact_q) / fabs(exact_q);

        T dx = hard_max<T>({abs(debug_xloc_q_1[2 * j] - debug_xloc_q_2[2 * j]),
                            abs(debug_xloc_q_1[2 * j] - debug_xloc_q[2 * j]),
                            abs(debug_xloc_q[2 * j] - debug_xloc_q_2[2 * j])});
        T dy = hard_max<T>(
            {abs(debug_xloc_q_1[2 * j + 1] - debug_xloc_q_2[2 * j + 1]),
             abs(debug_xloc_q_1[2 * j + 1] - debug_xloc_q[2 * j + 1]),
             abs(debug_xloc_q[2 * j + 1] - debug_xloc_q_2[2 * j + 1])});

        std::printf(
            "[cell:%d][q:%d(%10.8f,%10.8f)]Np_1d: %d, dh: %.5e, FD: "
            "%30.20e, Actual: %30.20e, "
            "Rel "
            "err: %20.10e\n",
            cell, quad, debug_xloc_q[2 * j], debug_xloc_q[2 * j + 1], Np_1d, dh,
            fd_q, exact_q, relerr_q);
      }
    }
  }
}

TEST(quad_level, playground) {
  int constexpr Np_1d = 4;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    finite_difference_check<Np_1d>(dh);
  }
}
