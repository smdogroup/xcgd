#include <limits>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"

constexpr int spatial_dim = 2;

template <typename T, class Mesh, class Quadrature>
class SingleQuadAnalysis {
 public:
  constexpr int static max_nnodes_per_element = Mesh::Np_1d * Mesh::Np_1d;

  // Constructor for regular analysis
  SingleQuadAnalysis(const Mesh& mesh, const Quadrature& quadrature)
      : mesh(mesh), quadrature(quadrature) {}

  // return the weight of j-th quadrature of i-th element
  auto weight(int elem, int quad) const {
    int i = elem;
    int j = quad;

    std::vector<T> pts, wts, ns;
    int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

    return std::make_tuple(wts[j], std::vector<T>{pts[2 * j], pts[2 * j + 1]});
  }

  // return the differential of the quadrature weight of the j-th quadrature of
  // the i-th element
  auto weight_deriv(int elem, int quad, std::vector<T> debug_p) const {
    int i = elem;
    int j = quad;

    // Create the element dfdphi
    std::vector<T> element_dw(max_nnodes_per_element, 0.0);
    std::vector<T> element_dx(max_nnodes_per_element, 0.0);
    std::vector<T> element_dy(max_nnodes_per_element, 0.0);

    std::vector<T> pts, wts, ns, pts_grad, wts_grad;
    int num_quad_pts =
        quadrature.get_quadrature_pts_grad(i, pts, wts, ns, pts_grad, wts_grad);

    xcgd_assert(j < num_quad_pts, "not enough number of quadrature points");

    int offset_wts = j * max_nnodes_per_element;
    int offset_pts = j * max_nnodes_per_element * Mesh::spatial_dim;

    for (int n = 0; n < max_nnodes_per_element; n++) {
      element_dw[n] = wts_grad[offset_wts + n];
      element_dx[n] = pts_grad[offset_pts + Mesh::spatial_dim * n];
      element_dy[n] = pts_grad[offset_pts + Mesh::spatial_dim * n + 1];
    }

    // TODO delete
    std::vector<T> element_p(max_nnodes_per_element, 0.0);
    const auto& lsf_mesh = mesh.get_lsf_mesh();
    int c = mesh.get_elem_cell(i);
    get_element_dfdphi<T, decltype(lsf_mesh), max_nnodes_per_element>(
        lsf_mesh, c, debug_p.data(), element_p.data());

    T dw = 0.0, dx = 0.0, dy = 0.0;
    for (int ii = 0; ii < max_nnodes_per_element; ii++) {
      dw += element_p[ii] * element_dw[ii];
      dx += element_p[ii] * element_dx[ii];
      dy += element_p[ii] * element_dy[ii];
    }

    return std::make_tuple(wts[j], dw,
                           std::vector<T>{pts[2 * j], pts[2 * j + 1]},
                           std::vector<T>{dx, dy});
  }

 private:
  const Mesh& mesh;
  const Quadrature& quadrature;
};

template <typename T, int Np_1d>
T directional_gradient_fd(int cell, int quad, double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;

  int nxy[2] = {3, 3};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh(grid, [](T x[]) {
    T n = 1.1;  // bad
    return (x[0] - 0.5) + n * (x[1] - 0.5);
  });

  Quadrature quadrature(mesh);

  using Analysis = SingleQuadAnalysis<T, Mesh, Quadrature>;

  Analysis analysis(mesh, quadrature);

  int ndv = grid.get_num_verts();
  std::vector<T> p(ndv, T(0.0));
  srand(42);
  for (int i = 0; i < ndv; i++) {
    p[i] = (double)rand() / RAND_MAX;
  }

  int elem = mesh.get_cell_elems().at(cell);

  // Exact
  auto [w_exact, dw_exact, pt_exact, dpt_exact] =
      analysis.weight_deriv(elem, quad, p);

  T dx_exact = dpt_exact[0];
  T dy_exact = dpt_exact[1];

  auto write_vtk = [&](int eval_num, T wt, std::vector<T> pt) {
    std::string name = "Np_" + std::to_string(Np_1d) + "_h_1e" +
                       std::to_string(int(log10(dh))) + "_" +
                       std::to_string(eval_num) + ".vtk";

    ToVTK<T, typeof(mesh.get_lsf_mesh())> vtk(mesh.get_lsf_mesh(),
                                              "grid_" + name);
    vtk.write_mesh();

    std::vector<T> node_indexing_v(mesh.get_lsf_mesh().get_num_nodes(),
                                   std::numeric_limits<T>::quiet_NaN());
    for (int n = 0; n < mesh.get_num_nodes(); n++) {
      int v = mesh.get_node_vert(n);
      node_indexing_v[v] = n;  // vert -> node
    }
    vtk.write_sol("node", node_indexing_v.data());
    vtk.write_sol("lsf", mesh.get_lsf_dof().data());

    std::vector<T> elem_indexing_v(mesh.get_lsf_mesh().get_num_elements(),
                                   std::numeric_limits<T>::quiet_NaN());
    for (int e = 0; e < mesh.get_num_elements(); e++) {
      int c = mesh.get_elem_cell(e);
      elem_indexing_v[c] = e;  // cell -> elem
    }

    vtk.write_cell_sol("elem", elem_indexing_v.data());

    FieldToVTKNew<T, 2> quad_vtk("quad_" + name);
    quad_vtk.add_mesh(pt);
    quad_vtk.write_mesh();

    quad_vtk.add_sol("wt", std::vector<T>{wt});
    quad_vtk.write_sol("wt");
  };

  // FD
  auto [w1, pt1] = analysis.weight(elem, quad);

  // Write to vtk
  if (dh > 0.9e-1 and dh < 1.1e-1) {
    write_vtk(0, w1, pt1);
  }

  auto& phi = mesh.get_lsf_dof();
  for (int i = 0; i < ndv; i++) {
    phi[i] += dh * p[i];
  }
  mesh.update_mesh();

  auto [w2, pt2] = analysis.weight(elem, quad);

  // Write to vtk
  if (dh > 0.9e-1 and dh < 1.1e-1) {
    write_vtk(1, w2, pt2);
  }

  T dw_fd = (w2 - w1) / dh;
  T dx_fd = (pt2[0] - pt1[0]) / dh;
  T dy_fd = (pt2[1] - pt1[1]) / dh;

  T relerr_w = fabs(dw_fd - dw_exact) / fabs(dw_exact);
  T relerr_x = fabs(dx_fd - dx_exact) / fabs(dx_exact);
  T relerr_y = fabs(dy_fd - dy_exact) / fabs(dy_exact);

  std::printf(
      "[cell:%d][q:%d][we/w1:%10f/%10f]Np_1d: %d, dh: %.5e, "
      "FD(w): %12.5e, Actual(w): %12.5e, Rel err(w): %12.5e, "
      "FD(x): %12.5e, Actual(x): %12.5e, Rel err(x): %12.5e, "
      "FD(y): %12.5e, Actual(y): %12.5e, Rel err(y): %12.5e"
      "\n",
      cell, quad, w_exact, w1, Np_1d, dh, dw_fd, dw_exact, relerr_w, dx_fd,
      dx_exact, relerr_x, dy_fd, dy_exact, relerr_y);

  return hard_max<T>({relerr_w, relerr_x, relerr_y});
}

template <typename T, int Np_1d>
T full_gradient_fd(int cell, int quad, double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;

  int nxy[2] = {3, 3};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh(grid, [](T x[]) {
    T n = 1.1;  // bad
    return (x[0] - 0.5) + n * (x[1] - 0.5);
  });

  Quadrature quadrature(mesh);

  using Analysis = SingleQuadAnalysis<T, Mesh, Quadrature>;

  Analysis analysis(mesh, quadrature);

  int ndv = grid.get_num_verts();

  int elem = mesh.get_cell_elems().at(cell);

  std::vector<T> w;
  std::vector<T> grad_fd, grad_exact;
  for (int i = 0; i < ndv; i++) {
    // Set direction
    std::vector<T> p(ndv, T(0.0));
    p[i] = 1.0;

    // Exact
    auto [w_exact, dw_exact, pt_exact, dpt_exact] =
        analysis.weight_deriv(elem, quad, p);

    // FD
    auto [w1, pt1] = analysis.weight(elem, quad);

    auto& phi = mesh.get_lsf_dof();
    for (int i = 0; i < ndv; i++) {
      phi[i] += dh * p[i];
    }
    mesh.update_mesh();

    auto [w2, pt2] = analysis.weight(elem, quad);

    grad_exact.push_back(dw_exact);
    grad_fd.push_back((w2 - w1) / dh);
    w.push_back(w_exact);
  }

  double zero_tol = 1e-12;
  for (int i = 0; i < ndv; i++) {
    if (fabs(grad_exact[i]) < zero_tol) grad_exact[i] = 0.0;
    if (fabs(grad_fd[i]) < zero_tol) grad_fd[i] = 0.0;
  }

  printf("ignore gradient entries with zero_tol : %.1e\n", zero_tol);

  T max_err = 0.0;
  for (int i = 0; i < ndv; i++) {
    T abserr = fabs(grad_exact[i] - grad_fd[i]);
    T relerr = abserr / fabs(grad_exact[i]);
    printf(
        "[Np=%d][dh:%10.2e]w[%2d]:%15.5e, grad_exact[%2d]:%15.5e, "
        "grad_fd[%2d]:%15.5e, "
        "relerr:%15.5e\n",
        Np_1d, dh, i, w[i], i, grad_exact[i], i, grad_fd[i], relerr);

    if (fabs(grad_exact[i]) > zero_tol and relerr > max_err) max_err = relerr;
  }
  return max_err;
}

template <int Np_1d>
void directional_gradient_sweep() {
  double tol = 1e-6;

  int cell = 2;
  int quad = 0;

  double min_err = 1e20;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    double err = directional_gradient_fd<double, Np_1d>(cell, quad, dh);
    if (err < min_err) min_err = err;
  }
  EXPECT_LE(min_err, tol);
}

template <int Np_1d>
void full_gradient_sweep() {
  double tol = 1e-6;

  int cell = 2;
  int quad = 0;

  double min_err = 1e20;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    double err = full_gradient_fd<double, 2>(cell, quad, dh);
    if (err < min_err) min_err = err;
  }
  EXPECT_LE(min_err, tol);
}

TEST(quad_level, DirectionalGradNp2) { directional_gradient_sweep<2>(); }
TEST(quad_level, DirectionalGradNp4) { directional_gradient_sweep<4>(); }
TEST(quad_level, FullGradNp2) { full_gradient_sweep<2>(); }
TEST(quad_level, FullGradNp4) { full_gradient_sweep<4>(); }
