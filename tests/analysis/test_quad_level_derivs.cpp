#include <limits>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"

template <typename T, class Mesh, class Quadrature>
class SingleQuadAnalysis {
 public:
  constexpr int static max_nnodes_per_element = Mesh::Np_1d * Mesh::Np_1d;

  // Constructor for regular analysis
  SingleQuadAnalysis(const Mesh& mesh, const Quadrature& quadrature)
      : mesh(mesh), quadrature(quadrature) {}

  // return the weight of j-th quadrature of i-th element
  T weight(int elem, int quad) const {
    int i = elem;
    int j = quad;

    std::vector<T> pts, wts, ns;
    int num_quad_pts = quadrature.get_quadrature_pts(i, pts, wts, ns);

    return wts[j];
  }

  // return the differential of the quadrature weight of the j-th quadrature of
  // the i-th element
  auto weight_deriv(int elem, int quad, std::vector<T> debug_p) const {
    int i = elem;
    int j = quad;

    // Create the element dfdphi
    std::vector<T> element_dw(max_nnodes_per_element, 0.0);

    std::vector<T> pts, wts, ns, pts_grad, wts_grad;
    int num_quad_pts =
        quadrature.get_quadrature_pts_grad(i, pts, wts, ns, pts_grad, wts_grad);

    int offset_wts = j * max_nnodes_per_element;

    for (int n = 0; n < max_nnodes_per_element; n++) {
      element_dw[n] = wts_grad[offset_wts + n];
    }

    // TODO delete
    std::vector<T> element_p(max_nnodes_per_element, 0.0);
    const auto& lsf_mesh = mesh.get_lsf_mesh();
    int c = mesh.get_elem_cell(i);
    get_element_dfdphi<T, decltype(lsf_mesh), max_nnodes_per_element>(
        lsf_mesh, c, debug_p.data(), element_p.data());

    T dw = 0.0;
    for (int ii = 0; ii < max_nnodes_per_element; ii++) {
      dw += element_p[ii] * element_dw[ii];
    }

    return dw;
  }

 private:
  const Mesh& mesh;
  const Quadrature& quadrature;
};

template <typename T, int Np_1d>
T finite_difference_check(int cell, int quad, double dh = 1e-6) {
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh(grid, [](T x[]) {
    T n = 0.201;  // bad
    // T n = 0.199;  // good
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
  T dw_exact = analysis.weight_deriv(elem, quad, p);

  // FD
  T w1 = analysis.weight(elem, quad);

  auto& phi = mesh.get_lsf_dof();
  for (int i = 0; i < ndv; i++) {
    phi[i] += dh * p[i];
  }
  mesh.update_mesh();

  T w2 = analysis.weight(elem, quad);

  T dw_fd = (w2 - w1) / dh;
  T relerr = fabs(dw_fd - dw_exact) / fabs(dw_exact);

  std::printf(
      "[cell:%d][q:%d]Np_1d: %d, dh: %.5e, FD: %30.20e, Actual: %30.20e, Rel "
      "err: %20.10e\n",
      cell, quad, Np_1d, dh, dw_fd, dw_exact, relerr);

  return relerr;
}

TEST(quad_level, playground) {
  int constexpr Np_1d = 4;
  double tol = 1e-6;

  int cell = 2;
  int quad = 0;

  double min_err = 1e20;
  for (double dh :
       std::vector<double>{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15}) {
    double err = finite_difference_check<double, Np_1d>(cell, quad, dh);
    if (err < min_err) min_err = err;
  }
  EXPECT_LE(min_err, tol);
}
