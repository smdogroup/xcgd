#include <vector>

#include "analysis.h"
#include "elements/element_commons.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"
#include "utils/vtk.h"

class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(double k = 0.4, double b = 0.7) : k(k), b(b) {}

  template <typename T>
  T operator()(const T* x) const {
    return -k * x[0] + x[1] - b;
  }

 private:
  double k, b;
};

template <typename T, int Np_1d, class Mesh, class Basis, class Quadrature>
void test_lsf_quad_grad(Mesh& mesh, Basis& basis, Quadrature& quadrature,
                        double tol = 1e-8, double h = 1e-6) {
  constexpr int spatial_dim = Mesh::spatial_dim;
  constexpr int max_nnodes_per_element = Basis::max_nnodes_per_element;

  std::vector<T>& lsf_dof = mesh.get_lsf_dof();

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> pts, wts, wns, pts_grad, wts_grad, wns_grad;

    int num_quad_pts = -1;
    if (Quadrature::quad_type == QuadPtType::SURFACE) {
      num_quad_pts = quadrature.get_quadrature_pts_grad(
          elem, pts, wts, wns, pts_grad, wts_grad, wns_grad);
    } else {
      num_quad_pts = quadrature.get_quadrature_pts_grad(elem, pts, wts, wns,
                                                        pts_grad, wts_grad);
    }

    FieldToVTK<T, spatial_dim> vtk("quadratures.vtk");
    vtk.add_scalar_field(pts, wts);
    vtk.write_vtk();

    T elem_p[max_nnodes_per_element];
    for (int i = 0; i < max_nnodes_per_element; i++) {
      elem_p[i] = T(rand()) / RAND_MAX;
    }

    std::vector<T> p(lsf_dof.size(), 0.0);
    int cell = mesh.get_elem_cell(elem);
    add_element_res<T, 1, GridMesh<T, Np_1d>, Basis>(mesh.get_lsf_mesh(), cell,
                                                     elem_p, p.data());

    for (int i = 0; i < p.size(); i++) {
      lsf_dof[i] -= h * p[i];
    }

    std::vector<T> pts1, wts1, wns1;
    int num_quad_pts1 = quadrature.get_quadrature_pts(elem, pts1, wts1, wns1);

    FieldToVTK<T, spatial_dim> vtk1("quadratures1.vtk");
    vtk1.add_scalar_field(pts1, wts1);
    vtk1.write_vtk();

    for (int i = 0; i < p.size(); i++) {
      lsf_dof[i] += 2.0 * h * p[i];
    }

    std::vector<T> pts2, wts2, wns2;
    int num_quad_pts2 = quadrature.get_quadrature_pts(elem, pts2, wts2, wns2);

    for (int i = 0; i < p.size(); i++) {
      lsf_dof[i] -= 2.0 * h * p[i];
    }

    if (num_quad_pts != num_quad_pts1 or num_quad_pts != num_quad_pts2) {
      char msg[256];
      std::snprintf(msg, 256, "number of quadrature points changed: %d, %d, %d",
                    num_quad_pts, num_quad_pts1, num_quad_pts2);
      throw std::runtime_error(msg);
    }

    std::vector<T> wts_grad_fd(num_quad_pts, 0.0);
    std::vector<T> wts_grad_ad(num_quad_pts, 0.0);

    std::vector<T> pts_grad_fd(spatial_dim * num_quad_pts, 0.0);
    std::vector<T> pts_grad_ad(spatial_dim * num_quad_pts, 0.0);

    std::vector<T> wns_grad_fd(spatial_dim * num_quad_pts, 0.0);
    std::vector<T> wns_grad_ad(spatial_dim * num_quad_pts, 0.0);

    for (int i = 0; i < num_quad_pts1; i++) {
      wts_grad_fd[i] += (wts2[i] - wts1[i]) / (2.0 * h);
      for (int j = 0; j < max_nnodes_per_element; j++) {
        wts_grad_ad[i] += wts_grad[i * max_nnodes_per_element + j] * elem_p[j];
      }
      for (int d = 0; d < spatial_dim; d++) {
        int index = spatial_dim * i + d;
        pts_grad_fd[index] += (pts2[index] - pts1[index]) / (2.0 * h);
        if (not wns_grad.empty()) {
          wns_grad_fd[index] += (wns2[index] - wns1[index]) / (2.0 * h);
        }

        for (int j = 0; j < max_nnodes_per_element; j++) {
          pts_grad_ad[index] +=
              pts_grad[d + spatial_dim * (j + i * max_nnodes_per_element)] *
              elem_p[j];
          if (not wns_grad.empty()) {
            wns_grad_ad[index] +=
                wns_grad[d + spatial_dim * (j + i * max_nnodes_per_element)] *
                elem_p[j];
          }
        }
      }
    }

    double max_err = 0.0;

    bool verbose = true;
    if (verbose) {
      std::printf("elem: %d, num_quad_pts: %d\n", elem, num_quad_pts);
      std::printf("weights:\n");
      std::printf("    %25s%25s%25s\n", "fd", "ad", "rel.err");
    }
    for (int i = 0; i < num_quad_pts; i++) {
      double dw_err = fabs((wts_grad_ad[i] - wts_grad_fd[i]) / wts_grad_fd[i]);
      if (wts_grad_ad[i] == wts_grad_fd[i]) dw_err = 0.0;
      if (dw_err > max_err) max_err = dw_err;
      if (verbose) {
        std::printf("[%2d]%25.10e%25.10e%25.10e\n", i, wts_grad_fd[i],
                    wts_grad_ad[i], dw_err);
      }
    }
    if (verbose) {
      std::printf("points:\n");
      std::printf("    %25s%25s%25s\n", "fd", "ad", "rel.err");
    }
    for (int i = 0; i < num_quad_pts; i++) {
      for (int d = 0; d < spatial_dim; d++) {
        int index = spatial_dim * i + d;
        double dxi_err = fabs((pts_grad_ad[index] - pts_grad_fd[index]) /
                              pts_grad_fd[index]);
        if (pts_grad_ad[index] == pts_grad_fd[index]) dxi_err = 0.0;
        if (dxi_err > max_err) max_err = dxi_err;
        if (verbose) {
          std::printf("[%2d]%25.10e%25.10e%25.10e\n", i, pts_grad_fd[index],
                      pts_grad_ad[index], dxi_err);
        }
      }
    }

    if (verbose) {
      std::printf("norms:\n");
      std::printf("    %25s%25s%25s\n", "fd", "ad", "rel.err");
    }
    for (int i = 0; i < num_quad_pts; i++) {
      for (int d = 0; d < spatial_dim; d++) {
        int index = spatial_dim * i + d;
        double dxi_err = fabs((wns_grad_ad[index] - wns_grad_fd[index]) /
                              wns_grad_fd[index]);
        if (wns_grad_ad[index] == wns_grad_fd[index]) dxi_err = 0.0;
        if (dxi_err > max_err) max_err = dxi_err;
        if (verbose) {
          std::printf("[%2d]%25.10e%25.10e%25.10e\n", i, wns_grad_fd[index],
                      wns_grad_ad[index], dxi_err);
        }
      }
    }

    std::printf("max error: %25.10e\n", max_err);
    EXPECT_NEAR(max_err, 0.0, tol);
  }
}

template <int Np_1d, bool use_finite_cell_mesh>
void test_bulk(double tol = 1e-8, double h = 1e-6) {
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh =
      typename std::conditional<use_finite_cell_mesh, FiniteCellMesh<T, Np_1d>,
                                CutMesh<T, Np_1d>>::type;
  using Basis = GDBasis2D<T, Mesh>;
  using LSF = Line;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  test_lsf_quad_grad<T, Np_1d>(mesh, basis, quadrature, tol, h);
}

template <int Np_1d, bool use_finite_cell_mesh>
void test_surf(double tol = 1e-8, double h = 1e-6) {
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh =
      typename std::conditional<use_finite_cell_mesh, FiniteCellMesh<T, Np_1d>,
                                CutMesh<T, Np_1d>>::type;
  using Basis = GDBasis2D<T, Mesh>;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE, Grid, Np_1d, Mesh>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);

  auto lsf = [](T x[]) {
    T n = 0.21;
    return (x[0] - 0.5) + n * (x[1] - 0.5);
  };

  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  test_lsf_quad_grad<T, Np_1d>(mesh, basis, quadrature, tol, h);
}

TEST(adjoint, DISABLED_GDBulkQuadratureGradNp2) {
  test_bulk<2, false>(1e-6);
  test_bulk<2, true>(1e-6);
}

TEST(adjoint, DISABLED_GDBulkQuadratureGradNp4) {
  test_bulk<4, false>(1e-6);
  test_bulk<4, true>(1e-6);
}

TEST(adjoint, DISABLED_GDBulkQuadratureGradNp6) {
  test_bulk<6, false>(1e-6);
  test_bulk<6, true>(1e-6);
}

TEST(adjoint, DISABLED_GDSurfQuadratureGradNp2) {
  test_surf<2, false>(1e-6);
  test_surf<2, true>(1e-6);
}

TEST(adjoint, DISABLED_GDSurfQuadratureGradNp4) {
  test_surf<4, false>(1e-6);
  test_surf<4, true>(1e-6);
}

TEST(adjoint, DISABLED_GDSurfQuadratureGradNp6) {
  test_surf<6, false>(1e-6);
  test_surf<6, true>(1e-6);
}
