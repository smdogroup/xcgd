#include <vector>

#include "analysis.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"
#include "utils/vtk.h"

class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(double k = 0.4, double b = 0.7) : k(k), b(b) {}

  template <typename T>
  T operator()(const algoim::uvector<T, spatial_dim>& x) const {
    return -k * x(0) + x(1) - b;
  }

  template <typename T>
  algoim::uvector<T, spatial_dim> grad(const algoim::uvector<T, 2>& x) const {
    return algoim::uvector<T, spatial_dim>(-k, 1.0);
  }

 private:
  double k, b;
};

TEST(adjoint, GDLSFQuadratureGradient) {
  constexpr int Np_1d = 4;
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;

  constexpr int spatial_dim = Mesh::spatial_dim;
  constexpr int nodes_per_element = Basis::nodes_per_element;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  std::vector<T>& lsf_dof = mesh.get_lsf_dof();

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> pts, wts, pts_grad, wts_grad;
    int num_quad_pts =
        quadrature.get_quadrature_pts_grad(elem, pts, wts, pts_grad, wts_grad);

    FieldToVTK<T, spatial_dim> vtk("quadratures.vtk");
    vtk.add_scalar_field(pts, wts);
    vtk.write_vtk();

    T elem_p[nodes_per_element];
    for (int i = 0; i < nodes_per_element; i++) {
      elem_p[i] = T(rand()) / RAND_MAX;
    }

    std::vector<T> p(lsf_dof.size(), 0.0);
    int cell = mesh.get_elem_cell(elem);
    add_element_res<T, 1, GridMesh<T, Np_1d>, Basis>(mesh.get_lsf_mesh(), cell,
                                                     elem_p, p.data());

    double h = 1e-6;
    double tol = 1e-8;

    for (int i = 0; i < p.size(); i++) {
      lsf_dof[i] -= h * p[i];
    }

    std::vector<T> pts1, wts1;
    int num_quad_pts1 = quadrature.get_quadrature_pts(elem, pts1, wts1);

    FieldToVTK<T, spatial_dim> vtk1("quadratures1.vtk");
    vtk1.add_scalar_field(pts1, wts1);
    vtk1.write_vtk();

    for (int i = 0; i < p.size(); i++) {
      lsf_dof[i] += 2.0 * h * p[i];
    }

    std::vector<T> pts2, wts2;
    int num_quad_pts2 = quadrature.get_quadrature_pts(elem, pts2, wts2);

    for (int i = 0; i < p.size(); i++) {
      lsf_dof[i] -= 2.0 * h * p[i];
    }

    if (num_quad_pts != num_quad_pts1 or num_quad_pts != num_quad_pts2) {
      char msg[256];
      std::snprintf(msg, 256, "number of quadrature points changed: %d, %d, %d",
                    num_quad_pts, num_quad_pts1, num_quad_pts2);
      throw std::runtime_error(msg);
    }

    std::vector<T> pts_grad_fd(spatial_dim * num_quad_pts, 0.0);
    std::vector<T> wts_grad_fd(num_quad_pts, 0.0);
    std::vector<T> pts_grad_ad(spatial_dim * num_quad_pts, 0.0);
    std::vector<T> wts_grad_ad(num_quad_pts, 0.0);

    for (int i = 0; i < num_quad_pts1; i++) {
      wts_grad_fd[i] += (wts2[i] - wts1[i]) / (2.0 * h);
      for (int j = 0; j < nodes_per_element; j++) {
        wts_grad_ad[i] += wts_grad[i * nodes_per_element + j] * elem_p[j];
      }
      for (int d = 0; d < spatial_dim; d++) {
        int index = spatial_dim * i + d;
        pts_grad_fd[index] += (pts2[index] - pts1[index]) / (2.0 * h);
        for (int j = 0; j < nodes_per_element; j++) {
          pts_grad_ad[index] +=
              pts_grad[d + spatial_dim * (j + i * nodes_per_element)] *
              elem_p[j];
        }
      }
    }

    double max_err = 0.0;

    std::printf("elem: %d, num_quad_pts: %d\n", elem, num_quad_pts);
    std::printf("weights:\n");
    std::printf("    %25s%25s%25s\n", "fd", "ad", "rel.err");
    for (int i = 0; i < num_quad_pts; i++) {
      double dw_err = fabs((wts_grad_ad[i] - wts_grad_fd[i]) / wts_grad_fd[i]);
      if (wts_grad_ad[i] == wts_grad_fd[i]) dw_err = 0.0;
      if (dw_err > max_err) max_err = dw_err;
      std::printf("[%2d]%25.10e%25.10e%25.10e\n", i, wts_grad_fd[i],
                  wts_grad_ad[i], dw_err);
    }
    std::printf("points:\n");
    std::printf("    %25s%25s%25s\n", "fd", "ad", "rel.err");
    for (int i = 0; i < num_quad_pts; i++) {
      for (int d = 0; d < spatial_dim; d++) {
        int index = spatial_dim * i + d;
        double dxi_err = fabs((pts_grad_ad[index] - pts_grad_fd[index]) /
                              pts_grad_fd[index]);
        if (pts_grad_ad[index] == pts_grad_fd[index]) dxi_err = 0.0;
        if (dxi_err > max_err) max_err = dxi_err;
        std::printf("[%2d]%25.10e%25.10e%25.10e\n", i, pts_grad_fd[index],
                    pts_grad_ad[index], dxi_err);
      }
    }
    EXPECT_NEAR(max_err, 0.0, tol);
  }
}