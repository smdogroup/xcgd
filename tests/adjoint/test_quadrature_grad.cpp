#include <vector>

#include "analysis.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"

class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(double k = 0.9, double b = 0.1) : k(k), b(b) {}

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

TEST(adjoint, JacPsiProduct) {
  constexpr int Np_1d = 2;
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = Basis::Mesh;
  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;

  using Physics = LinearElasticity<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  constexpr int spatial_dim = Basis::spatial_dim;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Mesh lsf_mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh, lsf_mesh);

  std::vector<T>& lsf_dof = mesh.get_lsf_dof();
  std::vector<T> p(lsf_dof.size());

  int elem = 0;

  std::vector<T> pts, wts;
  quadrature.get_quadrature_pts(elem, pts, wts);

  for (int i = 0; i < p.size(); i++) {
    p[i] = T(rand()) / RAND_MAX;
  }

  double h = 1e-6;

  for (int i = 0; i < p.size(); i++) {
    lsf_dof[i] -= h * p[i];
  }

  std::vector<T> pts1, wts1;
  int num_quad_pts1 = quadrature.get_quadrature_pts(elem, pts1, wts1);

  for (int i = 0; i < p.size(); i++) {
    lsf_dof[i] += 2.0 * h * p[i];
  }

  std::vector<T> pts2, wts2;
  int num_quad_pts2 = quadrature.get_quadrature_pts(elem, pts2, wts2);

  if (num_quad_pts1 != num_quad_pts2) {
    char msg[256];
    std::snprintf(msg, 256, "number of quadrature points changed: %d -> %d",
                  num_quad_pts1, num_quad_pts2);
    throw std::runtime_error(msg);
  }

  T dxidphi_fd = 0.0, dwdphi_fd = 0.0;
  for (int i = 0; i < num_quad_pts1; i++) {
    dwdphi_fd += (wts2[i] - wts1[i]) / (2.0 * h);
    for (int d = 0; d < spatial_dim; d++) {
      dxidphi_fd +=
          (pts2[spatial_dim * i + d] - pts1[spatial_dim * i + d]) / (2.0 * h);
    }
  }

  T dxidphi_ad = 0.0, dwdphi_ad = 0.0;

  std::printf("dwdphi_fd:  %25.15e\n", dwdphi_fd);
  std::printf("dxidphi_fd: %25.15e\n", dxidphi_fd);
}