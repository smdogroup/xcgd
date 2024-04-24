#include <vector>

#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "physics/volume.h"
#include "test_commons.h"

#define PI 3.141592653589793

class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(double k = 0.4, double b = 0.7) : k(k), b(b) {}

  template <typename T>
  T operator()(const algoim::uvector<T, spatial_dim>& x) const {
    return -k * x(0) + x(1) - b;
  }

 private:
  double k, b;
};

template <typename T, class Basis>
T eval_det(Basis& basis, int elem, const T* element_xloc, std::vector<T> pt) {
  constexpr int spatial_dim = Basis::spatial_dim;
  constexpr int nodes_per_element = Basis::nodes_per_element;

  std::vector<T> N, Nxi, Nxixi;

  basis.eval_basis_grad(elem, pt, N, Nxi, Nxixi);

  A2D::Mat<T, spatial_dim, spatial_dim> J;
  A2D::Vec<T, spatial_dim> xloc;
  interp_val_grad<T, Basis, spatial_dim>(element_xloc, N.data(), Nxi.data(),
                                         &xloc, &J);

  T detJ = 0.0;
  A2D::MatDet(J, detJ);

  T tmp = 0.0;
  for (int i = 0; i < nodes_per_element; i++) {
    tmp += Nxixi[spatial_dim * spatial_dim * i] * element_xloc[spatial_dim * i];
  }

  return detJ;
}

TEST(adjoint, determinantGrad) {
  constexpr int Np_1d = 4;
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;

  using Physics = LinearElasticity<T, Basis::spatial_dim>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);
  std::vector<T> dof(mesh.get_num_nodes());
  for (int i = 0; i < dof.size(); i++) {
    dof[i] = T(i);
  }
  interp.to_vtk("interp.vtk", dof.data());
  ToVTK<T, Mesh> vtk(mesh, "mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> dof(mesh.get_num_nodes(), 0.0);
    int nodes[Mesh::nodes_per_element];
    mesh.get_elem_dof_nodes(elem, nodes);
    for (int i = 0; i < Mesh::nodes_per_element; i++) {
      dof[nodes[i]] = 1.0;
    }
    char name[256];
    std::snprintf(name, 256, "elem_%05d", elem);
    vtk.write_sol(name, dof.data());
  }

  T E = 30.0, nu = 0.3;
  Physics physics(E, nu);

  int i = 16;

  constexpr int spatial_dim = Basis::spatial_dim;
  constexpr int nodes_per_element = Basis::nodes_per_element;

  T element_xloc[spatial_dim * nodes_per_element];
  get_element_xloc<T, Mesh, Basis>(mesh, i, element_xloc);

  std::vector<T> pts, wts, pts_grad, wts_grad;
  int num_quad_pts =
      quadrature.get_quadrature_pts_grad(i, pts, wts, pts_grad, wts_grad);

  std::vector<T> N, Nxi, Nxixi;
  basis.eval_basis_grad(i, pts, N, Nxi, Nxixi);

  for (int j = 0; j < num_quad_pts; j++) {
    int offset_nxi = j * nodes_per_element * spatial_dim;
    A2D::Mat<T, spatial_dim, spatial_dim> J;
    interp_val_grad<T, Basis, spatial_dim>(element_xloc, nullptr,
                                           &Nxi[offset_nxi], nullptr, &J);

    A2D::Vec<T, spatial_dim> det_grad{};
    det_deriv<T, Basis>(element_xloc, Nxixi.data(), J, det_grad);

    T h = 1e-2;
    std::vector<T> p{0.7, -0.45};
    std::vector<T> pt1(spatial_dim);
    std::vector<T> pt2(spatial_dim);

    T det_grad_exact = 0.0;
    for (int d = 0; d < spatial_dim; d++) {
      // p[d] = 0.5 + (double)rand() / RAND_MAX;
      pt1[d] = pts[spatial_dim * j + d] - h * p[d];
      pt2[d] = pts[spatial_dim * j + d] + h * p[d];
      det_grad_exact += det_grad(d) * p[d];
    }

    T det_grad_fd = (eval_det(basis, i, element_xloc, pt2) -
                     eval_det(basis, i, element_xloc, pt1)) /
                    (2.0 * h);

    T err = (det_grad_exact - det_grad_fd) / det_grad_fd;
    std::printf("[%2d] exact: %20.10e, fd: %20.10e, relerr: %20.10e\n", j,
                det_grad_exact, det_grad_fd, err);
  }
}

template <typename T, class Mesh, class Quadrature, class Basis>
void quadratures_to_vtk(Mesh& mesh, Quadrature& quadrature, Basis& basis,
                        std::string vtk_name) {
  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);
  std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
  interp.to_vtk(vtk_name, dummy.data());
}

template <typename T, class Basis, class Mesh, class Quadrature, class Physics>
void test_jac_psi_product(Basis& basis, Mesh& mesh, Quadrature& quadrature,
                          Physics& physics, double h = 1e-6,
                          double tol = 1e-6) {
  int ndof = Physics::dof_per_node * mesh.get_num_nodes();
  int ndv = mesh.get_lsf_mesh().get_num_nodes();
  std::vector<T>& dvs = mesh.get_lsf_dof();

  std::vector<T> dof(ndof), psi(ndof), res1(ndof, 0.0), res2(ndof, 0.0);
  std::vector<T> dfdphi(ndv, 0.0), dadphi(ndv, 0.0), p(ndv);

  for (int i = 0; i < ndof; i++) {
    dof[i] = (T)rand() / RAND_MAX;
    psi[i] = (T)rand() / RAND_MAX;
  }
  for (int i = 0; i < ndv; i++) {
    p[i] = (T)rand() / RAND_MAX;
  }

  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  Analysis analysis(mesh, quadrature, basis, physics);
  analysis.LSF_jacobian_adjoint_product(dof.data(), psi.data(), dfdphi.data());

  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;
  Volume vol;
  VolAnalysis vol_analysis(mesh, quadrature, basis, vol);
  std::vector<T> dummy(ndv);

  vol_analysis.LSF_volume_derivatives(dadphi.data());

  for (int i = 0; i < ndv; i++) {
    dvs[i] -= h * p[i];
  }
  analysis.residual(nullptr, dof.data(), res1.data());
  T a1 = vol_analysis.energy(nullptr, dummy.data());

  for (int i = 0; i < ndv; i++) {
    dvs[i] += 2.0 * h * p[i];
  }
  analysis.residual(nullptr, dof.data(), res2.data());
  T a2 = vol_analysis.energy(nullptr, dummy.data());

  double dfdphi_fd = 0.0;
  double dadphi_fd = (a2 - a1) / 2.0 / h;
  for (int i = 0; i < ndof; i++) {
    dfdphi_fd += psi[i] * (res2[i] - res1[i]) / (2.0 * h);
  }

  double dfdphi_adjoint = 0.0;
  double dadphi_exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdphi_adjoint += dfdphi[i] * p[i];
    dadphi_exact += dadphi[i] * p[i];
  }

  std::printf("dfdphi_fd:      %25.15e\n", dfdphi_fd);
  std::printf("dfdphi_adjoint: %25.15e\n", dfdphi_adjoint);
  EXPECT_NEAR((dfdphi_fd - dfdphi_adjoint) / dfdphi_adjoint, 0.0, tol);

  std::printf("area grad fd:    %25.15e\n", dadphi_fd);
  std::printf("area grad exact: %25.15e\n", dadphi_exact);
  EXPECT_NEAR((dadphi_fd - dadphi_exact) / dadphi_exact, 0.0, tol);
}

TEST(adjoint, JacPsiProductElasticity) {
  constexpr int Np_1d = 4;
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Physics = LinearElasticity<T, Basis::spatial_dim>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  T E = 30.0, nu = 0.3;
  Physics physics(E, nu);

  test_jac_psi_product<T>(basis, mesh, quadrature, physics);
}

template <typename T, class Grid>
std::vector<T> create_initial_topology(Grid& grid, int m, int n) {
  const T* lxy = grid.get_lxy();
  int ndv = grid.get_num_verts();
  std::vector<T> x0(ndv, 0.0);
  for (int i = 0; i < ndv; i++) {
    T xloc[Grid::spatial_dim];
    grid.get_vert_xloc(i, xloc);
    x0[i] = (cos(xloc[0] / lxy[0] * 2.0 * PI * m) - 0.5) *
                (cos(xloc[1] / lxy[1] * 2.0 * PI * n) - 0.5) * 2.0 / 3.0 -
            0.5;
  }
  return x0;
}

TEST(adjoint, JacPsiProductPoisson) {
  constexpr int Np_1d = 4;
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using CutMesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, CutMesh>;
  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Physics = PoissonPhysics<T, Basis::spatial_dim>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  CutMesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  Physics physics;

  test_jac_psi_product<T>(basis, mesh, quadrature, physics);
}

TEST(adjoint, JacPsiProductElasticityPeriodic) {
  constexpr int Np_1d = 4;
  using T = double;

  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Physics = LinearElasticity<T, Basis::spatial_dim>;
  using Filter = HelmholtzFilter<T, Np_1d>;

  int nxy[2] = {32, 16};
  T lxy[2] = {2.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  T E = 1e2, nu = 0.3;
  Physics physics(E, nu);

  T r0 = 0.05;
  Filter filter(r0, grid);

  std::vector<T> x = create_initial_topology<T>(grid, 5, 3);
  filter.apply(x.data(), mesh.get_lsf_dof().data());
  mesh.update_mesh();

  test_jac_psi_product<T>(basis, mesh, quadrature, physics);
}