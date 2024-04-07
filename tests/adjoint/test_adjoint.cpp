#include <vector>

#include "analysis.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "test_commons.h"

class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(double k = 0.4, double b = 0.1) : k(k), b(b) {}

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

  // using Physics = LinearElasticity<T, Basis::spatial_dim>;
  using Physics = PoissonPhysics<T, Basis::spatial_dim>;

  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Mesh lsf_mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh, lsf_mesh);

  T E = 30.0, nu = 0.3;
  // Physics physics(E, nu);
  Physics physics;

  double h = 1e-6;
  int ndof = Physics::dof_per_node * mesh.get_num_nodes();
  int ndv = quadrature.get_lsf_mesh().get_num_nodes();
  std::vector<T>& dvs = mesh.get_lsf_dof();

  std::vector<T> dof(ndof), psi(ndof), res1(ndof, 0.0), res2(ndof, 0.0);
  std::vector<T> dfdx(ndv), p(ndv);

  for (int i = 0; i < ndof; i++) {
    dof[i] = (T)rand() / RAND_MAX;
    psi[i] = (T)rand() / RAND_MAX;
  }
  for (int i = 0; i < ndv; i++) {
    p[i] = (T)rand() / RAND_MAX;
  }

  Analysis analysis(mesh, quadrature, basis, physics);
  analysis.LSF_jacobian_adjoint_product(dof.data(), psi.data(), dfdx.data());

  for (int i = 0; i < ndv; i++) {
    dvs[i] -= h * p[i];
  }
  analysis.residual(nullptr, dof.data(), res1.data());

  for (int i = 0; i < ndv; i++) {
    dvs[i] += 2.0 * h * p[i];
  }
  analysis.residual(nullptr, dof.data(), res2.data());

  double dfdx_fd = 0.0, dfdx_adjoint;
  for (int i = 0; i < ndof; i++) {
    dfdx_fd += psi[i] * (res2[i] - res1[i]) / (2.0 * h);
    dfdx_adjoint += dfdx[i] * p[i];
  }

  std::printf("dfdx_fd:      %25.15e\n", dfdx_fd);
  std::printf("dfdx_adjoint: %25.15e\n", dfdx_adjoint);
}