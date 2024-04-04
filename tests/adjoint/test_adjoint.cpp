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

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Mesh lsf_mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh, lsf_mesh);

  T E = 30.0, nu = 0.3;
  Physics physics(E, nu);

  int ndof = Physics::dof_per_node * mesh.get_num_nodes();
  int ndof_lsf = quadrature.get_lsf_mesh().get_num_nodes();

  std::vector<T> dof(ndof), psi(ndof), dfdx(ndof_lsf);

  for (int i = 0; i < ndof; i++) {
    dof[i] = (T)rand() / RAND_MAX;
    psi[i] = (T)rand() / RAND_MAX;
  }
  Analysis analysis(mesh, quadrature, basis, physics);

  analysis.LSF_jacobian_adjoint_product(dof.data(), psi.data(), dfdx.data());

  std::cout << "dfdx:\n";
  for (auto v : dfdx) std::cout << v << "\n";
}