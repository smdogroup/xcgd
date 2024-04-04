#include <vector>

#include "analysis.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "test_commons.h"

template <typename T>
class Circle {
 public:
  Circle(T* center, T radius, bool flip = false) {
    x0[0] = center[0];
    x0[1] = center[1];
    r = radius;
    if (flip) {
      sign = -1.0;
    }
  }

  T operator()(const algoim::uvector<T, 2>& x) const {
    return sign * ((x(0) - x0[0]) * (x(0) - x0[0]) +
                   (x(1) - x0[1]) * (x(1) - x0[1]) - r * r);
  }
  algoim::uvector<T, 2> grad(const algoim::uvector<T, 2>& x) const {
    return algoim::uvector<T, 2>(2.0 * sign * (x(0) - x0[0]),
                                 2.0 * sign * (x(1) - x0[1]));
  }

 private:
  T x0[2];
  T r;
  double sign = 1.0;
};

TEST(adjoint, JacPsiProduct) {
  constexpr int Np_1d = 2;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = typename Basis::Mesh;
  using Physics = LinearElasticity<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {2, 2};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};

  T center[2] = {0.0, 0.0};
  T r = 0.5;

  bool flip = false;
  Circle lsf(center, r, flip);

  Grid grid(nxy, lxy, xy0);
  Mesh mesh(grid, lsf);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  T E = 30.0, nu = 0.3;
  Physics physics(E, nu);

  int ndof = Physics::dof_per_node * mesh.get_num_nodes();
  int ndof_lsf = quadrature.get_lsf_mesh().get_num_nodes();

  std::vector<T> dof(ndof), psi(ndof), dfdx(ndof_lsf);

  Analysis analysis(mesh, quadrature, basis, physics);

  analysis.LSF_jacobian_adjoint_product(dof.data(), psi.data(), dfdx.data());

  std::cout << "dfdx:\n";
  for (auto v : dfdx) std::cout << v << "\n";
}