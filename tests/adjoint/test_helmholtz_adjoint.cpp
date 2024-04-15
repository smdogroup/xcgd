
#include "analysis.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/helmholtz.h"
#include "test_commons.h"

template <typename T>
class Circle {
 public:
  Circle(T *center, T radius, bool flip = false) {
    x0[0] = center[0];
    x0[1] = center[1];
    r = radius;
    if (flip) {
      sign = -1.0;
    }
  }

  T operator()(const algoim::uvector<T, 2> &x) const {
    return sign * ((x(0) - x0[0]) * (x(0) - x0[0]) +
                   (x(1) - x0[1]) * (x(1) - x0[1]) - r * r);
  }
  algoim::uvector<T, 2> grad(const algoim::uvector<T, 2> &x) const {
    return algoim::uvector<T, 2>(2.0 * sign * (x(0) - x0[0]),
                                 2.0 * sign * (x(1) - x0[1]));
  }

 private:
  T x0[2];
  T r;
  double sign = 1.0;
};

TEST(adjoint, JacPsiProductHelmholtz) {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = Basis::Mesh;
  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  T pt0[2] = {0.5, 0.5};
  T r = 0.5;
  Grid grid(nxy, lxy);
  Circle lsf(pt0, r);
  Mesh mesh(grid, lsf);
  Mesh lsf_mesh(grid);
  Quadrature quadrature(mesh, lsf_mesh);
  Basis basis(mesh);

  // Export quadrature points
  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);
  interp.to_vtk("helmholtz_quadratures.vtk");

  auto xfunc = [pt0, r](T *xloc) {
    T rx2 = (xloc[0] - pt0[0]) * (xloc[0] - pt0[0]) +
            (xloc[1] - pt0[1]) * (xloc[1] - pt0[1]);
    T r2 = r * r;
    if (rx2 < r2) {
      return sqrt(rx2 / r2);
    } else {
      return 0.0;
    }
  };

  using Physics = HelmholtzPhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  T r0 = 5.0;
  Physics physics(r0);
  Analysis analysis(mesh, quadrature, basis, physics);

  int ndof = mesh.get_num_nodes();
  int ndv = mesh.get_num_nodes();
  double h = 1e-6;
  double tol = 1e-7;

  std::vector<T> x(ndv, 0.0), p(ndv, 0.0), dfdx(ndv, 0.0);
  std::vector<T> dof(ndof, 0.0), psi(ndof, 0.0), res1(ndof, 0.0),
      res2(ndof, 0.0);
  for (int i = 0; i < ndv; i++) {
    // set x
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    x[i] = xfunc(xloc);

    // set perturbation
    p[i] = (T)rand() / RAND_MAX;
  }

  for (int i = 0; i < ndof; i++) {
    dof[i] = (T)rand() / RAND_MAX;
    psi[i] = (T)rand() / RAND_MAX;
  }

  analysis.jacobian_adjoint_product(x.data(), dof.data(), psi.data(),
                                    dfdx.data());

  for (int i = 0; i < ndv; i++) {
    x[i] -= h * p[i];
  }
  analysis.residual(x.data(), dof.data(), res1.data());

  for (int i = 0; i < ndv; i++) {
    x[i] += 2.0 * h * p[i];
  }
  analysis.residual(x.data(), dof.data(), res2.data());

  double dfdx_fd = 0.0;
  for (int i = 0; i < ndof; i++) {
    dfdx_fd += psi[i] * (res2[i] - res1[i]) / (2.0 * h);
  }

  double dfdx_adjoint = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdx_adjoint += dfdx[i] * p[i];
  }

  std::printf("dfdx_fd:      %25.15e\n", dfdx_fd);
  std::printf("dfdx_adjoint: %25.15e\n", dfdx_adjoint);
  EXPECT_NEAR((dfdx_fd - dfdx_adjoint) / dfdx_adjoint, 0.0, tol);
}