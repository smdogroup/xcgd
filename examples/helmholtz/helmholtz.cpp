
#include "physics/helmholtz.h"

#include <string>
#include <vector>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/mesh.h"
#include "utils/vtk.h"

template <typename T, class Quadrature, class Basis, class Func>
void solve_helmholtz(T r0, const Func &xfunc, Basis &basis, std::string name) {
  using Physics = HelmholtzPhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  Quadrature quadrature(basis.mesh);
  Physics physics(r0);
  Analysis analysis(quadrature, basis, physics);

  int ndof = basis.mesh.get_num_nodes();

  // Set up Jacobian matrix
  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivityFunctor(
      basis.mesh.get_num_nodes(), basis.mesh.get_num_elements(),
      basis.mesh.nodes_per_element,
      [&basis](int elem, int *nodes) {
        basis.mesh.get_elem_dof_nodes(elem, nodes);
      },
      &rowp, &cols);

  int nnz = rowp[basis.mesh.get_num_nodes()];
  BSRMat *jac_bsr = new BSRMat(basis.mesh.get_num_nodes(), nnz, rowp, cols);

  // set x
  std::vector<T> x(ndof, 0.0);
  for (int i = 0; i < basis.mesh.get_num_nodes(); i++) {
    T xloc[Basis::spatial_dim];
    basis.mesh.get_node_xloc(i, xloc);
    x[i] = xfunc(xloc);
  }

  // Compute Jacobian matrix
  std::vector<T> dof(ndof, 0.0);
  analysis.jacobian(x.data(), dof.data(), jac_bsr);

  // Store right hand size to sol
  std::vector<T> sol(ndof, 0.0);
  analysis.residual(x.data(), dof.data(), sol.data());
  for (int i = 0; i < sol.size(); i++) {
    sol[i] *= -1.0;
  }

  T x_2norm = 0.0, rhs_2norm = 0.0;
  for (int i = 0; i < ndof; i++) {
    x_2norm += x[i] * x[i];
    rhs_2norm += sol[i] * sol[i];
  }
  printf("|x|:    %20.10e\n", x_2norm);
  printf("|rhs|:  %20.10e\n", rhs_2norm);

  jac_bsr->write_mtx("K_" + name + ".mtx");

  // Apply bcs to Jacobian matrix
  CSCMat *jac_csc = SparseUtils::bsr_to_csc(jac_bsr);

  std::vector<T> rhs = sol;

  // solve
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T> *chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  chol->factor();
  chol->solve(sol.data());

  // Check error
  // res = Ku - rhs
  std::vector<T> Ku(sol.size());
  jac_bsr->axpy(sol.data(), Ku.data());
  T err = 0.0;
  for (int i = 0; i < Ku.size(); i++) {
    err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
  }
  std::printf("||Ku - f||: %25.15e\n", sqrt(err));

  // Write to vtk
  ToVTK<T, typename Basis::Mesh> vtk(basis.mesh, name + ".vtk");
  vtk.write_mesh();
  vtk.write_sol("x", x.data());
  vtk.write_sol("u", sol.data());
  vtk.write_sol("rhs", rhs.data());
}

void solve_helmholtz_fem() {
  using T = double;
  using Quadrature = QuadrilateralQuadrature<T>;
  using Basis = QuadrilateralBasis<T>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;
  int nxy[2] = {64, 64};
  T lxy[2] = {3.0, 3.0};
  T pt0[2] = {1.5, 1.5};
  T r = 1.0, r0 = 5.0;
  create_2d_rect_quad_mesh(nxy, lxy, &num_elements, &num_nodes, &element_nodes,
                           &xloc);

  Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Basis basis(mesh);

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

  solve_helmholtz<T, Quadrature, Basis>(r0, xfunc, basis, "fe");
}

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

void solve_helmholtz_gd() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;
  int nxy[2] = {64, 64};
  T lxy[2] = {1.0, 1.0};
  T pt0[2] = {0.5, 0.5};
  T r = 0.5;
  Grid grid(nxy, lxy);
  Circle lsf(pt0, r);
  Basis::Mesh mesh(grid, lsf);
  Basis basis(mesh);

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

  T r0 = 5.0;
  solve_helmholtz<T, Quadrature, Basis>(r0, xfunc, basis, "gd");

  // Test
  ToVTK<T, Basis::Mesh> vtk(mesh, "gd_mesh.vtk");
  vtk.write_mesh();

  // for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
  //   std::vector<T> dof(mesh.get_num_nodes(), 0.0);
  //   int nodes[Basis::Mesh::nodes_per_element];
  //   mesh.get_elem_dof_nodes(elem, nodes);
  //   for (int i = 0; i < Basis::Mesh::nodes_per_element; i++) {
  //     dof[nodes[i]] = 1.0;
  //   }
  //   char name[256];
  //   std::snprintf(name, 256, "elem_%05d", elem);
  //   vtk.write_sol(name, dof.data());
  // }
}

int main(int argc, char *argv[]) {
  solve_helmholtz_fem();
  solve_helmholtz_gd();

  return 0;
}