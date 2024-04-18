#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

template <typename T, class Mesh, class Quadrature, class Basis, class LSFMesh,
          class LSFQuadrature, class LSFBasis>
class Topo {
 private:
  using HelmholtzFilter = HelmholtzFilter<T, LSFMesh, LSFQuadrature, LSFBasis>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;
  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;

 public:
  Topo(HelmholtzFilter& filter, StaticElastic& elastic)
      : filter(filter),
        elastic(elastic),
        vol_analysis(elastic.get_mesh(), elastic.get_quadrature(),
                     elastic.get_basis(), vol),
        mesh(elastic.get_mesh()),
        lsf_mesh(filter.get_mesh()),
        phi(mesh.get_lsf_dof()) {}

  T eval_compliance_grad(const std::vector<T>& x, std::vector<T>& grad) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }
    filter.apply(x.data(), phi.data());
    mesh.update_mesh();

    sol = elastic.solve();
    std::vector<T> psi = sol;
    for (T& p : psi) p *= -1.0;

    T comp = 2.0 * elastic.get_analysis().energy(nullptr, sol.data());

    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), 0.0);
    elastic.get_analysis().LSF_jacobian_adjoint_product(sol.data(), psi.data(),
                                                        grad.data());
    filter.applyGradient(x.data(), grad.data(), grad.data());
    return comp;
  }

  T eval_area_grad(const std::vector<T>& x, std::vector<T>& grad) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    filter.apply(x.data(), phi.data());
    mesh.update_mesh();

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    T area = vol_analysis.energy(nullptr, dummy.data());

    grad.resize(x.size());
    std::fill(grad.begin(), grad.end(), 0.0);
    vol_analysis.LSF_volume_derivatives(grad.data());
    filter.applyGradient(x.data(), grad.data(), grad.data());
    return area;
  }

  std::vector<T>& get_sol() { return sol; }

 private:
  HelmholtzFilter& filter;
  StaticElastic& elastic;
  Volume vol;
  VolAnalysis vol_analysis;

  Mesh& mesh;
  LSFMesh& lsf_mesh;

  std::vector<T>& phi;  // LSF values (filtered design variables)

  std::vector<T> sol;  // nodal displacement solution
};

void mesh_test() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  int constexpr spatial_dim = Grid::spatial_dim;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using LSFQuadrature = GDGaussQuadrature2D<T, Np_1d>;
  using LSFMesh = GridMesh<T, Np_1d>;
  using LSFBasis = GDBasis2D<T, LSFMesh>;

  using Filter = HelmholtzFilter<T, LSFMesh, LSFQuadrature, LSFBasis>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;

  using Physics = VolumePhysics<T, spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {32, 16};
  T lxy[2] = {2.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  LSFMesh lsf_mesh = mesh.get_lsf_mesh();
  LSFQuadrature lsf_quadrature(lsf_mesh);
  LSFBasis lsf_basis(lsf_mesh);
  T r0 = 0.1;
  Filter filter(r0, lsf_mesh, lsf_quadrature, lsf_basis);

  std::vector<int> bc_nodes = mesh.get_left_boundary_nodes();
  std::vector<int> load_nodes = mesh.get_right_boundary_nodes();
  std::vector<int> bc_dof(spatial_dim * bc_nodes.size());
  std::vector<int> load_dof(load_nodes.size());
  std::vector<T> load_vals(load_nodes.size(), 0.0);

  for (int i = 0; i < bc_nodes.size(); i++) {
    for (int d = 0; d < spatial_dim; d++) {
      bc_dof[spatial_dim * i + d] = spatial_dim * bc_nodes[i] + d;
    }
  }

  for (int i = 0; i < load_nodes.size(); i++) {
    load_dof[i] = spatial_dim * load_nodes[i] + 1;
    load_vals[i] = -1.0;
  }
  T E = 1e2, nu = 0.3;
  StaticElastic static_elastic(E, nu, mesh, quadrature, basis, bc_dof, load_dof,
                               load_vals);

  using Topo =
      Topo<T, Mesh, Quadrature, Basis, LSFMesh, LSFQuadrature, LSFBasis>;

  Topo topo(filter, static_elastic);

  int ndv = lsf_mesh.get_num_nodes();
  std::vector<T> x(ndv, 0.0);

  int m = 1, n = 1;
  for (int i = 0; i < ndv; i++) {
    T xloc[spatial_dim];
    lsf_mesh.get_node_xloc(i, xloc);
    x[i] = cos(xloc[0] / lxy[0] * 2.0 * PI * m) *
               cos(xloc[1] / lxy[1] * 2.0 * PI * n) -
           0.5;
  }

  std::vector<T>& phi = mesh.get_lsf_dof();

  filter.apply(x.data(), phi.data());
  mesh.update_mesh();

  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);
  std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
  interp.to_vtk("quadratures.vtk", dummy.data());

  ToVTK<T, GridMesh<T, Np_1d>> lsf_vtk(mesh.get_lsf_mesh(), "lsf_mesh.vtk");
  lsf_vtk.write_mesh();
  lsf_vtk.write_sol("x", x.data());
  lsf_vtk.write_sol("phi", phi.data());

  ToVTK<T, Mesh> vtk(mesh, "cut_mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("phi", mesh.get_lsf_nodes().data());
  vtk.write_sol("x", mesh.get_lsf_nodes(x).data());

  ToVTK<T, Mesh> elas_vtk(mesh, "elastic.vtk");
  elas_vtk.write_mesh();

  std::vector<T> p(ndv, 0.0);
  for (int i = 0; i < ndv; i++) {
    p[i] = T(rand()) / RAND_MAX;
  }

  double h = 1e-6;
  for (int i = 0; i < ndv; i++) {
    x[i] -= h * p[i];
  }
  std::vector<T> gcomp, garea;
  T c1 = topo.eval_compliance_grad(x, gcomp);
  T a1 = topo.eval_area_grad(x, garea);

  for (int i = 0; i < ndv; i++) {
    x[i] += 2.0 * h * p[i];
  }
  T c2 = topo.eval_compliance_grad(x, gcomp);
  T a2 = topo.eval_area_grad(x, garea);

  T gcomp_adjoint = (c2 - c1) / 2.0 / h;
  T garea_adjoint = (a2 - a1) / 2.0 / h;

  for (int i = 0; i < ndv; i++) {
    x[i] -= 2.0 * h * p[i];
  }

  T comp = topo.eval_compliance_grad(x, gcomp);
  T area = topo.eval_area_grad(x, garea);

  std::vector<T> sol = topo.get_sol();
  T gcomp_fd = 0.0, garea_fd = 0.0;
  for (int i = 0; i < ndv; i++) {
    gcomp_fd += gcomp[i] * p[i];
    garea_fd += garea[i] * p[i];
  }

  std::printf("compliance:      %25.15e\n", comp);
  std::printf("gradient fd:     %25.15e\n", gcomp_fd);
  std::printf("gradientadjoint: %25.15e\n", gcomp_adjoint);
  std::printf("relative error:  %25.15e\n",
              (gcomp_fd - gcomp_adjoint) / gcomp_adjoint);
  std::printf("\n");

  std::printf("area:            %25.15e\n", area);
  std::printf("gradient fd:     %25.15e\n", garea_fd);
  std::printf("gradientadjoint: %25.15e\n", garea_adjoint);
  std::printf("relative error:  %25.15e\n",
              (garea_fd - garea_adjoint) / garea_adjoint);

  elas_vtk.write_vec("sol", sol.data());
  elas_vtk.write_sol("phi", mesh.get_lsf_nodes().data());
  elas_vtk.write_sol("x", mesh.get_lsf_nodes(x).data());
}

int main() {
  mesh_test();
  return 0;
}