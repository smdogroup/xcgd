#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/stress.h"
#include "test_commons.h"

TEST(topo, stress_evaluation_regular_grid) {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {32, 32};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid, [](double x[]) {
    double region1 = x[0] - 0.4;  // <= 0
    double region2 = x[1] - 0.4;  // <= 0
    return hard_min<double>({region1, region2});
  });

  Quadrature quadrature(mesh);
  Basis basis(mesh);
  int nnodes = mesh.get_num_nodes();
  int ndof = nnodes * Basis::spatial_dim;
  T E = 100.0, nu = 0.3;

  auto int_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    A2D::Vec<T, Basis::spatial_dim> intf;
    intf(1) = -1.0;
    return intf;
  };

  StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_fun)> elastic(
      E, nu, mesh, quadrature, basis, int_fun);

  std::vector<int> bc_dof;
  for (int bc_node : mesh.get_upper_boundary_nodes()) {
    for (int d = 0; d < Basis::spatial_dim; d++) {
      bc_dof.push_back(bc_node * Basis::spatial_dim + d);
    }
  }

  std::vector<int> load_dof;
  for (int load_node : mesh.get_right_boundary_nodes()) {
    load_dof.push_back(load_node * Basis::spatial_dim + 1);
  }

  std::vector<T> sol =
      elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), 0.0));

  ToVTK<T, Mesh> vtk(mesh, "lbracket_case.vtk");
  vtk.write_mesh();
  vtk.write_vec("u", sol.data());
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  using Physics = LinearElasticity2DVonMisesStress<T>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  Physics physics(E, nu);
  Analysis analysis(mesh, quadrature, basis, physics);

  auto [xloc_q, stress_q] = analysis.interpolate_energy(sol.data());
  auto [_, lsf_q] =
      analysis.template interpolate<1>(mesh.get_lsf_nodes().data());
  FieldToVTKNew<T, Basis::spatial_dim> field_vtk("lbracket_case_quad.vtk");
  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();
  field_vtk.add_sol("VonMises", stress_q);
  field_vtk.add_sol("lsf", lsf_q);

  field_vtk.write_sol("VonMises");
  field_vtk.write_sol("lsf");
}
