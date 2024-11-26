#include <algorithm>
#include <stdexcept>

#include "analysis.h"
#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "utils/vtk.h"

template <typename T, class Mesh>
void write_vtk(std::string name, Mesh mesh, std::vector<T> sol = {}) {
  ToVTK<T, Mesh> to_vtk(mesh, name);
  to_vtk.write_mesh();

  std::vector<double> elems(mesh.get_num_elements());
  std::vector<double> cells(mesh.get_num_elements());

  for (int i = 0; i < mesh.get_num_elements(); i++) {
    elems[i] = double(i);
    cells[i] = double(mesh.get_elem_cell(i));
  }

  if (sol.size()) {
    to_vtk.write_vec("sol", sol.data());
  }

  to_vtk.write_cell_sol("elem", elems.data());
  to_vtk.write_cell_sol("cell", cells.data());
}

void test_two_sided_problem() {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{};
  };
  using Physics = LinearElasticity<T, Basis::spatial_dim, typeof(int_func)>;

  T E1 = 100.0, nu1 = 0.3;
  Physics physics_l(E1, nu1, int_func);

  T E2 = 1.0, nu2 = 0.2;
  Physics physics_r(E2, nu2, int_func);

  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics, true>;

  int nxy[2] = {100, 20};
  T lxy[2] = {5.0, 1.0};
  Grid grid(nxy, lxy);

  Mesh mesh_l(grid, [](T* xloc) { return xloc[0] + xloc[1] - 3.0; });
  Mesh mesh_r(grid);
  for (int i = 0; i < grid.get_num_verts(); i++) {
    mesh_r.get_lsf_dof()[i] = -mesh_l.get_lsf_dof()[i];
  }
  mesh_r.update_mesh();

  write_vtk<T>("mesh_left.vtk", mesh_l);
  write_vtk<T>("mesh_right.vtk", mesh_r);

  Quadrature quadrature_l(mesh_l);
  Quadrature quadrature_r(mesh_r);

  Basis basis_l(mesh_l);
  Basis basis_r(mesh_r);

  Analysis analysis_l(mesh_l, quadrature_l, basis_l, physics_l);
  Analysis analysis_r(mesh_r, quadrature_r, basis_r, physics_r);

  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int ndof_total = grid.get_num_verts() * Physics::dof_per_node;
  int *rowp = nullptr, *cols = nullptr;
  int constexpr max_nnodes_per_element = 2 * Mesh::max_nnodes_per_element;
  SparseUtils::CSRFromConnectivityFunctor(
      grid.get_num_verts(), grid.get_num_cells(), max_nnodes_per_element,
      [&mesh_l, &mesh_r](int cell, int* verts) -> int {
        const auto& cell_elems_l = mesh_l.get_cell_elems();
        const auto& cell_elems_r = mesh_r.get_cell_elems();

        int nverts = 0;
        int verts_work[max_nnodes_per_element];

        if (cell_elems_l.count(cell)) {
          nverts += mesh_l.get_cell_dof_verts(cell, verts_work);
        }

        if (cell_elems_r.count(cell)) {
          nverts += mesh_r.get_cell_dof_verts(cell, verts_work + nverts);
        }

        std::set<int> verts_set(verts_work, verts_work + nverts);

        int i = 0;
        for (auto it = verts_set.begin(); it != verts_set.end(); it++, i++) {
          verts[i] = *it;
        }

        nverts = verts_set.size();

        return nverts;
      },
      &rowp, &cols);

  int nnz = rowp[grid.get_num_verts()];
  BSRMat* jac_bsr = new BSRMat(grid.get_num_verts(), nnz, rowp, cols);

  // Compute Jacobian matrix
  std::vector<T> zeros(ndof_total, 0.0);
  analysis_l.jacobian(nullptr, zeros.data(), jac_bsr, true);
  analysis_r.jacobian(nullptr, zeros.data(), jac_bsr, false);

  jac_bsr->write_mtx("two_sided_jacobian.mtx");

  // BC verts
  std::vector<int> bc_nodes = mesh_l.get_left_boundary_nodes();
  std::vector<int> bc_verts(bc_nodes.size());
  std::transform(bc_nodes.begin(), bc_nodes.end(), bc_verts.begin(),
                 [mesh_l](int n) { return mesh_l.get_node_vert(n); });
  std::vector<int> bc_dof;
  bc_dof.reserve(Physics::dof_per_node * bc_verts.size());
  for (int vert : bc_verts) {
    for (int d = 0; d < Physics::dof_per_node; d++) {
      bc_dof.push_back(Physics::dof_per_node * vert + d);
    }
  }

  // Apply bcs to jacobian
  jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
  CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
  jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

  auto load_func = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    A2D::Vec<T, Basis::spatial_dim> intf;
    intf(0) = 1.0;
    return intf;
  };
  using LoadPhysics =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>;
  using LoadQuadrature =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;
  using LoadAnalysis =
      GalerkinAnalysis<T, Mesh, LoadQuadrature, Basis, LoadPhysics, true>;

  LoadPhysics load_physics(load_func);
  std::set<int> load_elements;
  for (int iy = 0; iy < nxy[1]; iy++) {
    int c = grid.get_coords_cell(nxy[0] - 1, iy);
    load_elements.insert(mesh_r.get_cell_elems().at(c));
  }
  LoadQuadrature load_quadrature(mesh_r, load_elements);
  LoadAnalysis load_analysis(mesh_r, load_quadrature, basis_r, load_physics);

  // Compute rhs
  std::vector<T> rhs(ndof_total, 0.0);
  load_analysis.residual(nullptr, zeros.data(), rhs.data());
  for (int i = 0; i < rhs.size(); i++) {
    rhs[i] *= -1.0;
  }

  // Factorize Jacobian matrix
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T>* chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  chol->factor();
  std::vector<T> sol = rhs;
  chol->solve(sol.data());

  // Check error
  // res = Ku - rhs
  std::vector<T> Ku(sol.size());
  jac_bsr->axpy(sol.data(), Ku.data());
  T err = 0.0;
  for (int i = 0; i < Ku.size(); i++) {
    err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
  }
  std::printf("[Debug] Linear elasticity residual:\n");
  std::printf("||Ku - f||_2: %25.15e\n", sqrt(err));

  // This is cheating - we generate the vtk for the combined mesh
  GridMesh<T, Np_1d> gmesh(grid);
  ToVTK<T, typeof(gmesh)> to_vtk(gmesh, "combined.vtk");
  to_vtk.write_mesh();
  to_vtk.write_vec("rhs", rhs.data());
  to_vtk.write_vec("sol", sol.data());
}

void test_two_sided_app() {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto int_func = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return A2D::Vec<T, Basis::spatial_dim>{};
  };
  using StaticApp =
      StaticElasticErsatz<T, Mesh, Quadrature, Basis, typeof(int_func)>;

  T E = 100.0, nu = 0.3;

  int nxy[2] = {100, 20};
  T lxy[2] = {5.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid, [](T* xloc) { return xloc[0] + xloc[1] - 3.0; });
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  StaticApp static_app(E, nu, mesh, quadrature, basis, int_func, 1e-3);

  write_vtk<T>("app_mesh_left.vtk", static_app.get_mesh());
  write_vtk<T>("app_mesh_right.vtk", static_app.get_mesh_ersatz());

  // BC
  std::vector<int> bc_nodes = mesh.get_left_boundary_nodes();
  std::vector<int> bc_verts(bc_nodes.size());
  std::transform(bc_nodes.begin(), bc_nodes.end(), bc_verts.begin(),
                 [mesh](int n) { return mesh.get_node_vert(n); });
  std::vector<int> bc_dof;
  bc_dof.reserve(StaticApp::Physics::dof_per_node * bc_verts.size());
  for (int vert : bc_verts) {
    for (int d = 0; d < StaticApp::Physics::dof_per_node; d++) {
      bc_dof.push_back(StaticApp::Physics::dof_per_node * vert + d);
    }
  }

  // Load
  auto load_func = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    A2D::Vec<T, Basis::spatial_dim> intf;
    intf(0) = 1.0;
    return intf;
  };
  using LoadPhysics =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>;
  using LoadQuadrature =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;
  using LoadAnalysis =
      GalerkinAnalysis<T, Mesh, LoadQuadrature, Basis, LoadPhysics, true>;

  LoadPhysics load_physics(load_func);
  std::set<int> load_elements;
  for (int iy = 0; iy < nxy[1]; iy++) {
    int c = grid.get_coords_cell(nxy[0] - 1, iy);
    load_elements.insert(static_app.get_mesh_ersatz().get_cell_elems().at(c));
  }
  LoadQuadrature load_quadrature(static_app.get_mesh_ersatz(), load_elements);
  LoadAnalysis load_analysis(static_app.get_mesh_ersatz(), load_quadrature,
                             static_app.get_basis_ersatz(), load_physics);

  // Solve
  std::vector<T> sol =
      static_app.solve(bc_dof, std::vector<T>(bc_dof.size(), T(0.0)),
                       std::tuple<LoadAnalysis>{load_analysis});

  // Export to vtk
  GridMesh<T, Np_1d> gmesh(grid);
  ToVTK<T, typeof(gmesh)> to_vtk(gmesh, "app_combined.vtk");
  to_vtk.write_mesh();
  to_vtk.write_vec("sol", sol.data());
}

int main() {
  test_two_sided_problem();
  test_two_sided_app();
}
