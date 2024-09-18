#include <stdexcept>

#include "apps/static_elastic.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/json.h"

template <typename T, typename Mesh>
void write_vtk(const Mesh& mesh, const char* vtkname,
               const std::vector<T>& sol) {
  assert(sol.size() == mesh.get_num_nodes() * Mesh::spatial_dim);

  ToVTK<T, Mesh> vtk(mesh, vtkname);
  vtk.write_mesh();

  std::vector<T> elem_indices(mesh.get_num_elements());
  for (int i = 0; i < mesh.get_num_elements(); i++) {
    elem_indices[i] = T(i);
  }
  vtk.write_cell_sol("elem_indices", elem_indices.data());
  vtk.write_vec("displacement", sol.data());
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  // for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
  //   std::vector<T> dof(mesh.get_num_nodes(), 0.0);
  //   int nodes[Mesh::max_nnodes_per_element];
  //   int nnodes = mesh.get_elem_dof_nodes(elem, nodes);
  //   for (int i = 0; i < nnodes; i++) {
  //     dof[nodes[i]] = 1.0;
  //   }
  //   char name[256];
  //   std::snprintf(name, 256, "stencil_elem_%05d", elem);
  //   vtk.write_sol(name, dof.data());
  // }
}

template <typename T, int Np_1d>
void test_regression_static(json j) {
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;

  int nxy[2] = {j["nx"], j["ny"]};
  T lxy[2] = {j["lx"], j["ly"]};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  StaticElastic elastic(j["E"], j["nu"], mesh, quadrature, basis);

  std::vector<T> lsf_dof = j["lsf_dof"];
  EXPECT_EQ(lsf_dof.size(), mesh.get_lsf_dof().size());
  mesh.get_lsf_dof() = lsf_dof;
  mesh.update_mesh();

  std::vector<T> sol =
      elastic.solve(j["bc_dof"], j["load_dof"], j["load_vals"]);

  char vtkname[256];
  std::snprintf(vtkname, 256, "regression_Np_1d_%d.vtk", Np_1d);
  write_vtk<T>(mesh, vtkname, sol);

  EXPECT_EQ(sol.size(), 2 * mesh.get_num_nodes());
  EXPECT_VEC_EQ(sol.size(), sol, j["u"]);
}

TEST(regression, static) {
  using T = double;
  std::string json_path = "./prob.json";

  json j = read_json(json_path);
  switch (int(j["Np_1d"])) {
    case 2:
      test_regression_static<T, 2>(j);
      break;
    case 4:
      test_regression_static<T, 4>(j);
      break;
    default:
      char msg[256];
      std::snprintf(msg, 256,
                    "Np_1d = %d loaded from json is not precompiled by the "
                    "test code, you may want to manually add it in the source "
                    "code and compile the test again",
                    int(j["Np_1d"]));
      throw std::runtime_error(msg);
      break;
  }
}
