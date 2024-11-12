#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/vtk.h"

template <typename T, int spatial_dim>
std::vector<int> get_dof_vec_from_nodes(std::vector<int> nodes,
                                        std::vector<int> spatial = {0, 1}) {
  std::vector<int> bc_dof;
  for (auto node : nodes) {
    for (int d : spatial) {
      bc_dof.push_back(spatial_dim * node + d);
    }
  }
  return bc_dof;
}

template <typename T, class Vectors>
std::vector<T> concat_vectors(const Vectors& vectors) {
  std::vector<T> ret;
  for (auto vec : vectors) {
    ret.insert(ret.end(), vec.begin(), vec.end());
  }
  return ret;
}

template <int Np_1d>
void test_elastic_app() {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  int nxy[2] = {5, 5};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};
  Grid grid(nxy, lxy, xy0);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);
  int nnodes = mesh.get_num_nodes();
  int ndof = nnodes * Basis::spatial_dim;
  T E = 100.0, nu = 0.3;

  // Cases 1: 0-displacement on all boundary nodes with body force
  {
    auto int_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
      A2D::Vec<T, Basis::spatial_dim> intf;
      intf(0) = xloc(0);
      std::printf("xloc: (%15.5f, %15.5f), fx: %15.5f\n", xloc(0), xloc(1),
                  intf(0));
      return intf;
    };
    StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_fun)> elastic(
        E, nu, mesh, quadrature, basis, int_fun);
    std::vector<int> bc_dof;
    for (auto nodes :
         {mesh.get_left_boundary_nodes(), mesh.get_right_boundary_nodes(),
          mesh.get_upper_boundary_nodes(), mesh.get_lower_boundary_nodes()}) {
      auto t = get_dof_vec_from_nodes<T, Basis::spatial_dim>(nodes);
      bc_dof.insert(bc_dof.end(), t.begin(), t.end());
    }
    std::vector<T> sol =
        elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), 0.0), {}, {});
    ToVTK<T, Mesh> vtk(mesh, "elastic_case1.vtk");
    vtk.write_mesh();
    vtk.write_vec("u", sol.data());
  }

  // Case 2: 0-displacement on left and some displacement on right, no body
  // force
  {
    auto int_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
      A2D::Vec<T, Basis::spatial_dim> intf;
      return intf;
    };
    StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_fun)> elastic(
        E, nu, mesh, quadrature, basis, int_fun);

    std::vector<int> left_dof = get_dof_vec_from_nodes<T, Basis::spatial_dim>(
        mesh.get_left_boundary_nodes());
    std::vector<int> right_dof = get_dof_vec_from_nodes<T, Basis::spatial_dim>(
        mesh.get_right_boundary_nodes(), {0});

    std::vector<T> left_vals(left_dof.size(), 0.0);
    std::vector<T> right_vals(right_dof.size(), 1.0);

    std::vector<int> bc_dof = left_dof;
    bc_dof.insert(bc_dof.end(), right_dof.begin(), right_dof.end());

    std::vector<T> bc_vals = left_vals;
    bc_vals.insert(bc_vals.end(), right_vals.begin(), right_vals.end());

    std::vector<T> sol = elastic.solve(bc_dof, bc_vals, {}, {});
    ToVTK<T, Mesh> vtk(mesh, "elastic_case2.vtk");
    vtk.write_mesh();
    vtk.write_vec("u", sol.data());
  }
}

TEST(apps, Elastic) { test_elastic_app<2>(); }
