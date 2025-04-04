#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"

template <typename T, int Np_1d>
void execute() {
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Physics = VolumePhysics<T, Basis::spatial_dim>;  // dummy physics
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {8, 8};
  T lxy[2] = {8.0, 8.0};
  T xy0[2] = {-4.0, -4.0};
  Grid grid(nxy, lxy, xy0);

  T r = 3.5;

  Mesh cut_mesh(grid, [r](T* x) { return x[0] * x[0] + x[1] * x[1] - r * r; });
  Quadrature cut_quad(cut_mesh);
  Basis cut_basis(cut_mesh);
  Physics physics;

  Analysis cut_analysis(cut_mesh, cut_quad, cut_basis, physics);

  Mesh grid_mesh(grid);
  Quadrature grid_quad(grid_mesh);
  Basis grid_basis(grid_mesh);

  Analysis grid_analysis(grid_mesh, grid_quad, grid_basis, physics);

  // Mesh
  {
    ToVTK<T, Mesh> grid_vtk(grid_mesh, "pi_visualization_mesh.vtk");
    grid_vtk.write_mesh();
  }

  // cut quads
  {
    std::vector<T> cut_dof(cut_mesh.get_num_nodes(), T(0.0));  // dummy dof
    auto [xloc_q, cut_dof_q] = cut_analysis.interpolate(cut_dof.data());

    FieldToVTKNew<T, 2> cut_quads("pi_visualization_cut_quads.vtk");
    cut_quads.add_mesh(xloc_q);
    cut_quads.write_mesh();
  }

  // grid quads
  {
    std::vector<T> grid_dof(grid_mesh.get_num_nodes(), T(0.0));  // dummy dof
    auto [xloc_q, grid_dof_q] = grid_analysis.interpolate(grid_dof.data());

    std::vector<T> is_inside_q(grid_dof_q.size(), 0.0);
    for (int i = 0; i < grid_dof_q.size(); i++) {
      T x = xloc_q[2 * i];
      T y = xloc_q[2 * i + 1];
      if (x * x + y * y <= r * r) {
        is_inside_q[i] = 1.0;
      }
    }

    FieldToVTKNew<T, 2> grid_quads("pi_visualization_grid_quads.vtk");
    grid_quads.add_mesh(xloc_q);
    grid_quads.write_mesh();
    grid_quads.add_sol("is_inside", is_inside_q);
    grid_quads.write_sol("is_inside");
  }
}

int main() { execute<double, 4>(); }
