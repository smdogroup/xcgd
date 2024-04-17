#include "elements/gd_vandermonde.h"
#include "utils/vtk.h"

template <int spatial_dim>
class Circle {
 public:
  Circle(double *center, double radius, bool flip = false) {
    for (int d = 0; d < spatial_dim; d++) {
      x0[d] = center[d];
    }
    r = radius;
    if (flip) {
      sign = -1.0;
    }
  }

  template <typename T>
  T operator()(const algoim::uvector<T, spatial_dim> &x) const {
    return sign * ((x(0) - x0[0]) * (x(0) - x0[0]) +
                   (x(1) - x0[1]) * (x(1) - x0[1]) - r * r);
  }

 private:
  double x0[spatial_dim];
  double r;
  double sign = 1.0;
};

int main() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using LSF = Circle<Grid::spatial_dim>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  int nxy[2] = {96, 64};
  T lxy[2] = {1.5, 1.0};

  double center[2] = {0.75, 0.5};
  double r = 0.3;

  LSF lsf(center, r, false);

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  std::vector<T> &lsf_dof = mesh.get_lsf_dof();

  for (int i = 0; i < lsf_dof.size(); i++) {
    lsf_dof[i] += 0.1 * (T(rand()) / RAND_MAX - 0.5);
  }
  mesh.update_mesh();

  ToVTK<T, GridMesh<T, Np_1d>> lsf_vtk(mesh.get_lsf_mesh(), "lsf_mesh.vtk");
  lsf_vtk.write_mesh();
  lsf_vtk.write_sol("lsf", mesh.get_lsf_dof().data());

  ToVTK<T, Mesh> vtk(mesh, "cut_mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
}
