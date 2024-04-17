#include "analysis.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"
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

  using Physics = VolumePhysics<T, Grid::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  int nxy[2] = {48, 64};
  T lxy[2] = {3.0, 3.5};

  double center[2] = {1.2, 1.3};
  double r = 1.0;

  LSF lsf(center, r, false);

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  Physics physics;
  Analysis analysis(mesh, quadrature, basis, physics);

  int ndv = mesh.get_lsf_mesh().get_num_nodes();
  std::vector<T> &dvs = mesh.get_lsf_dof();
  std::vector<T> p(ndv, 0.0);

  double h = 1e-6;
  for (int i = 0; i < ndv; i++) {
    p[i] = (T)rand() / RAND_MAX;
  }

  std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
  std::printf("LSF area: %20.10e\n", analysis.energy(nullptr, dummy.data()));

  for (int i = 0; i < ndv; i++) {
    dvs[i] -= h * p[i];
  }
  T a1 = analysis.energy(nullptr, dummy.data());

  for (int i = 0; i < ndv; i++) {
    dvs[i] += 2.0 * h * p[i];
  }
  T a2 = analysis.energy(nullptr, dummy.data());

  T dfdx_fd = (a2 - a1) / 2.0 / h;

  std::vector<T> dfdx(ndv, 0.0);
  analysis.LSF_volume_derivatives(dfdx.data());

  T dfdx_exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdx_exact += dfdx[i] * p[i];
  }

  std::printf("area gradient fd:    %25.15e\n", dfdx_fd);
  std::printf("area gradient exact: %25.15e\n", dfdx_exact);

  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);

  interp.to_vtk("quadratures.vtk", dummy.data());

  ToVTK<T, GridMesh<T, Np_1d>> lsf_vtk(mesh.get_lsf_mesh(), "lsf_mesh.vtk");
  lsf_vtk.write_mesh();
  lsf_vtk.write_sol("lsf", mesh.get_lsf_dof().data());

  ToVTK<T, Mesh> vtk(mesh, "cut_mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
}
