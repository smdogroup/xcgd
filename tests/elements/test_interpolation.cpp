#include <string>

#include "elements/element_utils.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/mesher.h"
#include "utils/vtk.h"

class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(double k = 0.9, double b = 0.1) : k(k), b(b) {}

  template <typename T>
  T operator()(const algoim::uvector<T, spatial_dim>& x) const {
    return -k * x(0) + x(1) - b;
  }

  template <typename T>
  algoim::uvector<T, spatial_dim> grad(const algoim::uvector<T, 2>& x) const {
    return algoim::uvector<T, spatial_dim>(-k, 1.0);
  }

 private:
  double k, b;
};

template <typename T, class Mesh, class Quadrature, class Basis>
void interpolate_dof_at_quadratures(const Mesh& mesh,
                                    const Quadrature& quadrature,
                                    const Basis& basis,
                                    const std::string& name) {
  using Interpolator = Interpolator<T, Quadrature, Basis>;

  // Write mesh
  ToVTK<T, Mesh> vtk(mesh, name + "_mesh.vtk");
  std::vector<T> dof(mesh.get_num_nodes());
  for (int i = 0; i < dof.size(); i++) {
    dof[i] = T(i);
  }
  vtk.write_mesh();
  vtk.write_sol("dof", dof.data());

  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> dof(mesh.get_num_nodes(), 0.0);
    int nodes[Basis::Mesh::nodes_per_element];
    mesh.get_elem_dof_nodes(elem, nodes);
    for (int i = 0; i < Basis::Mesh::nodes_per_element; i++) {
      dof[nodes[i]] = 1.0;
    }
    char name[256];
    std::snprintf(name, 256, "elem_%05d", elem);
    vtk.write_sol(name, dof.data());
  }

  // Write interpolation field
  Interpolator interpolator(mesh, quadrature, basis);
  interpolator.to_vtk(name + "_field.vtk", dof.data());
}

TEST(elements, GDSamplingCutMeshGaussQuadratures) {
  constexpr int Np_1d = 2;
  constexpr int samples_1d = 10;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using LSF = Line;
  using Quadrature = GDSampler2D<T, samples_1d, Mesh>;
  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  interpolate_dof_at_quadratures<T>(mesh, quadrature, basis, "gd_sampling_lsf");
}

TEST(elements, GDInterpolationGaussQuadrature) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;

  int nxy[2] = {20, 20};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  interpolate_dof_at_quadratures<T>(mesh, quadrature, basis,
                                    "gd_interpolation_gauss");
}

TEST(elements, GDInterpolationLSFQuadrature) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using GridMesh = GridMesh<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;

  int nxy[2] = {20, 20};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  interpolate_dof_at_quadratures<T>(mesh, quadrature, basis,
                                    "gd_interpolation_lsf");
}
