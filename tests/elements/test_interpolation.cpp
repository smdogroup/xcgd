#include <string>

#include "elements/element_utils.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/mesh.h"
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

template <typename T, class Quadrature, class Basis>
void interpolate_dof_at_quadratures(const Quadrature& quadrature,
                                    const Basis& basis,
                                    const std::string& name) {
  using Interpolator = Interpolator<T, Quadrature, Basis>;
  using Mesh = typename Basis::Mesh;
  const Mesh& mesh = basis.mesh;

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
  Interpolator interpolator(quadrature, basis);
  interpolator.to_vtk(name + "_field.vtk", dof.data());
}

// TODO: finish this
TEST(elements, InterpolationQuad) {}

TEST(elements, SampleGDGaussLSF) {
  constexpr int Np_1d = 2;
  constexpr int samples_1d = 10;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = Basis::Mesh;
  using LSF = Line;
  using Quadrature = GDSampler2D<T, samples_1d, Mesh>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  interpolate_dof_at_quadratures<T>(quadrature, basis, "sample_gd_gauss_lsf");
}

TEST(elements, QuadGDGaussLSF) {
  constexpr int Np_1d = 4;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = Basis::Mesh;
  using LSF = Line;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;

  int nxy[2] = {20, 20};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  interpolate_dof_at_quadratures<T>(quadrature, basis, "quad_gd_gauss_lsf");
}

TEST(elements, QuadGDLSFLSF) {
  constexpr int Np_1d = 2;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = Basis::Mesh;
  using LSF = Line;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d, LSF>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};
  LSF lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);
  Quadrature quadrature(mesh, lsf);

  interpolate_dof_at_quadratures<T>(quadrature, basis, "quad_gd_lsf_lsf");
}
