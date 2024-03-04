/*
Interpolate the field using different bases
*/

#include "elements/fe_quadrilateral.h"
#include "elements/gd_commons.h"
#include "utils/mesh.h"
#include "utils/vtk.h"

using T = double;

template <int samples_1d_, class Mesh_>
class StructuredSampling2D final
    : public QuadratureBase<T, samples_1d_ * samples_1d_, Mesh_> {
 private:
  using QuadratureBase = QuadratureBase<T, samples_1d_ * samples_1d_, Mesh_>;

 public:
  static constexpr int samples_1d = samples_1d_;
  static constexpr int samples = samples_1d * samples_1d;
  using QuadratureBase::num_quadrature_pts;
  using typename QuadratureBase::Mesh;

  StructuredSampling2D(Mesh& mesh) : QuadratureBase(mesh) {
    int nxy[2] = {samples_1d - 1, samples_1d - 1};
    T lxy[2] = {-1.0, 1.0};
    grid = new StructuredGrid2D<T>(nxy, lxy);
  }

  void get_quadrature_pts(int _, T pts[], T wts[]) const {}

  void get_sampling_pts(T pts[]) {
    for (int i = 0; i < samples; i++) {
      grid->get_vert_xloc(i, pts);
      pts += Mesh::spatial_dim;
    }
  }

 private:
  StructuredGrid2D<T>* grid;
};

int main() {
  int constexpr samples = 10;
  using Basis = QuadrilateralBasis<T>;
  using Sampling = StructuredSampling2D<samples, typename Basis::Mesh>;

  // Create a coarse mesh
  int num_elements, num_nodes;
  int* element_nodes;
  double* xloc;
  int nxy[2] = {1, 1};
  double lxy[2] = {1.0, 1.0};
  create_2d_rect_quad_mesh(nxy, lxy, &num_elements, &num_nodes, &element_nodes,
                           &xloc);
  typename Basis::Mesh mesh(num_elements, num_nodes, element_nodes, xloc);

  // Interpolate using the sampling grid
  Basis basis(mesh);

  std::vector<T> element_dof = {1, 2, 3, 4};

  Sampling sampling(mesh);

  T N[Sampling::samples];
  int elem = 0;
  std::vector<T> pts(Sampling::samples * Basis::spatial_dim);
  std::vector<T> vals(Sampling::samples);

  sampling.get_sampling_pts(pts.data());
  basis.eval_basis_grad(elem, pts.data(), N, nullptr);

  for (int i = 0; i < Sampling::samples; i++) {
    int offset_n = i * Basis::nodes_per_element;
    T val = 0.0;
    interp_val_grad<T, Basis>(0, element_dof.data(), &N[offset_n], nullptr,
                              &val, nullptr);
    vals[i] = val;
  }

  FieldToVTK<T, Basis::spatial_dim> vtk;
  vtk.add_scalar_field(pts, vals);
  vtk.write_vtk();

  return 0;
}