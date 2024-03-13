#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "interp_data.h"
#include "physics/physics_commons.h"
#include "test_commons.h"
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
  using QuadratureBase::num_quadrature_pts;
  static constexpr int samples = num_quadrature_pts;
  using typename QuadratureBase::Mesh;

  StructuredSampling2D(Mesh& mesh) : QuadratureBase(mesh) {
    int nxy[2] = {samples_1d - 1, samples_1d - 1};
    T lxy[2] = {1.0, 1.0};
    T xy0[2] = {-1.0, -1.0};
    grid = new StructuredGrid2D<T>(nxy, lxy, xy0);
  }

  void get_quadrature_pts(int _, T pts[], T wts[]) const {
    for (int i = 0; i < samples; i++) {
      grid->get_vert_xloc(i, pts);
      pts += Mesh::spatial_dim;
    }
  }

 private:
  StructuredGrid2D<T>* grid;
};

template <class Mesh, class Quadrature, class Basis>
void interpolate(const std::string name, Mesh& mesh, std::vector<T>& ptx,
                 std::vector<T>& vals) {
  int constexpr dof_per_node = 1;
  int constexpr data_per_node = 0;
  using Physics =
      PhysicsBase<T, Mesh::spatial_dim, data_per_node, dof_per_node>;
  using Analysis = GalerkinAnalysis<T, Quadrature, Basis, Physics>;

  Quadrature quadrature(mesh);
  Basis basis(mesh);
  Physics physics;
  Analysis analysis(quadrature, basis, physics);

  std::vector<T> dof(mesh.get_num_nodes(), 0.0);
  for (int i = 0; i < mesh.get_num_nodes(); i++) {
    dof[i] = T(i);
  }

  ToVTK<T, Mesh> mesh_vtk(mesh, name + "_mesh.vtk");
  mesh_vtk.write_mesh();
  mesh_vtk.write_sol("scalar", dof.data());

  FieldToVTK<T, Basis::spatial_dim> field_vtk(name + "_field.vtk");
  for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
    std::vector<T> element_dof(Mesh::nodes_per_element);
    std::vector<T> element_xloc(Mesh::nodes_per_element * Basis::spatial_dim);
    std::vector<T> pts(Quadrature::num_quadrature_pts * Basis::spatial_dim);
    std::vector<T> wts(Quadrature::num_quadrature_pts);
    std::vector<T> _vals(Quadrature::num_quadrature_pts);
    std::vector<T> _ptx(Quadrature::num_quadrature_pts * Basis::spatial_dim);
    T N[Mesh::nodes_per_element * Quadrature::num_quadrature_pts];

    analysis.template get_element_vars<dof_per_node>(elem, dof.data(),
                                                     element_dof.data());

    analysis.get_element_xloc(elem, element_xloc.data());
    quadrature.get_quadrature_pts(elem, pts.data(), wts.data());
    basis.template eval_basis_grad<Quadrature::num_quadrature_pts>(
        elem, pts.data(), N, nullptr);

    for (int i = 0; i < Quadrature::num_quadrature_pts; i++) {
      int offset_n = i * Basis::nodes_per_element;

      T val = 0.0;
      interp_val_grad<T, Basis>(elem, element_dof.data(), &N[offset_n], nullptr,
                                &val, nullptr);
      _vals[i] = val;
      A2D::Vec<T, Basis::spatial_dim> xloc;
      interp_val_grad<T, Basis, Basis::spatial_dim>(
          elem, element_xloc.data(), &N[offset_n], nullptr, &xloc, nullptr);
      for (int d = 0; d < Basis::spatial_dim; d++) {
        _ptx[i * Basis::spatial_dim + d] = xloc[d];
      }
    }
    ptx.insert(ptx.end(), _ptx.begin(), _ptx.end());
    vals.insert(vals.end(), _vals.begin(), _vals.end());
  }

  field_vtk.add_scalar_field(ptx, vals);
  field_vtk.write_vtk();
}

TEST(Interpolation, Quad) {
  using Mesh = FEMesh<T, 2, 4>;
  using Quadrature = StructuredSampling2D<QuadInterpData::samples_1d, Mesh>;
  using Basis = QuadrilateralBasis<T, Mesh>;

  // Create a coarse mesh
  int num_elements, num_nodes;
  int* element_nodes;
  double* xloc;

  create_2d_rect_quad_mesh(QuadInterpData::nxy, QuadInterpData::lxy,
                           &num_elements, &num_nodes, &element_nodes, &xloc);
  Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  std::vector<T> ptx, vals;
  interpolate<Mesh, Quadrature, Basis>("quad", mesh, ptx, vals);

  for (int i = 0; i < ptx.size(); i++) {
    EXPECT_NEAR(ptx[i], QuadInterpData::ptx[i], 1e-15);
  }
  for (int i = 0; i < vals.size(); i++) {
    EXPECT_NEAR(vals[i], QuadInterpData::vals[i], 1e-15);
  }
}

TEST(Interpolation, GD) {
  int constexpr Np_1d = GDInterpData::Np_1d;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;
  using Quadrature = StructuredSampling2D<GDInterpData::samples_1d, Mesh>;
  using Basis = GDBasis2D<T, Np_1d, Mesh>;

  Grid grid(GDInterpData::nxy, GDInterpData::lxy);
  Mesh mesh(grid);
  std::vector<T> ptx, vals;
  interpolate<Mesh, Quadrature, Basis>("gd", mesh, ptx, vals);

  for (int i = 0; i < ptx.size(); i++) {
    EXPECT_NEAR(ptx[i], GDInterpData::ptx[i], 1e-15);
  }
  for (int i = 0; i < vals.size(); i++) {
    EXPECT_NEAR(vals[i], GDInterpData::vals[i], 1e-15);
  }
}

TEST(Interpolation, Demo) {
  constexpr int samples_1d = 10;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;
  using Quadrature = StructuredSampling2D<samples_1d, Mesh>;
  using Basis = GDBasis2D<T, Np_1d, Mesh>;

  int nxy[2] = {3, 3};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  std::vector<T> ptx, vals;
  interpolate<Mesh, Quadrature, Basis>("demo", mesh, ptx, vals);
}
