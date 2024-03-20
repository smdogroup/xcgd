#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/gd_vandermonde.h"
#include "interp_data.h"
#include "physics/physics_commons.h"
#include "test_commons.h"
#include "utils/mesh.h"
#include "utils/vtk.h"

template <typename T, int samples_1d_, class Mesh_>
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
    T lxy[2] = {2.0, 2.0};
    T xy0[2] = {-1.0, -1.0};
    grid = new StructuredGrid2D<T>(nxy, lxy, xy0);
  }

  void get_quadrature_pts(int _, T pts[], T __[]) const {
    for (int i = 0; i < samples; i++) {
      grid->get_vert_xloc(i, pts);
      pts += Mesh::spatial_dim;
    }
  }

 private:
  StructuredGrid2D<T>* grid;
};

template <typename T, class Mesh, class Quadrature, class Basis>
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

template <typename T, int samples_1d_, class Mesh_>
class SamplerGD final
    : public QuadratureBase<T, samples_1d_ * samples_1d_, Mesh_> {
 private:
  using QuadratureBase = QuadratureBase<T, samples_1d_ * samples_1d_, Mesh_>;

 public:
  static constexpr int samples_1d = samples_1d_;
  using QuadratureBase::num_quadrature_pts;
  static constexpr int samples = num_quadrature_pts;
  using typename QuadratureBase::Mesh;

  SamplerGD(const Mesh& mesh) : QuadratureBase(mesh) {}

  void get_quadrature_pts(int elem, T pts[], T _[]) const {
    T xymin[2], xymax[2];
    get_computational_coordinates_limits(elem, xymin, xymax);

    T lxy[2], xy0[2];
    for (int d = 0; d < Mesh::spatial_dim; d++) {
      xy0[d] = xymin[d] + 0.05 * (xymax[d] - xymin[d]);
      lxy[d] = 0.95 * (xymax[d] - xymin[d]);
    }
    int nxy[2] = {samples_1d - 1, samples_1d - 1};
    StructuredGrid2D<T> grid(nxy, lxy, xy0);

    for (int i = 0; i < samples; i++) {
      grid.get_vert_xloc(i, pts);
      pts += Mesh::spatial_dim;
    }
  }

 private:
  void get_computational_coordinates_limits(int elem, T* xymin,
                                            T* xymax) const {
    int constexpr spatial_dim = Mesh::spatial_dim;
    T xy_min[spatial_dim], xy_max[spatial_dim];
    T uv_min[spatial_dim], uv_max[spatial_dim];
    this->mesh.get_elem_node_ranges(elem, xy_min, xy_max);
    this->mesh.get_elem_vert_ranges(elem, uv_min, uv_max);

    T hx = (uv_max[0] - uv_min[0]) / (xy_max[0] - xy_min[0]);
    T hy = (uv_max[1] - uv_min[1]) / (xy_max[1] - xy_min[1]);
    T wt = 4.0 * hx * hy;

    T cx = (2.0 * uv_min[0] - xy_min[0] - xy_max[0]) / (xy_max[0] - xy_min[0]);
    T dx = 2.0 * hx;
    T cy = (2.0 * uv_min[1] - xy_min[1] - xy_max[1]) / (xy_max[1] - xy_min[1]);
    T dy = 2.0 * hy;

    xymin[0] = cx;
    xymin[1] = cy;
    xymax[0] = cx + dx;
    xymax[1] = cy + dy;
  }
};

/**
 * @brief Given a mesh and dof, interpolate the field.
 *
 * Note: This is useful for sanity check and debugging.
 */
template <typename T, int samples_1d, class Basis>
class Interpolator {
  using Mesh = typename Basis::Mesh;
  using Sampler = SamplerGD<T, samples_1d, Mesh>;

  static int constexpr dof_per_node = 1;
  static int constexpr data_per_node = 0;
  static int constexpr spatial_dim = Mesh::spatial_dim;

  using Physics = PhysicsBase<T, spatial_dim, data_per_node, dof_per_node>;
  using Analysis = GalerkinAnalysis<T, Sampler, Basis, Physics>;

 public:
  Interpolator(Basis& basis)
      : mesh(basis.mesh),
        basis(basis),
        sampler(mesh),
        analysis(sampler, basis, physics) {}

  void to_vtk(const std::string name, T* dof) const {
    FieldToVTK<T, spatial_dim> field_vtk(name);

    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      std::vector<T> element_dof(Mesh::nodes_per_element);
      std::vector<T> element_xloc(Mesh::nodes_per_element * Basis::spatial_dim);
      std::vector<T> pts(Sampler::num_quadrature_pts * Basis::spatial_dim);
      std::vector<T> ptx(Sampler::num_quadrature_pts * Basis::spatial_dim);
      std::vector<T> vals(Sampler::num_quadrature_pts);

      T N[Mesh::nodes_per_element * Sampler::num_quadrature_pts];

      analysis.template get_element_vars<dof_per_node>(elem, dof,
                                                       element_dof.data());

      analysis.get_element_xloc(elem, element_xloc.data());
      sampler.get_quadrature_pts(elem, pts.data(), nullptr);
      basis.template eval_basis_grad<Sampler::num_quadrature_pts>(
          elem, pts.data(), N, nullptr);

      for (int i = 0; i < Sampler::num_quadrature_pts; i++) {
        int offset_n = i * Basis::nodes_per_element;

        T val = 0.0;
        interp_val_grad<T, Basis>(elem, element_dof.data(), &N[offset_n],
                                  nullptr, &val, nullptr);
        vals[i] = val;
        A2D::Vec<T, Basis::spatial_dim> xloc;
        interp_val_grad<T, Basis, Basis::spatial_dim>(
            elem, element_xloc.data(), &N[offset_n], nullptr, &xloc, nullptr);
        for (int d = 0; d < Basis::spatial_dim; d++) {
          ptx[i * Basis::spatial_dim + d] = xloc[d];
        }
        // TODO: delete
        if (i == 0) {
          std::printf("elem %2d, xloc: (%.3f, %.3f)\n", elem, xloc[0], xloc[1]);
        }
      }
      // TODO: delete
      // std::printf("elem %2d, elem: (%.3f, %.3f), pt: (%.3f, %.3f)\n", elem,
      //             element_xloc[0], element_xloc[1], ptx[0], ptx[1]);
      field_vtk.add_scalar_field(ptx, vals);
    }
    field_vtk.write_vtk();
  }

 private:
  const Mesh& mesh;
  const Basis& basis;
  Sampler sampler;
  Physics physics;
  Analysis analysis;
};

TEST(elements, InterpolationQuad) {
  using T = double;
  using Mesh = FEMesh<T, 2, 4>;
  using Quadrature = StructuredSampling2D<T, QuadInterpData::samples_1d, Mesh>;
  using Basis = QuadrilateralBasis<T, Mesh>;

  // Create a coarse mesh
  int num_elements, num_nodes;
  int* element_nodes;
  double* xloc;

  create_2d_rect_quad_mesh(QuadInterpData::nxy, QuadInterpData::lxy,
                           &num_elements, &num_nodes, &element_nodes, &xloc);
  Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  std::vector<T> ptx, vals;
  interpolate<T, Mesh, Quadrature, Basis>("quad", mesh, ptx, vals);

  for (int i = 0; i < ptx.size(); i++) {
    EXPECT_NEAR(ptx[i], QuadInterpData::ptx[i], 1e-15);
  }
  for (int i = 0; i < vals.size(); i++) {
    EXPECT_NEAR(vals[i], QuadInterpData::vals[i], 1e-15);
  }
}

TEST(elements, InterpolationGD) {
  using T = double;
  int constexpr Np_1d = GDInterpData::Np_1d;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;
  using Quadrature = StructuredSampling2D<T, GDInterpData::samples_1d, Mesh>;
  using Basis = GDBasis2D<T, Np_1d, Mesh>;

  Grid grid(GDInterpData::nxy, GDInterpData::lxy);
  Mesh mesh(grid);
  std::vector<T> ptx, vals;
  interpolate<T, Mesh, Quadrature, Basis>("gd", mesh, ptx, vals);

  for (int i = 0; i < ptx.size(); i++) {
    EXPECT_NEAR(ptx[i], GDInterpData::ptx[i], 1e-15);
  }
  for (int i = 0; i < vals.size(); i++) {
    EXPECT_NEAR(vals[i], GDInterpData::vals[i], 1e-15);
  }
}

TEST(elements, InterpolationDemo) {
  using T = double;
  constexpr int samples_1d = 10;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;
  using Quadrature = StructuredSampling2D<T, samples_1d, Mesh>;
  using Basis = GDBasis2D<T, Np_1d, Mesh>;

  int nxy[2] = {3, 3};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  std::vector<T> ptx, vals;
  interpolate<T, Mesh, Quadrature, Basis>("demo", mesh, ptx, vals);
}

template <typename T>
class Line {
 public:
  constexpr static int spatial_dim = 2;
  Line(T k = 0.9, T b = 0.1) : k(k), b(b) {}

  T operator()(const algoim::uvector<T, spatial_dim>& x) const {
    return -k * x(0) + x(1) - b;
  }

  algoim::uvector<T, spatial_dim> grad(const algoim::uvector<T, 2>& x) const {
    return algoim::uvector<T, spatial_dim>(-k, 1.0);
  }

 private:
  T k, b;
};

TEST(elements, InterpolationDemo2) {
  constexpr int Np_1d = 2;
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Mesh = Basis::Mesh;

  int constexpr samples_1d = 10;
  using Interpolator = Interpolator<T, samples_1d, Basis>;

  int nxy[2] = {5, 5};
  T lxy[2] = {1.0, 1.0};

  T center[2] = {0.0, 0.0};

  Line<T> lsf;

  Grid grid(nxy, lxy);
  Mesh mesh(grid, lsf);
  Basis basis(mesh);

  // Write mesh
  ToVTK<T, Mesh> vtk(mesh, "gd_mesh_lsf.vtk");
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
  Interpolator interpolator(basis);
  interpolator.to_vtk("gd_mesh_lsf_field.vtk", dof.data());
}
