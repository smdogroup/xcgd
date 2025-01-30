#include <mpi.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <variant>

#include "ParOptOptimizer.h"
#include "analysis.h"
#include "apps/convolution_filter.h"
#include "apps/helmholtz_filter.h"
#include "apps/robust_projection.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "elements/lbracket_mesh.h"
#include "physics/grad_penalization.h"
#include "physics/stress.h"
#include "physics/volume.h"
#include "utils/argparser.h"
#include "utils/exceptions.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/timer.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

using fspath = std::filesystem::path;

template <typename T, int Np_1d, class Grid_>
class ProbMeshBase {
 public:
  using Grid = Grid_;
  using Mesh = CutMesh<T, Np_1d, Grid>;

  virtual T get_domain_area() = 0;
  virtual int get_nvars() = 0;
  virtual std::set<int> get_loaded_cells() = 0;
  virtual std::set<int> get_loaded_verts() = 0;
  virtual std::set<int> get_protected_verts() = 0;
  virtual std::vector<int> get_bc_nodes() = 0;
  virtual std::vector<T> expand(std::vector<T> x) = 0;  // expand xr -> x
  virtual std::vector<T> reduce(std::vector<T> x) = 0;  // reduce x -> xr
  virtual Grid& get_grid() = 0;
  virtual Mesh& get_mesh() = 0;
};

template <typename T, int Np_1d>
class CantileverMesh final
    : public ProbMeshBase<T, Np_1d, StructuredGrid2D<T>> {
 private:
  using Base = ProbMeshBase<T, Np_1d, StructuredGrid2D<T>>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  CantileverMesh(std::array<int, Grid::spatial_dim> nxy,
                 std::array<T, Grid::spatial_dim> lxy, double loaded_frac)
      : nxy(nxy),
        lxy(lxy),
        grid(nxy.data(), lxy.data()),
        mesh(grid),
        loaded_frac(loaded_frac) {
    // Find loaded cells
    for (int iy = 0; iy < nxy[1]; iy++) {
      T xloc[Grid::spatial_dim];
      int c = this->grid.get_coords_cell(nxy[0] - 1, iy);
      this->grid.get_cell_xloc(c, xloc);
      if (xloc[1] >= lxy[1] * (1.0 - loaded_frac) / 2.0 and
          xloc[1] <= lxy[1] * (1.0 + loaded_frac) / 2.0) {
        loaded_cells.insert(c);
      }
    }

    // Find loaded verts
    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      loaded_verts.insert(verts[1]);
      loaded_verts.insert(verts[2]);
    }

    N = this->grid.get_num_verts();
    Nr = N - loaded_verts.size();

    // Find xr -> x mapping
    expand_mapping.reserve(Nr);  // xr -> x
    for (int i = 0; i < N; i++) {
      if (!loaded_verts.count(i)) {
        expand_mapping.push_back(i);
      }
    }
  }

  int get_nvars() { return Nr; }

  T get_domain_area() { return lxy[0] * lxy[1]; }

  std::set<int> get_loaded_cells() { return loaded_cells; }
  std::set<int> get_loaded_verts() { return loaded_verts; }
  std::set<int> get_protected_verts() { return {}; }

  std::vector<int> get_bc_nodes() {
    return this->mesh.get_left_boundary_nodes();
  }

  std::vector<T> expand(std::vector<T> xr) {
    std::vector<T> x(N, 0.0);  // dv = 0.0 is material
    for (int i = 0; i < Nr; i++) {
      x[expand_mapping[i]] = xr[i];
    }
    return x;
  }

  std::vector<T> reduce(std::vector<T> x) {
    std::vector<T> xr(Nr);
    for (int i = 0; i < Nr; i++) {
      xr[i] = x[expand_mapping[i]];
    }
    return xr;
  }

  Grid& get_grid() { return grid; }

  Mesh& get_mesh() { return mesh; }

 private:
  std::array<int, Grid::spatial_dim> nxy;
  std::array<T, Grid::spatial_dim> lxy;
  Grid grid;
  Mesh mesh;

  double loaded_frac;
  std::set<int> loaded_cells, loaded_verts;
  int N, Nr;
  std::vector<int> reduce_mapping, expand_mapping;
};

template <typename T, int Np_1d>
class LbracketMesh final : public ProbMeshBase<T, Np_1d, StructuredGrid2D<T>> {
 private:
  using Base = ProbMeshBase<T, Np_1d, StructuredGrid2D<T>>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  LbracketMesh(std::array<int, Grid::spatial_dim> nxy,
               std::array<T, Grid::spatial_dim> lxy, double loaded_frac,
               double lbracket_frac)
      : nxy(nxy),
        lxy(lxy),
        grid(nxy.data(), lxy.data()),
        mesh(grid),
        loaded_frac(loaded_frac),
        domain_area(lxy[0] * lxy[1] *
                    (1.0 - (1.0 - lbracket_frac) * (1.0 - lbracket_frac))) {
    T ty = lxy[1] * lbracket_frac;

    // Find loaded cells and verts
    for (int iy = 0; iy < nxy[1]; iy++) {
      T xloc[Grid::spatial_dim];
      int c = this->grid.get_coords_cell(nxy[0] - 1, iy);
      this->grid.get_cell_xloc(c, xloc);
      if (xloc[1] >= ty - lxy[1] * loaded_frac and xloc[1] <= ty) {
        loaded_cells.insert(c);
      }
    }

    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      loaded_verts.insert(verts[1]);
      loaded_verts.insert(verts[2]);
    }

    // Find inactive verts
    T tx = lxy[0] * lbracket_frac;
    inactive_verts = loaded_verts;
    for (int v = 0; v < this->grid.get_num_verts(); v++) {
      T xloc[Grid::spatial_dim];
      this->grid.get_vert_xloc(v, xloc);
      if (xloc[0] > tx and xloc[1] > ty) {
        inactive_verts.insert(v);
      }
    }

    N = this->grid.get_num_verts();
    Nr = N - inactive_verts.size();

    // Find xr -> x mapping
    expand_mapping.reserve(Nr);  // xr -> x
    for (int i = 0; i < N; i++) {
      if (!inactive_verts.count(i)) {
        expand_mapping.push_back(i);
      }
    }
  }

  int get_nvars() { return Nr; }

  T get_domain_area() { return domain_area; }

  std::set<int> get_loaded_cells() { return loaded_cells; }
  std::set<int> get_loaded_verts() { return loaded_verts; }
  std::set<int> get_protected_verts() { return {}; }

  std::vector<int> get_bc_nodes() {
    return this->mesh.get_upper_boundary_nodes();
  }

  std::vector<T> expand(std::vector<T> xr) {
    std::vector<T> x(N, 1.0);  // lsf = 1.0 is void
    for (int i = 0; i < Nr; i++) {
      x[expand_mapping[i]] = xr[i];
    }

    for (int i : loaded_verts) {
      x[i] = -1.0;  // lsf = -1.0 is material
    }
    return x;
  }

  std::vector<T> reduce(std::vector<T> x) {
    std::vector<T> xr(Nr);
    for (int i = 0; i < Nr; i++) {
      xr[i] = x[expand_mapping[i]];
    }
    return xr;
  }

  Grid& get_grid() { return grid; }

  Mesh& get_mesh() { return mesh; }

 private:
  std::array<int, Grid::spatial_dim> nxy;
  std::array<T, Grid::spatial_dim> lxy;
  Grid grid;
  Mesh mesh;
  double loaded_frac, domain_area;
  std::set<int> loaded_cells, loaded_verts;
  std::set<int> inactive_cells, inactive_verts;
  int N, Nr;
  std::vector<int> reduce_mapping, expand_mapping;
};

// load_top: true to put load on the top of the lbracket arm, false to put load
// on the side
template <typename T, int Np_1d, bool load_top = false>
class LbracketGridMesh final
    : public ProbMeshBase<T, Np_1d, LbracketGrid2D<T>> {
 private:
  using Base = ProbMeshBase<T, Np_1d, LbracketGrid2D<T>>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  LbracketGridMesh(std::array<int, Grid::spatial_dim> nxy,
                   std::array<T, Grid::spatial_dim> lxy, double loaded_frac,
                   double lbracket_frac)
      : nx1(nxy[0]),
        nx2(static_cast<int>(nxy[0] * lbracket_frac)),
        ny1(static_cast<int>(nxy[1] * lbracket_frac)),
        ny2(nxy[1] - ny1),
        lx1(lxy[0]),
        ly1(lxy[1] * lbracket_frac),
        grid(nx1, nx2, ny1, ny2, lx1, ly1),
        mesh(grid),
        loaded_frac(loaded_frac),
        domain_area(lxy[0] * lxy[1] *
                    (1.0 - (1.0 - lbracket_frac) * (1.0 - lbracket_frac))) {
    // Find loaded cells and verts
    if constexpr (load_top) {
      for (int ix = 0; ix < nx1; ix++) {
        T xloc[Grid::spatial_dim];
        int c = this->grid.get_coords_cell(ix, ny1 - 1);
        this->grid.get_cell_xloc(c, xloc);
        if (xloc[0] >= lx1 * (1.0 - loaded_frac) and xloc[0] <= lx1) {
          loaded_cells.insert(c);
        }
      }
    } else {
      for (int iy = 0; iy < ny1; iy++) {
        T xloc[Grid::spatial_dim];
        int c = this->grid.get_coords_cell(nx1 - 1, iy);
        this->grid.get_cell_xloc(c, xloc);
        if (xloc[1] >= ly1 - lxy[1] * loaded_frac and xloc[1] <= ly1) {
          loaded_cells.insert(c);
        }
      }
    }

    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      if constexpr (load_top) {
        loaded_verts.insert(verts[2]);
        loaded_verts.insert(verts[3]);
      } else {
        loaded_verts.insert(verts[1]);
        loaded_verts.insert(verts[2]);
      }
    }

    for (int v : loaded_verts) {
      int ixy[2] = {-1, -1};
      grid.get_vert_coords(v, ixy);
      for (auto [dx, dy] : std::vector<std::pair<int, int>>{
               {0, 0}, {0, 1}, {0, -1}, {-1, 0}, {-1, -1}}) {
        if (grid.is_valid_vert(ixy[0] + dx, ixy[1] + dy)) {
          protected_verts.insert(
              grid.get_coords_vert(ixy[0] + dx, ixy[1] + dy));
        };
      }
    }

    N = this->grid.get_num_verts();

    // Find xr -> x mapping, identical mapping since we don't have non-design
    // verts any more
    Nr = N;
    expand_mapping.reserve(Nr);  // xr -> x
    for (int i = 0; i < N; i++) {
      expand_mapping.push_back(i);
    }
  }

  int get_nvars() { return Nr; }

  T get_domain_area() { return domain_area; }

  std::set<int> get_loaded_cells() { return loaded_cells; }
  std::set<int> get_loaded_verts() { return loaded_verts; }
  std::set<int> get_protected_verts() { return protected_verts; }

  std::vector<int> get_bc_nodes() {
    return this->mesh.get_upper_boundary_nodes();
  }

  std::vector<T> expand(std::vector<T> xr) {
    std::vector<T> x(N, -1.0);  // lsf = -1.0 is material
    for (int i = 0; i < Nr; i++) {
      x[expand_mapping[i]] = xr[i];
    }
    return x;
  }

  std::vector<T> reduce(std::vector<T> x) {
    std::vector<T> xr(Nr);
    for (int i = 0; i < Nr; i++) {
      xr[i] = x[expand_mapping[i]];
    }
    return xr;
  }

  Grid& get_grid() { return grid; }

  Mesh& get_mesh() { return mesh; }

 private:
  int nx1, nx2, ny1, ny2;
  T lx1, ly1;
  Grid grid;
  Mesh mesh;
  double loaded_frac, domain_area;
  std::set<int> loaded_cells, loaded_verts, protected_verts;
  std::set<int> inactive_cells, inactive_verts;
  int N, Nr;
  std::vector<int> reduce_mapping, expand_mapping;
};

template <typename T, int Np_1d, int Np_1d_filter, bool use_ersatz_,
          class Grid_>
class TopoAnalysis {
 public:
  static constexpr bool use_ersatz = use_ersatz_;
  static constexpr int get_spatial_dim() { return Grid_::spatial_dim; }

 private:
  using ProbMesh = ProbMeshBase<T, Np_1d, Grid_>;
  using Grid = typename ProbMesh::Grid;
  using Mesh = typename ProbMesh::Mesh;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid>;
  using Basis = GDBasis2D<T, Mesh>;
  using HFilter = HelmholtzFilter<T, Np_1d_filter, Grid>;
  using CFilter = ConvolutionFilter<T, Grid>;

  constexpr static auto int_func =
      [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        return A2D::Vec<T, Basis::spatial_dim>{};
      };
  constexpr static auto load_func =
      [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        A2D::Vec<T, Basis::spatial_dim> ret;
        ret(1) = -1.0;
        return ret;
      };

  using Elastic = typename std::conditional<
      use_ersatz,
      StaticElasticErsatz<T, Mesh, Quadrature, Basis, typeof(int_func), Grid>,
      StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_func)>>::type;
  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using Penalization = GradPenalization<T, Basis::spatial_dim>;
  using Stress = LinearElasticity2DVonMisesStress<T>;
  using StressKS = LinearElasticity2DVonMisesStressAggregation<T>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;
  using PenalizationAnalysis =
      GalerkinAnalysis<T, typename HFilter::Mesh, typename HFilter::Quadrature,
                       typename HFilter::Basis, Penalization>;
  using StressAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Stress>;
  using StressKSAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, StressKS, use_ersatz>;

  using LoadPhysics =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>;
  using LoadQuadrature =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;
  using LoadAnalysis =
      GalerkinAnalysis<T, Mesh, LoadQuadrature, Basis, LoadPhysics, use_ersatz>;

  int constexpr static spatial_dim = Basis::spatial_dim;

 public:
  TopoAnalysis(ProbMesh& prob_mesh, bool use_helmholtz_filter,
               int num_conv_filter_apply, T r0, T E, T nu, T penalty,
               T stress_ksrho, T yield_stress, bool use_robust_projection,
               double proj_beta, double proj_eta, std::string prefix,
               double compliance_scalar)
      : prob_mesh(prob_mesh),
        grid(prob_mesh.get_grid()),
        mesh(prob_mesh.get_mesh()),
        quadrature(mesh),
        basis(mesh),
        use_helmholtz_filter(use_helmholtz_filter),
        num_conv_filter_apply(num_conv_filter_apply),
        use_robust_projection(use_robust_projection),
        projector(proj_beta, proj_eta, grid.get_num_verts()),
        hfilter(r0, grid),
        cfilter(r0, grid),
        elastic(E, nu, mesh, quadrature, basis, int_func),
        vol_analysis(mesh, quadrature, basis, vol),
        pen(penalty),
        pen_analysis(hfilter.get_mesh(), hfilter.get_quadrature(),
                     hfilter.get_basis(), pen),
        stress(E, nu),
        stress_analysis(mesh, quadrature, basis, stress),
        stress_ks(stress_ksrho, E, nu, yield_stress),
        stress_ks_analysis(mesh, quadrature, basis, stress_ks),
        phi(mesh.get_lsf_dof()),
        prefix(prefix),
        cache({{"x", {}}, {"sol", {}}, {"chol", nullptr}}),
        compliance_scalar(compliance_scalar) {
    // Get loaded cells
    loaded_cells = prob_mesh.get_loaded_cells();
  }

  // construct the initial design using sine waves
  std::vector<T> create_initial_topology_sine(int period_x, int period_y,
                                              double offset) {
    const T* lxy = grid.get_lxy();
    int nverts = grid.get_num_verts();
    std::vector<T> dvs(nverts, 0.0);
    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      grid.get_vert_xloc(i, xloc);
      T x = xloc[0];
      T y = xloc[1];
      dvs[i] = -cos(x / lxy[0] * 2.0 * PI * period_x) *
                   cos(y / lxy[1] * 2.0 * PI * period_y) -
               offset;
      dvs[i] = dvs[i] / (1.0 + offset) / 2.0 +
               0.5;  // scale dvs uniformly so it's in [0, 1]
    }
    return dvs;
  }

  // Create nodal design variables for a domain with periodic holes
  std::vector<T> create_initial_topology(int nholes_x, int nholes_y, double r,
                                         bool cell_center = true) {
    const T* lxy = grid.get_lxy();
    int nverts = grid.get_num_verts();
    std::vector<T> lsf(nverts, 0.0);
    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      grid.get_vert_xloc(i, xloc);
      T x = xloc[0];
      T y = xloc[1];

      std::vector<T> lsf_vals;
      for (int ix = 0; ix < nholes_x; ix++) {
        for (int iy = 0; iy < nholes_y; iy++) {
          if (cell_center) {
            T x0 = lxy[0] / nholes_x / 2.0 * (2.0 * ix + 1.0);
            T y0 = lxy[1] / nholes_y / 2.0 * (2.0 * iy + 1.0);
            lsf_vals.push_back(r -
                               sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
          } else {
            T x0 = lxy[0] / (nholes_x - 1.0) * ix;
            T y0 = lxy[1] / (nholes_y - 1.0) * iy;
            lsf_vals.push_back(r -
                               sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
          }
        }
      }
      lsf[i] = hard_max(lsf_vals);
    }

    // // Normalize lsf so values are within [-1, 1]
    // T lsf_max = hard_max(lsf);
    // T lsf_min = hard_min(lsf);
    // for (int i = 0; i < nverts; i++) {
    //   if (lsf[i] < 0.0) {
    //     lsf[i] /= -lsf_min;
    //   } else {
    //     lsf[i] /= lsf_max;
    //   }
    // }

    return lsf;
  }

  // Map filtered x -> phi, recall that
  // x  -> filtered x -> phi
  // where x, filtered x are in [0, 1], and phi in h * fx - 0.5h, h = element
  // size
  void design_mapping_apply(const T* x, T* y) {
    double elem_h = 0.5 * (grid.get_h()[0] + grid.get_h()[1]);
    int ndv = phi.size();
    for (int i = 0; i < ndv; i++) {
      y[i] = elem_h * (x[i] - 0.5);
    }
  }

  void design_mapping_apply_gradient(const T* dfdy, T* dfdx) {
    double elem_h = 0.5 * (grid.get_h()[0] + grid.get_h()[1]);
    int ndv = phi.size();
    for (int i = 0; i < ndv; i++) {
      dfdx[i] = elem_h * dfdy[i];
    }
  }

  std::vector<T> update_mesh(const std::vector<T>& x) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    // Apply filter on design variables
    if (use_helmholtz_filter) {
      hfilter.apply(x.data(), phi.data());
    } else {
      cfilter.apply(x.data(), phi.data());
      for (int i = 0; i < num_conv_filter_apply - 1; i++) {
        cfilter.apply(phi.data(), phi.data());
      }
    }

    // Apply projection
    if (use_robust_projection) {
      projector.apply(phi.data(), phi.data());
    }

    // Save H(F(x))
    std::vector<T> HFx = phi;

    // Apply design mapping
    design_mapping_apply(phi.data(), phi.data());

    // Update the mesh given new phi value
    mesh.update_mesh();

    if constexpr (use_ersatz) {
      int nverts = grid.get_num_verts();
      auto& lsf_mesh = elastic.get_mesh_ersatz();
      auto& lsf_ersatz = lsf_mesh.get_lsf_dof();
      for (int i = 0; i < nverts; i++) {
        lsf_ersatz[i] = -phi[i];
      }
      lsf_mesh.update_mesh();
    }

    const std::unordered_map<int, int>& vert_nodes = mesh.get_vert_nodes();

    // Update bc dof
    bc_dof.clear();
    std::vector<int> bc_nodes = prob_mesh.get_bc_nodes();
    for (int n : bc_nodes) {
      if constexpr (use_ersatz) {
        n = mesh.get_node_vert(n);
      }
      for (int d = 0; d < spatial_dim; d++) {
        bc_dof.push_back(spatial_dim * n + d);
      }
    }

    // TODO: delete
    auto mesh = elastic.get_mesh();
    StencilToVTK<T, Mesh> stencil_vtk(mesh,
                                      fspath(prefix) / "debug_stencil.vtk");
    stencil_vtk.write_stencils(mesh.get_elem_nodes());

    return HFx;
  }

  std::vector<T> update_mesh_and_solve(
      const std::vector<T>& x, std::vector<T>& HFx,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol = nullptr) {
    // Solve the static problem
    HFx = update_mesh(x);

    VandermondeCondLogger::clear();
    VandermondeCondLogger::enable();

    try {
      LoadPhysics load_physics(load_func);
      std::set<int> load_elements;
      for (int i = 0; i < mesh.get_num_elements(); i++) {
        if (loaded_cells.count(mesh.get_elem_cell(i))) {
          load_elements.insert(i);
        }
      }
      LoadQuadrature load_quadrature(mesh, load_elements);
      LoadAnalysis load_analysis(mesh, load_quadrature, basis, load_physics);

      std::vector<T> sol =
          elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), T(0.0)),
                        std::tuple<LoadAnalysis>(load_analysis), chol);

      return sol;
    } catch (const StencilConstructionFailed& e) {
      std::printf(
          "StencilConstructionFailed error has been caught when calling "
          "update_mesh_and_solve(), dumping debug info in a vtk and "
          "throwing...\n");
      auto cut_mesh = elastic.get_mesh();
      ToVTK<T, Mesh> err_vtk(
          cut_mesh, fspath(prefix) / "stencil_construction_failed.vtk");
      err_vtk.write_mesh();

      std::vector<double> failed_elem(cut_mesh.get_num_elements(), 0.0);
      failed_elem[e.get_elem_index()] = 1.0;
      err_vtk.write_cell_sol("failed_element", failed_elem.data());

      throw e;
    }

    VandermondeCondLogger::disable();
  }

  auto eval_obj_con(const std::vector<T>& x) {
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol;
    std::vector<T> HFx;
    std::vector<T> sol = update_mesh_and_solve(x, HFx, &chol);

    T comp = std::inner_product(sol.begin(), sol.end(),
                                elastic.get_rhs().begin(), T(0.0)) *
             compliance_scalar;

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    T area = vol_analysis.energy(nullptr, dummy.data());
    T pterm = pen_analysis.energy(nullptr, phi.data());

    auto [xloc_q, stress_q] = eval_stress(sol);
    T max_stress = *std::max_element(stress_q.begin(), stress_q.end());
    T max_stress_ratio = max_stress / stress_ks.get_yield_stress();
    stress_ks.set_max_stress_ratio(max_stress_ratio);
    T ks_energy = stress_ks_analysis.energy(nullptr, sol.data());
    T ks_stress_ratio =
        log(ks_energy) / stress_ks.get_ksrho() + max_stress_ratio;

    // Save information
    cache["x"] = x;
    cache["sol"] = sol;
    cache["chol"] = chol;
    cache["ks_energy"] = ks_energy;
    cache["area"] = area;

    return std::make_tuple(HFx, comp, area, pterm, max_stress, max_stress_ratio,
                           ks_stress_ratio, sol, xloc_q, stress_q);
  }

  // only useful if ersatz material is used
  template <int dim>
  std::vector<T> grid_dof_to_cut_dof(const std::vector<T> u) {
    if (u.size() != mesh.get_grid().get_num_verts() * dim) {
      throw std::runtime_error(
          "grid_dof_to_cut_dof() only takes dof vectors of size nverts * dim "
          "(" +
          std::to_string(mesh.get_grid().get_num_verts() * dim) + "), got " +
          std::to_string(u.size()));
    }

    int nnodes = mesh.get_num_nodes();
    std::vector<T> u0(dim * nnodes);
    for (int i = 0; i < nnodes; i++) {
      int vert = mesh.get_node_vert(i);
      for (int d = 0; d < dim; d++) {
        u0[dim * i + d] = u[dim * vert + d];
      }
    }
    return u0;
  }

  template <int dim>
  std::vector<T> cut_dof_to_grid_dof(const std::vector<T> u0) {
    if (u0.size() != mesh.get_num_nodes() * dim) {
      throw std::runtime_error(
          "cut_dof_to_grid_dof() only takes dof vectors of size nnodes * dim "
          "(" +
          std::to_string(mesh.get_num_nodes() * dim) + "), got " +
          std::to_string(u0.size()));
    }

    int nnodes = mesh.get_num_nodes();
    std::vector<T> u(dim * mesh.get_grid().get_num_verts(), T(0.0));
    for (int i = 0; i < nnodes; i++) {
      int vert = mesh.get_node_vert(i);
      for (int d = 0; d < dim; d++) {
        u[dim * vert + d] = u0[dim * i + d];
      }
    }
    return u;
  }

  std::pair<std::vector<T>, std::vector<T>> eval_stress(
      const std::vector<T>& u) {
    if constexpr (use_ersatz) {
      return stress_analysis.interpolate_energy(
          grid_dof_to_cut_dof<spatial_dim>(u).data());
    } else {
      return stress_analysis.interpolate_energy(u.data());
    }
  }

  void eval_obj_con_gradient(const std::vector<T>& x, std::vector<T>& gcomp,
                             std::vector<T>& garea, std::vector<T>& gpen,
                             std::vector<T>& gstress) {
    T ks_energy = 0.0, area = 0.0;
    std::vector<T> sol;
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol;
    if (x == std::get<std::vector<T>>(cache["x"])) {
      sol = std::get<std::vector<T>>(cache["sol"]);
      chol = std::get<std::shared_ptr<SparseUtils::SparseCholesky<T>>>(
          cache["chol"]);
      ks_energy = std::get<T>(cache["ks_energy"]);
      area = std::get<T>(cache["area"]);
    } else {
      std::vector<T> HFx;
      sol = update_mesh_and_solve(x, HFx, &chol);
      ks_energy = stress_ks_analysis.energy(nullptr, sol.data());
      std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
      area = vol_analysis.energy(nullptr, dummy.data());
    }

    // Compliance function is self-adjoint with a sign flip
    std::vector<T> psi_comp = sol;
    for (T& p : psi_comp) p *= -1.0;

    // Evaluate the rhs of the adjoint equation for stress
    // adjoint equation is K * psi = -∂s/∂u
    std::vector<T> psi_stress(sol.size(), T(0.0));
    stress_ks_analysis.residual(nullptr, sol.data(), psi_stress.data());

    // Apply boundary conditions to the rhs
    for (int i : bc_dof) {
      psi_stress[i] = 0.0;
    }

    // Compute stress adjoints
    chol->solve(psi_stress.data());

    // Apply boundary conditions again to the adjoint variables
    for (int i : bc_dof) {
      psi_stress[i] = 0.0;
    }

    std::vector<T> psi_stress_neg = psi_stress;
    for (T& p : psi_stress) p *= -1.0;

    gcomp.resize(x.size());
    std::fill(gcomp.begin(), gcomp.end(), 0.0);
    elastic.get_analysis().LSF_jacobian_adjoint_product(
        sol.data(), psi_comp.data(), gcomp.data());
    if constexpr (use_ersatz) {
      elastic.get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), sol.data() /*this is effectively -psi*/, gcomp.data());
    }

    std::vector<T> Fx(x.size(), 0.0);
    if (use_helmholtz_filter) {
      hfilter.apply(x.data(), Fx.data());
    } else {
      cfilter.apply(x.data(), Fx.data());
    }

    design_mapping_apply_gradient(gcomp.data(), gcomp.data());
    if (use_robust_projection) {
      projector.applyGradient(Fx.data(), gcomp.data(), gcomp.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(gcomp.data(), gcomp.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(gcomp.data(), gcomp.data());
      }
    }

    std::transform(
        gcomp.begin(), gcomp.end(), gcomp.begin(),
        [this](const T& val) { return val * this->compliance_scalar; });

    garea.resize(x.size());
    std::fill(garea.begin(), garea.end(), 0.0);
    vol_analysis.LSF_volume_derivatives(garea.data());

    design_mapping_apply_gradient(garea.data(), garea.data());
    if (use_robust_projection) {
      projector.applyGradient(Fx.data(), garea.data(), garea.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(garea.data(), garea.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(garea.data(), garea.data());
      }
    }

    gpen.resize(x.size());
    std::fill(gpen.begin(), gpen.end(), 0.0);
    pen_analysis.residual(nullptr, phi.data(), gpen.data());

    design_mapping_apply_gradient(gpen.data(), gpen.data());
    if (use_robust_projection) {
      projector.applyGradient(Fx.data(), gpen.data(), gpen.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(gpen.data(), gpen.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(gpen.data(), gpen.data());
      }
    }

    gstress.resize(x.size());
    std::fill(gstress.begin(), gstress.end(), 0.0);

    // Explicit partials
    stress_ks_analysis.LSF_energy_derivatives(sol.data(), gstress.data());

    // Implicit derivatives via the adjoint variables
    elastic.get_analysis().LSF_jacobian_adjoint_product(
        sol.data(), psi_stress.data(), gstress.data());
    if constexpr (use_ersatz) {
      elastic.get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), psi_stress_neg.data(), gstress.data());
    }

    design_mapping_apply_gradient(gstress.data(), gstress.data());
    if (use_robust_projection) {
      projector.applyGradient(Fx.data(), gstress.data(), gstress.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(gstress.data(), gstress.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(gstress.data(), gstress.data());
      }
    }

    // Now gstress is really just denergy/dx, next, compute dks/dx:
    // dks/dx = (1.0 / energy * denergy/dx - 1.0 / area * darea/dx) / rho
    for (int i = 0; i < gstress.size(); i++) {
      // gstress[i] =
      //     (gstress[i] / ks_energy - garea[i] / area) / stress_ks.get_ksrho();
      gstress[i] = (gstress[i] / ks_energy) / stress_ks.get_ksrho();
    }
  }

  void write_quad_pts_to_vtk(const std::string vtk_path) {
    Interpolator<T, Quadrature, Basis> interp{mesh, quadrature, basis};
    interp.to_vtk(vtk_path);
  }

  void write_grid_vtk(const std::string vtk_path, const std::vector<T>& x,
                      const std::vector<T>& phi,
                      std::map<std::string, std::vector<T>&> node_sols = {},
                      std::map<std::string, std::vector<T>&> cell_sols = {},
                      std::map<std::string, std::vector<T>&> node_vecs = {},
                      std::map<std::string, std::vector<T>&> cell_vecs = {}) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    ToVTK<T, typename HFilter::Mesh> vtk(hfilter.get_mesh(), vtk_path);
    vtk.write_mesh();

    // Node solutions
    for (auto [name, vals] : node_sols) {
      if (vals.size() != grid.get_num_verts()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]node sol size doesn't match "
            "number of verts");
      }
      vtk.write_sol(name, vals.data());
    }

    vtk.write_sol("x", x.data());
    vtk.write_sol("phi", phi.data());

    // Node vectors
    for (auto [name, vals] : node_vecs) {
      if (vals.size() != spatial_dim * grid.get_num_verts()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]node vec size doesn't match "
            "number of verts * spatial_dim");
      }
      vtk.write_vec(name, vals.data());
    }

    // Cell solutions
    for (auto [name, vals] : cell_sols) {
      if (vals.size() != grid.get_num_cells()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]cell sol size doesn't match "
            "number of cells");
      }
      vtk.write_cell_sol(name, vals.data());
    }

    std::vector<T> loaded_cells_v(grid.get_num_cells(), 0.0);
    for (int c : loaded_cells) loaded_cells_v[c] = 1.0;
    vtk.write_cell_sol("loaded_cells", loaded_cells_v.data());

    // Cell vectors
    for (auto [name, vals] : cell_vecs) {
      if (vals.size() != spatial_dim * grid.get_num_cells()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]cell vec size doesn't match "
            "number of cells * spatial_dim");
      }
      vtk.write_cell_vec(name, vals.data());
    }
  }

  void write_cut_vtk(const std::string vtk_path, const std::vector<T>& x,
                     const std::vector<T>& phi,
                     std::map<std::string, std::vector<T>&> node_sols = {},
                     std::map<std::string, std::vector<T>&> cell_sols = {},
                     std::map<std::string, std::vector<T>&> node_vecs = {},
                     std::map<std::string, std::vector<T>&> cell_vecs = {}) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("size doesn't match for x and phi, dim(x): " +
                               std::to_string(x.size()) +
                               ", dim(phi): " + std::to_string(phi.size()));
    }

    const Mesh& mesh = elastic.get_mesh();
    ToVTK<T, Mesh> vtk(mesh, vtk_path);
    vtk.write_mesh();

    // Node solutions
    for (auto [name, vals] : node_sols) {
      if (vals.size() != mesh.get_num_nodes()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_cut_vtk()]node sol size doesn't match "
            "number of nodes");
      }
      vtk.write_sol(name, vals.data());
    }
    vtk.write_sol("x", mesh.get_lsf_nodes(x).data());
    vtk.write_sol("phi", mesh.get_lsf_nodes().data());

    std::vector<T> bc_nodes_v(mesh.get_num_nodes(), 0.0);
    const auto& bc_nodes = prob_mesh.get_bc_nodes();
    for (int n : bc_nodes) {
      bc_nodes_v[n] = 1.0;
    }
    vtk.write_sol("bc_nodes", bc_nodes_v.data());

    // Node vectors
    for (auto [name, vals] : node_vecs) {
      if (vals.size() != spatial_dim * mesh.get_num_nodes()) {
        throw std::runtime_error("[TopoAnalysis::write_cut_vtk()]node vec " +
                                 name +
                                 " size doesn't match "
                                 "number of nodes * spatial_dim");
      }
      vtk.write_vec(name, vals.data());
    }

    // Cell solutions
    int nelems = mesh.get_num_elements();
    std::vector<T> is_cut_elem_v(nelems, 0.0);
    for (int i = 0; i < nelems; i++) {
      if (mesh.is_cut_elem(i)) {
        is_cut_elem_v[i] = 1.0;
      }
    }
    vtk.write_cell_sol("is_cut_element", is_cut_elem_v.data());
    for (auto [name, vals] : cell_sols) {
      if (vals.size() != mesh.get_num_elements()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_cut_vtk()]cell sol size doesn't match "
            "number of elements");
      }
      vtk.write_cell_sol(name, vals.data());
    }

    std::vector<double> conds(mesh.get_num_elements());
    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      conds[elem] = VandermondeCondLogger::get_conds().at(elem);
    }
    vtk.write_cell_sol("cond", conds.data());

    auto& stencils = mesh.get_elem_nodes();
    std::map<int, std::vector<int>> degenerate_stencils;
    std::vector<double> nstencils(stencils.size(), -1);
    for (auto& [elem, stencil] : stencils) {
      int num = stencil.size();
      nstencils[elem] = num;
      if (num < Mesh::max_nnodes_per_element) {
        degenerate_stencils[elem] = stencil;
      }
    }
    vtk.write_cell_sol("nstencils", nstencils.data());

    // Cell vectors
    for (auto [name, vals] : cell_vecs) {
      if (vals.size() != mesh.get_num_elements()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_cut_vtk()]cell vec size doesn't match "
            "number of elements * spatial_dim");
      }
      vtk.write_cell_vec(name, vals.data());
    }

    // Save stencils and degenerate stencils
    auto [base, suffix] = split_path(vtk_path);
    StencilToVTK<T, Mesh> stencil_vtk(mesh, base + "_stencils" + suffix);
    stencil_vtk.write_stencils(stencils);
    StencilToVTK<T, Mesh> degen_stencil_vtk(mesh,
                                            base + "_degen_stencils" + suffix);
    degen_stencil_vtk.write_stencils(degenerate_stencils);
  }

  void write_prob_json(const std::string json_path, const ConfigParser& parser,
                       const std::vector<T>& x) {
    json j;
    j["cfg"] = parser.get_options();
    j["lsf_dof"] = mesh.get_lsf_dof();
    j["bc_dof"] = bc_dof;
    j["loaded_cells"] = loaded_cells;
    j["dvs"] = x;
    write_json(json_path, j);
  }

  std::vector<T>& get_phi() { return phi; }
  std::vector<T>& get_rhs() { return elastic.get_rhs(); }
  ProbMesh& get_prob_mesh() { return prob_mesh; }

 private:
  ProbMesh& prob_mesh;
  Grid& grid;
  Mesh& mesh;
  Quadrature quadrature;
  Basis basis;

  bool use_helmholtz_filter;
  int num_conv_filter_apply;
  bool use_robust_projection;
  RobustProjection<T> projector;
  HFilter hfilter;
  CFilter cfilter;
  Elastic elastic;
  Volume vol;
  VolAnalysis vol_analysis;
  Penalization pen;
  PenalizationAnalysis pen_analysis;
  Stress stress;
  StressAnalysis stress_analysis;
  StressKS stress_ks;
  StressKSAnalysis stress_ks_analysis;

  std::vector<T>& phi;  // LSF values (filtered design variables)

  std::string prefix;

  std::vector<int> bc_dof;

  std::set<int> loaded_cells;

  std::map<std::string,
           std::variant<T, std::vector<T>,
                        std::shared_ptr<SparseUtils::SparseCholesky<T>>>>
      cache;
  double compliance_scalar;
};

template <typename T, class TopoAnalysis>
class TopoProb : public ParOptProblem {
 public:
  TopoProb(TopoAnalysis& topo, std::string prefix, const ConfigParser& parser)
      : ParOptProblem(MPI_COMM_SELF),
        topo(topo),
        prefix(prefix),
        parser(parser),
        domain_area(topo.get_prob_mesh().get_domain_area()),
        area_frac(parser.get_double_option("area_frac")),
        stress_ratio_ub_init(
            parser.get_double_option("stress_ratio_upper_bound_init")),
        stress_ratio_ub_final(
            parser.get_double_option("stress_ratio_upper_bound_final")),
        stress_ratio_ub_decay_rate(
            parser.get_double_option("stress_ratio_upper_bound_decay_rate")),
        has_stress_constraint(parser.get_bool_option("has_stress_constraint")),
        has_stress_objective(parser.get_bool_option("has_stress_objective")),
        stress_objective_scalar(
            parser.get_double_option("stress_objective_scalar")),
        stress_objective_theta(
            parser.get_double_option("stress_objective_theta")),
        nvars(topo.get_prob_mesh().get_nvars()),
        ncon(has_stress_constraint ? 2 : 1),
        nineq(ncon),
        HFx_old(nvars, 0.0) {
    setProblemSizes(nvars, ncon, 0);
    setNumInequalities(nineq, 0);

    if (!std::filesystem::is_directory(prefix)) {
      std::filesystem::create_directory(prefix);
    }

    reset_counter();
  }

  void print_progress(T obj, T comp, T reg, T pterm, T vol_frac, T max_stress,
                      T max_stress_ratio, T ks_stress_ratio,
                      T ks_stress_ratio_ub, T HFx_change,
                      int header_every = 10) {
    std::ofstream progress_file(fspath(prefix) / fspath("optimization.log"),
                                std::ios::app);
    if (counter % header_every == 0) {
      char line[2048];
      std::snprintf(line, 2048,
                    "\n%4s%15s%15s%15s%15s%11s%15s%15s%15s%13s%14s%13s%15s\n",
                    "iter", "obj", "comp", "x regular", "grad pen", "vol(\%)",
                    "max(vm)", "max(vm/yield)", "ks(vm/yield)", "ks relerr(\%)",
                    "ks ub", "diffx_infty", "uptime(H:M:S)");
      std::cout << line;
      progress_file << line;
    }
    char line[2048];
    std::snprintf(
        line, 2048,
        "%4d%15.5e%15.5e%15.5e%15.5e%11.5f%15.5e%15.5e%15.5e%13.5f%14."
        "4e%13.5f%15s\n",
        counter, obj, comp, reg, pterm, 100.0 * vol_frac, max_stress,
        max_stress_ratio, ks_stress_ratio,
        (ks_stress_ratio - max_stress_ratio) / max_stress_ratio * 100.0,
        ks_stress_ratio_ub, HFx_change, watch.format_time(watch.lap()).c_str());
    std::cout << line;
    progress_file << line;
    progress_file.close();
  }

  void check_gradients(double dh) {
    is_gradient_check = true;
    checkGradients(dh);
    is_gradient_check = false;
    reset_counter();
    VandermondeCondLogger::clear();
  }

  void reset_counter() { counter = 0; }

  void getVarsAndBounds(ParOptVec* xvec, ParOptVec* lbvec, ParOptVec* ubvec) {
    T *xr, *lb, *ub;
    xvec->getArray(&xr);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set initial design
    std::vector<T> x0;
    std::string init_topo_json_path =
        parser.get_str_option("init_topology_from_json");
    if (!init_topo_json_path.empty()) {
      json j = read_json(init_topo_json_path);
      json j_this(parser.get_options());

      // Sanity checks to make sure the input json has the same basic settings
      // as the current case

      for (std::string key :
           std::vector<std::string>{"instance", "nx", "ny", "lx", "ly"}) {
        xcgd_assert(j["cfg"][key] == j_this[key], key + "mismatch");
      }

      x0 = std::vector<T>(j["dvs"]);
      xcgd_assert(x0.size() == nvars,
                  "design variable dimension mismatch, expect " +
                      std::to_string(nvars) + ", got " +
                      std::to_string(x0.size()));

    }

    else {
      std::string init_topology_method =
          parser.get_str_option("init_topology_method");

      if (init_topology_method == "circles") {
        x0 = topo.create_initial_topology(
            parser.get_int_option("init_topology_nholes_x"),
            parser.get_int_option("init_topology_nholes_y"),
            parser.get_double_option("init_topology_r"),
            parser.get_bool_option("init_topology_cell_center"));
      } else {  // init_topology_method == "sinusoidal"
        x0 = topo.create_initial_topology_sine(
            parser.get_int_option("init_topology_sine_period_x"),
            parser.get_int_option("init_topology_sine_period_y"),
            parser.get_double_option("init_topology_sine_offset"));
      }
    }

    std::vector<T> x0r = topo.get_prob_mesh().reduce(x0);

    // update mesh and bc dof, but don't perform the linear solve
    topo.update_mesh(topo.get_prob_mesh().expand(x0));

    for (int i = 0; i < nvars; i++) {
      xr[i] = x0r[i];
      lb[i] = 0.0;
      ub[i] = 1.0;
    }

    const auto& protected_verts = topo.get_prob_mesh().get_protected_verts();
    for (int i : protected_verts) {
      ub[i] =
          0.5;  // we prescribe x to be sufficient low for protected
                // verts such that material will be placed at those locations
    }
  }

  int evalObjCon(ParOptVec* xvec, T* fobj, T* cons) {
    T* xptr;
    xvec->getArray(&xptr);
    std::vector<T> xr(xptr, xptr + nvars);
    std::vector<T> x = topo.get_prob_mesh().expand(xr);

    // Update stress constraint
    stress_ratio_ub =
        stress_ratio_ub_init - stress_ratio_ub_decay_rate * counter;
    if (stress_ratio_ub < stress_ratio_ub_final) {
      stress_ratio_ub = stress_ratio_ub_final;
    }

    if (is_gradient_check) {
      stress_ratio_ub = 1.234;
    }

    // Regularization term
    double reg_coeff = parser.get_double_option("regularization_coeff") / nvars;
    T reg = 0.0;
    for (int i = 0; i < nvars; i++) {
      reg += xr[i] * (1.0 - xr[i]);
    }
    reg *= reg_coeff;

    // Save the elastic problem instance to json
    if (counter % parser.get_int_option("save_prob_json_every") == 0) {
      std::string json_path = fspath(prefix) / fspath("json") /
                              ((is_gradient_check ? "fdcheck_" : "opt_") +
                               std::to_string(counter) + ".json");
      topo.write_prob_json(json_path, parser, x);
    }

    auto [HFx, comp, area, pterm, max_stress, max_stress_ratio, ks_stress_ratio,
          u, xloc_q, stress_q] = topo.eval_obj_con(x);

    // Evaluate the design change ||phi_new - phi_old||_infty
    std::transform(
        HFx_old.begin(), HFx_old.end(), HFx.begin(), HFx_old.begin(),
        [](T a, T b) { return abs(b - a); });  // HFx_old = abs(HFx_old - HFx)
    T HFx_change = *std::max_element(HFx_old.begin(), HFx_old.end());
    HFx_old = HFx;

    if (has_stress_objective) {
      *fobj =
          (1.0 - stress_objective_theta) * comp +
          stress_objective_theta * stress_objective_scalar * ks_stress_ratio +
          pterm + reg;
    } else {
      *fobj = comp + pterm + reg;
    }
    cons[0] = 1.0 - area / (domain_area * area_frac);  // >= 0
    if (has_stress_constraint) {
      cons[1] = 1.0 - ks_stress_ratio / stress_ratio_ub;  // >= 0
    }

    if (counter % parser.get_int_option("write_vtk_every") == 0) {
      // Write design to vtk
      std::string vtk_path =
          fspath(prefix) / fspath("grid_" + std::to_string(counter) + ".vtk");
      std::vector<double> protected_verts_v(x.size(), 0.0);
      const auto& protected_verts = topo.get_prob_mesh().get_protected_verts();
      for (auto v : protected_verts) {
        protected_verts_v[v] = 1.0;
      }

      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_grid_vtk(vtk_path, x, topo.get_phi(),
                            {{"protected_verts", protected_verts_v}}, {},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(vtk_path, x, topo.get_phi(),
                            {{"protected_verts", protected_verts_v}}, {}, {},
                            {});
      }

      // Write cut mesh to vtk
      vtk_path =
          fspath(prefix) / fspath("cut_" + std::to_string(counter) + ".vtk");
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_cut_vtk(vtk_path, x, topo.get_phi(), {}, {}, {}, {});
      } else {
        topo.write_cut_vtk(vtk_path, x, topo.get_phi(), {}, {},
                           {{"displacement", u}}, {});
      }

      // Write quadrature-level data
      vtk_path =
          fspath(prefix) / fspath("quad_" + std::to_string(counter) + ".vtk");
      FieldToVTKNew<T, TopoAnalysis::get_spatial_dim()> field_vtk(vtk_path);
      field_vtk.add_mesh(xloc_q);
      field_vtk.write_mesh();
      field_vtk.add_sol("VonMises", stress_q);
      field_vtk.write_sol("VonMises");
    }

    // write quadrature to vtk for gradient check
    if (is_gradient_check) {
      std::string vtk_name = "fdcheck_quad_" + std::to_string(counter) + ".vtk";
      topo.write_quad_pts_to_vtk(fspath(prefix) / fspath(vtk_name));

      vtk_name = "fdcheck_grid_" + std::to_string(counter) + ".vtk";

      std::vector<T> vert_loaded_or_not(
          topo.get_prob_mesh().get_grid().get_num_verts(), 0.0);
      for (int i : topo.get_prob_mesh().get_loaded_verts()) {
        vert_loaded_or_not[i] = 1.0;
      }

      std::vector<T> cell_loaded_or_not(
          topo.get_prob_mesh().get_grid().get_num_cells(), 0.0);
      for (int i : topo.get_prob_mesh().get_loaded_cells()) {
        cell_loaded_or_not[i] = 1.0;
      }

      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(),
                            {{"loaded_verts", vert_loaded_or_not}},
                            {{"loaded_cells", cell_loaded_or_not}},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(),
                            {{"loaded_verts", vert_loaded_or_not}},
                            {{"loaded_cells", cell_loaded_or_not}}, {}, {});
      }

      vtk_name = "fdcheck_cut_" + std::to_string(counter) + ".vtk";
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, topo.get_phi(),
                           {}, {}, {}, {});
      } else {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, topo.get_phi(),
                           {}, {}, {{"displacement", u}}, {});
      }
    }

    // print optimization progress
    print_progress(*fobj, comp, reg, pterm, area / domain_area, max_stress,
                   max_stress_ratio, ks_stress_ratio, stress_ratio_ub,
                   HFx_change);

    counter++;

    return 0;
  }

  int evalObjConGradient(ParOptVec* xvec, ParOptVec* gvec, ParOptVec** Ac) {
    T* xptr;
    xvec->getArray(&xptr);
    std::vector<T> xr(xptr, xptr + nvars);
    std::vector<T> x = topo.get_prob_mesh().expand(xr);

    T *g, *c1, *c2;
    gvec->getArray(&g);
    gvec->zeroEntries();

    Ac[0]->getArray(&c1);
    Ac[0]->zeroEntries();

    if (has_stress_constraint) {
      Ac[1]->getArray(&c2);
      Ac[1]->zeroEntries();
    }

    std::vector<T> gcomp, garea, gpen, gstress;
    topo.eval_obj_con_gradient(x, gcomp, garea, gpen, gstress);

    std::vector<T> gcompr = topo.get_prob_mesh().reduce(gcomp);
    std::vector<T> garear = topo.get_prob_mesh().reduce(garea);
    std::vector<T> gpenr = topo.get_prob_mesh().reduce(gpen);
    std::vector<T> gstressr;

    if (has_stress_objective or has_stress_constraint) {
      gstressr = topo.get_prob_mesh().reduce(gstress);
    }

    std::vector<T> greg(nvars, 0.0);
    double reg_coeff = parser.get_double_option("regularization_coeff") / nvars;
    for (int i = 0; i < nvars; i++) {
      greg[i] = reg_coeff * (1.0 - 2.0 * xr[i]);
    }

    if (has_stress_objective) {
      T a = 1.0 - stress_objective_theta;
      T b = stress_objective_theta * stress_objective_scalar;
      for (int i = 0; i < nvars; i++) {
        g[i] = a * gcompr[i] + b * gstressr[i] + gpenr[i] + greg[i];
      }
    } else {
      for (int i = 0; i < nvars; i++) {
        g[i] = gcompr[i] + gpenr[i] + greg[i];
      }
    }

    for (int i = 0; i < nvars; i++) {
      c1[i] = -garear[i] / (domain_area * area_frac);
    }

    if (has_stress_constraint) {
      for (int i = 0; i < nvars; i++) {
        c2[i] = -gstressr[i] / stress_ratio_ub;
      }
    }

    return 0;
  }

  // Dummy method
  ParOptQuasiDefMat* createQuasiDefMat() {
    int nwblock = 0;
    return new ParOptQuasiDefBlockMat(this, nwblock);
  }

 private:
  TopoAnalysis& topo;
  std::string prefix;
  const ConfigParser& parser;
  double domain_area = 0.0;
  double area_frac = 0.0;
  double stress_ratio_ub_init = 0.0;
  double stress_ratio_ub_final = 0.0;
  double stress_ratio_ub_decay_rate = 0.0;
  double stress_ratio_ub = 0.0;
  bool has_stress_constraint = false;
  bool has_stress_objective = false;
  double stress_objective_scalar = 0.0;
  double stress_objective_theta = 0.0;
  int nvars = -1;
  int ncon = -1;
  int nineq = -1;

  std::vector<T> HFx_old;

  int counter = -1;
  StopWatch watch;
  bool is_gradient_check = false;
};

template <int Np_1d, bool use_ersatz, bool use_lbracket_grid>
void execute(int argc, char* argv[]) {
  constexpr int Np_1d_filter = Np_1d > 2 ? 4 : 2;
  MPI_Init(&argc, &argv);

  using T = double;
  using Grid = typename std::conditional<use_lbracket_grid, LbracketGrid2D<T>,
                                         StructuredGrid2D<T>>::type;
  using TopoAnalysis = TopoAnalysis<T, Np_1d, Np_1d_filter, use_ersatz, Grid>;

  bool smoke_test = false;
  if (argc > 2 and "--smoke" == std::string(argv[2])) {
    std::printf("This is a smoke test\n");
    smoke_test = true;
  }

  // Create config parser
  std::string cfg_path{argv[1]};
  ConfigParser parser{cfg_path};

  std::string prefix = parser.get_str_option("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
    if (smoke_test) {
      prefix = "smoke_" + prefix;
    } else {
      prefix = "opt_" + prefix;
    }
  }

  if (!std::filesystem::is_directory(prefix)) {
    std::filesystem::create_directory(prefix);
  }

  std::string json_dir = fspath(prefix) / fspath("json");
  if (!std::filesystem::is_directory(json_dir)) {
    std::filesystem::create_directory(json_dir);
  }

  std::filesystem::copy(
      cfg_path,
      fspath(prefix) / fspath(std::filesystem::absolute(cfg_path).filename()));

  std::string init_topo_json_path =
      parser.get_str_option("init_topology_from_json");
  if (!init_topo_json_path.empty()) {
    std::filesystem::copy(
        init_topo_json_path,
        fspath(prefix) /
            fspath(std::filesystem::absolute(init_topo_json_path).filename()));
  }

  // Set up grid
  std::array<int, 2> nxy = {parser.get_int_option("nx"),
                            parser.get_int_option("ny")};
  if (smoke_test) {
    nxy[0] = 9;
    nxy[1] = 5;
  }
  std::array<T, 2> lxy = {parser.get_double_option("lx"),
                          parser.get_double_option("ly")};

  // Set up analysis
  std::string instance = parser.get_str_option("instance");
  if (!(instance == "lbracket" or instance == "cantilever")) {
    throw std::runtime_error(
        "expect lbracket or cantilever for option instance, got " + instance);
  }

  std::shared_ptr<ProbMeshBase<T, Np_1d, Grid>> prob_mesh;
  double loaded_frac = parser.get_double_option("loaded_frac");
  if constexpr (use_lbracket_grid) {
    double lbracket_frac = parser.get_double_option("lbracket_frac");
    prob_mesh = std::make_shared<LbracketGridMesh<T, Np_1d>>(
        nxy, lxy, loaded_frac, lbracket_frac);
  } else {
    if (instance == "cantilever") {
      prob_mesh =
          std::make_shared<CantileverMesh<T, Np_1d>>(nxy, lxy, loaded_frac);
    } else if (instance == "lbracket") {
      double lbracket_frac = parser.get_double_option("lbracket_frac");
      prob_mesh = std::make_shared<LbracketMesh<T, Np_1d>>(
          nxy, lxy, loaded_frac, lbracket_frac);
    } else {
      throw std::runtime_error("invalid instance " + instance);
    }
  }

  bool use_helmholtz_filter = parser.get_bool_option("use_helmholtz_filter");
  int num_conv_filter_apply = parser.get_int_option("num_conv_filter_apply");
  T r0 = parser.get_double_option("filter_r0");
  T E = parser.get_double_option("E");
  T nu = parser.get_double_option("nu");
  bool use_robust_projection = parser.get_bool_option("use_robust_projection");
  double robust_proj_beta = parser.get_double_option("robust_proj_beta");
  double robust_proj_eta = parser.get_double_option("robust_proj_eta");
  T penalty = parser.get_double_option("grad_penalty_coeff");
  T stress_ksrho = parser.get_double_option("stress_ksrho");
  T yield_stress = parser.get_double_option("yield_stress");
  double compliance_scalar = parser.get_double_option("compliance_scalar");

  TopoAnalysis topo{*prob_mesh,
                    use_helmholtz_filter,
                    num_conv_filter_apply,
                    r0,
                    E,
                    nu,
                    penalty,
                    stress_ksrho,
                    yield_stress,
                    use_robust_projection,
                    robust_proj_beta,
                    robust_proj_eta,
                    prefix,
                    compliance_scalar};

  TopoProb<T, TopoAnalysis>* prob =
      new TopoProb<T, TopoAnalysis>(topo, prefix, parser);
  prob->incref();

  double dh = parser.get_double_option("grad_check_fd_h");
  prob->check_gradients(dh);

  if (parser.get_bool_option("check_grad_and_exit")) {
    return;
  }

  // Set options
  ParOptOptions* options = new ParOptOptions;
  options->incref();
  ParOptOptimizer::addDefaultOptions(options);

  int max_it = smoke_test ? 10 : parser.get_int_option("max_it");

  options->setOption("algorithm", "mma");
  options->setOption("mma_max_iterations", max_it);
  options->setOption("mma_init_asymptote_offset",
                     parser.get_double_option("mma_init_asymptote_offset"));
  options->setOption("mma_move_limit",
                     parser.get_double_option("mma_move_limit"));
  options->setOption("max_major_iters",
                     parser.get_int_option("max_major_iters"));
  options->setOption("penalty_gamma",
                     parser.get_double_option("penalty_gamma"));
  options->setOption("qn_subspace_size",
                     parser.get_int_option("qn_subspace_size"));
  options->setOption("qn_type", parser.get_str_option("qn_type").c_str());
  options->setOption("abs_res_tol", parser.get_double_option("abs_res_tol"));
  options->setOption("starting_point_strategy",
                     parser.get_str_option("starting_point_strategy").c_str());
  options->setOption("barrier_strategy",
                     parser.get_str_option("barrier_strategy").c_str());
  options->setOption("use_line_search",
                     parser.get_bool_option("use_line_search"));
  options->setOption("output_file",
                     (fspath(prefix) / fspath("paropt.out")).c_str());
  options->setOption("tr_output_file",
                     (fspath(prefix) / fspath("paropt.tr")).c_str());
  options->setOption("mma_output_file",
                     (fspath(prefix) / fspath("paropt.mma")).c_str());

  ParOptOptimizer* opt = new ParOptOptimizer(prob, options);
  opt->incref();

  opt->optimize();

  prob->decref();
  options->decref();
  opt->decref();

  MPI_Finalize();
}

int main(int argc, char* argv[]) {
  VandermondeCondLogger::enable();

  if (argc == 1) {
    std::printf("Usage: ./topo level_set.cfg [--smoke]\n");
    exit(0);
  }

  std::string cfg_path{argv[1]};
  ConfigParser parser{cfg_path};
  int Np_1d = parser.get_int_option("Np_1d");
  bool use_ersatz = parser.get_bool_option("use_ersatz");
  bool use_lbracket_grid = parser.get_bool_option("use_lbracket_grid");

  if (Np_1d % 2) {
    std::printf("[Error]Invalid input, expect even Np_1d, got %d\n", Np_1d);
    exit(-1);
  }

  switch (Np_1d) {
    case 2:
      if (use_ersatz) {
        if (use_lbracket_grid) {
          execute<2, true, true>(argc, argv);
        } else {
          execute<2, true, false>(argc, argv);
        }
      } else {
        if (use_lbracket_grid) {
          execute<2, false, true>(argc, argv);
        }
      }
      break;

    case 4:
      if (use_ersatz) {
        if (use_lbracket_grid) {
          execute<4, true, true>(argc, argv);
        } else {
          execute<4, true, false>(argc, argv);
        }

      } else {
        if (use_lbracket_grid) {
          execute<4, false, true>(argc, argv);
        } else {
          execute<4, false, false>(argc, argv);
        }
      }
      break;

    default:
      std::printf(
          "Np_1d = %d is not pre-compiled, enumerate it in the source code if "
          "you intend to use this combination.\n",
          Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
