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
#include "apps/helmholtz_filter.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "lbracket_mesh.h"
#include "physics/grad_penalization.h"
#include "physics/stress.h"
#include "physics/volume.h"
#include "utils/argparser.h"
#include "utils/exceptions.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
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

  std::vector<int> get_bc_nodes() {
    return this->mesh.get_left_boundary_nodes();
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

template <typename T, int Np_1d>
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
    for (int iy = 0; iy < ny1; iy++) {
      T xloc[Grid::spatial_dim];
      int c = this->grid.get_coords_cell(nx1 - 1, iy);
      this->grid.get_cell_xloc(c, xloc);
      if (xloc[1] >= ly1 - lxy[1] * loaded_frac and xloc[1] <= ly1) {
        loaded_cells.insert(c);
      }
    }

    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      loaded_verts.insert(verts[1]);
      loaded_verts.insert(verts[2]);
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
  std::set<int> loaded_cells, loaded_verts;
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
  using Filter = HelmholtzFilter<T, Np_1d_filter, Grid>;

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
      GalerkinAnalysis<T, typename Filter::Mesh, typename Filter::Quadrature,
                       typename Filter::Basis, Penalization>;
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
  TopoAnalysis(ProbMesh& prob_mesh, T r0, T E, T nu, T penalty, T ksrho,
               bool use_robust_projection, double proj_beta, double proj_eta,
               std::string prefix)
      : prob_mesh(prob_mesh),
        grid(prob_mesh.get_grid()),
        mesh(prob_mesh.get_mesh()),
        quadrature(mesh),
        basis(mesh),
        filter(r0, grid, use_robust_projection, proj_beta, proj_eta),
        elastic(E, nu, mesh, quadrature, basis, int_func),
        vol_analysis(mesh, quadrature, basis, vol),
        pen(penalty),
        pen_analysis(filter.get_mesh(), filter.get_quadrature(),
                     filter.get_basis(), pen),
        stress(E, nu),
        stress_analysis(mesh, quadrature, basis, stress),
        stress_ks(ksrho, E, nu),
        stress_ks_analysis(mesh, quadrature, basis, stress_ks),
        phi(mesh.get_lsf_dof()),
        prefix(prefix),
        cache({{"x", {}}, {"sol", {}}, {"chol", nullptr}}) {
    // Get loaded cells
    loaded_cells = prob_mesh.get_loaded_cells();

    // TODO: delete
    coefs.resize(grid.get_num_verts() * Basis::spatial_dim);
    for (int i = 0; i < coefs.size(); i++) {
      coefs[i] = (T)rand() / RAND_MAX;
    }
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

  void update_mesh(const std::vector<T>& x) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    // Update mesh based on new LSF
    filter.apply(x.data(), phi.data());
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
  }

  std::vector<T> update_mesh_and_solve(
      const std::vector<T>& x,
      const std::vector<T>& fdv /*TODO: delete this fake design variable*/,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol = nullptr) {
    // Solve the static problem
    update_mesh(x);

    DegenerateStencilLogger::clear();
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

      std::vector<T> manual_rhs = fdv;
      if (manual_rhs.size()) {
        for (int i = 0; i < manual_rhs.size(); i++) {
          manual_rhs[i] *= coefs[i];
        }
      }
      std::vector<T> sol = elastic.solve(
          bc_dof, std::vector<T>(bc_dof.size(), T(0.0)),
          std::tuple<LoadAnalysis>(load_analysis), chol, manual_rhs);

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

  // TODO: delete
  T eval_stress_for_partial(const std::vector<T> x, const std::vector<T> sol,
                            T area, T max_stress_val) {
    update_mesh(x);
    return eval_stress_ks(max_stress_val, area, sol);
  }

  std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> eval_obj_con(
      const std::vector<T>& x, T& comp, T& area, T& pen, T& max_stress_val,
      T& ks_stress_val,
      std::vector<T> fdv = {} /*TODO: delete this fake design variable*/) {
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol;
    std::vector<T> sol = update_mesh_and_solve(x, fdv, &chol);
    cache["x"] = x;
    cache["sol"] = sol;
    cache["chol"] = chol;

    comp = std::inner_product(sol.begin(), sol.end(), elastic.get_rhs().begin(),
                              T(0.0));

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    area = vol_analysis.energy(nullptr, dummy.data());
    pen = pen_analysis.energy(nullptr, phi.data());

    auto [xloc_q, stress_q] = eval_stress(sol);
    max_stress_val = *std::max_element(stress_q.begin(), stress_q.end());

    ks_stress_val = eval_stress_ks(max_stress_val, area, sol);

    // ks_stress_val =
    //     0.5 * std::inner_product(sol.begin(), sol.end(), sol.begin(),
    //                              T(0.0));  // TODO: delete
    // ks_stress_val = comp;  // TODO: delete

    // ks_stress_val =
    //     std::inner_product(sol.begin(), sol.end(), coefs.begin(), 0.0);

    x_saved = x;  // TODO: delete
    return {sol, xloc_q, stress_q};
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

  T eval_stress_ks(T max_stress_val, T area, const std::vector<T>& u) {
    stress_ks.set_von_mises_max_stress(max_stress_val);

    return stress_ks_analysis.energy(nullptr, u.data());  // TODO: delete this
    return log(stress_ks_analysis.energy(nullptr, u.data()) / area) /
               stress_ks.get_ksrho() +
           max_stress_val;
  }

  void eval_obj_con_gradient(const std::vector<T>& x, std::vector<T>& gcomp,
                             std::vector<T>& garea, std::vector<T>& gpen,
                             std::vector<T>& gstress,
                             /*TODO: delete the following arguments*/
                             std::vector<T>& gstress_partial,
                             std::vector<T>& stress_adjoint_rhs,
                             std::vector<T>& stress_adjoint,
                             std::vector<T> fdv = {},
                             std::vector<T>* gstress_fdv = nullptr,
                             bool force_recompute = false /*TODO: delete*/) {
    std::vector<T> sol;
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol;
    if (x == std::get<std::vector<T>>(cache["x"]) and !force_recompute) {
      sol = std::get<std::vector<T>>(cache["sol"]);
      chol = std::get<std::shared_ptr<SparseUtils::SparseCholesky<T>>>(
          cache["chol"]);
    } else {
      sol = update_mesh_and_solve(x, fdv, &chol);
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

    // TODO: delete
    stress_adjoint_rhs = psi_stress;

    // Compute stress adjoints
    chol->solve(psi_stress.data());

    // Apply boundary conditions again to the adjoint variables
    for (int i : bc_dof) {
      psi_stress[i] = 0.0;
    }

    std::vector<T> psi_stress_neg = psi_stress;
    for (T& p : psi_stress) p *= -1.0;

    // TODO: delete
    stress_adjoint = psi_stress;

    gcomp.resize(x.size());
    std::fill(gcomp.begin(), gcomp.end(), 0.0);
    elastic.get_analysis().LSF_jacobian_adjoint_product(
        sol.data(), psi_comp.data(), gcomp.data());
    if constexpr (use_ersatz) {
      elastic.get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), sol.data() /*this is effectively -psi*/, gcomp.data());
    }
    filter.applyGradient(x.data(), gcomp.data(), gcomp.data());

    garea.resize(x.size());
    std::fill(garea.begin(), garea.end(), 0.0);
    vol_analysis.LSF_volume_derivatives(garea.data());
    filter.applyGradient(x.data(), garea.data(), garea.data());

    gpen.resize(x.size());
    std::fill(gpen.begin(), gpen.end(), 0.0);
    pen_analysis.residual(nullptr, phi.data(), gpen.data());
    filter.applyGradient(x.data(), gpen.data(), gpen.data());

    gstress.resize(x.size());
    std::fill(gstress.begin(), gstress.end(), 0.0);

    // Explicit partials
    if constexpr (use_ersatz) {
      std::vector<T> sol0 = grid_dof_to_cut_dof<spatial_dim>(sol);
      stress_ks_analysis.LSF_energy_derivatives(sol0.data(), gstress.data());
    } else {
      stress_ks_analysis.LSF_energy_derivatives(sol.data(), gstress.data());
    }

    // Implicit derivatives via the adjoint variables
    elastic.get_analysis().LSF_jacobian_adjoint_product(
        sol.data(), psi_stress.data(), gstress.data());
    if constexpr (use_ersatz) {
      elastic.get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), psi_stress_neg.data(), gstress.data());
    }

    filter.applyGradient(x.data(), gstress.data(), gstress.data());

    // TODO: delete
    gstress_partial.resize(x.size());
    if constexpr (use_ersatz) {
      std::vector<T> sol0 = grid_dof_to_cut_dof<spatial_dim>(sol);
      stress_ks_analysis.LSF_energy_derivatives(sol0.data(),
                                                gstress_partial.data());
    } else {
      stress_ks_analysis.LSF_energy_derivatives(sol.data(),
                                                gstress_partial.data());
    }
    filter.applyGradient(x.data(), gstress_partial.data(),
                         gstress_partial.data());  // TODO: delete

    // TODO: delete
    if (gstress_fdv) {
      gstress_fdv->resize(fdv.size());
      if (fdv.size() != psi_stress.size()) {
        throw std::runtime_error("incompatible dimensions");
      }
      for (int i = 0; i < psi_stress.size(); i++) {
        (*gstress_fdv)[i] = coefs[i] * psi_stress_neg[i];
      }
    }

    x_saved = x;  // TODO: delete
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

    ToVTK<T, typename Filter::Mesh> vtk(filter.get_mesh(), vtk_path);
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

    std::vector<double> nstencils(mesh.get_num_elements(),
                                  Mesh::Np_1d * Mesh::Np_1d);
    auto degenerate_stencils = DegenerateStencilLogger::get_stencils();
    for (auto e : degenerate_stencils) {
      int elem = e.first;
      nstencils[elem] = e.second.size();
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
  }

  void write_prob_json(const std::string json_path,
                       const ConfigParser& parser) {
    json j;
    j["Np_1d"] = parser.get_int_option("Np_1d");
    j["E"] = parser.get_double_option("E");
    j["nu"] = parser.get_double_option("nu");
    j["nx"] = parser.get_int_option("nx");
    j["ny"] = parser.get_int_option("ny");
    j["lx"] = parser.get_double_option("lx");
    j["ly"] = parser.get_double_option("ly");
    j["lsf_dof"] = mesh.get_lsf_dof();
    j["bc_dof"] = bc_dof;
    j["loaded_cells"] = loaded_cells;
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

  Filter filter;
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
           std::variant<std::vector<T>,
                        std::shared_ptr<SparseUtils::SparseCholesky<T>>>>
      cache;

  std::vector<T> coefs;  // TODO: delete

  // TODO: delete the following debug code
 public:
  std::vector<T> x_saved;
};

template <typename T, class TopoAnalysis>
class TopoProb : public ParOptProblem {
 public:
  TopoProb(TopoAnalysis& topo, double area_frac, double stress_target,
           std::string prefix, const ConfigParser& parser)
      : ParOptProblem(MPI_COMM_SELF),
        nvars(topo.get_prob_mesh().get_nvars()),
        ncon(2),
        nineq(2),
        topo(topo),
        prefix(prefix),
        parser(parser),
        domain_area(topo.get_prob_mesh().get_domain_area()),
        area_frac(area_frac),
        stress_target(stress_target) {
    setProblemSizes(nvars, ncon, 0);
    setNumInequalities(nineq, 0);

    if (!std::filesystem::is_directory(prefix)) {
      std::filesystem::create_directory(prefix);
    }

    reset_counter();
  }

  void print_progress(T comp, T pterm, T vol_frac, T max_stress_val,
                      T ks_stress_val, int header_every = 10) {
    std::ofstream progress_file(fspath(prefix) / fspath("optimization.log"),
                                std::ios::app);
    if (counter % header_every == 1) {
      char phead[30];
      std::snprintf(phead, 30, "gradx penalty(c:%9.2e)",
                    parser.get_double_option("grad_penalty_coeff"));
      char line[2048];
      std::snprintf(line, 2048, "\n%5s%20s%30s%20s%20s%20s\n", "iter", "comp",
                    phead, "vol (\%)", "stress_max", "stress_ks");
      std::cout << line;
      progress_file << line;
    }
    char line[2048];
    std::snprintf(line, 2048, "%5d%20.10e%30.10e%20.5f%20.10e%20.10e\n",
                  counter, comp, pterm, 100.0 * vol_frac, max_stress_val,
                  ks_stress_val);
    std::cout << line;
    progress_file << line;
    progress_file.close();
  }

  void check_gradients(double dh) {
    is_gradient_check = true;
    checkGradients(dh);
    is_gradient_check = false;
    reset_counter();
    DegenerateStencilLogger::clear();
    VandermondeCondLogger::clear();
  }

  void reset_counter() { counter = 0; }

  void getVarsAndBounds(ParOptVec* xvec, ParOptVec* lbvec, ParOptVec* ubvec) {
    T *xr, *lb, *ub;
    xvec->getArray(&xr);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    std::vector<T> x0 = topo.create_initial_topology(
        parser.get_int_option("init_topology_nholes_x"),
        parser.get_int_option("init_topology_nholes_y"),
        parser.get_double_option("init_topology_r"),
        parser.get_bool_option("init_topology_cell_center"));
    std::vector<T> x0r = topo.get_prob_mesh().reduce(x0);

    // update mesh and bc dof, but don't perform the linear solve
    topo.update_mesh(topo.get_prob_mesh().expand(x0));

    double ubval = parser.get_double_option("opt_x_ub");
    double lbval = parser.get_double_option("opt_x_lb");
    for (int i = 0; i < nvars; i++) {
      xr[i] = x0r[i];
      ub[i] = ubval;
      lb[i] = lbval;
    }

    const auto& loaded_verts = topo.get_prob_mesh().get_loaded_verts();
    for (int i : loaded_verts) {
      ub[i] = 1e-3;  // we prescribe x < 0 for loaded verts
    }
  }

  int evalObjCon(ParOptVec* xvec, T* fobj, T* cons) {
    // Save the elastic problem instance to json
    if (counter % parser.get_int_option("save_prob_json_every") == 0) {
      std::string json_path = fspath(prefix) / fspath("json") /
                              ((is_gradient_check ? "fdcheck_" : "opt_") +
                               std::to_string(counter) + ".json");
      topo.write_prob_json(json_path, parser);
    }

    T* xptr;
    xvec->getArray(&xptr);
    std::vector<T> xr(xptr, xptr + nvars);
    std::vector<T> x = topo.get_prob_mesh().expand(xr);

    T comp, area, pterm, max_stress_val, ks_stress_val;
    auto [u, xloc_q, stress_q] =
        topo.eval_obj_con(x, comp, area, pterm, max_stress_val, ks_stress_val);
    *fobj = comp + pterm;
    cons[0] = 1.0 - area / (domain_area * area_frac);  // >= 0
    cons[1] = 1.0 - ks_stress_val / stress_target;     // >= 0

    if (counter % parser.get_int_option("write_vtk_every") == 0) {
      // Write design to vtk
      std::string vtk_path =
          fspath(prefix) / fspath("grid_" + std::to_string(counter) + ".vtk");
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_grid_vtk(vtk_path, x, topo.get_phi(), {}, {},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(vtk_path, x, topo.get_phi(), {}, {}, {}, {});
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
      std::vector<T> dvr(nvars, 0.0);
      std::vector<T> dv = topo.get_prob_mesh().expand(dvr);
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(), {{"dv", dv}}, {},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(), {{"dv", dv}}, {}, {}, {});
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
    print_progress(comp, pterm, area / domain_area, max_stress_val,
                   ks_stress_val);

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

    Ac[1]->getArray(&c2);
    Ac[1]->zeroEntries();

    std::vector<T> gcomp, garea, gpen, gstress;
    std::vector<T> stress_adjoint_rhs, stress_adjoint,
        gstress_partial;  // TODO: delete
    topo.eval_obj_con_gradient(x, gcomp, garea, gpen, gstress, gstress_partial,
                               stress_adjoint_rhs, stress_adjoint);

    std::vector<T> gcompr = topo.get_prob_mesh().reduce(gcomp);
    std::vector<T> garear = topo.get_prob_mesh().reduce(garea);
    std::vector<T> gpenr = topo.get_prob_mesh().reduce(gpen);
    std::vector<T> gstressr = topo.get_prob_mesh().reduce(gstress);

    for (int i = 0; i < nvars; i++) {
      g[i] = gcompr[i] + gpenr[i];
      c1[i] = -garear[i] / (domain_area * area_frac);
      c2[i] = -gstressr[i] / stress_target;
    }

    // TODO: delete
    std::string vtk_path =
        fspath(prefix) /
        fspath("adjoint_debug_" + std::to_string(counter) + ".vtk");
    if constexpr (TopoAnalysis::use_ersatz) {
      topo.write_grid_vtk(vtk_path, x, topo.get_phi(), {}, {},
                          {{"stress_adjoint_rhs", stress_adjoint_rhs},
                           {"stress_adjoint", stress_adjoint}},
                          {});
    } else {
      topo.write_cut_vtk(vtk_path, x, topo.get_phi(), {}, {},
                         {{"stress_adjoint_rhs", stress_adjoint_rhs},
                          {"stress_adjoint", stress_adjoint}},
                         {});
    }

    return 0;
  }

  // Dummy method
  ParOptQuasiDefMat* createQuasiDefMat() {
    int nwblock = 0;
    return new ParOptQuasiDefBlockMat(this, nwblock);
  }

 private:
  int nvars = -1;
  int ncon = -1;
  int nineq = -1;

  TopoAnalysis& topo;

  std::string prefix;
  const ConfigParser& parser;

  int counter = -1;

  double domain_area = 0.0;
  double area_frac = 0.0;
  double stress_target = 0.0;

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

  T r0 = parser.get_double_option("helmholtz_r0");
  T E = parser.get_double_option("E");
  T nu = parser.get_double_option("nu");
  bool use_robust_projection = parser.get_bool_option("use_robust_projection");
  double robust_proj_beta = parser.get_double_option("robust_proj_beta");
  double robust_proj_eta = parser.get_double_option("robust_proj_eta");
  T penalty = parser.get_double_option("grad_penalty_coeff");
  T ksrho = parser.get_double_option("stress_ksrho");

  TopoAnalysis topo{*prob_mesh,
                    r0,
                    E,
                    nu,
                    penalty,
                    ksrho,
                    use_robust_projection,
                    robust_proj_beta,
                    robust_proj_eta,
                    prefix};

  double area_frac = parser.get_double_option("area_frac");
  double stress_target = parser.get_double_option("stress_target");
  TopoProb<T, TopoAnalysis>* prob = new TopoProb<T, TopoAnalysis>(
      topo, area_frac, stress_target, prefix, parser);
  prob->incref();

  double dh = parser.get_double_option("grad_check_fd_h");
  prob->check_gradients(dh);

  // TODO: delete the following code block
  {
    std::vector<T> fdv(topo.get_rhs().size(), T(0.0));
    std::vector<T> p(fdv.size(), T(0.0));

    // fdv[0] = 1.0;
    // p[0] = 1.0;

    for (int i = 0; i < fdv.size(); i++) {
      fdv[i] = (T)rand() / RAND_MAX;
      p[i] = (T)rand() / RAND_MAX;
    }

    T comp, area, pterm, max_stress_val, ks_stress_val_1, ks_stress_val_2;
    auto [sol, xloc_q, stress_q] = topo.eval_obj_con(
        topo.x_saved, comp, area, pterm, max_stress_val, ks_stress_val_1, fdv);

    std::vector<T> gcomp, garea, gpen, gstress, gstress_fdv;
    std::vector<T> stress_adjoint_rhs, stress_adjoint,
        gstress_partial;  // TODO: delete
    std::vector<T> x1 = topo.x_saved;
    topo.eval_obj_con_gradient(x1, gcomp, garea, gpen, gstress, gstress_partial,
                               stress_adjoint_rhs, stress_adjoint, fdv,
                               &gstress_fdv, true);

    for (int i = 0; i < fdv.size(); i++) {
      fdv[i] += p[i] * dh;
    }
    topo.eval_obj_con(x1, comp, area, pterm, max_stress_val, ks_stress_val_2,
                      fdv);

    T stress_fd = (ks_stress_val_2 - ks_stress_val_1) / dh;
    T stress_exact = 0.0;

    // assemble derivative w.r.t. fake dv
    if (sol.size() != gstress_fdv.size())
      throw std::runtime_error("sol and gstress_fdv size mismatch");

    // for (int i = 0; i < gstress_fdv.size(); i++) {
    //   gstress_fdv[i] += sol[i];
    // }

    for (int i = 0; i < fdv.size(); i++) {
      stress_exact += p[i] * gstress_fdv[i];
    }

    std::printf("Gradient verification on a fake dv:\n");
    std::printf("fake FD: %30.20e, Actual: %30.20e, Rel err: %20.10e\n",
                stress_fd, stress_exact,
                abs(stress_fd - stress_exact) / stress_exact);

    std::vector<T> xp(x1.size());
    for (int i = 0; i < xp.size(); i++) {
      xp[i] = (T)rand() / RAND_MAX;
    }

    std::vector<T> x2(x1.size());
    for (int i = 0; i < x2.size(); i++) {
      x2[i] = x1[i] + dh * xp[i];
    }

    T stress_partial_1 =
        topo.eval_stress_for_partial(x1, sol, area, max_stress_val);
    T stress_partial_2 =
        topo.eval_stress_for_partial(x2, sol, area, max_stress_val);

    T stress_partial_fd = (stress_partial_2 - stress_partial_1) / dh;
    T stress_partial_exact = 0.0;
    for (int i = 0; i < gstress_partial.size(); i++) {
      stress_partial_exact += gstress_partial[i] * xp[i];
    }

    // TODO: delete
    std::printf(
        "stress partial FD: %30.20e, Actual: %30.20e, Rel err: %20.10e\n",
        stress_partial_fd, stress_partial_exact,
        fabs(stress_partial_fd - stress_partial_exact) /
            fabs(stress_partial_exact));
  }

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
  DegenerateStencilLogger::enable();
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
