#include <mpi.h>

#include <filesystem>
#include <set>
#include <string>

#include "ParOptOptimizer.h"
#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/grad_penalization.h"
#include "physics/volume.h"
#include "utils/argparser.h"
#include "utils/exceptions.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

using fspath = std::filesystem::path;

template <typename T, int Np_1d>
class ProbMeshBase {
 public:
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;

  ProbMeshBase(int nxy[2], T lxy[2]) : grid(nxy, lxy), mesh(grid) {}

  virtual std::set<int> get_loaded_cells() = 0;
  virtual std::vector<int> get_bc_nodes() = 0;
  virtual std::vector<T> expand(std::vector<T> x) = 0;  // expand xr -> x
  virtual std::vector<T> reduce(std::vector<T> x) = 0;  // reduce x -> xr
  virtual int get_nvars() = 0;

  Grid grid;
  Mesh mesh;
};

template <typename T, int Np_1d>
struct CantileverMesh : public ProbMeshBase<T, Np_1d> {
 private:
  using Base = ProbMeshBase<T, Np_1d>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  CantileverMesh(std::array<int, Grid::spatial_dim> nxy,
                 std::array<T, Grid::spatial_dim> lxy, double loaded_frac)
      : Base(nxy.data(), lxy.data()),
        nxy(nxy),
        lxy(lxy),
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

  std::set<int> get_loaded_cells() { return loaded_cells; }

  std::vector<int> get_bc_nodes() {
    return this->mesh.get_left_boundary_nodes();
  }

  std::vector<T> expand(std::vector<T> xr) {
    std::vector<T> x(N, -1.0);
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

  int get_nvars() { return Nr; }

 private:
  std::array<int, Grid::spatial_dim> nxy;
  std::array<T, Grid::spatial_dim> lxy;
  double loaded_frac;
  std::set<int> loaded_cells, loaded_verts;
  int N, Nr;
  std::vector<int> reduce_mapping, expand_mapping;
};

template <typename T, int Np_1d, int Np_1d_filter, bool use_ersatz_>
class TopoAnalysis {
 public:
  static constexpr bool use_ersatz = use_ersatz_;

 private:
  using ProbMesh = ProbMeshBase<T, Np_1d>;
  using Grid = typename ProbMesh::Grid;
  using Mesh = typename ProbMesh::Mesh;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Filter = HelmholtzFilter<T, Np_1d_filter>;

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
      StaticElasticErsatz<T, Mesh, Quadrature, Basis, typeof(int_func)>,
      StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_func)>>::type;
  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using Penalization = GradPenalization<T, Basis::spatial_dim>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;
  using PenalizationAnalysis =
      GalerkinAnalysis<T, typename Filter::Mesh, typename Filter::Quadrature,
                       typename Filter::Basis, Penalization>;
  using LoadPhysics =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>;
  using LoadQuadrature =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;
  using LoadAnalysis =
      GalerkinAnalysis<T, Mesh, LoadQuadrature, Basis, LoadPhysics, use_ersatz>;

  int constexpr static spatial_dim = Basis::spatial_dim;

 public:
  TopoAnalysis(ProbMesh& prob_mesh, T r0, T E, T nu, T penalty,
               bool use_robust_projection, double proj_beta, double proj_eta,
               std::string prefix)
      : prob_mesh(prob_mesh),
        grid(prob_mesh.grid),
        mesh(prob_mesh.mesh),
        quadrature(mesh),
        basis(mesh),
        filter(r0, grid, use_robust_projection, proj_beta, proj_eta),
        elastic(E, nu, mesh, quadrature, basis, int_func),
        vol_analysis(mesh, quadrature, basis, vol),
        pen(penalty),
        pen_analysis(filter.get_mesh(), filter.get_quadrature(),
                     filter.get_basis(), pen),
        phi(mesh.get_lsf_dof()),
        prefix(prefix) {
    // Get loaded cells
    loaded_cells = prob_mesh.get_loaded_cells();
  }

  // Create nodal design variables for a domain with periodic holes
  std::vector<T> create_initial_topology(int nholes_x, int nholes_y, double r,
                                         bool cell_center = true) {
    const T* lxy = mesh.get_grid().get_lxy();
    int nverts = mesh.get_grid().get_num_verts();
    std::vector<T> lsf(nverts, 0.0);
    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      mesh.get_grid().get_vert_xloc(i, xloc);
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

    // Normalize lsf so values are within [-1, 1]
    T lsf_max = hard_max(lsf);
    T lsf_min = hard_min(lsf);
    for (int i = 0; i < nverts; i++) {
      if (lsf[i] < 0.0) {
        lsf[i] /= -lsf_min;
      } else {
        lsf[i] /= lsf_max;
      }
    }

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

  std::vector<T> update_mesh_and_solve(const std::vector<T>& x) {
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
      std::vector<T> sol =
          elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), T(0.0)),
                        std::tuple<LoadAnalysis>(load_analysis));

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

  std::vector<T> eval_compliance_area(const std::vector<T>& x, T& comp,
                                      T& area) {
    std::vector<T> sol = update_mesh_and_solve(x);

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    // WARNING: this holds only when body force is zero!
    comp = 2.0 * elastic.get_analysis().energy(nullptr, sol.data());
    // if constexpr (use_ersatz) {
    //   comp += 2.0 * elastic.get_analysis_ersatz().energy(nullptr,
    //   sol.data());
    // }
    area = vol_analysis.energy(nullptr, dummy.data());

    return sol;
  }

  T eval_grad_penalization(const std::vector<T>& x) {
    return pen_analysis.energy(nullptr, x.data());
  }

  void eval_grad_penalization_grad(const std::vector<T>& x, std::vector<T>& g) {
    g.resize(x.size());
    pen_analysis.residual(nullptr, x.data(), g.data());
  }

  void eval_compliance_area_grad(const std::vector<T>& x, std::vector<T>& gcomp,
                                 std::vector<T>& garea) {
    std::vector<T> sol = update_mesh_and_solve(x);

    // compliance is self-adjoint
    std::vector<T> psi = sol;
    for (T& p : psi) p *= -1.0;

    gcomp.resize(x.size());
    std::fill(gcomp.begin(), gcomp.end(), 0.0);
    elastic.get_analysis().LSF_jacobian_adjoint_product(sol.data(), psi.data(),
                                                        gcomp.data());
    // if constexpr (use_ersatz) {
    //   elastic.get_analysis_ersatz().LSF_jacobian_adjoint_product(
    //       sol.data(), psi.data(), gcomp.data());
    // }
    filter.applyGradient(x.data(), gcomp.data(), gcomp.data());

    garea.resize(x.size());
    std::fill(garea.begin(), garea.end(), 0.0);
    vol_analysis.LSF_volume_derivatives(garea.data());
    filter.applyGradient(x.data(), garea.data(), garea.data());
  }

  std::vector<T> grid_sol_to_cut_sol(const std::vector<T>& grid_sol) {
    int nnodes = mesh.get_num_nodes();
    std::vector<T> cut_sol(spatial_dim * nnodes);
    for (int n = 0; n < nnodes; n++) {
      int v = mesh.get_node_vert(n);
      for (int d = 0; d < spatial_dim; d++) {
        cut_sol[spatial_dim * n + d] = grid_sol[spatial_dim * v + d];
      }
    }
    return cut_sol;
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

    std::vector<T> loaded_cells_v(mesh.get_grid().get_num_cells(), 0.0);
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
    std::vector<T> dof_bcs(spatial_dim * mesh.get_num_nodes(), 0.0);
    for (int i : bc_dof) dof_bcs[i] = 1.0;
    vtk.write_vec("bc-dof", dof_bcs.data());

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

  std::vector<T>& phi;  // LSF values (filtered design variables)

  std::string prefix;

  std::vector<int> bc_dof;

  std::set<int> loaded_cells;
};

template <typename T, class TopoAnalysis>
class TopoProb : public ParOptProblem {
 public:
  TopoProb(TopoAnalysis& topo, double domain_area, double area_frac,
           std::string prefix, const ConfigParser& parser)
      : ParOptProblem(MPI_COMM_SELF),
        nvars(topo.get_prob_mesh().get_nvars()),
        ncon(1),
        nineq(1),
        topo(topo),
        prefix(prefix),
        parser(parser),
        domain_area(domain_area),
        area_frac(area_frac) {
    setProblemSizes(nvars, ncon, 0);
    setNumInequalities(nineq, 0);

    if (!std::filesystem::is_directory(prefix)) {
      std::filesystem::create_directory(prefix);
    }

    reset_counter();
  }

  void print_progress(T comp, T pterm, T vol_frac, int header_every = 10) {
    if (counter % header_every == 1) {
      char phead[30];
      std::snprintf(phead, 30, "gradx penalty(c:%9.2e)",
                    parser.get_double_option("grad_penalty_coeff"));
      std::printf("\n%5s%20s%30s%20s\n", "iter", "comp", phead, "vol (\%)");
    }
    std::printf("%5d%20.10e%30.10e%20.5f\n", counter, comp, pterm,
                100.0 * vol_frac);
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

    for (int i = 0; i < nvars; i++) {
      xr[i] = x0r[i];
      lb[i] = -1.0;
      ub[i] = 1.0;
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

    T comp, area, pterm;
    std::vector<T> u = topo.eval_compliance_area(x, comp, area);
    pterm = topo.eval_grad_penalization(x);
    *fobj = comp + pterm;
    cons[0] = 1.0 - area / (domain_area * area_frac);  // >= 0

    if (counter % parser.get_int_option("write_vtk_every") == 0) {
      // Write design to vtk
      std::string vtk_name = "grid_" + std::to_string(counter) + ".vtk";
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(), {}, {},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(), {}, {}, {}, {});
      }

      // Write cut mesh to vtk
      vtk_name = "cut_" + std::to_string(counter) + ".vtk";
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, topo.get_phi(),
                           {}, {}, {}, {});
      } else {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, topo.get_phi(),
                           {}, {}, {{"displacement", u}}, {});
      }
    }

    // write quadrature to vtk for gradient check
    if (is_gradient_check) {
      std::string vtk_name = "fdcheck_quad_" + std::to_string(counter) + ".vtk";
      topo.write_quad_pts_to_vtk(fspath(prefix) / fspath(vtk_name));

      vtk_name = "fdcheck_grid_" + std::to_string(counter) + ".vtk";
      if constexpr (TopoAnalysis::use_ersatz) {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(), {}, {},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            topo.get_phi(), {}, {}, {}, {});
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
    print_progress(comp, pterm, area / domain_area);

    counter++;

    return 0;
  }

  int evalObjConGradient(ParOptVec* xvec, ParOptVec* gvec, ParOptVec** Ac) {
    T* xptr;
    xvec->getArray(&xptr);
    std::vector<T> xr(xptr, xptr + nvars);
    std::vector<T> x = topo.get_prob_mesh().expand(xr);

    T *g, *c;
    gvec->getArray(&g);
    gvec->zeroEntries();

    Ac[0]->getArray(&c);
    Ac[0]->zeroEntries();

    std::vector<T> gcomp, garea, gpen;
    topo.eval_compliance_area_grad(x, gcomp, garea);
    topo.eval_grad_penalization_grad(x, gpen);

    std::vector<T> gcompr = topo.get_prob_mesh().reduce(gcomp);
    std::vector<T> garear = topo.get_prob_mesh().reduce(garea);
    std::vector<T> gpenr = topo.get_prob_mesh().reduce(gpen);

    for (int i = 0; i < nvars; i++) {
      g[i] = gcompr[i] + gpenr[i];
      c[i] = -garear[i] / (domain_area * area_frac);
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

  bool is_gradient_check = false;
};

template <int Np_1d, bool use_ersatz>
void execute(int argc, char* argv[]) {
  constexpr int Np_1d_filter = Np_1d > 2 ? 4 : 2;
  MPI_Init(&argc, &argv);

  using T = double;
  using TopoAnalysis = TopoAnalysis<T, Np_1d, Np_1d_filter, use_ersatz>;

  bool smoke_test = false;
  if (argc > 2 and "--smoke" == std::string(argv[2])) {
    std::printf("This is a smoke test\n");
    smoke_test = true;
  }

  // Make the prefix
  std::string prefix = get_local_time();
  if (smoke_test) {
    prefix = "smoke_" + prefix;
  } else {
    prefix = "opt_" + prefix;
  }
  if (!std::filesystem::is_directory(prefix)) {
    std::filesystem::create_directory(prefix);
  }

  std::string json_dir = fspath(prefix) / fspath("json");
  if (!std::filesystem::is_directory(json_dir)) {
    std::filesystem::create_directory(json_dir);
  }

  std::string cfg_path{argv[1]};
  ConfigParser parser{cfg_path};
  std::filesystem::copy(
      cfg_path,
      fspath(prefix) / fspath(std::filesystem::absolute(cfg_path).filename()));

  // Set up grid
  std::array<int, 2> nxy = {parser.get_int_option("nx"),
                            parser.get_int_option("ny")};
  double loaded_frac = parser.get_double_option("loaded_frac");
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

  // TODO: lift this once lbracket is implemented
  if (instance == "lbracket") {
    throw std::runtime_error("lbracket is WIP");
  }

  T r0 = parser.get_double_option("helmholtz_r0");
  T E = parser.get_double_option("E");
  T nu = parser.get_double_option("nu");
  bool use_robust_projection = parser.get_bool_option("use_robust_projection");
  double robust_proj_beta = parser.get_double_option("robust_proj_beta");
  double robust_proj_eta = parser.get_double_option("robust_proj_eta");
  T penalty = parser.get_double_option("grad_penalty_coeff");

  CantileverMesh<T, Np_1d> prob_mesh(nxy, lxy, loaded_frac);

  TopoAnalysis topo{prob_mesh,
                    r0,
                    E,
                    nu,
                    penalty,
                    use_robust_projection,
                    robust_proj_beta,
                    robust_proj_eta,
                    prefix};

  double domain_area = lxy[0] * lxy[1];
  double area_frac = parser.get_double_option("area_frac");
  TopoProb<T, TopoAnalysis>* prob = new TopoProb<T, TopoAnalysis>(
      topo, domain_area, area_frac, prefix, parser);
  prob->incref();

  prob->check_gradients(1e-6);

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

  if (Np_1d % 2) {
    std::printf("[Error]Invalid input, expect even Np_1d, got %d\n", Np_1d);
    exit(-1);
  }

  switch (Np_1d) {
    case 2:
      if (use_ersatz) {
        execute<2, true>(argc, argv);
      } else {
        execute<2, false>(argc, argv);
      }
      break;

    case 4:
      if (use_ersatz) {
        execute<4, true>(argc, argv);
      } else {
        execute<4, false>(argc, argv);
      }
      break;

    case 6:
      if (use_ersatz) {
        execute<6, true>(argc, argv);
      } else {
        execute<6, false>(argc, argv);
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
