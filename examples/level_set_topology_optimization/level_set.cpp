#include <mpi.h>

#include <filesystem>

#include "ParOptOptimizer.h"
#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/grad_penalization.h"
#include "physics/volume.h"
#include "utils/parser.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

using fspath = std::filesystem::path;

template <typename T, int Np_1d_filter, class Mesh, class Quadrature,
          class Basis>
class TopoAnalysis {
 private:
  using Grid = StructuredGrid2D<T>;
  using Filter = HelmholtzFilter<T, Np_1d_filter>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;
  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using Penalization = GradPenalization<T, Basis::spatial_dim>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;
  using PenalizationAnalysis =
      GalerkinAnalysis<T, typename Filter::Mesh, typename Filter::Quadrature,
                       typename Filter::Basis, Penalization>;

  int constexpr static spatial_dim = Grid::spatial_dim;

 public:
  TopoAnalysis(T r0, T E, T nu, T penalty, bool use_robust_projection,
               double proj_beta, double proj_eta, Grid& grid, Mesh& mesh,
               Quadrature& quadrature, Basis& basis)
      : filter(r0, grid, use_robust_projection, proj_beta, proj_eta),
        elastic(E, nu, mesh, quadrature, basis),
        vol_analysis(mesh, quadrature, basis, vol),
        pen(penalty),
        pen_analysis(filter.get_mesh(), filter.get_quadrature(),
                     filter.get_basis(), pen),
        mesh(mesh),
        quadrature(quadrature),
        basis(basis),
        phi(mesh.get_lsf_dof()) {
    const T* xy0 = grid.get_xy0();
    const T* lxy = grid.get_lxy();

    // Get bcs and load verts
    double tol = 1e-10;
    double frac = 0.1;
    for (int i = 0; i < grid.get_num_verts(); i++) {
      T xloc[spatial_dim];
      grid.get_vert_xloc(i, xloc);
      if (xloc[0] - xy0[0] < tol) {
        bc_verts.push_back(i);
      } else if (xy0[0] + lxy[0] - xloc[0] < tol and
                 xloc[1] - (xy0[1] + (0.5 + frac) * lxy[1]) < tol and
                 xloc[1] - (xy0[1] + (0.5 - frac) * lxy[1]) > -tol) {
        load_verts.push_back(i);
      }
    }
  }

  // Create nodal design variables for a domain with periodic holes
  std::vector<T> create_initial_topology(int m, int n, double t,
                                         bool take_abs = true) {
    const T* lxy = mesh.get_grid().get_lxy();
    int ndv = mesh.get_grid().get_num_verts();
    std::vector<T> x0(ndv, 0.0);
    for (int i = 0; i < ndv; i++) {
      T xloc[spatial_dim];
      mesh.get_grid().get_vert_xloc(i, xloc);
      // x0[i] = (cos(xloc[0] / lxy[0] * 2.0 * PI * m) - 0.5) *
      //             (cos(xloc[1] / lxy[1] * 2.0 * PI * n) - 0.5) * 2.0 / 3.0 -
      //         0.5;
      T val;
      if (take_abs) {
        val = abs(cos(xloc[0] / lxy[0] * 2.0 * PI * m) *
                  sin(xloc[1] / lxy[1] * 2.0 * PI * n)) -
              t;
        if (val < 0) {
          x0[i] = val / t;
        } else {
          x0[i] = val / (1.0 - t);
        }
      } else {
        if (n % 2) {
          val = cos(xloc[0] / lxy[0] * 2.0 * PI * m) *
                    cos(xloc[1] / lxy[1] * 2.0 * PI * n) -
                t;
        } else {
          val = -cos(xloc[0] / lxy[0] * 2.0 * PI * m) *
                    cos(xloc[1] / lxy[1] * 2.0 * PI * n) -
                t;
        }
        if (val < 0) {
          x0[i] = val / (1.0 + t);
        } else {
          x0[i] = val / (1.0 - t);
        }
      }
    }
    return x0;
  }

  void update_mesh(const std::vector<T>& x) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    // Update mesh based on new LSF
    filter.apply(x.data(), phi.data());
    mesh.update_mesh();
    const std::unordered_map<int, int>& vert_nodes = mesh.get_vert_nodes();

    // Update bc dof
    bc_dof.clear();
    for (int bc_vert : bc_verts) {
      if (vert_nodes.count(bc_vert)) {
        for (int d = 0; d < spatial_dim; d++) {
          bc_dof.push_back(spatial_dim * vert_nodes.at(bc_vert) + d);
        }
      }
    }

    // Update load dof
    load_dof.clear();
    active_load_verts.clear();
    for (int load_vert : load_verts) {
      if (vert_nodes.count(load_vert) and x[load_vert] < 0.0) {
        active_load_verts.push_back(load_vert);
        load_dof.push_back(spatial_dim * vert_nodes.at(load_vert) +
                           1);  // y dir
      }
    }
  }

  std::vector<T> update_mesh_and_solve(const std::vector<T>& x) {
    // Solve the static problem
    // Note: bc_dof might change from iteration to iteration, but load_dof
    // should constant if optimizer bounds are applied properly
    update_mesh(x);

    T force = 1.0;
    std::vector<T> load_vals(active_load_verts.size(),
                             -force / active_load_verts.size());
    std::vector<T> sol = elastic.solve(bc_dof, load_dof, load_vals);
    return sol;
  }

  std::vector<T> eval_compliance_area(const std::vector<T>& x, T& comp,
                                      T& area) {
    std::vector<T> sol = update_mesh_and_solve(x);

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    comp = 2.0 * elastic.get_analysis().energy(nullptr, sol.data());
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
    filter.applyGradient(x.data(), gcomp.data(), gcomp.data());

    garea.resize(x.size());
    std::fill(garea.begin(), garea.end(), 0.0);
    vol_analysis.LSF_volume_derivatives(garea.data());
    filter.applyGradient(x.data(), garea.data(), garea.data());
  }

  void write_quad_pts_to_vtk(const std::string vtk_path) {
    Interpolator<T, Quadrature, Basis> interp(mesh, quadrature, basis);
    interp.to_vtk(vtk_path);
  }

  void write_design_to_vtk(const std::string vtk_path, const std::vector<T>& x,
                           const std::vector<T>& phi) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    ToVTK<T, typename Filter::Mesh> vtk(filter.get_mesh(), vtk_path);
    vtk.write_mesh();
    vtk.write_sol("x", x.data());
    vtk.write_sol("phi", phi.data());

    std::vector<T> bcs(x.size(), 0.0);
    std::vector<T> loads(x.size(), 0.0);

    for (int v : bc_verts) bcs[v] = 1.0;
    for (int v : load_verts) loads[v] = 1.0;

    vtk.write_sol("bc-verts", bcs.data());
    vtk.write_sol("load-verts", loads.data());
  }

  void write_cut_design_to_vtk(const std::string vtk_path,
                               const std::vector<T>& x,
                               const std::vector<T>& phi,
                               const std::vector<T>& u) {
    if (x.size() != phi.size() or
        u.size() != spatial_dim * mesh.get_num_nodes()) {
      throw std::runtime_error("sizes don't match");
    }

    ToVTK<T, Mesh> vtk(elastic.get_mesh(), vtk_path);
    vtk.write_mesh();
    vtk.write_vec("displacement", u.data());
    vtk.write_sol("x", mesh.get_lsf_nodes(x).data());
    vtk.write_sol("phi", mesh.get_lsf_nodes().data());

    std::vector<T> dof_bcs(spatial_dim * mesh.get_num_nodes(), 0.0);
    std::vector<T> dof_load(spatial_dim * mesh.get_num_nodes(), 0.0);

    for (int i : bc_dof) dof_bcs[i] = 1.0;
    for (int i : load_dof) dof_load[i] = 1.0;
    vtk.write_vec("bc-dof", dof_bcs.data());
    vtk.write_vec("load-dof", dof_load.data());
  }

  std::vector<T>& get_phi() { return phi; }
  std::vector<int>& get_active_load_verts() { return active_load_verts; }

 private:
  Filter filter;
  StaticElastic elastic;
  Volume vol;
  VolAnalysis vol_analysis;
  Penalization pen;
  PenalizationAnalysis pen_analysis;

  Mesh& mesh;
  Quadrature& quadrature;
  Basis& basis;
  std::vector<T>& phi;  // LSF values (filtered design variables)

  std::vector<int> bc_verts;
  std::vector<int> load_verts;
  std::vector<int> active_load_verts;

  std::vector<int> bc_dof;
  std::vector<int> load_dof;
};

template <typename T, class TopoAnalysis>
class TopoProb : public ParOptProblem {
 public:
  TopoProb(TopoAnalysis& topo, double domain_area, double area_frac,
           std::string prefix, const ConfigParser& parser)
      : ParOptProblem(MPI_COMM_SELF),
        nvars(topo.get_phi().size()),
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
  }

  void reset_counter() { counter = 0; }

  void getVarsAndBounds(ParOptVec* xvec, ParOptVec* lbvec, ParOptVec* ubvec) {
    ParOptScalar *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    std::vector<T> x0 = topo.create_initial_topology(
        parser.get_int_option("init_topology_m"),
        parser.get_int_option("init_topology_n"),
        parser.get_double_option("init_topology_t"),
        parser.get_bool_option("init_topology_abs"));

    // update mesh and bc/load dof, but don't perform the linear solve
    topo.update_mesh(x0);

    for (int i = 0; i < nvars; i++) {
      x[i] = x0[i];
      lb[i] = -1.0;
      ub[i] = 1.0;
    }

    const std::vector<int>& active_load_verts = topo.get_active_load_verts();
    for (int v : active_load_verts) {
      ub[v] = -1.0e-5;  // effectively 0
    }
  }

  int evalObjCon(ParOptVec* xvec, ParOptScalar* fobj, ParOptScalar* cons) {
    counter++;

    ParOptScalar* xptr;
    xvec->getArray(&xptr);
    std::vector<ParOptScalar> x(xptr, xptr + nvars);

    T comp, area, pterm;
    std::vector<T> u = topo.eval_compliance_area(x, comp, area);
    pterm = topo.eval_grad_penalization(x);
    *fobj = comp + pterm;
    cons[0] = 1.0 - area / (domain_area * area_frac);  // >= 0

    if (counter % parser.get_int_option("write_vtk_every") == 0) {
      // Write design to vtk
      std::string vtk_name = "design_" + std::to_string(counter) + ".vtk";
      topo.write_design_to_vtk(fspath(prefix) / fspath(vtk_name), x,
                               topo.get_phi());

      // Write cut mesh to vtk
      vtk_name = "cut_" + std::to_string(counter) + ".vtk";
      topo.write_cut_design_to_vtk(fspath(prefix) / fspath(vtk_name), x,
                                   topo.get_phi(), u);
    }

    // write quadrature to vtk for gradient check
    if (is_gradient_check) {
      std::string vtk_name = "fdcheck_quad_" + std::to_string(counter) + ".vtk";
      topo.write_quad_pts_to_vtk(fspath(prefix) / fspath(vtk_name));

      vtk_name = "fdcheck_design_" + std::to_string(counter) + ".vtk";
      topo.write_design_to_vtk(fspath(prefix) / fspath(vtk_name), x,
                               topo.get_phi());

      vtk_name = "fdcheck_cut_" + std::to_string(counter) + ".vtk";
      topo.write_cut_design_to_vtk(fspath(prefix) / fspath(vtk_name), x,
                                   topo.get_phi(), u);
    }

    // print optimization progress
    print_progress(comp, pterm, area / domain_area);

    return 0;
  }

  int evalObjConGradient(ParOptVec* xvec, ParOptVec* gvec, ParOptVec** Ac) {
    ParOptScalar* xptr;
    xvec->getArray(&xptr);
    std::vector<ParOptScalar> x(xptr, xptr + nvars);

    ParOptScalar *g, *c;
    gvec->getArray(&g);
    gvec->zeroEntries();

    Ac[0]->getArray(&c);
    Ac[0]->zeroEntries();

    std::vector<ParOptScalar> gcomp, garea, gpen;
    topo.eval_compliance_area_grad(x, gcomp, garea);
    topo.eval_grad_penalization_grad(x, gpen);

    for (int i = 0; i < nvars; i++) {
      g[i] = gcomp[i] + gpen[i];
      c[i] = -garea[i] / (domain_area * area_frac);
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

template <int Np_1d, int Np_1d_filter>
void execute(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  using T = double;
  using Grid = StructuredGrid2D<T>;
  int constexpr spatial_dim = Grid::spatial_dim;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using TopoAnalysis = TopoAnalysis<T, Np_1d_filter, Mesh, Quadrature, Basis>;

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

  std::string cfg_path(argv[1]);
  ConfigParser parser(cfg_path);
  std::filesystem::copy(
      cfg_path,
      fspath(prefix) / fspath(std::filesystem::absolute(cfg_path).filename()));

  // Set up grid
  int nxy[2] = {parser.get_int_option("nx"), parser.get_int_option("ny")};
  if (smoke_test) {
    nxy[0] = 9;
    nxy[1] = 5;
  }
  T lxy[2] = {parser.get_double_option("lx"), parser.get_double_option("ly")};
  Grid grid(nxy, lxy);

  // Set up analysis mesh
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  // Set up analysis
  T r0 = parser.get_double_option("helmholtz_r0"), E = 1e2, nu = 0.3;
  bool use_robust_projection = parser.get_bool_option("use_robust_projection");
  double robust_proj_beta = parser.get_double_option("robust_proj_beta");
  double robust_proj_eta = parser.get_double_option("robust_proj_eta");
  T penalty = parser.get_double_option("grad_penalty_coeff");
  TopoAnalysis topo(r0, E, nu, penalty, use_robust_projection, robust_proj_beta,
                    robust_proj_eta, grid, mesh, quadrature, basis);

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
  options->setOption("mma_max_iterations", parser.get_int_option("max_it"));
  options->setOption("mma_move_limit",
                     parser.get_double_option("mma_move_limit"));
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
  if (argc == 1) {
    std::printf("Usage: ./level_set level_set.cfg [--smoke]\n");
    exit(0);
  }

  std::string cfg_path(argv[1]);
  ConfigParser parser(cfg_path);
  int Np_1d = parser.get_int_option("Np_1d");
  int Np_1d_filter = parser.get_int_option("Np_1d_filter");

  if (Np_1d % 2) {
    std::printf("[Error]Invalid input, expect even Np_1d, got %d\n", Np_1d);
    exit(-1);
  }
  if (Np_1d_filter % 2) {
    std::printf("[Error]Invalid input, expect even Np_1d_filter, got %d\n",
                Np_1d_filter);
    exit(-1);
  }

  switch (Np_1d) {
    case 2:
      switch (Np_1d_filter) {
        case 2:
          execute<2, 2>(argc, argv);
          break;

        case 4:
          execute<2, 4>(argc, argv);
          break;

        case 6:
          execute<2, 6>(argc, argv);
          break;

        default:
          std::printf(
              "(Np_1d, Np_1d_filter) = (%d, %d) and not pre-compiled, "
              "enumerate it in the source code if you intend to use this "
              "combination.\n",
              Np_1d, Np_1d_filter);
          exit(-1);
          break;
      }
      break;

    case 4:
      switch (Np_1d_filter) {
        case 2:
          execute<4, 2>(argc, argv);
          break;

        case 4:
          execute<4, 4>(argc, argv);
          break;

        case 6:
          execute<4, 6>(argc, argv);
          break;

        default:
          std::printf(
              "(Np_1d, Np_1d_filter) = (%d, %d) and not pre-compiled, "
              "enumerate it in the source code if you intend to use this "
              "combination.\n",
              Np_1d, Np_1d_filter);
          exit(-1);
          break;
      }
      break;

    case 6:
      switch (Np_1d_filter) {
        case 2:
          execute<6, 2>(argc, argv);
          break;

        case 4:
          execute<6, 4>(argc, argv);
          break;

        case 6:
          execute<6, 6>(argc, argv);
          break;

        default:
          std::printf(
              "(Np_1d, Np_1d_filter) = (%d, %d) and not pre-compiled, "
              "enumerate it in the source code if you intend to use this "
              "combination.\n",
              Np_1d, Np_1d_filter);
          exit(-1);
          break;
      }
      break;

    default:
      std::printf(
          "(Np_1d, Np_1d_filter) = (%d, %d) and not pre-compiled, "
          "enumerate it in the source code if you intend to use this "
          "combination.\n",
          Np_1d, Np_1d_filter);
      exit(-1);
      break;
  }

  return 0;
}
