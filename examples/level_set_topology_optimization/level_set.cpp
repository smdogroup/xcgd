#include "ParOptOptimizer.h"
#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/volume.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

using fspath = std::filesystem::path;

template <typename T, class Mesh, class Quadrature, class Basis>
class TopoAnalysis {
 private:
  int constexpr static Np_1d_filter = 4;
  using Grid = StructuredGrid2D<T>;
  using Filter = HelmholtzFilter<T, Np_1d_filter>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;
  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;

  int constexpr static spatial_dim = Grid::spatial_dim;

 public:
  TopoAnalysis(T r0, T E, T nu, Grid& grid, Mesh& mesh, Quadrature& quadrature,
               Basis& basis)
      : filter(r0, grid),
        elastic(E, nu, mesh, quadrature, basis),
        vol_analysis(mesh, quadrature, basis, vol),
        mesh(mesh),
        phi(mesh.get_lsf_dof()) {
    const T* xy0 = grid.get_xy0();
    const T* lxy = grid.get_lxy();

    // Get bcs and load verts
    double tol = 1e-10;
    double frac = 0.3;
    for (int i = 0; i < grid.get_num_verts(); i++) {
      T xloc[spatial_dim];
      grid.get_vert_xloc(i, xloc);
      if (xloc[0] - xy0[0] < tol) {
        bc_verts.push_back(i);
      } else if (xy0[0] + lxy[0] - xloc[0] < tol and
                 xloc[1] - xy0[1] - frac * lxy[1] < tol) {
        load_verts.push_back(i);
      }
    }

    // Get load dof
    const std::unordered_map<int, int>& vert_nodes = mesh.get_vert_nodes();
    for (int load_vert : load_verts) {
      load_dof.push_back(spatial_dim * vert_nodes.at(load_vert) + 1);  // y dir
    }
    T force = 1.0;
    load_vals = std::vector<T>(load_verts.size(), -force / load_verts.size());
  }

  // Create nodal design variables for a domain with periodic holes
  std::vector<T> create_initial_topology(int m, int n) {
    const T* lxy = mesh.get_grid().get_lxy();
    int ndv = mesh.get_grid().get_num_verts();
    std::vector<T> x0(ndv, 0.0);
    for (int i = 0; i < ndv; i++) {
      T xloc[spatial_dim];
      mesh.get_grid().get_vert_xloc(i, xloc);
      x0[i] = (cos(xloc[0] / lxy[0] * 2.0 * PI * m) - 0.5) *
                  (cos(xloc[1] / lxy[1] * 2.0 * PI * n) - 0.5) * 2.0 / 3.0 -
              0.5;
    }
    return x0;
  }

  std::vector<T> update_mesh_and_solve(const std::vector<T>& x,
                                       bool write_vtk = true) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    // Update mesh based on new LSF
    filter.apply(x.data(), phi.data());
    mesh.update_mesh();

    // Update bcs
    std::vector<int> bc_dof;
    const std::unordered_map<int, int>& vert_nodes = mesh.get_vert_nodes();
    for (int bc_vert : bc_verts) {
      if (vert_nodes.count(bc_vert)) {
        for (int d = 0; d < spatial_dim; d++) {
          bc_dof.push_back(spatial_dim * vert_nodes.at(bc_vert) + d);
        }
      }
    }

    // Solve the static problem
    // Note: bc_dof might change from iteration to iteration, but load_dof
    // remains constant
    std::vector<T> sol = elastic.solve(bc_dof, load_dof, load_vals);
    return sol;
  }

  void eval_compliance_area(const std::vector<T>& x, T& comp, T& area) {
    std::vector<T> sol = update_mesh_and_solve(x);

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    comp = 2.0 * elastic.get_analysis().energy(nullptr, sol.data());
    area = vol_analysis.energy(nullptr, dummy.data());
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

  void write_design_to_vtk(const std::string vtk_path, const std::vector<T>& x,
                           const std::vector<T>& phi) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    ToVTK<T, typename Filter::Mesh> vtk(filter.get_mesh(), vtk_path);
    vtk.write_mesh();
    vtk.write_sol("x", x.data());
    vtk.write_sol("phi", phi.data());
  }

  std::vector<T>& get_phi() { return phi; }
  std::vector<int>& get_load_verts() { return load_verts; }

 private:
  Filter filter;
  StaticElastic elastic;
  Volume vol;
  VolAnalysis vol_analysis;
  Mesh& mesh;
  std::vector<T>& phi;  // LSF values (filtered design variables)

  std::vector<int> bc_verts;
  std::vector<int> load_verts;
  std::vector<int> load_dof;
  std::vector<T> load_vals;
};

template <typename T, class TopoAnalysis>
class TopoProb : public ParOptProblem {
 public:
  TopoProb(TopoAnalysis& topo, double max_area)
      : ParOptProblem(MPI_COMM_SELF),
        nvars(topo.get_phi().size()),
        ncon(1),
        nineq(1),
        topo(topo),
        prefix("results"),
        max_area(max_area) {
    setProblemSizes(nvars, ncon, 0);
    setNumInequalities(nineq, 0);

    if (!std::filesystem::is_directory(prefix)) {
      std::filesystem::create_directory(prefix);
    }

    reset_counter();
  }

  void reset_counter() { counter = 0; }

  void getVarsAndBounds(ParOptVec* xvec, ParOptVec* lbvec, ParOptVec* ubvec) {
    ParOptScalar *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    std::vector<T> x0 = topo.create_initial_topology(5, 3);

    for (int i = 0; i < nvars; i++) {
      x[i] = x0[i];
      lb[i] = -1.0;
      ub[i] = 1.0;
    }

    const std::vector<int>& load_verts = topo.get_load_verts();
    for (int v : load_verts) {
      ub[v] = -0.1;
    }
  }

  int evalObjCon(ParOptVec* xvec, ParOptScalar* fobj, ParOptScalar* cons) {
    counter++;

    ParOptScalar* xptr;
    xvec->getArray(&xptr);
    std::vector<ParOptScalar> x(xptr, xptr + nvars);

    T comp, area;
    topo.eval_compliance_area(x, comp, area);
    *fobj = comp;
    cons[0] = 1.0 - area / max_area;  // >= 0

    // Write design to vtk
    std::string vtk_name = "design_" + std::to_string(counter) + ".vtk";
    topo.write_design_to_vtk(fspath(prefix) / fspath(vtk_name), x,
                             topo.get_phi());

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

    std::vector<ParOptScalar> gcomp, garea;
    topo.eval_compliance_area_grad(x, gcomp, garea);

    for (int i = 0; i < nvars; i++) {
      g[i] = gcomp[i];
      c[i] = -garea[i] / max_area;
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
  int counter = -1;

  double max_area = 0.0;
};

void mesh_test(int argc, char* argv[]) {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  int constexpr spatial_dim = Grid::spatial_dim;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using TopoAnalysis = TopoAnalysis<T, Mesh, Quadrature, Basis>;

  bool smoke_test = false;
  if (argc > 1 and "--smoke" == std::string(argv[1])) {
    smoke_test = true;
  }

  // Set up grid
  int nxy[2] = {128, 32};
  if (smoke_test) {
    nxy[0] = 8;
    nxy[1] = 4;
  }
  T lxy[2] = {2.0, 1.0};
  Grid grid(nxy, lxy);

  // Set up analysis mesh
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  // Set up analysis
  T r0 = 0.05, E = 1e2, nu = 0.3;
  TopoAnalysis topo(r0, E, nu, grid, mesh, quadrature, basis);

  double max_area = lxy[0] * lxy[1] * 0.4;
  TopoProb<T, TopoAnalysis>* prob =
      new TopoProb<T, TopoAnalysis>(topo, max_area);
  prob->incref();

  prob->checkGradients(1e-6);
  prob->reset_counter();

  // Set options
  ParOptOptions* options = new ParOptOptions;
  options->incref();
  ParOptOptimizer::addDefaultOptions(options);

  int max_it = smoke_test ? 10 : 200;

  options->setOption("algorithm", "mma");
  options->setOption("mma_max_iterations", max_it);
  options->setOption("mma_move_limit", 0.05);
  options->setOption("output_file", "paropt.out");
  options->setOption("tr_output_file", "paropt.tr");
  options->setOption("mma_output_file", "paropt.mma");

  ParOptOptimizer* opt = new ParOptOptimizer(prob, options);
  opt->incref();

  opt->optimize();

  prob->decref();
  options->decref();
  opt->decref();
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  mesh_test(argc, argv);
  MPI_Finalize();
  return 0;
}