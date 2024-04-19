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

template <typename T, class Mesh, class Quadrature, class Basis, class LSFMesh,
          class LSFQuadrature, class LSFBasis>
class TopoAnalysis {
 private:
  using HelmholtzFilter = HelmholtzFilter<T, LSFMesh, LSFQuadrature, LSFBasis>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;
  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;

 public:
  TopoAnalysis(HelmholtzFilter& filter, StaticElastic& elastic)
      : filter(filter),
        elastic(elastic),
        vol_analysis(elastic.get_mesh(), elastic.get_quadrature(),
                     elastic.get_basis(), vol),
        mesh(elastic.get_mesh()),
        lsf_mesh(filter.get_mesh()),
        phi(mesh.get_lsf_dof()) {}

  void eval_compliance_area(const std::vector<T>& x, T& comp, T& area) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    filter.apply(x.data(), phi.data());
    mesh.update_mesh();
    std::vector<T> sol = elastic.solve();

    std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
    comp = 2.0 * elastic.get_analysis().energy(nullptr, sol.data());
    area = vol_analysis.energy(nullptr, dummy.data());
  }

  void eval_compliance_area_grad(const std::vector<T>& x, std::vector<T>& gcomp,
                                 std::vector<T>& garea) {
    if (x.size() != phi.size()) {
      throw std::runtime_error("sizes don't match");
    }

    filter.apply(x.data(), phi.data());
    mesh.update_mesh();
    std::vector<T> sol = elastic.solve();

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
    ToVTK<T, LSFMesh> vtk(lsf_mesh, vtk_path);
    vtk.write_mesh();
    vtk.write_sol("x", x.data());
    vtk.write_sol("phi", phi.data());
  }

  std::vector<T>& get_phi() { return phi; }
  LSFMesh& get_lsf_mesh() { return lsf_mesh; }

 private:
  HelmholtzFilter& filter;
  StaticElastic& elastic;
  Volume vol;
  VolAnalysis vol_analysis;

  Mesh& mesh;
  LSFMesh& lsf_mesh;

  std::vector<T>& phi;  // LSF values (filtered design variables)
};

// Create nodal design variables for a domain with periodic holes
template <typename T, class Mesh>
std::vector<T> create_initial_topology(const Mesh& lsf_mesh, int m = 1,
                                       int n = 1) {
  const T* lxy = lsf_mesh.get_grid().get_lxy();
  int ndv = lsf_mesh.get_num_nodes();
  std::vector<T> x0(ndv, 0.0);
  for (int i = 0; i < ndv; i++) {
    T xloc[Mesh::spatial_dim];
    lsf_mesh.get_node_xloc(i, xloc);
    x0[i] = (cos(xloc[0] / lxy[0] * 2.0 * PI * m) - 0.5) *
                (cos(xloc[1] / lxy[1] * 2.0 * PI * n) - 0.5) * 2.0 / 3.0 -
            0.5;
  }
  return x0;
}

template <typename T, class TopoAnalysis>
class TopoProb : public ParOptProblem {
 public:
  TopoProb(TopoAnalysis& topo)
      : ParOptProblem(MPI_COMM_SELF),
        nvars(topo.get_phi().size()),
        ncon(1),
        nineq(1),
        topo(topo),
        prefix("results"),
        counter(0) {
    setProblemSizes(nvars, ncon, 0);
    setNumInequalities(nineq, 0);

    if (!std::filesystem::is_directory(prefix)) {
      std::filesystem::create_directory(prefix);
    }
  }

  void getVarsAndBounds(ParOptVec* xvec, ParOptVec* lbvec, ParOptVec* ubvec) {
    ParOptScalar *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    std::vector<T> x0 = create_initial_topology<T>(topo.get_lsf_mesh(), 4, 2);

    for (int i = 0; i < nvars; i++) {
      x[i] = x0[i];
      lb[i] = -1.0;
      ub[i] = 1.0;
    }
  }

  int evalObjCon(ParOptVec* xvec, ParOptScalar* fobj, ParOptScalar* cons) {
    counter++;

    ParOptScalar* xptr;
    xvec->getArray(&xptr);
    std::vector<ParOptScalar> x(xptr, xptr + nvars);
    topo.eval_compliance_area(x, *fobj, cons[0]);

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
      c[i] = garea[i];
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
};

void mesh_test() {
  using T = double;
  int constexpr Np_1d = 4;
  using Grid = StructuredGrid2D<T>;
  int constexpr spatial_dim = Grid::spatial_dim;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  using LSFQuadrature = GDGaussQuadrature2D<T, Np_1d>;
  using LSFMesh = GridMesh<T, Np_1d>;
  using LSFBasis = GDBasis2D<T, LSFMesh>;

  using Filter = HelmholtzFilter<T, LSFMesh, LSFQuadrature, LSFBasis>;
  using StaticElastic = StaticElastic<T, Mesh, Quadrature, Basis>;

  using Physics = VolumePhysics<T, spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;

  // Set up grid
  int nxy[2] = {128, 64};
  T lxy[2] = {2.0, 1.0};
  Grid grid(nxy, lxy);

  // Set up analysis mesh
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  // Set up lsf mesh
  LSFMesh lsf_mesh = mesh.get_lsf_mesh();
  LSFQuadrature lsf_quadrature(lsf_mesh);
  LSFBasis lsf_basis(lsf_mesh);

  // Set up filter
  T r0 = 0.05;
  Filter filter(r0, lsf_mesh, lsf_quadrature, lsf_basis);

  // Create
  std::vector<T> x = create_initial_topology<T>(lsf_mesh, 4, 2);
  std::vector<T>& phi = mesh.get_lsf_dof();
  filter.apply(x.data(), phi.data());
  mesh.update_mesh();

  // Set up bcs and loads
  std::vector<int> bc_nodes = mesh.get_left_boundary_nodes();
  std::vector<int> load_nodes = mesh.get_right_boundary_nodes();
  std::vector<int> bc_dof(spatial_dim * bc_nodes.size());
  std::vector<int> load_dof(load_nodes.size());
  std::vector<T> load_vals(load_nodes.size(), 0.0);
  for (int i = 0; i < bc_nodes.size(); i++) {
    for (int d = 0; d < spatial_dim; d++) {
      bc_dof[spatial_dim * i + d] = spatial_dim * bc_nodes[i] + d;
    }
  }
  for (int i = 0; i < load_nodes.size(); i++) {
    load_dof[i] = spatial_dim * load_nodes[i] + 1;
    load_vals[i] = -1.0;
  }

  ToVTK<T, GridMesh<T, Np_1d>> lsf_vtk(lsf_mesh, "lsf_mesh.vtk");
  lsf_vtk.write_mesh();
  lsf_vtk.write_sol("x", x.data());
  lsf_vtk.write_sol("phi", phi.data());

  // Export quadrature points to a separate vtk file
  using Interpolator = Interpolator<T, Quadrature, Basis>;
  Interpolator interp(mesh, quadrature, basis);
  std::vector<T> dummy(mesh.get_num_nodes(), 0.0);
  interp.to_vtk("quadratures.vtk", dummy.data());

  ToVTK<T, Mesh> vtk(mesh, "cut_mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("x", mesh.get_lsf_nodes(x).data());
  vtk.write_sol("phi", mesh.get_lsf_nodes().data());

  // Write boundary condition to vtk
  std::vector<T> sol_bcs(spatial_dim * mesh.get_num_nodes(), 0.0);
  std::vector<T> sol_load(spatial_dim * mesh.get_num_nodes(), 0.0);
  for (int bc : bc_dof) sol_bcs[bc] = 1.0;
  for (int l : load_dof) sol_load[l] = 1.0;
  vtk.write_vec("bc", sol_bcs.data());
  vtk.write_vec("load", sol_load.data());

  int ndv = x.size();
  std::vector<T> p(ndv, 0.0);
  for (int i = 0; i < ndv; i++) {
    p[i] = T(rand()) / RAND_MAX;
  }

  double h = 1e-6;
  for (int i = 0; i < ndv; i++) {
    x[i] -= h * p[i];
  }

  T E = 1e2, nu = 0.3;
  StaticElastic static_elastic(E, nu, mesh, quadrature, basis, bc_dof, load_dof,
                               load_vals);

  using TopoAnalysis = TopoAnalysis<T, Mesh, Quadrature, Basis, LSFMesh,
                                    LSFQuadrature, LSFBasis>;

  TopoAnalysis topo(filter, static_elastic);

  TopoProb<T, TopoAnalysis>* prob = new TopoProb<T, TopoAnalysis>(topo);
  prob->incref();

  prob->checkGradients(1e-6);
  exit(-1);

  // Set options
  ParOptOptions* options = new ParOptOptions;
  options->incref();
  ParOptOptimizer::addDefaultOptions(options);

  options->setOption("algorithm", "mma");
  options->setOption("mma_max_iterations", 200);
  options->setOption("output_file", "paropt.out");
  options->setOption("tr_output_file", "paropt.tr");
  options->setOption("mma_output_file", "paropt.mma");

  ParOptOptimizer* opt = new ParOptOptimizer(prob, options);
  opt->incref();

  T c1, a1;
  std::vector<T> gcomp, garea;
  topo.eval_compliance_area(x, c1, a1);
  topo.eval_compliance_area_grad(x, gcomp, garea);

  for (int i = 0; i < ndv; i++) {
    x[i] += 2.0 * h * p[i];
  }
  T c2, a2;
  topo.eval_compliance_area(x, c2, a2);
  topo.eval_compliance_area_grad(x, gcomp, garea);

  T gcomp_adjoint = (c2 - c1) / 2.0 / h;
  T garea_adjoint = (a2 - a1) / 2.0 / h;

  for (int i = 0; i < ndv; i++) {
    x[i] -= 2.0 * h * p[i];
  }

  T comp, area;
  topo.eval_compliance_area(x, comp, area);
  topo.eval_compliance_area_grad(x, gcomp, garea);

  T gcomp_fd = 0.0, garea_fd = 0.0;
  for (int i = 0; i < ndv; i++) {
    gcomp_fd += gcomp[i] * p[i];
    garea_fd += garea[i] * p[i];
  }

  std::printf("compliance:      %25.15e\n", comp);
  std::printf("gradient fd:     %25.15e\n", gcomp_fd);
  std::printf("gradientadjoint: %25.15e\n", gcomp_adjoint);
  std::printf("relative error:  %25.15e\n",
              (gcomp_fd - gcomp_adjoint) / gcomp_adjoint);
  std::printf("\n");

  std::printf("area:            %25.15e\n", area);
  std::printf("gradient fd:     %25.15e\n", garea_fd);
  std::printf("gradientadjoint: %25.15e\n", garea_adjoint);
  std::printf("relative error:  %25.15e\n",
              (garea_fd - garea_adjoint) / garea_adjoint);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  mesh_test();
  MPI_Finalize();
  return 0;
}