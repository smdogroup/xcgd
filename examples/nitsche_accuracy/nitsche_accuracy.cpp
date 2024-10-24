#include <algorithm>
#include <cassert>
#include <exception>
#include <memory>
#include <set>
#include <string>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/cut_bcs.h"
#include "physics/poisson.h"
#include "physics/volume.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/vtk.h"

template <typename T, int spatial_dim>
class L2normBulk final : public PhysicsBase<T, spatial_dim, 0, 1> {
 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ____) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * val * val;
  }
};

template <typename T, int spatial_dim>
class L2normSurf final : public PhysicsBase<T, spatial_dim, 0, 1> {
  static_assert(spatial_dim == 2,
                "This part is not yet implemented properly for 3D");

 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ___) const {
    T dt_val[spatial_dim] = {nrm_ref[1], -nrm_ref[0]};

    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> dt(dt_val);
    A2D::Vec<T, spatial_dim> JTJdt;

    T scale;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, dt, JTJdt);
    A2D::VecDot(dt, JTJdt, scale);
    scale = sqrt(scale);

    return weight * scale * val * val;
  }
};

#define PI 3.14159265358979323846

double hard_max(std::vector<double> vals) {
  return *std::max_element(vals.begin(), vals.end());
}

double ks_max(std::vector<double> vals, double ksrho = 50.0) {
  double umax = hard_max(vals);
  std::vector<double> eta(vals.size());
  std::transform(vals.begin(), vals.end(), eta.begin(),
                 [umax, ksrho](double x) { return exp(ksrho * (x - umax)); });
  return log(std::accumulate(eta.begin(), eta.end(), 0.0)) / ksrho + umax;
}

template <typename T, class Mesh>
void write_vtk(std::string vtkpath, const Mesh& mesh, const std::vector<T>& sol,
               const std::vector<T>& exact_sol, bool save_stencils) {
  ToVTK<T, Mesh> vtk(mesh, vtkpath);
  vtk.write_mesh();

  std::vector<double> nstencils(mesh.get_num_elements(),
                                Mesh::Np_1d * Mesh::Np_1d);
  auto degenerate_stencils = DegenerateStencilLogger::get_stencils();
  for (auto e : degenerate_stencils) {
    int elem = e.first;
    nstencils[elem] = e.second.size();
  }
  vtk.write_cell_sol("nstencils", nstencils.data());
  vtk.write_sol("u", sol.data());
  vtk.write_sol("u_exact", exact_sol.data());
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  if (save_stencils) {
    for (auto e : degenerate_stencils) {
      int elem = e.first;
      std::vector<int> nodes = e.second;
      std::vector<T> dof(mesh.get_num_nodes(), 0.0);
      for (int n : nodes) {
        dof[n] = 1.0;
      }
      char name[256];
      std::snprintf(name, 256, "degenerate_stencil_elem_%05d", elem);
      vtk.write_sol(name, dof.data());
    }
  }
}

template <typename T, class Mesh, class Analysis>
void write_field_vtk(std::string field_vtkpath, const Mesh& mesh,
                     const Analysis& analysis, const std::vector<T>& sol) {
  FieldToVTKNew<T, Mesh::spatial_dim> field_vtk(field_vtkpath);

  auto [xloc_q, dof_q] = analysis.interpolate_dof(sol.data());

  field_vtk.add_mesh(xloc_q);
  field_vtk.add_sol("sol", dof_q);

  field_vtk.write_mesh();
  field_vtk.write_sol("sol");
}

enum class ProbInstance { Circle, Wedge, Image };

template <typename T, typename Vec>
T exact_solution(ProbInstance instance, Vec xloc) {
  if (instance == ProbInstance::Circle) {
    T r2 = xloc[0] * xloc[0] + xloc[1] * xloc[1];
    T r = sqrt(r2);
    return r2 * sin(r);
  } else {  // Wedge and image
    return sin(xloc[0] * 4.0 * PI) * sin(xloc[1] * 4.0 * PI);
  }
}

// Get the l2 error of the numerical Poisson solution
template <int Np_1d>
void solve_poisson_problem(std::string prefix, ProbInstance instance,
                           std::string image_json, int nxy_val,
                           bool save_stencils, double nitsche_eta) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using QuadratureBulk = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER>;
  using QuadratureBCs = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto source_fun = [instance](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    if (instance == ProbInstance::Circle) {
      T r2 = xloc[0] * xloc[0] + xloc[1] * xloc[1];
      T r = sqrt(r2);
      return -(4.0 - r2) * sin(r) - 5.0 * r * cos(r);
    } else {  // Wedge and image
      return 32.0 * PI * PI * sin(xloc[0] * 4.0 * PI) * sin(xloc[1] * 4.0 * PI);
    }
  };

  auto bc_fun = [instance](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return exact_solution<T>(instance, xloc);
  };

  using PoissonBulk = PoissonPhysics<T, Basis::spatial_dim, typeof(source_fun)>;
  using PoissonBCs = CutDirichlet<T, Basis::spatial_dim, typeof(bc_fun)>;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PoissonBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PoissonBCs>;

  using BSRMat = GalerkinBSRMat<T, PoissonBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  DegenerateStencilLogger::enable();

  std::shared_ptr<Grid> grid;
  std::shared_ptr<Mesh> mesh;
  if (instance == ProbInstance::Circle) {
    int nxy[2] = {nxy_val, nxy_val};
    double R = 2.0 * PI;
    double lxy[2] = {3.0 * R, 3.0 * R};
    double xy0[2] = {-1.5 * R, -1.5 * R};
    grid = std::make_shared<Grid>(nxy, lxy, xy0);
    mesh = std::make_shared<Mesh>(*grid, [R](double x[]) {
      return x[0] * x[0] + x[1] * x[1] - R * R;  // <= 0
    });
  } else if (instance == ProbInstance::Wedge) {
    int nxy[2] = {nxy_val, nxy_val};
    double lxy[2] = {1.0, 1.0};
    double angle = PI / 6.0;
    grid = std::make_shared<Grid>(nxy, lxy);
    mesh = std::make_shared<Mesh>(*grid, [angle](double x[]) {
      T region1 = sin(angle) * (x[0] - 1.0) + cos(angle) * x[1];  // <= 0
      T region2 = 1e-6 - x[0];
      T region3 = 1e-6 - x[1];
      return hard_max({region1, region2, region3});
    });
  } else {  // Image
    json j;
    try {
      j = read_json(image_json);
    } catch (const std::exception& e) {
      std::cout << "failed to load the json file \"" + image_json +
                       "\" with the following exception message:\n";
      std::cout << std::string(e.what()) << "\n";
    }
    std::vector<double> lsf_dof = j["lsf_dof"];
    int nxy[2] = {j["nxy"], j["nxy"]};
    double lxy[2] = {1.0, 1.0};
    grid = std::make_shared<Grid>(nxy, lxy);
    mesh = std::make_shared<Mesh>(*grid);
    if (mesh->get_lsf_dof().size() != lsf_dof.size()) {
      std::string msg =
          "Attempting to populate the LSF dof from input image, but the "
          "dimensions don't match, the mesh has " +
          std::to_string(mesh->get_lsf_dof().size()) +
          " LSF nodes, but the input json has " +
          std::to_string(lsf_dof.size()) + " entries.";
      throw std::runtime_error(msg.c_str());
    }
    mesh->get_lsf_dof() = lsf_dof;
    mesh->update_mesh();
  }

  QuadratureBulk quadrature_bulk(*mesh);
  QuadratureBCs quadrature_bcs(*mesh);
  Basis basis(*mesh);

  PoissonBulk poisson_bulk(source_fun);
  PoissonBCs poisson_bcs(nitsche_eta, bc_fun);

  AnalysisBulk analysis_bulk(*mesh, quadrature_bulk, basis, poisson_bulk);
  AnalysisBCs analysis_bcs(*mesh, quadrature_bcs, basis, poisson_bcs);

  int ndof = mesh->get_num_nodes();

  // Set up the Jacobian matrix for Poisson's problem with Nitsche's boundary
  // conditions
  int *rowp = nullptr, *cols = nullptr;
  SparseUtils::CSRFromConnectivityFunctor(
      mesh->get_num_nodes(), mesh->get_num_elements(),
      mesh->max_nnodes_per_element,
      [&mesh](int elem, int* nodes) -> int {
        return mesh->get_elem_dof_nodes(elem, nodes);
      },
      &rowp, &cols);
  int nnz = rowp[mesh->get_num_nodes()];
  BSRMat* jac_bsr = new BSRMat(ndof, nnz, rowp, cols);
  std::vector<T> zeros(ndof, 0.0);
  analysis_bulk.jacobian(nullptr, zeros.data(), jac_bsr);
  jac_bsr->write_mtx(std::filesystem::path(prefix) /
                     std::filesystem::path("poisson_jac.mtx"));
  analysis_bcs.jacobian(nullptr, zeros.data(), jac_bsr,
                        false);  // Add bcs contribution
  jac_bsr->write_mtx(std::filesystem::path(prefix) /
                     std::filesystem::path("poisson_jac_with_nitsche.mtx"));
  CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);

  // Set up the right hand side
  std::vector<T> rhs(ndof, 0.0);

  analysis_bulk.residual(nullptr, zeros.data(), rhs.data());
  analysis_bcs.residual(nullptr, zeros.data(), rhs.data());
  for (int i = 0; i < ndof; i++) {
    rhs[i] *= -1.0;
  }

  // Solve
  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T>* chol =
      new SparseUtils::SparseCholesky<T>(jac_csc);
  chol->factor();
  std::vector<T> sol = rhs;
  chol->solve(sol.data());

  // Get exact solution
  std::vector<T> sol_exact(ndof, 0.0);
  for (int i = 0; i < ndof; i++) {
    T xloc[Basis::spatial_dim];
    mesh->get_node_xloc(i, xloc);
    sol_exact[i] = exact_solution<T>(instance, xloc);
  }

  // Compute the L2 norm of the solution field (not vector)
  std::vector<T> diff(sol.size());
  for (int i = 0; i < sol.size(); i++) {
    diff[i] = (sol[i] - sol_exact[i]);
  }

  GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis,
                   L2normBulk<T, Basis::spatial_dim>>
      integrator_bulk(*mesh, quadrature_bulk, basis, {});
  GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis,
                   L2normSurf<T, Basis::spatial_dim>>
      integrator_bcs(*mesh, quadrature_bcs, basis, {});

  std::vector<T> ones(sol.size(), 1.0);
  T area = integrator_bulk.energy(nullptr, ones.data());
  T perimeter = integrator_bcs.energy(nullptr, ones.data());

  T err_l2norm_bulk = integrator_bulk.energy(nullptr, diff.data());
  T err_l2norm_bcs = integrator_bcs.energy(nullptr, diff.data());

  T l2norm_bulk = integrator_bulk.energy(nullptr, sol_exact.data());
  T l2norm_bcs = integrator_bcs.energy(nullptr, sol_exact.data());

  // Get numerical and exact solutions at quadrature points
  auto [xlocq_bulk, solq_bulk] = analysis_bulk.interpolate_dof(sol.data());
  auto [xlocq_bcs, solq_bcs] = analysis_bcs.interpolate_dof(sol.data());

  std::vector<T> solq_bulk_exact(solq_bulk.size()),
      solq_bcs_exact(solq_bcs.size());

  for (int i = 0; i < solq_bulk.size(); i++) {
    solq_bulk_exact[i] =
        exact_solution<T>(instance, &xlocq_bulk[Basis::spatial_dim * i]);
  }

  for (int i = 0; i < solq_bcs.size(); i++) {
    solq_bcs_exact[i] =
        exact_solution<T>(instance, &xlocq_bcs[Basis::spatial_dim * i]);
  }

  json j = {// {"sol", sol},
            // {"sol_exact", sol_exact},
            // {"lsf", mesh->get_lsf_nodes()},
            // {"solq_bulk", solq_bulk},
            // {"solq_bulk_exact", solq_bulk_exact},
            // {"solq_bcs", solq_bcs},
            // {"solq_bcs_exact", solq_bcs_exact},
            {"err_l2norm_bulk", err_l2norm_bulk},
            {"err_l2norm_bcs", err_l2norm_bcs},
            {"l2norm_bulk", l2norm_bulk},
            {"l2norm_bcs", l2norm_bcs},
            {"area", area},
            {"perimeter", perimeter}};

  char json_name[256];
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);
  write_vtk<T>(
      std::filesystem::path(prefix) / std::filesystem::path("poisson.vtk"),
      *mesh, sol, sol_exact, save_stencils);

  write_field_vtk<T>(std::filesystem::path(prefix) /
                         std::filesystem::path("field_poisson_bulk.vtk"),
                     *mesh, analysis_bulk, sol);

  write_field_vtk<T>(std::filesystem::path(prefix) /
                         std::filesystem::path("field_poisson_bcs.vtk"),
                     *mesh, analysis_bcs, sol);
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--save-degenerate-stencils", 0);
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--nxy", 64);
  p.add_argument<double>("--nitsche_eta", 1e6);
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--instance", "circle",
                              {"circle", "wedge", "image"});
  p.add_argument<std::string>("--image_json", "image.json");
  p.parse_args(argc, argv);

  bool save_stencils = p.get<int>("save-degenerate-stencils");

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int nxy_val = p.get<int>("nxy");
  double nitsche_eta = p.get<double>("nitsche_eta");
  ProbInstance instance = std::map<std::string, ProbInstance>{
      {"circle", ProbInstance::Circle},
      {"wedge", ProbInstance::Wedge},
      {"image", ProbInstance::Image}}[p.get<std::string>("instance")];
  std::string image_json = p.get<std::string>("image_json");
  switch (Np_1d) {
    case 2:
      solve_poisson_problem<2>(prefix, instance, image_json, nxy_val,
                               save_stencils, nitsche_eta);
      break;
    case 4:
      solve_poisson_problem<4>(prefix, instance, image_json, nxy_val,
                               save_stencils, nitsche_eta);
      break;
    case 6:
      solve_poisson_problem<6>(prefix, instance, image_json, nxy_val,
                               save_stencils, nitsche_eta);
      break;
    case 8:
      solve_poisson_problem<8>(prefix, instance, image_json, nxy_val,
                               save_stencils, nitsche_eta);
      break;
    case 10:
      solve_poisson_problem<10>(prefix, instance, image_json, nxy_val,
                                save_stencils, nitsche_eta);
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
