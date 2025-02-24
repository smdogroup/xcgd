#include "apps/poisson_app.h"
#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/vtk.h"

#define PI 3.14159265358979323846
#define k (1.4 * PI)

template <int Np_1d, bool use_finite_cell_mesh>
void execute(std::string prefix, int nxy, std::string instance, bool save_vtk) {
  using T = double;
  using Grid = StructuredGrid2D<T>;

  using Mesh =
      typename std::conditional<use_finite_cell_mesh, FiniteCellMesh<T, Np_1d>,
                                CutMesh<T, Np_1d>>::type;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;
  using Basis = GDBasis2D<T, Mesh>;
  constexpr int spatial_dim = Basis::spatial_dim;

  auto poisson_exact_fun = [](const A2D::Vec<T, spatial_dim>& xloc) {
    return sin(k * xloc(0)) * sin(k * xloc(1));
  };

  auto poisson_stress_fun = [](const A2D::Vec<T, spatial_dim>& xloc) {
    A2D::Vec<T, spatial_dim> ret;
    ret(0) = k * cos(k * xloc(0)) * sin(k * xloc(1));
    ret(1) = k * sin(k * xloc(0)) * cos(k * xloc(1));
    return ret;
  };

  auto poisson_source_fun = [](const A2D::Vec<T, spatial_dim>& xloc) {
    return -2.0 * k * k * sin(k * xloc(0)) * sin(k * xloc(1));
  };

  int nx_ny[2] = {nxy, nxy};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nx_ny, lxy);

  std::shared_ptr<Mesh> mesh;

  if (instance == "square") {
    mesh = std::make_shared<Mesh>(grid);
  } else if (instance == "circle") {
    mesh = std::make_shared<Mesh>(grid, [](T* x) {
      return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) -
             0.489 * 0.489;
    });
  }

  Quadrature quadrature(*mesh);
  Basis basis(*mesh);

  double nitsche_eta = 1e8;
  PoissonApp<T, Mesh, Quadrature, Basis, typeof(poisson_source_fun)>
      poisson_app(*mesh, quadrature, basis, poisson_source_fun);
  PoissonNitscheApp<T, Mesh, Quadrature, Basis, typeof(poisson_source_fun),
                    typeof(poisson_exact_fun)>
      poisson_nitsche_app(*mesh, quadrature, basis, poisson_source_fun,
                          poisson_exact_fun, nitsche_eta);

  // Solve
  std::vector<T> sol;

  if (instance == "square") {
    std::vector<int> dof_bcs;
    std::vector<T> dof_vals;

    for (auto nodes :
         {mesh->get_left_boundary_nodes(), mesh->get_right_boundary_nodes(),
          mesh->get_upper_boundary_nodes(), mesh->get_lower_boundary_nodes()}) {
      for (int node : nodes) {
        T xloc[spatial_dim];
        mesh->get_node_xloc(node, xloc);
        dof_bcs.push_back(node);
        dof_vals.push_back(poisson_exact_fun(xloc));
      }
    }

    sol = poisson_app.solve(dof_bcs, dof_vals);
  } else {
    sol = poisson_nitsche_app.solve();
  }

  // Evaluate norm errors
  using EnergyNormPhysics =
      PoissonEnergyNorm<T, spatial_dim, typeof(poisson_exact_fun),
                        typeof(poisson_stress_fun)>;
  using EnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, EnergyNormPhysics>;

  EnergyNormPhysics val_norm_physics(poisson_exact_fun, poisson_stress_fun, 1.0,
                                     0.0);
  EnergyNormPhysics stress_norm_physics(poisson_exact_fun, poisson_stress_fun,
                                        0.0, 1.0);
  EnergyNormPhysics energy_norm_physics(poisson_exact_fun, poisson_stress_fun,
                                        1.0, 1.0);

  EnergyNormAnalysis val_norm_analysis(*mesh, quadrature, basis,
                                       val_norm_physics);
  EnergyNormAnalysis stress_norm_analysis(*mesh, quadrature, basis,
                                          stress_norm_physics);
  EnergyNormAnalysis energy_norm_analysis(*mesh, quadrature, basis,
                                          energy_norm_physics);

  json j = {
      {"val_norm", sqrt(val_norm_analysis.energy(nullptr, sol.data()))},
      {"stress_norm", sqrt(stress_norm_analysis.energy(nullptr, sol.data()))},
      {"energy_norm", sqrt(energy_norm_analysis.energy(nullptr, sol.data()))}};

  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);

  // VTK
  if (save_vtk) {
    ToVTK<T, Mesh> vtk(*mesh, std::filesystem::path(prefix) /
                                  std::filesystem::path("cut.vtk"));
    vtk.write_mesh();
    vtk.write_sol("u", sol.data());
    vtk.write_sol("lsf", mesh->get_lsf_nodes().data());

    FieldToVTKNew<T, spatial_dim> sampling_vtk(
        std::filesystem::path(prefix) / std::filesystem::path("sampling.vtk"));

    // using Sampler = GDSampler2D<T, 15, Mesh>;
    // Sampler sampler(*mesh, 0.0, 0.0);

    using Sampler = Quadrature;
    Sampler& sampler = quadrature;

    using EnergyNormSamplerAnalysis =
        GalerkinAnalysis<T, Mesh, Sampler, Basis, EnergyNormPhysics>;

    EnergyNormSamplerAnalysis val_norm_sampler_analysis(*mesh, sampler, basis,
                                                        val_norm_physics);
    EnergyNormSamplerAnalysis stress_norm_sampler_analysis(
        *mesh, sampler, basis, stress_norm_physics);
    EnergyNormSamplerAnalysis energy_norm_sampler_analysis(
        *mesh, sampler, basis, energy_norm_physics);

    std::vector<T> xloc_samples =
        val_norm_sampler_analysis.interpolate_energy(sol.data()).first;
    sampling_vtk.add_mesh(xloc_samples);
    sampling_vtk.write_mesh();

    std::vector<T> val_norm_samples =
        val_norm_sampler_analysis.interpolate_energy(sol.data()).second;
    std::vector<T> stress_norm_samples =
        stress_norm_sampler_analysis.interpolate_energy(sol.data()).second;
    std::vector<T> energy_norm_samples =
        energy_norm_sampler_analysis.interpolate_energy(sol.data()).second;

    sampling_vtk.add_sol("val_norm", val_norm_samples);
    sampling_vtk.write_sol("val_norm");

    sampling_vtk.add_sol("stress_norm", stress_norm_samples);
    sampling_vtk.write_sol("stress_norm");

    sampling_vtk.add_sol("energy_norm", energy_norm_samples);
    sampling_vtk.write_sol("energy_norm");
  }
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--nxy", 32);
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--instance", "square");
  p.add_argument<int>("--use-finite-cell-mesh", 0);
  p.add_argument<int>("--save-vtk", 1);
  p.parse_args(argc, argv);

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int nxy = p.get<int>("nxy");
  std::string instance = p.get<std::string>("instance");
  bool use_finite_cell_mesh = p.get<int>("use-finite-cell-mesh");
  bool save_vtk = p.get<int>("save-vtk");

  switch (Np_1d) {
    case 2:
      if (use_finite_cell_mesh) {
        execute<2, true>(prefix, nxy, instance, save_vtk);
      } else {
        execute<2, false>(prefix, nxy, instance, save_vtk);
      }
      break;
    case 4:
      if (use_finite_cell_mesh) {
        execute<4, true>(prefix, nxy, instance, save_vtk);
      } else {
        execute<4, false>(prefix, nxy, instance, save_vtk);
      }
      break;
    case 6:
      if (use_finite_cell_mesh) {
        execute<6, true>(prefix, nxy, instance, save_vtk);
      } else {
        execute<6, false>(prefix, nxy, instance, save_vtk);
      }
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
