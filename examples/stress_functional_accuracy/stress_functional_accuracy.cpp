#include "apps/poisson_app.h"
#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/stress.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/vtk.h"

#define PI 3.14159265358979323846
#define k (1.4 * PI)

// only useful if ersatz material is used
template <typename T, int dim, class Mesh>
std::vector<T> grid_dof_to_cut_dof(const Mesh& mesh, const std::vector<T> u) {
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

template <int Np_1d, bool use_finite_cell_mesh, bool use_ersatz = false>
void execute_elasticity(std::string prefix, int nxy, double ersatz_ratio) {
  using T = double;
  using Grid = StructuredGrid2D<T>;

  using Mesh =
      typename std::conditional<use_finite_cell_mesh, FiniteCellMesh<T, Np_1d>,
                                CutMesh<T, Np_1d>>::type;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;
  using Basis = GDBasis2D<T, Mesh>;
  constexpr int spatial_dim = Basis::spatial_dim;

  auto elasticity_source = [](const A2D::Vec<T, spatial_dim>& xloc) {
    A2D::Vec<T, spatial_dim> ret;
    return ret;
  };

  T Tx = 10.0, R = 1.0;

  auto elasticity_stress_exact = [Tx, R](const A2D::Vec<T, spatial_dim>& xloc) {
    T theta = atan(xloc[1] / xloc[0]);
    T r = sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);

    T R2 = R * R;
    T r2 = r * r;
    T R4 = R * R * R * R;
    T r4 = r * r * r * r;

    T Sr = Tx / 2.0 * (1.0 - R2 / r2) +
           Tx / 2.0 * (1.0 + 3.0 * R4 / r4 - 4.0 * R2 / r2) * cos(2.0 * theta);
    T St = Tx / 2.0 * (1.0 + R2 / r2) -
           Tx / 2.0 * (1.0 + 3.0 * R4 / r4) * cos(2.0 * theta);
    T Srt =
        -Tx / 2.0 * (1.0 - 3.0 * R4 / r4 + 2.0 * R2 / r2) * sin(2.0 * theta);

    T C = cos(theta);
    T S = sin(theta);

    T Sx = C * C * Sr - 2.0 * C * S * Srt + S * S * St;
    T Sy = S * S * Sr + 2.0 * C * S * Srt + C * C * St;
    T Sxy = C * S * (Sr - St) + Srt * (C * C - S * S);

    A2D::SymMat<T, spatial_dim> Stensor;
    Stensor(0, 0) = Sx;
    Stensor(1, 1) = Sy;
    Stensor(0, 1) = Sxy;
    return Stensor;
  };

  using PhysicsApp = typename std::conditional<
      use_ersatz,
      StaticElasticErsatz<T, Mesh, Quadrature, Basis, typeof(elasticity_source),
                          Grid>,
      StaticElastic<T, Mesh, Quadrature, Basis,
                    typeof(elasticity_source)>>::type;

  int nx_ny[2] = {nxy, nxy};
  T L = 4.0;
  T dh = L / nxy;
  T lxy[2] = {L, L};
  Grid grid(nx_ny, lxy);

  Mesh mesh(grid, [R](T* x) { return R * R - x[0] * x[0] - x[1] * x[1]; });

  Quadrature quadrature(mesh);
  Basis basis(mesh);

  double E = 1e5, nu = 0.3;

  std::shared_ptr<PhysicsApp> physics_app;
  if constexpr (use_ersatz) {
    physics_app = std::make_shared<PhysicsApp>(E, nu, mesh, quadrature, basis,
                                               elasticity_source, ersatz_ratio);
  } else {
    physics_app = std::make_shared<PhysicsApp>(E, nu, mesh, quadrature, basis,
                                               elasticity_source);
  }

  // Set bcs
  std::vector<int> dof_bcs;
  std::vector<T> dof_vals;

  for (int node : mesh.get_left_boundary_nodes()) {
    T xloc[spatial_dim];
    mesh.get_node_xloc(node, xloc);
    dof_vals.push_back(0.0);
    if (use_ersatz) {
      node = mesh.get_node_vert(node);
    }
    dof_bcs.push_back(2 * node);
  }

  for (int node : mesh.get_lower_boundary_nodes()) {
    T xloc[spatial_dim];
    mesh.get_node_xloc(node, xloc);
    dof_vals.push_back(0.0);
    if (use_ersatz) {
      node = mesh.get_node_vert(node);
    }
    dof_bcs.push_back(2 * node + 1);
  }

  auto load_func_top =
      [elasticity_stress_exact](const A2D::Vec<T, spatial_dim>& xloc) {
        A2D::SymMat<T, spatial_dim> stress = elasticity_stress_exact(xloc);

        A2D::Vec<T, spatial_dim> ret;
        ret(0) = stress(0, 1);
        ret(1) = stress(1, 1);
        return ret;
      };
  auto load_func_right =
      [elasticity_stress_exact](const A2D::Vec<T, spatial_dim>& xloc) {
        A2D::SymMat<T, spatial_dim> stress = elasticity_stress_exact(xloc);

        A2D::Vec<T, spatial_dim> ret;
        ret(0) = stress(0, 0);
        ret(1) = stress(0, 1);
        return ret;
      };

  using LoadPhysicsTop =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func_top)>;
  using LoadPhysicsRight =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func_right)>;

  LoadPhysicsTop load_physics_top(load_func_top);
  LoadPhysicsRight load_physics_right(load_func_right);

  using LoadQuadratureTop =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::TOP, Mesh>;
  using LoadAnalysisTop = GalerkinAnalysis<T, Mesh, LoadQuadratureTop, Basis,
                                           LoadPhysicsTop, use_ersatz>;

  using LoadQuadratureRight =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;
  using LoadAnalysisRight =
      GalerkinAnalysis<T, Mesh, LoadQuadratureRight, Basis, LoadPhysicsRight,
                       use_ersatz>;

  std::set<int> load_elements_top, load_elements_right;
  const auto& cell_elems = mesh.get_cell_elems();
  for (int i = 0; i < nxy; i++) {
    load_elements_top.insert(cell_elems.at(grid.get_coords_cell(i, nxy - 1)));
    load_elements_right.insert(cell_elems.at(grid.get_coords_cell(nxy - 1, i)));
  }

  LoadQuadratureTop load_quadrature_top(mesh, load_elements_top);
  LoadQuadratureRight load_quadrature_right(mesh, load_elements_right);

  LoadAnalysisTop load_analysis_top(mesh, load_quadrature_top, basis,
                                    load_physics_top);
  LoadAnalysisRight load_analysis_right(mesh, load_quadrature_right, basis,
                                        load_physics_right);

  std::vector<T> sol = physics_app->solve(
      dof_bcs, dof_vals,
      std::make_tuple(load_analysis_top, load_analysis_right));

  // Grid vtk
  {
    using GMesh = GridMesh<T, Np_1d, Grid>;
    GMesh gmesh(grid);
    ToVTK<T, GMesh> grid_vtk((gmesh), std::filesystem::path(prefix) /
                                          std::filesystem::path("grid.vtk"));
    grid_vtk.write_mesh();
    grid_vtk.write_sol("lsf", mesh.get_lsf_dof().data());

    if (use_ersatz) {
      grid_vtk.write_vec("sol", sol.data());
    }
  }

  if (use_ersatz) {
    sol = grid_dof_to_cut_dof<T, spatial_dim, Mesh>(mesh, sol);
  }

  // Evaluate norm errors
  using EnergyNormPhysics =
      LinearElasticityEnergyNormError<T, spatial_dim,
                                      typeof(elasticity_stress_exact)>;
  using EnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, EnergyNormPhysics>;

  EnergyNormPhysics stress_norm_physics(E, nu, elasticity_stress_exact);
  EnergyNormAnalysis stress_norm_analysis(mesh, quadrature, basis,
                                          stress_norm_physics);

  json j = {
      {"stress_norm", sqrt(stress_norm_analysis.energy(nullptr, sol.data()))}};
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);

  // Cut vtk
  {
    ToVTK<T, Mesh> cut_vtk(
        mesh, std::filesystem::path(prefix) / std::filesystem::path("cut.vtk"));
    cut_vtk.write_mesh();

    std::vector<T> load_elems_v(mesh.get_num_elements(), 0.0);
    for (int e : load_elements_top) load_elems_v[e] = 1.0;
    for (int e : load_elements_right) load_elems_v[e] = 1.0;

    cut_vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
    cut_vtk.write_vec("sol", sol.data());
    cut_vtk.write_vec("rhs", physics_app->get_rhs().data());
    cut_vtk.write_cell_sol("loaded_elems", load_elems_v.data());
  }

  // evaluate stress at quadratures
  {
    using StrainStress = LinearElasticity2DStrainStress<T>;
    using StrainStressAnalysis =
        GalerkinAnalysis<T, Mesh, Quadrature, Basis, StrainStress>;

    StrainStress strain_stress(E, nu);
    StrainStressAnalysis strain_stress_analysis(mesh, quadrature, basis,
                                                strain_stress);

    std::vector<StrainStressType> types = {
        StrainStressType::sx, StrainStressType::sy, StrainStressType::sxy,
        StrainStressType::ex, StrainStressType::ey, StrainStressType::exy};
    std::vector<std::string> names = {"sx", "sy", "sxy", "ex", "ey", "exy"};

    FieldToVTKNew<T, spatial_dim> quad_vtk(std::filesystem::path(prefix) /
                                           std::filesystem::path("quad.vtk"));

    std::vector<T> xloc_q =
        strain_stress_analysis.interpolate_energy(sol.data()).first;
    quad_vtk.add_mesh(xloc_q);
    quad_vtk.write_mesh();

    // Energy norm at each quadrature points
    std::vector<T> stress_norm_q =
        stress_norm_analysis.interpolate_energy(sol.data()).second;
    quad_vtk.add_sol("stress_norm_err", stress_norm_q);
    quad_vtk.write_sol("stress_norm_err");

    for (int i = 0; i < 3; i++) {
      strain_stress.set_type(types[i]);
      quad_vtk.add_sol(
          names[i],
          strain_stress_analysis.interpolate_energy(sol.data()).second);
      quad_vtk.write_sol(names[i]);

      // Compute exact stress components
      int num_quads = xloc_q.size() / spatial_dim;
      std::vector<T> exact;
      for (int q = 0; q < num_quads; q++) {
        T xloc[spatial_dim] = {xloc_q[spatial_dim * q],
                               xloc_q[spatial_dim * q + 1]};
        exact.push_back(elasticity_stress_exact(xloc)[i]);
      }

      quad_vtk.add_sol(names[i] + "_exact", exact);
      quad_vtk.write_sol(names[i] + "_exact");
    }
  }
}

template <int Np_1d, bool use_finite_cell_mesh>
void execute_poisson(std::string prefix, int nxy, std::string instance,
                     bool save_vtk) {
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
      PoissonEnergyNormError<T, spatial_dim, typeof(poisson_exact_fun),
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

template <int Np_1d, bool use_finite_cell_mesh>
void execute(std::string physics, std::string prefix, int nxy,
             std::string instance, bool save_vtk, bool use_ersatz,
             double ersatz_ratio) {
  if (physics == "poisson") {
    execute_poisson<Np_1d, use_finite_cell_mesh>(prefix, nxy, instance,
                                                 save_vtk);
  } else {
    if (use_ersatz) {
      execute_elasticity<Np_1d, use_finite_cell_mesh, true>(prefix, nxy,
                                                            ersatz_ratio);
    } else {
      execute_elasticity<Np_1d, use_finite_cell_mesh, false>(prefix, nxy, 0.0);
    }
  }
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--nxy", 32);
  p.add_argument<std::string>("--physics", "elasticity",
                              {"poisson", "elasticity"});
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--instance", "square");
  p.add_argument<int>("--use-finite-cell-mesh", 0);
  p.add_argument<int>("--save-vtk", 1);
  p.add_argument<int>("--use-ersatz", 0);
  p.add_argument<double>("--ersatz-ratio", 1e-6);
  p.parse_args(argc, argv);

  std::string physics = p.get<std::string>("physics");
  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = physics + "_" + get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int nxy = p.get<int>("nxy");
  std::string instance = p.get<std::string>("instance");
  bool use_finite_cell_mesh = p.get<int>("use-finite-cell-mesh");
  bool save_vtk = p.get<int>("save-vtk");
  bool use_ersatz = p.get<int>("use-ersatz");
  double ersatz_ratio = p.get<double>("ersatz-ratio");

  switch (Np_1d) {
    case 2:
      if (use_finite_cell_mesh) {
        execute<2, true>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                         ersatz_ratio);
      } else {
        execute<2, false>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                          ersatz_ratio);
      }
      break;
    case 4:
      if (use_finite_cell_mesh) {
        execute<4, true>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                         ersatz_ratio);
      } else {
        execute<4, false>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                          ersatz_ratio);
      }
      break;
    case 6:
      if (use_finite_cell_mesh) {
        execute<6, true>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                         ersatz_ratio);
      } else {
        execute<6, false>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                          ersatz_ratio);
      }
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
