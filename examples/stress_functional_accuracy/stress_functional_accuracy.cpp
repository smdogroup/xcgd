#include "apps/nitsche.h"
#include "apps/poisson_app.h"
#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "physics/stress.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/vtk.h"

#define PI 3.14159265358979323846
#define k (1.2345 * PI)

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

template <typename T, class Mesh, class Quadrature, class Basis,
          class StressFunc>
void eval_bulk_stress(std::string vtk_path, T E, T nu, Mesh& mesh,
                      Quadrature& quadrature, Basis& basis, std::vector<T> sol,
                      StressFunc& stress_func) {
  int constexpr spatial_dim = Basis::spatial_dim;

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

  FieldToVTKNew<T, spatial_dim> quad_vtk(vtk_path);

  std::vector<T> xloc_q =
      strain_stress_analysis.interpolate_energy(sol.data()).first;
  quad_vtk.add_mesh(xloc_q);
  quad_vtk.write_mesh();

  // Evaluate norm errors
  using EnergyNormPhysics =
      LinearElasticityEnergyNormError<T, spatial_dim, StressFunc>;
  using EnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, EnergyNormPhysics>;

  EnergyNormPhysics stress_norm_physics(E, nu, stress_func);
  EnergyNormAnalysis stress_norm_analysis(mesh, quadrature, basis,
                                          stress_norm_physics);
  // Energy norm at each quadrature points
  std::vector<T> stress_norm_q =
      stress_norm_analysis.interpolate_energy(sol.data()).second;
  quad_vtk.add_sol("stress_norm_err", stress_norm_q);
  quad_vtk.write_sol("stress_norm_err");

  std::vector<int> I = {0, 1, 0};
  std::vector<int> J = {0, 1, 1};

  for (int i = 0; i < 3; i++) {
    strain_stress.set_type(types[i]);
    quad_vtk.add_sol(
        names[i], strain_stress_analysis.interpolate_energy(sol.data()).second);
    quad_vtk.write_sol(names[i]);

    // Compute exact stress components
    int num_quads = xloc_q.size() / spatial_dim;
    std::vector<T> exact;
    for (int q = 0; q < num_quads; q++) {
      T xloc[spatial_dim] = {xloc_q[spatial_dim * q],
                             xloc_q[spatial_dim * q + 1]};

      exact.push_back(stress_func(xloc)(I[i], J[i]));
    }

    quad_vtk.add_sol(names[i] + "_exact", exact);
    quad_vtk.write_sol(names[i] + "_exact");
  }
}

template <typename T, class Mesh, class Quadrature, class Basis,
          class StressFunc, class LSFGradFunc>
void eval_interface_stress(std::string vtk_path, T E, T nu, Mesh& mesh,
                           Quadrature& quadrature, Basis& basis,
                           std::vector<T> sol, StressFunc& stress_func,
                           LSFGradFunc& lsf_grad_func) {
  int constexpr spatial_dim = Basis::spatial_dim;
  using SurfQuadrature = typename Quadrature::InterfaceQuad;
  SurfQuadrature surf_quadrature(mesh);

  using SurfStress = LinearElasticity2DSurfStress<T>;
  using SurfStressAnalysis =
      GalerkinAnalysis<T, Mesh, SurfQuadrature, Basis, SurfStress>;

  SurfStress surf_stress(E, nu);
  SurfStressAnalysis surf_stress_analysis(mesh, surf_quadrature, basis,
                                          surf_stress);

  std::vector<SurfStressType> surf_stress_types = {SurfStressType::normal,
                                                   SurfStressType::tangent};
  std::vector<std::string> surf_stress_type_names = {"normal_stress",
                                                     "tangent_stress"};

  FieldToVTKNew<T, spatial_dim> surf_quad_vtk(vtk_path);

  std::vector<T> xloc_surf_q =
      surf_stress_analysis.interpolate_energy(sol.data()).first;
  surf_quad_vtk.add_mesh(xloc_surf_q);
  surf_quad_vtk.write_mesh();

  // Evaluate norm errors
  using EnergyNormPhysics =
      LinearElasticityEnergyNormError<T, spatial_dim, StressFunc>;
  using EnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, SurfQuadrature, Basis, EnergyNormPhysics>;

  EnergyNormPhysics stress_norm_physics(E, nu, stress_func);
  EnergyNormAnalysis stress_norm_analysis(mesh, surf_quadrature, basis,
                                          stress_norm_physics);
  // Energy norm at each quadrature points
  std::vector<T> stress_norm_q =
      stress_norm_analysis.interpolate_energy(sol.data()).second;
  surf_quad_vtk.add_sol("stress_norm_err", stress_norm_q);
  surf_quad_vtk.write_sol("stress_norm_err");

  // Get exact boundary Stress
  int num_quads = xloc_surf_q.size() / spatial_dim;
  std::map<std::string, std::vector<T>> exact = {{"normal_stress", {}},
                                                 {"tangent_stress", {}}};
  for (int q = 0; q < num_quads; q++) {
    T xloc[spatial_dim] = {xloc_surf_q[spatial_dim * q],
                           xloc_surf_q[spatial_dim * q + 1]};

    auto S = stress_func(xloc);
    std::vector<T> lsf_grad = lsf_grad_func(xloc);

    // Compute normal and tangent stress
    T nrm = sqrt(lsf_grad[0] * lsf_grad[0] + lsf_grad[1] * lsf_grad[1]);
    T Cos = lsf_grad[0] / nrm;
    T Sin = lsf_grad[1] / nrm;

    exact["normal_stress"].push_back(S(0, 0) * Cos * Cos + S(1, 1) * Sin * Sin +
                                     2.0 * S(0, 1) * Sin * Cos);
    exact["tangent_stress"].push_back((S(0, 0) - S(1, 1)) * Sin * Cos +
                                      S(0, 1) * (Sin * Sin - Cos * Cos));
  }

  for (int i = 0; i < 2; i++) {
    auto t = surf_stress_types[i];
    auto n = surf_stress_type_names[i];
    surf_stress.set_type(t);
    surf_quad_vtk.add_sol(n + "_exact", exact[n]);
    surf_quad_vtk.add_sol(
        n, surf_stress_analysis.interpolate_energy(sol.data()).second);

    surf_quad_vtk.write_sol(n + "_exact");
    surf_quad_vtk.write_sol(n);
  }
}

template <int Np_1d, bool use_finite_cell_mesh, bool use_ersatz = false>
void execute_bulk_elasticity(std::string prefix, int nxy, double ersatz_ratio) {
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

  auto elasticity_stress_exact_polar =
      [Tx, R](const A2D::Vec<T, spatial_dim>& xloc) {
        T theta = atan(xloc[1] / xloc[0]);
        T r = sqrt(xloc[0] * xloc[0] + xloc[1] * xloc[1]);

        T R2 = R * R;
        T r2 = r * r;
        T R4 = R * R * R * R;
        T r4 = r * r * r * r;

        T Sr =
            Tx / 2.0 * (1.0 - R2 / r2) +
            Tx / 2.0 * (1.0 + 3.0 * R4 / r4 - 4.0 * R2 / r2) * cos(2.0 * theta);
        T St = Tx / 2.0 * (1.0 + R2 / r2) -
               Tx / 2.0 * (1.0 + 3.0 * R4 / r4) * cos(2.0 * theta);
        T Srt = -Tx / 2.0 * (1.0 - 3.0 * R4 / r4 + 2.0 * R2 / r2) *
                sin(2.0 * theta);
        A2D::SymMat<T, spatial_dim> Stensor;
        Stensor(0, 0) = Sr;
        Stensor(1, 1) = St;
        Stensor(0, 1) = Srt;
        return Stensor;
      };

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
  // TODO: this code can be replaced by calling eval_bulk_stress()
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

    std::vector<int> I = {0, 1, 0};
    std::vector<int> J = {0, 1, 1};

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
        exact.push_back(elasticity_stress_exact(xloc)(I[i], J[i]));
      }

      quad_vtk.add_sol(names[i] + "_exact", exact);
      quad_vtk.write_sol(names[i] + "_exact");
    }
  }

  // Evaluate stress at interface
  {
    using SurfQuadrature =
        GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE, Grid, Np_1d, Mesh>;
    SurfQuadrature surf_quadrature(mesh);

    using SurfStress = LinearElasticity2DSurfStress<T>;
    using SurfStressAnalysis =
        GalerkinAnalysis<T, Mesh, SurfQuadrature, Basis, SurfStress>;

    SurfStress surf_stress(E, nu);
    SurfStressAnalysis surf_stress_analysis(mesh, surf_quadrature, basis,
                                            surf_stress);

    std::vector<SurfStressType> surf_stress_types = {SurfStressType::normal,
                                                     SurfStressType::tangent};
    std::vector<std::string> surf_stress_type_names = {"normal_stress",
                                                       "tangent_stress"};

    FieldToVTKNew<T, spatial_dim> surf_quad_vtk(
        std::filesystem::path(prefix) / std::filesystem::path("surf_quad.vtk"));

    std::vector<T> xloc_surf_q =
        surf_stress_analysis.interpolate_energy(sol.data()).first;
    surf_quad_vtk.add_mesh(xloc_surf_q);
    surf_quad_vtk.write_mesh();

    // Get exact boundary Stress
    int num_quads = xloc_surf_q.size() / spatial_dim;
    std::map<std::string, std::vector<T>> exact = {{"normal_stress", {}},
                                                   {"tangent_stress", {}}};
    for (int q = 0; q < num_quads; q++) {
      T xloc[spatial_dim] = {xloc_surf_q[spatial_dim * q],
                             xloc_surf_q[spatial_dim * q + 1]};

      auto Stensor = elasticity_stress_exact_polar(xloc);
      exact["normal_stress"].push_back(Stensor(0, 0));
      exact["tangent_stress"].push_back(Stensor(0, 1));
    }

    for (int i = 0; i < 2; i++) {
      auto t = surf_stress_types[i];
      auto n = surf_stress_type_names[i];
      surf_stress.set_type(t);
      surf_quad_vtk.add_sol(n + "_exact", exact[n]);
      surf_quad_vtk.add_sol(
          n, surf_stress_analysis.interpolate_energy(sol.data()).second);

      surf_quad_vtk.write_sol(n + "_exact");
      surf_quad_vtk.write_sol(n);
    }
  }
}

template <int Np_1d, bool use_finite_cell_mesh>
void execute_interface_elasticity(std::string prefix, int nxy,
                                  double nitsche_eta) {
  using T = double;
  using Grid = StructuredGrid2D<T>;

  using Mesh =
      typename std::conditional<use_finite_cell_mesh, FiniteCellMesh<T, Np_1d>,
                                CutMesh<T, Np_1d>>::type;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;
  using Basis = GDBasis2D<T, Mesh>;
  constexpr int spatial_dim = Basis::spatial_dim;

  auto elasticity_exact_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    A2D::Vec<T, Basis::spatial_dim> u;
    u(0) = sin(k * xloc(0)) * sin(k * xloc(1));
    u(1) = cos(k * xloc(0)) * cos(k * xloc(1));

    return u;
  };

  auto stress_general_fun = [](T E, T nu,
                               const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    constexpr int spatial_dim = Basis::spatial_dim;
    constexpr int dof_per_node = Basis::spatial_dim;

    T mu = 0.5 * E / (1.0 + nu);
    T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

    double k2 = k * k;

    A2D::Mat<T, dof_per_node, spatial_dim> grad;
    T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
    T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

    grad(0, 0) = ux;
    grad(0, 1) = uy;
    grad(1, 0) = vx;
    grad(1, 1) = vy;

    A2D::SymMat<T, spatial_dim> Etensor, Stensor;

    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, Etensor);
    A2D::SymIsotropic(mu, lambda, Etensor, Stensor);
    return Stensor;
  };

  auto intf_general_fun = [](T E, T nu,
                             const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    constexpr int spatial_dim = Basis::spatial_dim;
    constexpr int dof_per_node = Basis::spatial_dim;

    T mu = 0.5 * E / (1.0 + nu);
    T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

    double k2 = k * k;

    A2D::Mat<T, dof_per_node, spatial_dim> grad;
    T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
    T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

    grad(0, 0) = ux;
    grad(0, 1) = uy;
    grad(1, 0) = vx;
    grad(1, 1) = vy;

    T uxx = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T uxy = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T uyx = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T uyy = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vxx = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T vxy = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vyx = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vyy = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));

    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>> grad_obj(grad);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj));

    // Spartials(i, j) = ∂S(i, j)/∂x(j)
    A2D::Mat<T, dof_per_node, spatial_dim> Spartials;

    for (int i = 0; i < spatial_dim; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        grad_obj.bvalue().zero();
        E_obj.bvalue().zero();
        S_obj.bvalue().zero();
        S_obj.bvalue()(i, j) = 1.0;

        stack.reverse();

        // ∂S(i, j)/∂x(j) = ∂S(i, j)/∂grad * ∂grad/∂x(j)
        auto& bgrad = grad_obj.bvalue();

        if (j == 0) {
          Spartials(i, j) = bgrad(0, 0) * uxx + bgrad(0, 1) * uyx +
                            bgrad(1, 0) * vxx + bgrad(1, 1) * vyx;
        } else {
          Spartials(i, j) = bgrad(0, 0) * uxy + bgrad(0, 1) * uyy +
                            bgrad(1, 0) * vxy + bgrad(1, 1) * vyy;
        }
      }
    }

    A2D::Vec<T, dof_per_node> intf;
    intf(0) = -(Spartials(0, 0) + Spartials(0, 1));
    intf(1) = -(Spartials(1, 0) + Spartials(1, 1));

    return intf;
  };

  T L = 1.0;
  T R = 0.31;
  auto lsf_fun = [R, L](const T* x) {
    return R * R - (x[0] - 0.5 * L) * (x[0] - 0.5 * L) -
           (x[1] - 0.5 * L) * (x[1] - 0.5 * L);
  };
  auto lsf_grad_fun = [L](const T* x) {
    std::vector<T> ret = {-2.0 * (x[0] - 0.5 * L), -2.0 * (x[1] - 0.5 * L)};
    return ret;
  };

  auto surf_stress_general_fun =
      [lsf_grad_fun](T E, T nu, const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        constexpr int spatial_dim = Basis::spatial_dim;
        constexpr int dof_per_node = Basis::spatial_dim;

        T mu = 0.5 * E / (1.0 + nu);
        T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

        double k2 = k * k;

        A2D::Mat<T, dof_per_node, spatial_dim> grad;
        T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
        T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
        T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
        T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

        grad(0, 0) = ux;
        grad(0, 1) = uy;
        grad(1, 0) = vx;
        grad(1, 1) = vy;

        A2D::SymMat<T, spatial_dim> Etensor, Stensor;

        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, Etensor);
        A2D::SymIsotropic(mu, lambda, Etensor, Stensor);

        A2D::Vec<T, spatial_dim> S_surf;

        std::vector<T> lsf_grad = lsf_grad_fun(get_data(xloc));
        A2D::Vec<T, spatial_dim> nrm;
        nrm(0) = lsf_grad[0];
        nrm(1) = lsf_grad[1];

        A2D::VecNormalize(nrm, nrm);
        T Cos = nrm(0);
        T Sin = nrm(1);

        // Normal stress
        S_surf(0) = Stensor(0, 0) * Cos * Cos + Stensor(1, 1) * Sin * Sin +
                    2.0 * Stensor(0, 1) * Sin * Cos;

        // Tangent stress
        S_surf(1) = (Stensor(0, 0) - Stensor(1, 1)) * Sin * Cos +
                    Stensor(0, 1) * (Sin * Sin - Cos * Cos);

        return S_surf;
      };

  T E1 = 10.0, nu1 = 0.3;
  T E2 = 10.0, nu2 = 0.3;
  auto elasticity_intf_fun = [E1, nu1, E2, nu2, lsf_fun, intf_general_fun](
                                 const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    if (lsf_fun(xloc.get_data()) < 0.0) {
      return intf_general_fun(E1, nu1, xloc);
    } else {
      return intf_general_fun(E2, nu2, xloc);
    }
  };

  auto elasticity_stress_fun =
      [E1, nu1, E2, nu2, lsf_fun,
       stress_general_fun](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        if (lsf_fun(xloc.get_data()) < 0.0) {
          return stress_general_fun(E1, nu1, xloc);
        } else {
          return stress_general_fun(E2, nu2, xloc);
        }
      };

  auto stress_fun_l = [E1, nu1, stress_general_fun](
                          const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return stress_general_fun(E1, nu1, xloc);
  };

  auto stress_fun_r = [E2, nu2, stress_general_fun](
                          const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return stress_general_fun(E2, nu2, xloc);
  };

  auto surf_stress_fun_l = [E1, nu1, surf_stress_general_fun](
                               const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return surf_stress_general_fun(E1, nu1, xloc);
  };

  auto surf_stress_fun_r = [E2, nu2, surf_stress_general_fun](
                               const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    return surf_stress_general_fun(E2, nu2, xloc);
  };

  using PhysicsBulk =
      LinearElasticity<T, spatial_dim, typeof(elasticity_intf_fun)>;
  using PhysicsInterface = LinearElasticityInterface<T, spatial_dim>;

  constexpr int dof_per_node = PhysicsBulk::dof_per_node;

  using PhysicsApp = NitscheTwoSidedApp<T, Mesh, Quadrature, Basis, PhysicsBulk,
                                        PhysicsInterface>;

  int nx_ny[2] = {nxy, nxy};
  T dh = L / nxy;
  T lxy[2] = {L, L};
  Grid grid(nx_ny, lxy);

  Mesh mesh_primary(grid, lsf_fun);

  Quadrature quadrature(mesh_primary);
  Basis basis(mesh_primary);

  std::shared_ptr<PhysicsApp> physics_app;
  PhysicsBulk physics_bulk_primary(E1, nu1, elasticity_intf_fun);
  PhysicsBulk physics_bulk_secondary(E2, nu2, elasticity_intf_fun);
  PhysicsInterface physics_interface(nitsche_eta, E1, nu1, E2, nu2);

  physics_app = std::make_shared<PhysicsApp>(
      mesh_primary, quadrature, basis, physics_bulk_primary,
      physics_bulk_secondary, physics_interface);

  // Set bcs
  std::vector<int> dof_bcs;
  std::vector<T> dof_vals;

  for (auto nodes : {mesh_primary.get_left_boundary_nodes(),
                     mesh_primary.get_right_boundary_nodes(),
                     mesh_primary.get_upper_boundary_nodes(),
                     mesh_primary.get_lower_boundary_nodes()}) {
    for (int node : nodes) {
      T xloc[Basis::spatial_dim];
      mesh_primary.get_node_xloc(node, xloc);
      auto vals = elasticity_exact_fun(A2D::Vec<T, Basis::spatial_dim>(xloc));
      for (int d = 0; d < Basis::spatial_dim; d++) {
        dof_bcs.push_back(Basis::spatial_dim * node + d);
        dof_vals.push_back(vals(d));
      }
    }
  }

  std::vector<T> sol = physics_app->solve(dof_bcs, dof_vals);

  // Split sol
  auto& mesh_secondary = physics_app->get_secondary_mesh();

  std::vector<T> sol_primary(
      sol.begin(), sol.begin() + dof_per_node * mesh_primary.get_num_nodes());
  std::vector<T> sol_secondary(
      sol.begin() + dof_per_node * mesh_primary.get_num_nodes(), sol.end());
  xcgd_assert(
      sol_secondary.size() == dof_per_node * mesh_secondary.get_num_nodes(),
      "dimension of secondary sol does not match number of secondary nodes");

  // Cut mesh
  {
    ToVTK<T, typeof(mesh_primary)> primary_cut_vtk(
        mesh_primary, std::filesystem::path(prefix) /
                          std::filesystem::path("cut_primary.vtk"));
    ToVTK<T, typeof(mesh_secondary)> secondary_cut_vtk(
        mesh_secondary, std::filesystem::path(prefix) /
                            std::filesystem::path("cut_secondary.vtk"));

    std::vector<T> sol_exact_primary;
    for (int i = 0; i < mesh_primary.get_num_nodes(); i++) {
      T xloc[Basis::spatial_dim];
      mesh_primary.get_node_xloc(i, xloc);
      auto vals = elasticity_exact_fun(A2D::Vec<T, Basis::spatial_dim>(xloc));
      for (int d = 0; d < Basis::spatial_dim; d++) {
        sol_exact_primary.push_back(vals(d));
      }
    }

    primary_cut_vtk.write_mesh();
    primary_cut_vtk.write_sol("lsf", mesh_primary.get_lsf_nodes().data());
    primary_cut_vtk.write_vec("sol", sol_primary.data());
    primary_cut_vtk.write_vec("sol_exact", sol_exact_primary.data());

    std::vector<T> sol_exact_secondary;
    for (int i = 0; i < mesh_secondary.get_num_nodes(); i++) {
      T xloc[Basis::spatial_dim];
      mesh_secondary.get_node_xloc(i, xloc);
      auto vals = elasticity_exact_fun(A2D::Vec<T, Basis::spatial_dim>(xloc));
      for (int d = 0; d < Basis::spatial_dim; d++) {
        sol_exact_secondary.push_back(vals(d));
      }
    }

    secondary_cut_vtk.write_mesh();
    secondary_cut_vtk.write_sol("lsf", mesh_secondary.get_lsf_nodes().data());
    secondary_cut_vtk.write_vec("sol", sol_secondary.data());
    secondary_cut_vtk.write_vec("sol_exact", sol_exact_secondary.data());
  }

  // Stencils
  {
    StencilToVTK<T, Mesh> primary_stencil_vtk(
        mesh_primary, std::filesystem::path(prefix) /
                          std::filesystem::path("stencils_primary.vtk"));
    primary_stencil_vtk.write_stencils(mesh_primary.get_elem_nodes());

    StencilToVTK<T, Mesh> secondary_stencil_vtk(
        mesh_secondary, std::filesystem::path(prefix) /
                            std::filesystem::path("stencils_secondary.vtk"));
    secondary_stencil_vtk.write_stencils(mesh_secondary.get_elem_nodes());
  }

  // Evaluate norm errors
  using EnergyNormPhysics =
      LinearElasticityEnergyNormError<T, spatial_dim,
                                      typeof(elasticity_stress_fun)>;
  using EnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, EnergyNormPhysics>;

  using InterfaceEnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, typeof(physics_app->get_interface_quadrature()),
                       Basis, EnergyNormPhysics>;

  EnergyNormPhysics stress_norm_physics_l(E1, nu1, elasticity_stress_fun);
  EnergyNormPhysics stress_norm_physics_r(E2, nu2, elasticity_stress_fun);

  EnergyNormAnalysis stress_norm_analysis_l(
      physics_app->get_primary_mesh(),
      physics_app->get_primary_bulk_quadrature(),
      physics_app->get_primary_basis(), stress_norm_physics_l);

  EnergyNormAnalysis stress_norm_analysis_r(
      physics_app->get_secondary_mesh(),
      physics_app->get_secondary_bulk_quadrature(),
      physics_app->get_secondary_basis(), stress_norm_physics_r);

  InterfaceEnergyNormAnalysis stress_norm_analysis_l_interface(
      physics_app->get_primary_mesh(), physics_app->get_interface_quadrature(),
      physics_app->get_primary_basis(), stress_norm_physics_l);

  InterfaceEnergyNormAnalysis stress_norm_analysis_r_interface(
      physics_app->get_secondary_mesh(),
      physics_app->get_interface_quadrature(),
      physics_app->get_secondary_basis(), stress_norm_physics_r);

  json j = {{"stress_norm_primary",
             sqrt(stress_norm_analysis_l.energy(nullptr, sol_primary.data()))},
            {"stress_norm_secondary", sqrt(stress_norm_analysis_r.energy(
                                          nullptr, sol_secondary.data()))},
            {"stress_norm_primary_interface",
             sqrt(stress_norm_analysis_l_interface.energy(nullptr,
                                                          sol_primary.data()))},

            {"stress_norm_secondary_interface",
             sqrt(stress_norm_analysis_r_interface.energy(
                 nullptr, sol_secondary.data()))}};
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);

  // Evaluate stress at quadratures
  {
    // left mesh
    eval_bulk_stress(
        std::filesystem::path(prefix) /
            std::filesystem::path("quad_primary.vtk"),
        E1, nu1, mesh_primary, physics_app->get_primary_bulk_quadrature(),
        physics_app->get_primary_basis(), sol_primary, stress_fun_l);

    // right mesh
    eval_bulk_stress(
        std::filesystem::path(prefix) /
            std::filesystem::path("quad_secondary.vtk"),
        E2, nu2, mesh_secondary, physics_app->get_secondary_bulk_quadrature(),
        physics_app->get_secondary_basis(), sol_secondary, stress_fun_r);
  }

  // Evaluate stress at interface
  {
    eval_interface_stress(std::filesystem::path(prefix) /
                              std::filesystem::path("interface_primary.vtk"),
                          E1, nu1, mesh_primary,
                          physics_app->get_interface_quadrature(),
                          physics_app->get_primary_basis(), sol_primary,
                          stress_fun_l, lsf_grad_fun);

    eval_interface_stress(std::filesystem::path(prefix) /
                              std::filesystem::path("interface_secondary.vtk"),
                          E2, nu2, mesh_secondary,
                          physics_app->get_interface_quadrature(),
                          physics_app->get_secondary_basis(), sol_secondary,
                          stress_fun_r, lsf_grad_fun);
  }
}

template <int Np_1d, bool use_finite_cell_mesh>
void execute_mms(std::string prefix, int nxy, std::string physics,
                 std::string instance, double nitsche_eta, bool save_vtk) {
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

  auto elasticity_exact_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    A2D::Vec<T, Basis::spatial_dim> u;
    u(0) = sin(k * xloc(0)) * sin(k * xloc(1));
    u(1) = cos(k * xloc(0)) * cos(k * xloc(1));

    return u;
  };

  T E = 100.0, nu = 0.3;
  auto elasticity_stress_fun =
      [E, nu](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        constexpr int spatial_dim = Basis::spatial_dim;
        constexpr int dof_per_node = Basis::spatial_dim;

        T mu = 0.5 * E / (1.0 + nu);
        T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

        double k2 = k * k;

        A2D::Mat<T, dof_per_node, spatial_dim> grad;
        T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
        T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
        T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
        T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

        grad(0, 0) = ux;
        grad(0, 1) = uy;
        grad(1, 0) = vx;
        grad(1, 1) = vy;

        A2D::SymMat<T, spatial_dim> Etensor, Stensor;

        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, Etensor);
        A2D::SymIsotropic(mu, lambda, Etensor, Stensor);
        return Stensor;
      };

  auto elasticity_intf_fun = [E,
                              nu](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    constexpr int spatial_dim = Basis::spatial_dim;
    constexpr int dof_per_node = Basis::spatial_dim;

    T mu = 0.5 * E / (1.0 + nu);
    T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

    double k2 = k * k;

    A2D::Mat<T, dof_per_node, spatial_dim> grad;
    T ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
    T uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
    T vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

    grad(0, 0) = ux;
    grad(0, 1) = uy;
    grad(1, 0) = vx;
    grad(1, 1) = vy;

    T uxx = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T uxy = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T uyx = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T uyy = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vxx = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));
    T vxy = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vyx = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
    T vyy = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));

    A2D::ADObj<A2D::Mat<T, dof_per_node, spatial_dim>> grad_obj(grad);
    A2D::ADObj<A2D::SymMat<T, spatial_dim>> E_obj, S_obj;

    auto stack = A2D::MakeStack(
        A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
        A2D::SymIsotropic(mu, lambda, E_obj, S_obj));

    // Spartials(i, j) = ∂S(i, j)/∂x(j)
    A2D::Mat<T, dof_per_node, spatial_dim> Spartials;

    for (int i = 0; i < spatial_dim; i++) {
      for (int j = 0; j < spatial_dim; j++) {
        grad_obj.bvalue().zero();
        E_obj.bvalue().zero();
        S_obj.bvalue().zero();
        S_obj.bvalue()(i, j) = 1.0;

        stack.reverse();

        // ∂S(i, j)/∂x(j) = ∂S(i, j)/∂grad * ∂grad/∂x(j)
        auto& bgrad = grad_obj.bvalue();

        if (j == 0) {
          Spartials(i, j) = bgrad(0, 0) * uxx + bgrad(0, 1) * uyx +
                            bgrad(1, 0) * vxx + bgrad(1, 1) * vyx;
        } else {
          Spartials(i, j) = bgrad(0, 0) * uxy + bgrad(0, 1) * uyy +
                            bgrad(1, 0) * vxy + bgrad(1, 1) * vyy;
        }
      }
    }

    A2D::Vec<T, dof_per_node> intf;
    intf(0) = -(Spartials(0, 0) + Spartials(0, 1));
    intf(1) = -(Spartials(1, 0) + Spartials(1, 1));

    return intf;
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

  using PoissonBulk =
      PoissonPhysics<T, Basis::spatial_dim, typeof(poisson_source_fun)>;
  using PoissonBCs =
      PoissonCutDirichlet<T, Basis::spatial_dim, typeof(poisson_exact_fun)>;

  using ElasticityBulk =
      LinearElasticity<T, spatial_dim, typeof(elasticity_intf_fun)>;

  using ElasticityBCs =
      LinearElasticityCutDirichlet<T, spatial_dim, spatial_dim,
                                   typeof(elasticity_exact_fun)>;

  PoissonBulk poisson_bulk(poisson_source_fun);
  PoissonBCs poisson_bcs(nitsche_eta, poisson_exact_fun);

  ElasticityBulk elasticity_bulk(E, nu, elasticity_intf_fun);
  ElasticityBCs elasticity_bcs(nitsche_eta, elasticity_exact_fun);

  PoissonApp<T, Mesh, Quadrature, Basis, typeof(poisson_source_fun)>
      poisson_app(*mesh, quadrature, basis, poisson_source_fun);

  NitscheBCsApp<T, Mesh, Quadrature, Basis, PoissonBulk, PoissonBCs>
      poisson_nitsche_app(*mesh, quadrature, basis, poisson_bulk, poisson_bcs);

  StaticElastic<T, Mesh, Quadrature, Basis, typeof(elasticity_intf_fun)>
      elasticity_app(E, nu, *mesh, quadrature, basis, elasticity_intf_fun);

  NitscheBCsApp<T, Mesh, Quadrature, Basis, ElasticityBulk, ElasticityBCs>
      elasticity_nitsche_app(*mesh, quadrature, basis, elasticity_bulk,
                             elasticity_bcs);

  // Solve
  std::vector<T> sol;

  if (instance == "square") {
    if (physics == "elasticity-mms") {
      throw std::runtime_error(
          "instance == \"square\" is not implemented for physics == "
          "elasticity-mms");
    }
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
    if (physics == "poisson") {
      sol = poisson_nitsche_app.solve();
    } else {
      sol = elasticity_nitsche_app.solve();
    }
  }

  // Evaluate norm errors
  using PoissonEnergyNormPhysics =
      PoissonEnergyNormError<T, spatial_dim, typeof(poisson_exact_fun),
                             typeof(poisson_stress_fun)>;

  using ElasticityEnergyNormPhysics =
      LinearElasticityEnergyNormError<T, spatial_dim,
                                      typeof(elasticity_stress_fun)>;
  using PoissonEnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, PoissonEnergyNormPhysics>;

  using ElasticityEnergyNormAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, ElasticityEnergyNormPhysics>;

  PoissonEnergyNormPhysics poisson_val_norm_physics(
      poisson_exact_fun, poisson_stress_fun, 1.0, 0.0);
  PoissonEnergyNormPhysics poisson_stress_norm_physics(
      poisson_exact_fun, poisson_stress_fun, 0.0, 1.0);
  PoissonEnergyNormPhysics poisson_energy_norm_physics(
      poisson_exact_fun, poisson_stress_fun, 1.0, 1.0);

  ElasticityEnergyNormPhysics elasticity_stress_norm_physics(
      E, nu, elasticity_stress_fun);

  PoissonEnergyNormAnalysis poisson_val_norm_analysis(*mesh, quadrature, basis,
                                                      poisson_val_norm_physics);
  PoissonEnergyNormAnalysis poisson_stress_norm_analysis(
      *mesh, quadrature, basis, poisson_stress_norm_physics);
  PoissonEnergyNormAnalysis poisson_energy_norm_analysis(
      *mesh, quadrature, basis, poisson_energy_norm_physics);

  ElasticityEnergyNormAnalysis elasticity_stress_norm_analysis(
      *mesh, quadrature, basis, elasticity_stress_norm_physics);

  json j;

  if (physics == "poisson") {
    j = {{"val_norm",
          sqrt(poisson_val_norm_analysis.energy(nullptr, sol.data()))},
         {"stress_norm",
          sqrt(poisson_stress_norm_analysis.energy(nullptr, sol.data()))},
         {"energy_norm",
          sqrt(poisson_energy_norm_analysis.energy(nullptr, sol.data()))}};
  } else {
    j = {{"stress_norm",
          sqrt(elasticity_stress_norm_analysis.energy(nullptr, sol.data()))}};
  }

  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);

  // VTK
  if (save_vtk) {
    ToVTK<T, Mesh> vtk(*mesh, std::filesystem::path(prefix) /
                                  std::filesystem::path("cut.vtk"));
    vtk.write_mesh();
    vtk.write_sol("u", sol.data());
    vtk.write_sol("lsf", mesh->get_lsf_nodes().data());

    FieldToVTKNew<T, spatial_dim> quad_vtk(std::filesystem::path(prefix) /
                                           std::filesystem::path("quad.vtk"));

    if (physics == "poisson") {
      std::vector<T> xloc_samples =
          poisson_val_norm_analysis.interpolate_energy(sol.data()).first;
      quad_vtk.add_mesh(xloc_samples);
      quad_vtk.write_mesh();

      std::vector<T> val_norm_samples =
          poisson_val_norm_analysis.interpolate_energy(sol.data()).second;
      std::vector<T> stress_norm_samples =
          poisson_stress_norm_analysis.interpolate_energy(sol.data()).second;
      std::vector<T> energy_norm_samples =
          poisson_energy_norm_analysis.interpolate_energy(sol.data()).second;

      quad_vtk.add_sol("val_norm", val_norm_samples);
      quad_vtk.write_sol("val_norm");

      quad_vtk.add_sol("stress_norm", stress_norm_samples);
      quad_vtk.write_sol("stress_norm");

      quad_vtk.add_sol("energy_norm", energy_norm_samples);
      quad_vtk.write_sol("energy_norm");
    } else {
      std::vector<T> xloc_samples =
          elasticity_stress_norm_analysis.interpolate_energy(sol.data()).first;
      quad_vtk.add_mesh(xloc_samples);
      quad_vtk.write_mesh();

      std::vector<T> stress_norm_samples =
          elasticity_stress_norm_analysis.interpolate_energy(sol.data()).second;

      quad_vtk.add_sol("stress_norm", stress_norm_samples);
      quad_vtk.write_sol("stress_norm");
    }
  }
}

template <int Np_1d, bool use_finite_cell_mesh>
void execute(std::string physics, std::string prefix, int nxy,
             std::string instance, bool save_vtk, bool use_ersatz,
             double ersatz_ratio, double nitsche_eta) {
  if (physics == "poisson" or physics == "elasticity-mms") {
    execute_mms<Np_1d, use_finite_cell_mesh>(prefix, nxy, physics, instance,
                                             nitsche_eta, save_vtk);
  } else if (physics == "elasticity-bulk") {
    xcgd_assert(instance == "circle",
                "instance must be circle for elasticity-bulk physics");
    if (use_ersatz) {
      execute_bulk_elasticity<Np_1d, use_finite_cell_mesh, true>(prefix, nxy,
                                                                 ersatz_ratio);
    } else {
      execute_bulk_elasticity<Np_1d, use_finite_cell_mesh, false>(prefix, nxy,
                                                                  0.0);
    }
  } else {  // physics == "elasticity-interface"
    xcgd_assert(instance == "circle",
                "instance must be circle for elasticity-interface physics");
    execute_interface_elasticity<Np_1d, use_finite_cell_mesh>(prefix, nxy,
                                                              nitsche_eta);
  }
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--nxy", 32);
  p.add_argument<std::string>(
      "--physics", "elasticity-mms",
      {"poisson", "elasticity-mms", "elasticity-bulk", "elasticity-interface"});
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--instance", "square", {"square", "circle"});
  p.add_argument<int>("--use-finite-cell-mesh", 0, {0, 1});
  p.add_argument<int>("--save-vtk", 1, {0, 1});
  p.add_argument<int>("--use-ersatz", 0, {0, 1});
  p.add_argument<double>("--ersatz-ratio", 1e-6);
  p.add_argument<double>("--nitsche-eta", 1e8);
  p.parse_args(argc, argv);

  std::string physics = p.get<std::string>("physics");
  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time() + "_" + physics;
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
  double nitsche_eta = p.get<double>("nitsche-eta");

  switch (Np_1d) {
    case 2:
      if (use_finite_cell_mesh) {
        execute<2, true>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                         ersatz_ratio, nitsche_eta);
      } else {
        execute<2, false>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                          ersatz_ratio, nitsche_eta);
      }
      break;
    case 4:
      if (use_finite_cell_mesh) {
        execute<4, true>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                         ersatz_ratio, nitsche_eta);
      } else {
        execute<4, false>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                          ersatz_ratio, nitsche_eta);
      }
      break;
    case 6:
      if (use_finite_cell_mesh) {
        execute<6, true>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                         ersatz_ratio, nitsche_eta);
      } else {
        execute<6, false>(physics, prefix, nxy, instance, save_vtk, use_ersatz,
                          ersatz_ratio, nitsche_eta);
      }
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
      break;
  }

  return 0;
}
