#include <stdexcept>
#include <uvector.hpp>

#include "analysis.h"
#include "apps/static_elastic.h"
#include "elements/gd_vandermonde.h"
#include "physics/stress.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/vtk.h"

template <int Np_1d>
void execute(ArgParser& p) {
  using T = double;
  using Grid = StructuredGrid2D<T>;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid>;
  using GridMesh = GridMesh<T, Np_1d, Grid>;
  using Mesh = CutMesh<T, Np_1d, Grid>;
  using Basis = GDBasis2D<T, Mesh>;

  json j = read_json(p.get<std::string>("image_json"));
  std::vector<double> lsf_dof = j["lsf_dof"];
  int nxy_val = j["nxy"];
  double lxy_val = 100.0;
  int nxy[2] = {nxy_val, nxy_val};
  double lxy[2] = {lxy_val, lxy_val};
  double h = lxy_val / int(j["nxy"]);
  Grid grid(nxy, lxy);

  GridMesh grid_mesh(grid);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  mesh.get_lsf_dof() = lsf_dof;
  mesh.update_mesh();

  constexpr static auto int_func =
      [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        A2D::Vec<T, Basis::spatial_dim> ret;
        // ret(-1) = -1.0;
        return ret;
      };
  constexpr static auto load_func =
      [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        A2D::Vec<T, Basis::spatial_dim> ret;
        ret(1) = -1.0;
        return ret;
      };
  using LoadPhysics =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>;
  using Elastic = StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_func)>;
  using LoadQuadrature =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;

  double E = 1.0, nu = 0.3;
  Elastic elastic(E, nu, mesh, quadrature, basis, int_func);

  LoadPhysics load_physics(load_func);
  std::set<int> load_elements;
  // load_elements = {/*1539, 1596,*/ 1646, 1687, 1723, 1753, 1781,
  //                  /*1806*/};
  for (int e = 0; e < mesh.get_num_elements(); e++) {
    double xloc[2];
    grid.get_cell_xloc(mesh.get_elem_cell(e), xloc);
    if (xloc[0] > lxy_val - 0.6 * h) {
      load_elements.insert(e);
    }
  }

  const auto& elastic_mesh = elastic.get_mesh();

  LoadQuadrature load_quadrature(elastic_mesh, load_elements);

  using LoadAnalysis =
      GalerkinAnalysis<T, Mesh, LoadQuadrature, Basis, LoadPhysics>;
  LoadAnalysis load_analysis(elastic_mesh, load_quadrature, elastic.get_basis(),
                             load_physics);

  /*
  std::vector<int> bc_nodes = {3066, 3065, 3067, 3068, 3069, 3070, 3072, 3071,
                               3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080};
  */

  // Obtain bc nodes
  std::vector<int> bc_nodes;
  for (int n = 0; n < mesh.get_num_nodes(); n++) {
    double xloc[2];
    mesh.get_node_xloc(n, xloc);
    if (xloc[1] > lxy_val - 0.1 * h) {
      bc_nodes.push_back(n);
    }
  }

  std::vector<int> bc_dof;
  for (int n : bc_nodes) {
    bc_dof.push_back(2 * n);
    bc_dof.push_back(2 * n + 1);
  }
  std::vector<T> sol =
      elastic.solve(bc_dof, std::vector<T>(bc_dof.size(), T(0.0)),
                    std::tuple<LoadAnalysis>(load_analysis));

  ToVTK<T, GridMesh> grid_vtk(grid_mesh, "analysis_verification_grid_Np_" +
                                             std::to_string(Np_1d) + ".vtk");

  grid_vtk.write_mesh();
  grid_vtk.write_sol("lsf", mesh.get_lsf_dof().data());

  ToVTK<T, Mesh> cut_vtk(
      mesh, "analysis_verification_cut_Np_" + std::to_string(Np_1d) + ".vtk");
  cut_vtk.write_mesh();
  cut_vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

  std::vector<double> bc_nodes_v(mesh.get_num_nodes(), 0.0);
  for (int i : bc_nodes) bc_nodes_v[i] = 1.0;
  cut_vtk.write_sol("bc_nodes", bc_nodes_v.data());

  cut_vtk.write_vec("displacement", sol.data());

  std::vector<double> load_elem_v(mesh.get_num_elements(), 0.0);
  for (int e : load_elements) load_elem_v[e] = 1.0;
  cut_vtk.write_cell_sol("loaded_elems", load_elem_v.data());

  using Stress = LinearElasticity2DVonMisesStress<T>;
  using StressAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Stress>;

  Stress stress(E, nu);
  StressAnalysis stress_analysis(mesh, quadrature, basis, stress);
  auto [xloc_q, stress_q] = stress_analysis.interpolate_energy(sol.data());
  auto [_, lsf_q] =
      stress_analysis.template interpolate<1>(mesh.get_lsf_nodes().data());

  FieldToVTKNew<T, 2> field_vtk("analysis_verification_quad_Np_" +
                                std::to_string(Np_1d) + ".vtk");
  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();
  field_vtk.add_sol("VonMises", stress_q);
  field_vtk.write_sol("VonMises");
  field_vtk.add_sol("lsf", lsf_q);
  field_vtk.write_sol("lsf");

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

  for (int i = 0; i < 6; i++) {
    strain_stress.set_type(types[i]);
    auto s = strain_stress_analysis.interpolate_energy(sol.data()).second;
    field_vtk.add_sol(names[i], s);
    field_vtk.write_sol(names[i]);
  }

  auto solq = elastic.get_analysis().interpolate(sol.data()).second;
  field_vtk.add_vec("displacement", solq);
  field_vtk.write_vec("displacement");

  using Sampler = GDSampler2D<T, 15, Mesh>;
  Sampler sampler(mesh, 0.0, 0.0);
  using SamplerVonMisesAnalysis =
      GalerkinAnalysis<T, Mesh, Sampler, Basis, Stress>;
  using SamplerStrainStressAnalysis =
      GalerkinAnalysis<T, Mesh, Sampler, Basis, StrainStress>;

  SamplerVonMisesAnalysis sampler_von_mises_analysis(mesh, sampler, basis,
                                                     stress);
  SamplerStrainStressAnalysis sampler_strain_stress_analysis(
      mesh, sampler, basis, strain_stress);

  auto [xloc_samples, sol_samples] =
      sampler_von_mises_analysis.template interpolate<2>(sol.data());

  FieldToVTKNew<T, 2> sampling_vtk("analysis_verification_sampling_Np_" +
                                   std::to_string(Np_1d) + ".vtk");

  sampling_vtk.add_mesh(xloc_samples);
  sampling_vtk.write_mesh();
  sampling_vtk.add_vec("displacement", sol_samples);
  sampling_vtk.write_vec("displacement");

  auto stress_samples =
      sampler_von_mises_analysis.interpolate_energy(sol.data()).second;
  sampling_vtk.add_sol("VonMises", stress_samples);
  sampling_vtk.write_sol("VonMises");

  for (int i = 0; i < 6; i++) {
    strain_stress.set_type(types[i]);
    auto sampling_s =
        sampler_strain_stress_analysis.interpolate_energy(sol.data()).second;
    sampling_vtk.add_sol(names[i], sampling_s);
    sampling_vtk.write_sol(names[i]);
  }
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<std::string>("--image_json", "lsf_dof.json");
  p.parse_args(argc, argv);

  int Np_1d = p.get<int>("Np_1d");

  switch (Np_1d) {
    case 2: {
      execute<2>(p);
      break;
    }
    case 4: {
      execute<4>(p);
      break;
    }
    case 6: {
      execute<6>(p);
      break;
    }
    case 8: {
      execute<8>(p);
      break;
    }
    default: {
      throw std::runtime_error("Np_1d = " + std::to_string(Np_1d) +
                               " is not pre-compiled");
    }
  }
}
