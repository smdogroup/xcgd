#include <stdexcept>
#include <uvector.hpp>

#include "analysis.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/stress.h"
#include "physics/volume.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/vtk.h"

int main(int argc, char *argv[]) {
  constexpr int Np_1d = 2;

  using T = double;
  using Grid = StructuredGrid2D<T>;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid>;
  using GridMesh = GridMesh<T, Np_1d, Grid>;
  using Mesh = CutMesh<T, Np_1d, Grid>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {1, 1};
  double lxy[2] = {1, 1};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  mesh.get_lsf_dof() = {-1.70609, 0.526867, 0.863312, 3.77184};
  mesh.update_mesh();

  std::vector<T> sol = {-311.808, -222.771, -310.697, -239.173,
                        -298.274, -239.471, -300.82,  -220.332};

  ToVTK<T, Mesh> cut_vtk(
      mesh, "single_element_study_cut_Np_" + std::to_string(Np_1d) + ".vtk");
  cut_vtk.write_mesh();
  cut_vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
  cut_vtk.write_vec("displacement", sol.data());

  using Stress = LinearElasticity2DVonMisesStress<T>;
  using StressAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Stress>;

  double E = 1.0, nu = 0.3;
  Stress stress(E, nu);
  StressAnalysis stress_analysis(mesh, quadrature, basis, stress);
  auto [xloc_q, stress_q] = stress_analysis.interpolate_energy(sol.data());

  FieldToVTKNew<T, 2> field_vtk("single_element_study_quad_Np_" +
                                std::to_string(Np_1d) + ".vtk");
  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();
  field_vtk.add_sol("VonMises", stress_q);
  field_vtk.write_sol("VonMises");

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
    auto [_, s] = strain_stress_analysis.interpolate_energy(sol.data());
    field_vtk.add_sol(names[i], s);
    field_vtk.write_sol(names[i]);
  }

  auto [_, solq] = stress_analysis.interpolate(sol.data());
  field_vtk.add_vec("displacement", solq);
  field_vtk.write_vec("displacement");

  using Sampler = GDSampler2D<T, 50, Mesh>;
  Sampler sampler(mesh, 0.0, 0.0);
  using SamplerAnalysis = GalerkinAnalysis<T, Mesh, Sampler, Basis, Stress>;

  SamplerAnalysis sampler_analysis(mesh, sampler, basis, stress);

  auto [xloc_samples, sol_samples] =
      sampler_analysis.interpolate<2>(sol.data());

  FieldToVTKNew<T, 2> sampling_vtk("single_element_study_sampling_Np_" +
                                   std::to_string(Np_1d) + ".vtk");

  sampling_vtk.add_mesh(xloc_samples);
  sampling_vtk.write_mesh();
  sampling_vtk.add_vec("displacement", sol_samples);
  sampling_vtk.write_vec("displacement");

  auto [__, stress_samples] = sampler_analysis.interpolate_energy(sol.data());
  sampling_vtk.add_sol("VonMises", stress_samples);
  sampling_vtk.write_sol("VonMises");

  return 0;
}
