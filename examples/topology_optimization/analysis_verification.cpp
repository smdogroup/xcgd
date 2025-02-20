#include <filesystem>
#include <stdexcept>
#include <uvector.hpp>

#include "analysis.h"
#include "apps/static_elastic.h"
#include "elements/gd_vandermonde.h"
#include "physics/stress.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/linalg.h"
#include "utils/timer.h"
#include "utils/vtk.h"

template <int Np_1d>
void execute(ArgParser& p) {
  using T = double;
  using Grid = StructuredGrid2D<T>;

  constexpr int spatial_dim = Grid::spatial_dim;

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

  // Perform the SPR on stress components
  {
    StopWatch watch;
    constexpr int nsamples_per_elem_1d = Np_1d - 1;
    constexpr int Amat_dim = Np_1d * Np_1d;  // for 2d problem
    using SPRSampler = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid,
                                         nsamples_per_elem_1d>;
    SPRSampler spr_sampler(mesh);

    using SPRAnalysis =
        GalerkinAnalysis<T, Mesh, SPRSampler, Basis, StrainStress>;
    SPRAnalysis spr_analysis(mesh, spr_sampler, basis, strain_stress);

    // Evaluate stress components on all sampling points
    // Note: this implementation relies on a stable, consistency order of
    // quadrature point query
    std::map<int, std::vector<T>> samples_xloc_map =
        spr_analysis.interpolate_energy_map(sol.data()).first;

    std::map<StrainStressType, std::map<int, std::vector<T>>> samples_scomp_map;
    std::vector<StrainStressType> stress_comps = {
        StrainStressType::sx, StrainStressType::sy, StrainStressType::sxy};

    for (auto scomp : stress_comps) {
      strain_stress.set_type(scomp);
      samples_scomp_map[scomp] =
          spr_analysis.interpolate_energy_map(sol.data()).second;
    }

    // Allocate recovered nodal stress values
    std::vector<T> sx_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<T> sy_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<T> sxy_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<int> num_recovered(mesh.get_num_nodes(), 0);

    // Debug
    std::vector<T> debug_one_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<T> debug_A_cond(mesh.get_num_nodes(), -1.0);

    // Loop over all patch assemply nodes
    for (int n = 0; n < mesh.get_num_nodes(); n++) {
      // Get patch elements
      std::set<int> patch_elems = mesh.get_node_patch_elems(n);

      // Skip the patch assembly node if it doesn't have enough patch elements
      int nsamples_estimate =
          patch_elems.size() * nsamples_per_elem_1d * nsamples_per_elem_1d;
      if (nsamples_estimate < Amat_dim) {
        /*
        std::printf("node %4d doesn't have enough patch elements (%d <%d)\n", n,
                    nsamples_estimate, Amat_dim);
        */
        continue;
      }

      // Get the dimensions bounding box for the element cell
      T patch_xloc_min[spatial_dim], patch_xloc_max[spatial_dim];
      mesh.get_node_patch_elems_node_ranges(n, patch_xloc_min, patch_xloc_max);
      T patch_dh[spatial_dim];
      for (int d = 0; d < spatial_dim; d++) {
        patch_dh[d] = patch_xloc_max[d] - patch_xloc_min[d];
      }

      // Get number of coordinates and values of sample points in all patch
      // elements of this assembly node
      std::vector<T> xloc_p;  // sample points in the patch frame
      std::vector<T> sx_p, sy_p, sxy_p;
      std::vector<T> debug_one_p;
      // Collect sample points from patch elements
      for (int e : patch_elems) {
        const std::vector<T>& xloc = samples_xloc_map[e];
        const std::vector<T>& sx = samples_scomp_map[StrainStressType::sx][e];
        const std::vector<T>& sy = samples_scomp_map[StrainStressType::sy][e];
        const std::vector<T>& sxy = samples_scomp_map[StrainStressType::sxy][e];

        int npts = xloc.size() / spatial_dim;

        // Convert sample points in patch frame
        for (int q = 0; q < npts; q++) {
          for (int d = 0; d < spatial_dim; d++) {
            int index = q * spatial_dim + d;
            xloc_p.push_back(
                (xloc[index] - patch_xloc_min[d]) / patch_dh[d] * 2.0 - 1.0);
          }

          sx_p.push_back(sx[q]);
          sy_p.push_back(sy[q]);
          sxy_p.push_back(sxy[q]);
          // debug_one_p.push_back(xloc[q * spatial_dim + 1]);
          debug_one_p.push_back(1.0);
        }
      }

      // Get number of sample points in all patch elements of this assembly node
      int nsamples = xloc_p.size() / spatial_dim;

      // Assemble the least-square system
      std::vector<T> Amat(Amat_dim * Amat_dim,
                          0.0);  // matrix entries stored column by column
      for (int s = 0; s < nsamples; s++) {
        // Get P entries for a sample point
        T x = xloc_p[spatial_dim * s];
        T y = xloc_p[spatial_dim * s + 1];
        std::array<T, Np_1d> xpows, ypows;
        for (int t = 0; t < Np_1d; t++) {
          xpows[t] = pow(x, t);
          ypows[t] = pow(y, t);
        }

        // Add PPT to A
        // for 2d problem, PT = [1, x, x^2, ..., y, yx, yx^2, ...]
        for (int col = 0; col < Amat_dim; col++) {
          int P_ix = col % Np_1d;
          int P_iy = col / Np_1d;
          for (int row = 0; row < Amat_dim; row++) {
            int PT_ix = row % Np_1d;
            int PT_iy = row / Np_1d;
            Amat[col * Amat_dim + row] +=
                xpows[P_ix] * ypows[P_iy] * xpows[PT_ix] * ypows[PT_iy];
          }
        }
      }

      // Debug: Save A
      bool debug = false;
      if (debug) {
        if (!std::filesystem::is_directory("debug")) {
          std::filesystem::create_directory("debug");
        }

        FILE* fp = std::fopen(
            (std::filesystem::path("debug") /
             std::filesystem::path("node_" + std::to_string(n) + ".mat"))
                .c_str(),
            "w+");

        for (auto val : Amat) {
          std::fprintf(fp, "%20.10e\n", val);
        }
        std::fclose(fp);
      }

      // Factor A
      DirectSolve<T> Afact(Amat_dim, Amat.data());

      // Debug: get condition number
      debug_A_cond[n] = Afact.cond();

      // rhs for the SPR least square linear system at assemply node n
      std::vector<T> bx(Amat_dim, 0.0), by(Amat_dim, 0.0), bxy(Amat_dim, 0.0),
          debug_bone(Amat_dim, 0.0);

      // Evaluate stress components on sampling points
      for (int s = 0; s < nsamples; s++) {
        // Get P entries for a sample point
        T x = xloc_p[spatial_dim * s];
        T y = xloc_p[spatial_dim * s + 1];
        std::array<T, Np_1d> xpows, ypows;
        for (int t = 0; t < Np_1d; t++) {
          xpows[t] = pow(x, t);
          ypows[t] = pow(y, t);
        }

        for (int row = 0; row < Amat_dim; row++) {
          int PT_ix = row % Np_1d;
          int PT_iy = row / Np_1d;
          T pv = xpows[PT_ix] * ypows[PT_iy];
          bx[row] += sx_p[s] * pv;
          by[row] += sy_p[s] * pv;
          bxy[row] += sxy_p[s] * pv;
          debug_bone[row] += debug_one_p[s] * pv;
        }
      }

      // Apply invA to stress components
      Afact.apply(bx.data());
      Afact.apply(by.data());
      Afact.apply(bxy.data());
      Afact.apply(debug_bone.data());

      // Get all nodes we add value to
      std::set<int> patch_nodes;
      for (int e : patch_elems) {
        int nodes[Mesh::corner_nodes_per_element];
        mesh.get_elem_corner_nodes(e, nodes);
        for (int i = 0; i < Mesh::corner_nodes_per_element; i++) {
          patch_nodes.insert(nodes[i]);
        }
      }

      // Add nodal contributions to nodal recovered values
      for (int pn : patch_nodes) {
        // Get nodal coordinates in the patch frame
        T pn_xloc[spatial_dim];
        mesh.get_node_xloc(pn, pn_xloc);
        T x = (pn_xloc[0] - patch_xloc_min[0]) / patch_dh[0] * 2.0 - 1.0;
        T y = (pn_xloc[1] - patch_xloc_min[1]) / patch_dh[1] * 2.0 - 1.0;
        std::array<T, Np_1d> xpows, ypows;
        for (int t = 0; t < Np_1d; t++) {
          xpows[t] = pow(x, t);
          ypows[t] = pow(y, t);
        }

        // val_recovered = PTa
        for (int row = 0; row < Amat_dim; row++) {
          int PT_ix = row % Np_1d;
          int PT_iy = row / Np_1d;
          T pv = xpows[PT_ix] * ypows[PT_iy];

          sx_recovered[pn] += bx[row] * pv;
          sy_recovered[pn] += by[row] * pv;
          sxy_recovered[pn] += bxy[row] * pv;
          debug_one_recovered[pn] += debug_bone[row] * pv;
        }
        num_recovered[pn]++;
      }
    }

    // Adjust recovered values by taking the average of multiple contributions
    int nnodes = mesh.get_num_nodes();
    std::vector<T> vm_recovered(nnodes, 0.0);
    for (int i = 0; i < nnodes; i++) {
      sx_recovered[i] /= num_recovered[i];
      sy_recovered[i] /= num_recovered[i];
      sxy_recovered[i] /= num_recovered[i];
      debug_one_recovered[i] /= num_recovered[i];

      vm_recovered[i] = sqrt(sx_recovered[i] * sx_recovered[i] -
                             sx_recovered[i] * sy_recovered[i] +
                             sy_recovered[i] * sy_recovered[i] +
                             3.0 * sxy_recovered[i] * sxy_recovered[i]);
    }

    std::printf("SPR execution time:%.5f ms\n", watch.lap() * 1000.0);

    // Export cut vtk
    ToVTK<T, Mesh> cut_vtk(
        mesh, "analysis_verification_cut_Np_" + std::to_string(Np_1d) + ".vtk");
    cut_vtk.write_mesh();
    cut_vtk.write_sol("lsf", mesh.get_lsf_nodes().data());

    std::vector<double> bc_nodes_v(mesh.get_num_nodes(), 0.0);
    for (int i : bc_nodes) bc_nodes_v[i] = 1.0;
    cut_vtk.write_sol("bc_nodes", bc_nodes_v.data());

    std::vector<double> num_recovered_v(num_recovered.begin(),
                                        num_recovered.end());
    cut_vtk.write_sol("num_recovered", num_recovered_v.data());
    cut_vtk.write_sol("debug_A_cond", debug_A_cond.data());

    cut_vtk.write_vec("displacement", sol.data());

    std::vector<double> load_elem_v(mesh.get_num_elements(), 0.0);
    for (int e : load_elements) load_elem_v[e] = 1.0;
    cut_vtk.write_cell_sol("loaded_elems", load_elem_v.data());

    // Export to vtk
    FieldToVTKNew<T, spatial_dim> spr_quad_vtk(
        "analysis_verification_spr_quad_Np_" + std::to_string(Np_1d) + ".vtk");

    spr_quad_vtk.add_mesh(elastic.get_analysis()
                              .template interpolate<1>(sx_recovered.data())
                              .first);
    spr_quad_vtk.write_mesh();

    strain_stress.set_type(StrainStressType::sx);
    spr_quad_vtk.add_sol("recovered_sx",
                         elastic.get_analysis()
                             .template interpolate<1>(sx_recovered.data())
                             .second);
    spr_quad_vtk.write_sol("recovered_sx");

    strain_stress.set_type(StrainStressType::sy);
    spr_quad_vtk.add_sol("recovered_sy",
                         elastic.get_analysis()
                             .template interpolate<1>(sy_recovered.data())
                             .second);
    spr_quad_vtk.write_sol("recovered_sy");

    strain_stress.set_type(StrainStressType::sxy);
    spr_quad_vtk.add_sol("recovered_sxy",
                         elastic.get_analysis()
                             .template interpolate<1>(sxy_recovered.data())
                             .second);
    spr_quad_vtk.write_sol("recovered_sxy");

    spr_quad_vtk.add_sol(
        "debug_recovered_one",
        elastic.get_analysis()
            .template interpolate<1>(debug_one_recovered.data())
            .second);
    spr_quad_vtk.write_sol("debug_recovered_one");

    spr_quad_vtk.add_sol("vm_recovered",
                         elastic.get_analysis()
                             .template interpolate<1>(vm_recovered.data())
                             .second);
    spr_quad_vtk.write_sol("vm_recovered");

    // Original sampling
    //
    auto [xloc_samples, sol_samples] =
        sampler_von_mises_analysis.template interpolate<2>(sol.data());

    FieldToVTKNew<T, spatial_dim> sampling_vtk(
        "analysis_verification_sampling_Np_" + std::to_string(Np_1d) + ".vtk");

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

    // SPR sampling
    FieldToVTKNew<T, spatial_dim> spr_sampling_vtk(
        "analysis_verification_spr_sampling_Np_" + std::to_string(Np_1d) +
        ".vtk");
    spr_sampling_vtk.add_mesh(xloc_samples);
    spr_sampling_vtk.write_mesh();

    strain_stress.set_type(StrainStressType::sx);
    spr_sampling_vtk.add_sol("recovered_sx",
                             sampler_strain_stress_analysis
                                 .template interpolate<1>(sx_recovered.data())
                                 .second);
    spr_sampling_vtk.write_sol("recovered_sx");

    strain_stress.set_type(StrainStressType::sy);
    spr_sampling_vtk.add_sol("recovered_sy",
                             sampler_strain_stress_analysis
                                 .template interpolate<1>(sy_recovered.data())
                                 .second);
    spr_sampling_vtk.write_sol("recovered_sy");

    strain_stress.set_type(StrainStressType::sxy);
    spr_sampling_vtk.add_sol("recovered_sxy",
                             sampler_strain_stress_analysis
                                 .template interpolate<1>(sxy_recovered.data())
                                 .second);
    spr_sampling_vtk.write_sol("recovered_sxy");

    spr_sampling_vtk.add_sol("recovered_vm",
                             sampler_strain_stress_analysis
                                 .template interpolate<1>(vm_recovered.data())
                                 .second);
    spr_sampling_vtk.write_sol("recovered_vm");
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
