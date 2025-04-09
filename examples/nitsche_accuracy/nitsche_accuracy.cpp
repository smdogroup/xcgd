#include <cassert>
#include <exception>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/argparser.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/vtk.h"

#define PI 3.14159265358979323846

template <typename T, int spatial_dim>
class L2normBulk final : public PhysicsBase<T, spatial_dim, 0, 1> {
 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ____) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * sqrt(val * val);
  }
};

template <typename T, int spatial_dim, int dim>
class VecL2normBulk final : public PhysicsBase<T, spatial_dim, 0, dim> {
 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, A2D::Vec<T, dim>& u,
           A2D::Mat<T, dim, spatial_dim>& ____) const {
    T detJ, dot;
    A2D::MatDet(J, detJ);
    A2D::VecDot(u, u, dot);
    return weight * detJ * sqrt(dot);
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

template <typename T, int spatial_dim, int dim>
class VecL2normSurf final : public PhysicsBase<T, spatial_dim, 0, dim> {
  static_assert(spatial_dim == 2,
                "This part is not yet implemented properly for 3D");

 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, A2D::Vec<T, dim>& u,
           A2D::Mat<T, dim, spatial_dim>& ____) const {
    T dt_val[spatial_dim] = {nrm_ref[1], -nrm_ref[0]};

    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> dt(dt_val);
    A2D::Vec<T, spatial_dim> JTJdt;

    T scale, dot;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, dt, JTJdt);
    A2D::VecDot(dt, JTJdt, scale);
    A2D::VecDot(u, u, dot);
    scale = sqrt(scale);

    return weight * scale * dot;
  }
};

enum class PhysicsType { Poisson, LinearElasticity };

template <typename T, class MeshBulk, class MeshBCs>
void write_vtk(std::string vtkpath, PhysicsType physics_type,
               const MeshBulk& mesh_bulk, const MeshBCs& mesh_bcs,
               const std::vector<T>& sol, const std::vector<T>& exact_sol,
               const std::vector<T>& source, bool save_stencils) {
  ToVTK<T, MeshBulk> vtk(mesh_bulk, vtkpath);
  vtk.write_mesh();

  std::vector<double> nstencils(mesh_bulk.get_num_elements(),
                                MeshBulk::Np_1d * MeshBulk::Np_1d);
  auto degenerate_stencils = DegenerateStencilLogger::get_stencils();
  for (auto e : degenerate_stencils) {
    int elem = e.first;
    nstencils[elem] = e.second.size();
  }
  vtk.write_cell_sol("nstencils", nstencils.data());
  vtk.write_sol("lsf", mesh_bulk.get_lsf_nodes().data());

  if (physics_type == PhysicsType::Poisson) {
    vtk.write_sol("u", sol.data());
    vtk.write_sol("u_exact", exact_sol.data());
    vtk.write_sol("source", source.data());
  } else {
    vtk.write_vec("u", sol.data());
    vtk.write_vec("u_exact", exact_sol.data());
    vtk.write_vec("internal_force", source.data());
  }

  if (save_stencils) {
    auto [base, suffix] = split_path(vtkpath);

    StencilToVTK<T, MeshBulk> bulk_stencil_vtk(
        mesh_bulk, base + "_bulk_stencils" + suffix);
    bulk_stencil_vtk.write_stencils(mesh_bulk.get_elem_nodes());

    std::unordered_map<int, std::vector<int>> bcs_stencils;
    for (int elem = 0; elem < mesh_bulk.get_num_elements(); elem++) {
      if (mesh_bulk.is_cut_elem(elem)) {
        bcs_stencils[elem] = mesh_bcs.get_nodes(elem);
      }
    }
    StencilToVTK<T, MeshBCs> bcs_stencil_vtk(mesh_bcs,
                                             base + "_bcs_stencils" + suffix);
    bcs_stencil_vtk.write_stencils(bcs_stencils);

    StencilToVTK<T, MeshBulk> degen_stencil_vtk(
        mesh_bulk, base + "_degen_stencils" + suffix);
    degen_stencil_vtk.write_stencils(degenerate_stencils);
  }
}

template <typename T, class Mesh, class Analysis>
void write_field_vtk(std::string field_vtkpath, PhysicsType physics_type,
                     const Mesh& mesh, const Analysis& analysis,
                     const std::vector<T>& sol) {
  FieldToVTKNew<T, Mesh::spatial_dim> field_vtk(field_vtkpath);

  auto [xloc_q, dof_q] = analysis.interpolate(sol.data());

  field_vtk.add_mesh(xloc_q);
  field_vtk.write_mesh();

  if (physics_type == PhysicsType::Poisson) {
    field_vtk.add_sol("sol", dof_q);
    field_vtk.write_sol("sol");
  } else {
    field_vtk.add_vec("sol", dof_q);
    field_vtk.write_vec("sol");
  }
}

// Given solution and f, we want to make sure the following equation holds:
// σij,j + fi = 0
template <typename T, int spatial_dim, class BcFun, class IntFun>
void test_consistency_elasticity(T E, T nu, const BcFun& bc_fun,
                                 const IntFun& int_fun, double rel_tol = 1e-5) {
  A2D::Vec<double, spatial_dim> xloc;
  xloc(0) = 0.14;
  xloc(1) = 0.32;

  // Compute fi
  auto intf = int_fun(xloc);

  // Create a function that evaluates σ so we can FD it to obtain gradient
  auto stress_fun = [E, nu, bc_fun](const A2D::Vec<T, spatial_dim>& xloc) {
    T mu = 0.5 * E / (1.0 + nu);
    T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

    A2D::Vec<T, spatial_dim> u = bc_fun(xloc);
    A2D::SymMat<T, spatial_dim> strain, stress;
    A2D::Mat<T, spatial_dim, spatial_dim> grad;

    double hc = 1e-30;

    using Tc = std::complex<T>;

    A2D::Vec<Tc, spatial_dim> xloc_dx(xloc), xloc_dy(xloc);
    xloc_dx(0) += Tc(0.0, hc);
    xloc_dy(1) += Tc(0.0, hc);

    grad(0, 0) = bc_fun(xloc_dx)(0).imag() / hc;  // dudx
    grad(0, 1) = bc_fun(xloc_dy)(0).imag() / hc;  // dudy
    grad(1, 0) = bc_fun(xloc_dx)(1).imag() / hc;  // dvdx
    grad(1, 1) = bc_fun(xloc_dy)(1).imag() / hc;  // dvdy

    A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad, strain);
    A2D::SymIsotropic(mu, lambda, strain, stress);

    return stress;
  };

  // Compute σij,j
  A2D::SymMat<T, spatial_dim> stress = stress_fun(xloc);
  double h = 1e-6;
  A2D::Vec<T, spatial_dim> xloc_d(xloc);
  A2D::Vec<double, spatial_dim> stress_grad;
  for (int i = 0; i < spatial_dim; i++) {
    for (int j = 0; j < spatial_dim; j++) {
      xloc_d(j) += h;
      A2D::SymMat<T, spatial_dim> stress_d = stress_fun(xloc_d);
      stress_grad(i) += (stress_d(i, j) - stress(i, j)) / h;
      xloc_d(j) -= h;
    }
  }

  // Compute error
  T err_2norm = 0.0, int_2norm = 0.0;
  for (int i = 0; i < spatial_dim; i++) {
    err_2norm += (stress_grad(i) + intf(i)) * (stress_grad(i) + intf(i));
    int_2norm += intf(i) * intf(i);
  }
  err_2norm = sqrt(err_2norm);
  int_2norm = sqrt(int_2norm);

  for (int i = 0; i < spatial_dim; i++) {
    std::printf("[i=%d]σij,j: %20.10e, fi: %20.10e, σij,j + fi: %20.10e\n", i,
                stress_grad(i), intf(i), stress_grad(i) + intf(i));
  }

  T rel_err = abs(err_2norm / int_2norm);
  std::printf("residual: %20.10e, normalized residual: %20.10e\n", err_2norm,
              rel_err);
  if (rel_err > rel_tol) {
    char msg[256];
    std::snprintf(msg, 256,
                  "Consistency check failed, normalized residual too large "
                  "(%.10e > %.10e)",
                  rel_err, rel_tol);
    throw std::runtime_error(msg);
  }
}

enum class ProbInstance { Circle, Wedge, Image };

// Solve the physical problem using Galerkin difference method with Nitsche's
// method
template <int Np_1d, PhysicsType physics_type>
void execute_accuracy_study(std::string prefix, ProbInstance instance,
                            std::string image_json, int nxy_val,
                            bool save_stencils, double nitsche_eta,
                            bool consistency_check) {
  using T = double;
  using Grid = StructuredGrid2D<T>;

  using QuadratureBulk = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid>;
  using QuadratureBCs = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE, Grid>;
  using MeshBulk = CutMesh<T, Np_1d, StructuredGrid2D<T>>;
  using MeshBCs = CutMesh<T, Np_1d, StructuredGrid2D<T>>;

  using BasisBulk = GDBasis2D<T, MeshBulk>;
  using BasisBCs = GDBasis2D<T, MeshBCs>;

  static_assert(BasisBulk::spatial_dim == BasisBCs::spatial_dim,
                "spatial_dim inconsistent between BasisBulk and BasisBCs");

  int constexpr spatial_dim = BasisBulk::spatial_dim;
  static_assert(spatial_dim == 2,
                "only 2-dimensional "
                "problem is "
                "implemented");

  // Construct exact solution using the method of manufactured solutions
  // Strong form of the poisson equation takes the form of
  // Δu = f
  auto poisson_bc_fun = [](const A2D::Vec<T, spatial_dim>& xloc) {
    double k = 1.9 * PI;
    return sin(k * xloc(0)) * sin(k * xloc(1));
  };

  auto poisson_source_fun = [](const A2D::Vec<T, spatial_dim>& xloc) {
    double k = 1.9 * PI;
    double k2 = k * k;
    return -2.0 * k2 * sin(k * xloc(0)) * sin(k * xloc(1));
  };

  // Strong form of the linear elasticity equation takes the form of
  // σij,j + fi = 0
  auto elasticity_bc_fun =
      []<typename T2>(const A2D::Vec<T2, spatial_dim>& xloc) {
        A2D::Vec<T2, spatial_dim> u;
        double k = 1.9 * PI;
        u(0) = sin(k * xloc(0)) * sin(k * xloc(1));
        u(1) = cos(k * xloc(0)) * cos(k * xloc(1));

        return u;
      };

  T E = 100.0, nu = 0.3;
  auto elasticity_int_fun =
      [E, nu]<typename T2>(const A2D::Vec<T2, spatial_dim>& xloc) {
        constexpr int dof_per_node = spatial_dim;

        T2 mu = 0.5 * E / (1.0 + nu);
        T2 lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

        double k = 1.9 * PI;
        double k2 = k * k;

        A2D::Mat<T2, dof_per_node, spatial_dim> grad;
        T2 ux = k * cos(k * xloc(0)) * sin(k * xloc(1));
        T2 uy = k * sin(k * xloc(0)) * cos(k * xloc(1));
        T2 vx = -k * sin(k * xloc(0)) * cos(k * xloc(1));
        T2 vy = -k * cos(k * xloc(0)) * sin(k * xloc(1));

        grad(0, 0) = ux;
        grad(0, 1) = uy;
        grad(1, 0) = vx;
        grad(1, 1) = vy;

        T2 uxx = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
        T2 uxy = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
        T2 uyx = k2 * cos(k * xloc(0)) * cos(k * xloc(1));
        T2 uyy = -k2 * sin(k * xloc(0)) * sin(k * xloc(1));
        T2 vxx = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));
        T2 vxy = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
        T2 vyx = k2 * sin(k * xloc(0)) * sin(k * xloc(1));
        T2 vyy = -k2 * cos(k * xloc(0)) * cos(k * xloc(1));

        A2D::ADObj<A2D::Mat<T2, dof_per_node, spatial_dim>> grad_obj(grad);
        A2D::ADObj<A2D::SymMat<T2, spatial_dim>> E_obj, S_obj;

        auto stack = A2D::MakeStack(
            A2D::MatGreenStrain<A2D::GreenStrainType::LINEAR>(grad_obj, E_obj),
            A2D::SymIsotropic(mu, lambda, E_obj, S_obj));

        // Spartials(i, j) = ∂S(i, j)/∂x(j)
        A2D::Mat<T2, dof_per_node, spatial_dim> Spartials;

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

        A2D::Vec<T2, dof_per_node> intf;
        intf(0) = -(Spartials(0, 0) + Spartials(0, 1));
        intf(1) = -(Spartials(1, 0) + Spartials(1, 1));
        return intf;
      };

  if (consistency_check and physics_type == PhysicsType::LinearElasticity) {
    test_consistency_elasticity<T, spatial_dim>(E, nu, elasticity_bc_fun,
                                                elasticity_int_fun);
  }

  using PhysicsBulk = typename std::conditional<
      physics_type == PhysicsType::Poisson,
      PoissonPhysics<T, spatial_dim, typeof(poisson_source_fun)>,
      LinearElasticity<T, spatial_dim, typeof(elasticity_int_fun)>>::type;
  using PhysicsBCs = typename std::conditional<
      physics_type == PhysicsType::Poisson,
      PoissonCutDirichlet<T, spatial_dim, typeof(poisson_bc_fun)>,
      LinearElasticityCutDirichlet<T, spatial_dim, spatial_dim,
                                   typeof(elasticity_bc_fun)>>::type;

  using AnalysisBulk =
      GalerkinAnalysis<T, MeshBulk, QuadratureBulk, BasisBulk, PhysicsBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, MeshBCs, QuadratureBCs, BasisBCs, PhysicsBCs>;

  using BSRMat = GalerkinBSRMat<T, PhysicsBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  std::shared_ptr<Grid> grid;

  std::shared_ptr<MeshBulk> mesh_bulk;
  std::shared_ptr<MeshBCs> mesh_bcs;

  switch (instance) {
    case ProbInstance::Circle: {
      int nxy[2] = {nxy_val, nxy_val};
      double R = 0.49;
      double lxy[2] = {1.0, 1.0};
      grid = std::make_shared<Grid>(nxy, lxy);

      mesh_bulk = std::make_shared<MeshBulk>(*grid, [R](double x[]) {
        return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) -
               R * R;  // <= 0
      });
      mesh_bcs = std::make_shared<MeshBCs>(*grid, [R](double x[]) {
        return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) -
               R * R;  // <= 0
      });
      break;
    }

    case ProbInstance::Wedge: {
      int nxy[2] = {nxy_val, nxy_val};
      double lxy[2] = {1.0, 1.0};
      double angle = PI / 6.0;
      grid = std::make_shared<Grid>(nxy, lxy);
      mesh_bulk = std::make_shared<MeshBulk>(*grid, [angle](double x[]) {
        T region1 = sin(angle) * (x[0] - 1.0) + cos(angle) * x[1];  // <= 0
        T region2 = 1e-6 - x[0];
        T region3 = 1e-6 - x[1];
        return hard_max<T>({region1, region2, region3});
      });
      mesh_bcs = std::make_shared<MeshBCs>(*grid, [angle](double x[]) {
        T region1 = sin(angle) * (x[0] - 1.0) + cos(angle) * x[1];  // <= 0
        T region2 = 1e-6 - x[0];
        T region3 = 1e-6 - x[1];
        return hard_max<T>({region1, region2, region3});
      });
      break;
    }

    case ProbInstance::Image: {
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

      mesh_bulk = std::make_shared<MeshBulk>(*grid);
      mesh_bcs = std::make_shared<MeshBCs>(*grid);

      if (mesh_bulk->get_lsf_dof().size() != lsf_dof.size()) {
        std::string msg =
            "Attempting to populate the LSF dof from input image, but the "
            "dimensions don't match, the mesh has " +
            std::to_string(mesh_bulk->get_lsf_dof().size()) +
            " LSF nodes, but the input json has " +
            std::to_string(lsf_dof.size()) + " entries.";
        throw std::runtime_error(msg.c_str());
      }
      mesh_bulk->get_lsf_dof() = lsf_dof;
      mesh_bulk->update_mesh();

      if (mesh_bcs->get_lsf_dof().size() != lsf_dof.size()) {
        std::string msg =
            "Attempting to populate the LSF dof from input image, but the "
            "dimensions don't match, the mesh has " +
            std::to_string(mesh_bcs->get_lsf_dof().size()) +
            " LSF nodes, but the input json has " +
            std::to_string(lsf_dof.size()) + " entries.";
        throw std::runtime_error(msg.c_str());
      }
      mesh_bcs->get_lsf_dof() = lsf_dof;
      mesh_bcs->update_mesh();

      break;
    }

    default: {
      throw std::runtime_error("Unknown instance");
      break;
    }
  }

  std::shared_ptr<PhysicsBulk> physics_bulk;
  std::shared_ptr<PhysicsBCs> physics_bcs;

  if constexpr (physics_type == PhysicsType::Poisson) {
    physics_bulk = std::make_shared<PhysicsBulk>(poisson_source_fun);
    physics_bcs = std::make_shared<PhysicsBCs>(nitsche_eta, poisson_bc_fun);
  } else {
    physics_bulk = std::make_shared<PhysicsBulk>(E, nu, elasticity_int_fun);
    physics_bcs =
        std::make_shared<PhysicsBCs>(nitsche_eta, E, nu, elasticity_bc_fun);
  }

  QuadratureBulk quadrature_bulk(*mesh_bulk);
  QuadratureBCs quadrature_bcs(*mesh_bcs);

  BasisBulk basis_bulk(*mesh_bulk);
  BasisBCs basis_bcs(*mesh_bcs);

  AnalysisBulk analysis_bulk(*mesh_bulk, quadrature_bulk, basis_bulk,
                             *physics_bulk);
  AnalysisBCs analysis_bcs(*mesh_bcs, quadrature_bcs, basis_bcs, *physics_bcs);

  xcgd_assert(mesh_bulk->get_num_nodes() == mesh_bcs->get_num_nodes(),
              "mesh_bulk and mesh_bcs have different num_nodes");
  xcgd_assert(mesh_bulk->get_num_elements() == mesh_bcs->get_num_elements(),
              "mesh_bulk and mesh_bcs have different num_elements");

  int nnodes = mesh_bulk->get_num_nodes();
  int nelems = mesh_bulk->get_num_elements();

  int ndof = nnodes * PhysicsBulk::dof_per_node;

  // Set up the Jacobian matrix for Poisson's problem with Nitsche's boundary
  // conditions
  int *rowp = nullptr, *cols = nullptr;
  static constexpr int max_nnodes_per_element =
      2 * MeshBulk::max_nnodes_per_element;
  SparseUtils::CSRFromConnectivityFunctor(
      nnodes, nelems, max_nnodes_per_element,
      [&mesh_bulk, &mesh_bcs](int elem, int* nodes) -> int {
        if (mesh_bulk->is_cut_elem(elem)) {
          int nodes_work[max_nnodes_per_element];

          int nnodes_this_elem = 0;
          nnodes_this_elem += mesh_bcs->get_elem_dof_nodes(elem, nodes_work);
          nnodes_this_elem += mesh_bulk->get_elem_dof_nodes(
              elem, nodes_work + nnodes_this_elem);

          for (int i = 0; i < nnodes_this_elem; i++) {
            nodes[i] = nodes_work[i];
          }

          return nnodes_this_elem;
        } else {
          return mesh_bulk->get_elem_dof_nodes(elem, nodes);
        }
      },
      &rowp, &cols);
  int nnz = rowp[nnodes];
  BSRMat* jac_bsr = new BSRMat(nnodes, nnz, rowp, cols);
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

  // Get exact solution and source
  std::vector<T> sol_exact(ndof, 0.0), source(ndof, 0.0);
  for (int i = 0; i < nnodes; i++) {
    T xloc[spatial_dim];
    mesh_bulk->get_node_xloc(i, xloc);

    if constexpr (physics_type == PhysicsType::Poisson) {
      sol_exact[i] = poisson_bc_fun(A2D::Vec<T, spatial_dim>(xloc));
      source[i] = poisson_source_fun(A2D::Vec<T, spatial_dim>(xloc));
    } else {
      auto bc = elasticity_bc_fun(A2D::Vec<T, spatial_dim>(xloc));
      auto s = elasticity_int_fun(A2D::Vec<T, spatial_dim>(xloc));
      for (int d = 0; d < spatial_dim; d++) {
        sol_exact[i * spatial_dim + d] = bc(d);
        source[i * spatial_dim + d] = s(d);
      }
    }
  }

  // Compute the L2 norm of the solution field (not vector)
  std::vector<T> diff(sol.size());
  for (int i = 0; i < sol.size(); i++) {
    diff[i] = (sol[i] - sol_exact[i]);
  }

  GalerkinAnalysis<
      T, MeshBulk, QuadratureBulk, BasisBulk,
      typename std::conditional<
          physics_type == PhysicsType::Poisson, L2normBulk<T, spatial_dim>,
          VecL2normBulk<T, spatial_dim, spatial_dim>>::type>
      integrator_bulk(*mesh_bulk, quadrature_bulk, basis_bulk, {});

  GalerkinAnalysis<
      T, MeshBCs, QuadratureBCs, BasisBCs,
      typename std::conditional<
          physics_type == PhysicsType::Poisson, L2normBulk<T, spatial_dim>,
          VecL2normBulk<T, spatial_dim, spatial_dim>>::type>
      integrator_bcs(*mesh_bcs, quadrature_bcs, basis_bcs, {});

  std::vector<T> ones(sol.size(), 1.0);
  T area = integrator_bulk.energy(nullptr, ones.data());
  T perimeter = integrator_bcs.energy(nullptr, ones.data());

  T err_l2norm_bulk = sqrt(integrator_bulk.energy(nullptr, diff.data()));
  T err_l2norm_bcs = sqrt(integrator_bcs.energy(nullptr, diff.data()));

  T l2norm_bulk = sqrt(integrator_bulk.energy(nullptr, sol_exact.data()));
  T l2norm_bcs = sqrt(integrator_bcs.energy(nullptr, sol_exact.data()));

  json j = {// {"sol", sol},
            // {"sol_exact", sol_exact},
            // {"lsf", mesh->get_lsf_nodes()},
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
      std::filesystem::path(prefix) / std::filesystem::path("solution.vtk"),
      physics_type, *mesh_bulk, *mesh_bcs, sol, sol_exact, source,
      save_stencils);

  // write_field_vtk<T>(std::filesystem::path(prefix) /
  //                        std::filesystem::path("field_solution_bulk.vtk"),
  //                    physics_type, *mesh, analysis_bulk, sol);
  //
  // write_field_vtk<T>(std::filesystem::path(prefix) /
  //                        std::filesystem::path("field_solution_bcs.vtk"),
  //                    physics_type, *mesh, analysis_bcs, sol);
}

int main(int argc, char* argv[]) {
  DegenerateStencilLogger::enable();

  ArgParser p;
  p.add_argument<int>("--save-stencils", 1);
  p.add_argument<int>("--consistency-check", 1);
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--nxy", 64);
  p.add_argument<double>("--nitsche_eta", 1e6);
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--physics_type", "poisson",
                              {"poisson", "linear_elasticity"});
  p.add_argument<std::string>("--instance", "circle",
                              {"circle", "wedge", "image"});
  p.add_argument<std::string>("--image_json", "image.json");
  p.parse_args(argc, argv);

  bool consistency_check = p.get<int>("consistency-check");
  bool save_stencils = p.get<int>("save-stencils");

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int nxy_val = p.get<int>("nxy");
  double nitsche_eta = p.get<double>("nitsche_eta");
  PhysicsType physics_type =
      std::map<std::string, PhysicsType>{
          {"poisson", PhysicsType::Poisson},
          {"linear_elasticity", PhysicsType::LinearElasticity}}
          .at(p.get<std::string>("physics_type"));

  ProbInstance instance = std::map<std::string, ProbInstance>{
      {"circle", ProbInstance::Circle},
      {"wedge", ProbInstance::Wedge},
      {"image",
       ProbInstance::Image}}.at(p.get<std::string>("instance"));
  std::string image_json = p.get<std::string>("image_json");

  if (physics_type == PhysicsType::Poisson) {
    switch (Np_1d) {
      case 2:
        execute_accuracy_study<2, PhysicsType::Poisson>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 4:
        execute_accuracy_study<4, PhysicsType::Poisson>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 6:
        execute_accuracy_study<6, PhysicsType::Poisson>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 8:
        execute_accuracy_study<8, PhysicsType::Poisson>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 10:
        execute_accuracy_study<10, PhysicsType::Poisson>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      default:
        printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
        exit(-1);
        break;
    }
  } else {  // Elasticity
    switch (Np_1d) {
      case 2:
        execute_accuracy_study<2, PhysicsType::LinearElasticity>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 4:
        execute_accuracy_study<4, PhysicsType::LinearElasticity>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 6:
        execute_accuracy_study<6, PhysicsType::LinearElasticity>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 8:
        execute_accuracy_study<8, PhysicsType::LinearElasticity>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      case 10:
        execute_accuracy_study<10, PhysicsType::LinearElasticity>(
            prefix, instance, image_json, nxy_val, save_stencils, nitsche_eta,
            consistency_check);
        break;
      default:
        printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
        exit(-1);
        break;
    }
  }

  return 0;
}
