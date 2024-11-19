#include <cassert>
#include <exception>
#include <string>

#include "apps/poisson_app.h"
#include "apps/static_elastic.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
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
    return weight * detJ * val * val;
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
    return weight * detJ * dot;
  }
};

template <typename T, int Np_1d>
class GridMeshDropOrder : public GridMesh<T, Np_1d> {
 private:
  using MeshBase = GDMeshBase<T, Np_1d>;

 public:
  using MeshBase::corner_nodes_per_element;
  using MeshBase::max_nnodes_per_element;
  using MeshBase::spatial_dim;
  using typename MeshBase::Grid;
  GridMeshDropOrder(const Grid& grid, int Np_bc)
      : GridMesh<T, Np_1d>(grid), Np_bc(Np_bc) {
    assert(Np_bc <= Np_1d);
    assert(Np_bc >= 2);
  }

  int get_elem_dof_nodes(
      int elem, int* nodes,
      std::vector<std::vector<bool>>* pstencil = nullptr) const {
    if (pstencil) {
      pstencil->clear();
      pstencil->resize(Np_1d);
      for (int I = 0; I < Np_1d; I++) {
        (*pstencil)[I] = std::vector<bool>(Np_1d, false);
        for (int J = 0; J < Np_1d; J++) {
          (*pstencil)[I][J] = false;
        }
      }
    }

    int tnodes[Np_1d * Np_1d];
    this->grid.template get_cell_ground_stencil<Np_1d>(elem, tnodes);

    int eij[2] = {-1, -1};
    this->grid.get_cell_coords(elem, eij);
    const int* nxy = this->grid.get_nxy();

    int nnodes = 0;
    for (int iy = 0; iy < Np_1d; iy++) {
      for (int ix = 0; ix < Np_1d; ix++) {
        int idx = ix + Np_1d * iy;

        // lower-left corner element
        if (eij[0] == 0 and eij[1] == 0) {
          if (ix >= Np_bc or iy >= Np_bc) continue;
        }
        // upper-left corner element
        else if (eij[0] == 0 and eij[1] == nxy[1] - 1) {
          if (ix >= Np_bc or iy < Np_1d - Np_bc) continue;
        }
        // lower-right corner element
        else if (eij[0] == nxy[0] - 1 and eij[1] == 0) {
          if (ix < Np_1d - Np_bc or iy >= Np_bc) continue;
        }
        // upper-right corner element
        else if (eij[0] == nxy[0] - 1 and eij[1] == nxy[1] - 1) {
          if (ix < Np_1d - Np_bc or iy < Np_1d - Np_bc) continue;
        }
        // left boundary elements
        else if (eij[0] == 0) {
          if (ix >= Np_bc) continue;
        }
        // right bonudary elements
        else if (eij[0] == nxy[0] - 1) {
          if (ix < Np_1d - Np_bc) continue;
        }
        // lower boundary elements
        else if (eij[1] == 0) {
          if (iy >= Np_bc) continue;
        }
        // upper boundary elements
        else if (eij[1] == nxy[0] - 1) {
          if (iy < Np_1d - Np_bc) continue;
        }

        nodes[nnodes] = tnodes[idx];
        if (pstencil) {
          (*pstencil)[ix][iy] = true;
        }
        nnodes++;
      }
    }

    if (nnodes != max_nnodes_per_element) {
      DegenerateStencilLogger::add(elem, nnodes, nodes);
    }

    return nnodes;
  }

 private:
  int Np_bc;
};

enum class PhysicsType { Poisson, LinearElasticity };

template <typename T, class Mesh>
void write_vtk(std::string vtkpath, const Mesh& mesh, const std::vector<T>& sol,
               const std::vector<T>& sol_exact, const std::vector<T>& source,
               bool save_stencils, PhysicsType physics_type) {
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
  if (physics_type == PhysicsType::Poisson) {
    vtk.write_sol("u", sol.data());
    vtk.write_sol("u_exact", sol_exact.data());
    vtk.write_sol("source", source.data());
  } else {
    vtk.write_vec("u", sol.data());
    vtk.write_vec("u_exact", sol_exact.data());
    vtk.write_vec("internal_force", source.data());
  }

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

// Given solution and f, we want to make sure the following equation holds:
// σij,j + fi = 0
template <typename T, class Basis, class BcFun, class IntFun>
void test_consistency_elasticity(T E, T nu, const BcFun& bc_fun,
                                 const IntFun& int_fun, double rel_tol = 1e-5) {
  A2D::Vec<double, Basis::spatial_dim> xloc;
  xloc(0) = 0.14;
  xloc(1) = 0.32;

  // Compute fi
  auto intf = int_fun(xloc);

  // Create a function that evaluates σ so we can FD it to obtain gradient
  auto stress_fun = [E, nu,
                     bc_fun](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    T mu = 0.5 * E / (1.0 + nu);
    T lambda = E * nu / ((1.0 + nu) * (1.0 - nu));

    A2D::Vec<T, Basis::spatial_dim> u = bc_fun(xloc);
    A2D::SymMat<T, Basis::spatial_dim> strain, stress;
    A2D::Mat<T, Basis::spatial_dim, Basis::spatial_dim> grad;

    double hc = 1e-30;

    using Tc = std::complex<T>;

    A2D::Vec<Tc, Basis::spatial_dim> xloc_dx(xloc), xloc_dy(xloc);
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
  A2D::SymMat<T, Basis::spatial_dim> stress = stress_fun(xloc);
  double h = 1e-6;
  A2D::Vec<T, Basis::spatial_dim> xloc_d(xloc);
  A2D::Vec<double, Basis::spatial_dim> stress_grad;
  for (int i = 0; i < Basis::spatial_dim; i++) {
    for (int j = 0; j < Basis::spatial_dim; j++) {
      xloc_d(j) += h;
      A2D::SymMat<T, Basis::spatial_dim> stress_d = stress_fun(xloc_d);
      stress_grad(i) += (stress_d(i, j) - stress(i, j)) / h;
      xloc_d(j) -= h;
    }
  }

  // Compute error
  T err_2norm = 0.0, int_2norm = 0.0;
  for (int i = 0; i < Basis::spatial_dim; i++) {
    err_2norm += (stress_grad(i) + intf(i)) * (stress_grad(i) + intf(i));
    int_2norm += intf(i) * intf(i);
  }
  err_2norm = sqrt(err_2norm);
  int_2norm = sqrt(int_2norm);

  for (int i = 0; i < Basis::spatial_dim; i++) {
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

// Get the l2 error of the numerical Poisson solution
template <int Np_1d, PhysicsType physics_type>
void execute_accuracy_study(std::string prefix, int nxy, int Np_bc,
                            bool save_stencils, bool consistency_check) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMeshDropOrder<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  auto poisson_exact_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    double k = 1.9 * PI;
    return sin(k * xloc(0)) * sin(k * xloc(1));
  };

  auto poisson_source_fun = [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
    double k = 1.9 * PI;
    double k2 = k * k;
    return -2.0 * k2 * sin(k * xloc(0)) * sin(k * xloc(1));
  };

  auto elasticity_exact_fun =
      []<typename T2>(const A2D::Vec<T2, Basis::spatial_dim>& xloc) {
        A2D::Vec<T2, Basis::spatial_dim> u;
        double k = 1.9 * PI;
        u(0) = sin(k * xloc(0)) * sin(k * xloc(1));
        u(1) = cos(k * xloc(0)) * cos(k * xloc(1));

        return u;
      };

  T E = 100.0, nu = 0.3;
  auto elasticity_int_fun =
      [E, nu]<typename T2>(const A2D::Vec<T2, Basis::spatial_dim>& xloc) {
        constexpr int spatial_dim = Basis::spatial_dim;
        constexpr int dof_per_node = Basis::spatial_dim;

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
    test_consistency_elasticity<T, Basis>(E, nu, elasticity_exact_fun,
                                          elasticity_int_fun);
  }

  using PhysicsApp = typename std::conditional<
      physics_type == PhysicsType::Poisson,
      PoissonApp<T, Mesh, Quadrature, Basis, typeof(poisson_source_fun)>,
      StaticElastic<T, Mesh, Quadrature, Basis,
                    typeof(elasticity_int_fun)>>::type;

  DegenerateStencilLogger::enable();

  int nx_ny[2] = {nxy, nxy};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};
  Grid grid(nx_ny, lxy, xy0);
  Mesh mesh(grid, Np_bc);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  std::shared_ptr<PhysicsApp> physics_app;

  if constexpr (physics_type == PhysicsType::Poisson) {
    physics_app = std::make_shared<PhysicsApp>(mesh, quadrature, basis,
                                               poisson_source_fun);
  } else {
    physics_app = std::make_shared<PhysicsApp>(E, nu, mesh, quadrature, basis,
                                               elasticity_int_fun);
  }

  int nnodes = mesh.get_num_nodes();
  int ndof = nnodes * PhysicsApp::Physics::dof_per_node;

  // Set bcs
  std::vector<int> dof_bcs;
  std::vector<T> dof_vals;

  for (auto nodes :
       {mesh.get_left_boundary_nodes(), mesh.get_right_boundary_nodes(),
        mesh.get_upper_boundary_nodes(), mesh.get_lower_boundary_nodes()}) {
    for (int node : nodes) {
      T xloc[Basis::spatial_dim];
      mesh.get_node_xloc(node, xloc);
      if constexpr (physics_type == PhysicsType::Poisson) {
        dof_bcs.push_back(node);
        dof_vals.push_back(poisson_exact_fun(xloc));
      } else {
        auto vals = elasticity_exact_fun(A2D::Vec<T, Basis::spatial_dim>(xloc));
        for (int d = 0; d < Basis::spatial_dim; d++) {
          dof_bcs.push_back(Basis::spatial_dim * node + d);
          dof_vals.push_back(vals(d));
        }
      }
    }
  }

  // Set exact solution and source
  std::vector<T> sol_exact(ndof, 0.0), source(ndof, 0.0);
  for (int i = 0; i < nnodes; i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    if constexpr (physics_type == PhysicsType::Poisson) {
      sol_exact[i] = poisson_exact_fun(xloc);
      source[i] = poisson_source_fun(xloc);
    } else {
      auto disp = elasticity_exact_fun(A2D::Vec<T, Basis::spatial_dim>(xloc));
      auto intf = elasticity_int_fun(A2D::Vec<T, Basis::spatial_dim>(xloc));
      for (int d = 0; d < Basis::spatial_dim; d++) {
        sol_exact[i * Basis::spatial_dim + d] = disp(d);
        source[i * Basis::spatial_dim + d] = intf(d);
      }
    }
  }

  // Solve
  std::vector<T> sol = physics_app->solve(dof_bcs, dof_vals);

  // Compute the L2 norm of the solution field (not vector)
  std::vector<T> diff(sol.size());
  for (int i = 0; i < sol.size(); i++) {
    diff[i] = (sol[i] - sol_exact[i]);
  }

  GalerkinAnalysis<
      T, Mesh, Quadrature, Basis,
      typename std::conditional<
          physics_type == PhysicsType::Poisson,
          L2normBulk<T, Basis::spatial_dim>,
          VecL2normBulk<T, Basis::spatial_dim, Basis::spatial_dim>>::type>
      integrator(mesh, quadrature, basis, {});

  std::vector<T> ones(sol.size(), 1.0);

  T energy = physics_app->get_analysis().energy(nullptr, sol.data());
  T energy_exact =
      physics_app->get_analysis().energy(nullptr, sol_exact.data());

  T area = integrator.energy(nullptr, ones.data());
  T err_l2norm = sqrt(integrator.energy(nullptr, diff.data()));
  T l2norm = sqrt(integrator.energy(nullptr, sol_exact.data()));

  json j = {{"energy", energy},
            {"energy_exact", energy_exact},
            {"err_l2norm", err_l2norm},
            {"l2norm", l2norm},
            {"area", area}};

  char json_name[256];
  write_json(std::filesystem::path(prefix) / std::filesystem::path("sol.json"),
             j);
  write_vtk<T>(
      std::filesystem::path(prefix) / std::filesystem::path("solution.vtk"),
      mesh, sol, sol_exact, source, save_stencils, physics_type);
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--save-degenerate-stencils", 1);
  p.add_argument<int>("--consistency-check", 1);
  p.add_argument<int>("--Np_1d", 4);
  p.add_argument<int>("--Np_bc", 4);
  p.add_argument<int>("--nxy", 6);
  p.add_argument<std::string>("--prefix", {});
  p.add_argument<std::string>("--physics_type", "poisson",
                              {"poisson", "linear_elasticity"});
  p.parse_args(argc, argv);

  bool consistency_check = p.get<int>("consistency-check");
  bool save_stencils = p.get<int>("save-degenerate-stencils");

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int Np_bc = p.get<int>("Np_bc");
  int nxy = p.get<int>("nxy");
  PhysicsType physics_type =
      std::map<std::string, PhysicsType>{
          {"poisson", PhysicsType::Poisson},
          {"linear_elasticity", PhysicsType::LinearElasticity}}
          .at(p.get<std::string>("physics_type"));

  if (physics_type == PhysicsType::Poisson) {
    switch (Np_1d) {
      case 2:
        execute_accuracy_study<2, PhysicsType::Poisson>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 4:
        execute_accuracy_study<4, PhysicsType::Poisson>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 6:
        execute_accuracy_study<6, PhysicsType::Poisson>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 8:
        execute_accuracy_study<8, PhysicsType::Poisson>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 10:
        execute_accuracy_study<10, PhysicsType::Poisson>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      default:
        printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
        exit(-1);
        break;
    }
  } else {  // LinearElasticity

    switch (Np_1d) {
      case 2:
        execute_accuracy_study<2, PhysicsType::LinearElasticity>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 4:
        execute_accuracy_study<4, PhysicsType::LinearElasticity>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 6:
        execute_accuracy_study<6, PhysicsType::LinearElasticity>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 8:
        execute_accuracy_study<8, PhysicsType::LinearElasticity>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      case 10:
        execute_accuracy_study<10, PhysicsType::LinearElasticity>(
            prefix, nxy, Np_bc, save_stencils, consistency_check);
        break;
      default:
        printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
        exit(-1);
        break;
    }
  }

  return 0;
}
