#include <complex>
#include <functional>
#include <memory>
#include <numeric>
#include <string>

#include "analysis.h"
#include "elements/fe_quadrilateral.h"
#include "elements/fe_tetrahedral.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "interface_analysis.h"
#include "physics/linear_elasticity.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/mesher.h"

using T = double;

template <class Physics, class Mesh, class Quadrature, class Basis>
void test_physics_fd(std::tuple<Mesh *, Quadrature *, Basis *> tuple,
                     Physics &physics, double h = 1e-6, double tol = 1e-14,
                     bool check_res_only = false) {
  Mesh *mesh = std::get<0>(tuple);
  std::shared_ptr<Mesh> mesh_slave;  // used only for interface physics
  Quadrature *quadrature = std::get<1>(tuple);
  Basis *basis = std::get<2>(tuple);

  constexpr bool is_interface_physics = Physics::is_interface_physics;
  if constexpr (is_interface_physics) {
    static_assert(Mesh::is_cut_mesh,
                  "cut mesh must be used to test interface physics");

    auto &grid = mesh->get_grid();
    mesh_slave = std::make_shared<Mesh>(grid);
    for (int i = 0; i < grid.get_num_verts(); i++) {
      mesh_slave->get_lsf_dof()[i] = -mesh->get_lsf_dof()[i];
    }
    mesh_slave->update_mesh();
  }

  int num_nodes = 0, num_elements = 0;

  if constexpr (is_interface_physics) {
    num_nodes = mesh->get_num_nodes() + mesh_slave->get_num_nodes();
    num_elements = mesh->get_grid().get_num_cells();
  } else {
    num_nodes = mesh->get_num_nodes();
    num_elements = mesh->get_num_elements();
  }

  // Set the number of degrees of freeom
  int ndof = Physics::dof_per_node * num_nodes;

  // Allocate space for the degrees of freeom
  T *dof = new T[ndof];
  T *dof1 = new T[ndof];
  T *dof2 = new T[ndof];

  T *res = new T[ndof];
  T *res1 = new T[ndof];
  T *res2 = new T[ndof];

  T *Jp = new T[ndof];
  T *Jp_axpy = new T[ndof];
  T *direction = new T[ndof];
  double *p = new double[ndof];

  for (int i = 0; i < ndof; i++) {
    direction[i] = (double)rand() / RAND_MAX;
    p[i] = (double)rand() / RAND_MAX;

    dof[i] = 0.01 * rand() / RAND_MAX;
    dof1[i] = dof[i] - h * direction[i];
    dof2[i] = dof[i] + h * direction[i];

    res[i] = 0.0;
    res1[i] = 0.0;
    res2[i] = 0.0;

    Jp[i] = 0.0;

    Jp_axpy[i] = 0.0;
  }

  // Allocate space for the residual
  using Analysis = typename std::conditional<
      is_interface_physics,
      InterfaceGalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>,
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>>::type;

  std::shared_ptr<Analysis> analysis;

  if constexpr (is_interface_physics) {
    analysis = std::make_shared<Analysis>(*mesh, *mesh_slave, *quadrature,
                                          *basis, physics);
  } else {
    analysis = std::make_shared<Analysis>(*mesh, *quadrature, *basis, physics);
  }

  T energy1 = analysis->energy(nullptr, dof1);
  T energy2 = analysis->energy(nullptr, dof2);

  analysis->residual(nullptr, dof, res);
  analysis->residual(nullptr, dof1, res1);
  analysis->residual(nullptr, dof2, res2);

  double dres_fd = (energy2 - energy1) / (2.0 * h);
  double dres_exact = 0.0;
  double dres_relerr = 0.0;

  for (int i = 0; i < ndof; i++) {
    dres_exact += res[i] * direction[i];
  }

  dres_relerr = (dres_exact - dres_fd) / dres_fd;
  if (dres_exact == 0 and dres_fd == 0) dres_relerr = 0.0;

  std::printf("\nDerivatives check for the residual\n");
  std::printf("finite difference derivatives: %25.15e\n", dres_fd);
  std::printf("exact derivatives:             %25.15e\n", dres_exact);
  std::printf("relative error:                %25.15e\n", dres_relerr);
  EXPECT_NEAR(dres_relerr, 0.0, tol);

  if (check_res_only) return;

  analysis->jacobian_product(nullptr, dof, direction, Jp);
  double dJp_fd = 0.0;
  double dJp_exact = 0.0;
  double dJp_relerr = 0.0;
  for (int i = 0; i < ndof; i++) {
    dJp_fd += p[i] * (res2[i] - res1[i]) / (2.0 * h);
    dJp_exact += Jp[i] * p[i];
  }

  dJp_relerr = (dJp_exact - dJp_fd) / dJp_fd;
  if (dJp_exact == 0 and dJp_fd == 0) dJp_relerr = 0.0;

  std::printf("\nDerivatives check for the Jacobian-vector product\n");
  std::printf("finite difference derivatives: %25.15e\n", dJp_fd);
  std::printf("exact derivatives:             %25.15e\n", dJp_exact);
  std::printf("relative error:                %25.15e\n", dJp_relerr);
  EXPECT_NEAR(dJp_relerr, 0.0, tol);

  int *rowp = nullptr, *cols = nullptr;

  if constexpr (is_interface_physics) {
    auto element_nodes_func = [mesh, mesh_slave](int cell,
                                                 int *nodes_g) -> int {
      auto mesh_m = *mesh;
      auto mesh_s = *mesh_slave;

      const auto &cell_elems_m = mesh_m.get_cell_elems();
      const auto &cell_elems_s = mesh_s.get_cell_elems();
      int num_master_nodes = mesh_m.get_num_nodes();

      int nnodes = 0;

      if (cell_elems_m.count(cell)) {
        nnodes += mesh_m.get_elem_dof_nodes(cell_elems_m.at(cell), nodes_g);
      }

      if (cell_elems_s.count(cell)) {
        int nodes_s[Mesh::max_nnodes_per_element];
        int nnodes_s =
            mesh_s.get_elem_dof_nodes(cell_elems_s.at(cell), nodes_s);
        for (int i = 0; i < nnodes_s; i++) {
          nodes_g[nnodes + i] = nodes_s[i] + num_master_nodes;
        }
        nnodes += nnodes_s;
      }

      return nnodes;
    };
    SparseUtils::CSRFromConnectivityFunctor(num_nodes, num_elements,
                                            Basis::max_nnodes_per_element,
                                            element_nodes_func, &rowp, &cols);
  } else {
    SparseUtils::CSRFromConnectivityFunctor(
        num_nodes, num_elements, Basis::max_nnodes_per_element,
        [mesh](int elem, int *nodes) -> int {
          return mesh->get_elem_dof_nodes(elem, nodes);
        },
        &rowp, &cols);
  }

  int nnz = rowp[num_nodes];
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  BSRMat *jac_bsr = new BSRMat(num_nodes, nnz, rowp, cols);
  analysis->jacobian(nullptr, dof, jac_bsr);
  jac_bsr->axpy(direction, Jp_axpy);

  double Jp_l1 = 0.0;
  double Jp_axpy_l1 = 0.0;
  for (int i = 0; i < ndof; i++) {
    Jp_l1 += Jp[i] * p[i];
    Jp_axpy_l1 += Jp_axpy[i] * p[i];
  }
  double Jp_relerr = (Jp_l1 - Jp_axpy_l1) / Jp_l1;
  if (Jp_l1 == 0 and Jp_axpy_l1 == 0) Jp_relerr = 0.0;

  std::printf("\nDerivatives check for the Jacobian matrix\n");
  std::printf("Jac-vec product:          %25.15e\n", Jp_l1);
  std::printf("Jac-vec product by axpy:  %25.15e\n", Jp_axpy_l1);
  std::printf("relative error:           %25.15e\n", Jp_relerr);
  EXPECT_NEAR(Jp_relerr, 0.0, tol);
}

template <int Np_1d = 4>
std::tuple<CutMesh<T, Np_1d> *,
           GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE> *,
           GDBasis2D<T, CutMesh<T, Np_1d>> *>
create_gd_lsf_surf_basis() {
  int constexpr nx = 8, ny = 8;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;

  int nxy[2] = {nx, ny};
  T lxy[2] = {3.0, 3.0};
  T pt0[2] = {1.5, 1.5};
  T r0 = 1.0;
  Grid *grid = new Grid(nxy, lxy);
  Mesh *mesh = new Mesh(*grid, [pt0, r0](T x[]) {
    return (x[0] - pt0[0]) * (x[0] - pt0[0]) +
           (x[1] - pt0[1]) * (x[1] - pt0[1]) - r0 * r0;
  });

  // ToVTK<T, Mesh> vtk(*mesh, "lsf_surf.vtk");
  // vtk.write_mesh();
  // vtk.write_sol("lsf", mesh->get_lsf_nodes().data());
  return {mesh, new Quadrature(*mesh), new Basis(*mesh)};
}

template <class Quadrature, class Basis>
void test_poisson_cut_dirichlet(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-5, double tol = 1e-12) {
  auto bc_fun = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    return xloc(0) * xloc(0) + xloc(1) * xloc(1);
  };
  using Physics = PoissonCutDirichlet<T, Basis::spatial_dim, typeof(bc_fun)>;
  Physics physics(1.23, bc_fun);
  test_physics_fd(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_poisson_cut_dirichlet_eig_k(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-5, double tol = 1e-12) {
  using Physics =
      PoissonCutDirichletNitscheParameterEigK<T, Basis::spatial_dim>;
  Physics physics;
  test_physics_fd(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_poisson_cut_dirichlet_eig_m(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-5, double tol = 1e-12) {
  using Physics =
      PoissonCutDirichletNitscheParameterEigM<T, Basis::spatial_dim>;
  Physics physics;
  test_physics_fd(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_elasticity_external_load(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-5, double tol = 1e-10) {
  if constexpr (Basis::spatial_dim == 2) {
    auto load_func = [](const A2D::Vec<T, 2> xloc) {
      A2D::Vec<T, 2> ret;
      ret(0) = -1.2 * sin(xloc(0)) + tan(xloc(1));
      ret(1) = 3.4 * exp(xloc(1)) * cos(xloc(0));
      return ret;
    };

    ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>
        physics_load(load_func);
    test_physics_fd(tuple, physics_load, h, tol, true);
  } else if constexpr (Basis::spatial_dim == 3) {
    auto load_func = [](const A2D::Vec<T, 3> xloc) {
      A2D::Vec<T, 3> ret;
      ret(0) = -1.2 * sin(xloc(0)) + tan(xloc(1)) * cos(xloc(2));
      ret(1) = 3.4 * exp(xloc(1)) * cos(xloc(0));
      ret(2) = 5.6 * sin(xloc(2));
      return ret;
    };

    ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>
        physics_load(load_func);
    test_physics_fd(tuple, physics_load, h, tol, true);

  } else {
    throw std::runtime_error(
        ("unknown Basis::spatial_dim: " + std::to_string(Basis::spatial_dim))
            .c_str());
  }
}

template <int dim, class Quadrature, class Basis>
void test_linear_elasticity_cut_dirichlet(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-5, double tol = 1e-10) {
  auto bc_fun = [](const A2D::Vec<T, Basis::spatial_dim> &xloc) {
    std::vector<T> data = {xloc(0), xloc(1), xloc(0) * xloc(1),
                           xloc(0) - xloc(1)};
    return A2D::Vec<T, dim>(data.data());
  };
  using Physics =
      LinearElasticityCutDirichlet<T, Basis::spatial_dim, dim, typeof(bc_fun)>;
  Physics physics(1.23, bc_fun);
  test_physics_fd(tuple, physics, h, tol);
}

template <class Quadrature, class Basis>
void test_linear_elasticity_interface(
    std::tuple<typename Basis::Mesh *, Quadrature *, Basis *> tuple,
    double h = 1e-5, double tol = 1e-10) {
  int constexpr dim = Basis::spatial_dim;
  using Physics = LinearElasticityInterface<T, Basis::spatial_dim>;
  Physics physics(50.0, 1.2, 0.3, 2.5, 0.2);
  test_physics_fd(tuple, physics, h, tol);
}

TEST(physics, PoissonCutDirichlet) {
  test_poisson_cut_dirichlet(create_gd_lsf_surf_basis(), 1e-5, 1e-10);
}

TEST(physics, PoissonCutDirichletEigK) {
  test_poisson_cut_dirichlet_eig_k(create_gd_lsf_surf_basis(), 1e-5, 1e-10);
}

TEST(physics, PoissonCutDirichletEigM) {
  test_poisson_cut_dirichlet_eig_m(create_gd_lsf_surf_basis(), 1e-5, 1e-10);
}

TEST(physics, LinearElasticityCutDirichlet) {
  test_linear_elasticity_cut_dirichlet<2>(create_gd_lsf_surf_basis());
  test_linear_elasticity_cut_dirichlet<3>(create_gd_lsf_surf_basis());
  test_linear_elasticity_cut_dirichlet<4>(create_gd_lsf_surf_basis());
}

TEST(physics, ElasticityExternalLoad) {
  test_elasticity_external_load(create_gd_lsf_surf_basis());
}

TEST(physics, LinearElasticityInterface) {
  test_linear_elasticity_interface(create_gd_lsf_surf_basis());
}
