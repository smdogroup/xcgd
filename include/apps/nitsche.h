/*
 * Implements an PDE solution app where the Dirichlet boundary conditions are
 * specified using the Nitsche's method
 * */
#pragma once

#include "analysis.h"
#include "interface_analysis.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/linalg.h"

template <typename T, class Mesh, class Quadrature, class Basis,
          class PhysicsBulk, class PhysicsBCs>
class NitscheBCsApp final {
 private:
  using QuadratureBulk = Quadrature;
  using QuadratureBCs = typename Quadrature::InterfaceQuad;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PhysicsBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PhysicsBCs>;

  static_assert(PhysicsBulk::dof_per_node == PhysicsBCs::dof_per_node,
                "dof_per_node mismatch between bulk physics and bcs physics");
  static constexpr int dof_per_node = PhysicsBulk::dof_per_node;
  using BSRMat = GalerkinBSRMat<T, dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  NitscheBCsApp(Mesh& mesh, Quadrature& quadrature, Basis& basis,
                PhysicsBulk& physics_bulk, PhysicsBCs& physics_bcs)
      : mesh(mesh),
        quadrature_bulk(quadrature),
        quadrature_bcs(mesh),
        basis(basis),
        physics_bulk(physics_bulk),
        physics_bcs(physics_bcs),
        analysis_bulk(mesh, quadrature_bulk, basis, physics_bulk),
        analysis_bcs(mesh, quadrature_bcs, basis, physics_bcs) {}

  std::vector<T> solve() {
    int nnodes = mesh.get_num_nodes();
    int nelems = mesh.get_num_elements();

    int ndof = nnodes * dof_per_node;

    // Set up the Jacobian matrix for Poisson's problem with Nitsche's boundary
    // conditions
    int *rowp = nullptr, *cols = nullptr;
    static constexpr int max_nnodes_per_element = Mesh::max_nnodes_per_element;
    SparseUtils::CSRFromConnectivityFunctor(
        nnodes, nelems, max_nnodes_per_element,
        [this](int elem, int* nodes) -> int {
          return this->mesh.get_elem_dof_nodes(elem, nodes);
        },
        &rowp, &cols);
    int nnz = rowp[nnodes];
    BSRMat* jac_bsr = new BSRMat(nnodes, nnz, rowp, cols);
    std::vector<T> zeros(ndof, 0.0);

    analysis_bulk.jacobian(nullptr, zeros.data(), jac_bsr);
    analysis_bcs.jacobian(nullptr, zeros.data(), jac_bsr,
                          false);  // Add bcs contribution
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

    if (jac_bsr) delete jac_bsr;
    if (jac_csc) delete jac_csc;
    if (chol) delete chol;

    return sol;
  }

 private:
  Mesh& mesh;
  QuadratureBulk& quadrature_bulk;
  QuadratureBCs quadrature_bcs;  // Note that this is not a reference

  Basis& basis;

  PhysicsBulk& physics_bulk;
  PhysicsBCs& physics_bcs;
  AnalysisBulk analysis_bulk;
  AnalysisBCs analysis_bcs;
};

/*
 * This class implements an app that solves a two-sided problem (for example,
 * static elastic problem for a two-material structure) that the interface
 * condition is enforced weakly via Nitsche's method.
 *
 * Note: Two mesh instances are contained: the primary mesh, and the secondary
 * mesh, which is the complement mesh with respect to the grid. However,
 * different from StaticElasticErsatz, the two meshes do not share DOFs on
 * overlapping verts. As a result, this class maintains mappings from mesh-local
 * dof indexing to the global dof indexing, which is different from the grid
 * vertex indexing.
 *
 * As a result, we have:
 *   num_nodes = num_primary_nodes + num_secondary_nodes
 *   num_elements = num_cells
 *   primary_node = global_node
 *   secondary_node + num_primary_nodes = global_node
 * */
template <typename T, class Mesh, class Quadrature, class Basis,
          class PhysicsBulk, class PhysicsInterface>
class NitscheTwoSidedApp final {
 private:
  using Grid = typename Mesh::Grid;
  using QuadratureBulk = Quadrature;
  using QuadratureInterface = typename Quadrature::InterfaceQuad;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PhysicsBulk>;
  using AnalysisInterface =
      InterfaceGalerkinAnalysis<T, Mesh, QuadratureInterface, Basis,
                                PhysicsInterface>;

  static_assert(
      PhysicsBulk::dof_per_node == PhysicsInterface::dof_per_node,
      "dof_per_node mismatch between bulk physics and interface physics");

  static constexpr int dof_per_node = PhysicsBulk::dof_per_node;
  static int constexpr max_nnodes_per_element =
      2 * Mesh::max_nnodes_per_element;

  using BSRMat = GalerkinBSRMat<T, dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  NitscheTwoSidedApp(Mesh& mesh, Quadrature& quadrature, Basis& basis,
                     PhysicsBulk& physics_bulk_primary,
                     PhysicsBulk& physics_bulk_secondary,
                     PhysicsInterface& physics_interface)
      : grid(mesh.get_grid()),
        mesh_m(mesh),
        mesh_s(grid),
        quadrature_bulk_m(quadrature),
        quadrature_bulk_s(mesh_s),
        quadrature_interface(mesh),
        basis_m(basis),
        basis_s(mesh_s),
        physics_bulk_m(physics_bulk_primary),
        physics_bulk_s(physics_bulk_secondary),
        physics_interface(physics_interface),
        analysis_bulk_m(mesh_m, quadrature_bulk_m, basis_m, physics_bulk_m),
        analysis_bulk_s(mesh_s, quadrature_bulk_s, basis_s, physics_bulk_s),
        analysis_interface(mesh_m, mesh_s, quadrature_interface, basis_m,
                           physics_interface) {
    update_mesh();
  }

  // Update primary mesh, secondary mesh and the dof mappings
  void update_mesh() {
    // Update level-set function for the secondary mesh
    for (int i = 0; i < grid.get_num_verts(); i++) {
      mesh_s.get_lsf_dof()[i] = -mesh_m.get_lsf_dof()[i];
    }
    mesh_s.update_mesh();
  }

  BSRMat* jacobian() {
    int num_nodes = mesh_m.get_num_nodes() + mesh_s.get_num_nodes();
    int num_elements = grid.get_num_cells();
    int node_offset = mesh_m.get_num_nodes();

    // Set up Jacobian matrix
    int *rowp = nullptr, *cols = nullptr;
    auto& mesh_m = this->mesh_m;
    auto& mesh_s = this->mesh_s;

    // Functor that obtain global nodes for each global element
    // Note: as explained in the class-level description, overlapping vertices
    // of the primary mesh and the secondary mesh do not share the same global
    // dof node indexing. But overlapping elements do, and we use the global
    // cell indexing for the global element indexing.
    auto element_nodes_func = [mesh_m, mesh_s](int cell, int* nodes_g) -> int {
      const auto& cell_elems_m = mesh_m.get_cell_elems();
      const auto& cell_elems_s = mesh_s.get_cell_elems();
      int num_primary_nodes = mesh_m.get_num_nodes();

      int nnodes = 0;

      if (cell_elems_m.count(cell)) {
        nnodes += mesh_m.get_elem_dof_nodes(cell_elems_m.at(cell), nodes_g);
      }

      if (cell_elems_s.count(cell)) {
        int nodes_s[max_nnodes_per_element];
        int nnodes_s =
            mesh_s.get_elem_dof_nodes(cell_elems_s.at(cell), nodes_s);
        for (int i = 0; i < nnodes_s; i++) {
          nodes_g[nnodes + i] = nodes_s[i] + num_primary_nodes;
        }
        nnodes += nnodes_s;
      }

      return nnodes;
    };

    SparseUtils::CSRFromConnectivityFunctor(num_nodes, num_elements,
                                            max_nnodes_per_element,
                                            element_nodes_func, &rowp, &cols);

    int nnz = rowp[num_nodes];
    BSRMat* jac_bsr = new BSRMat(num_nodes, nnz, rowp, cols);

    // Compute Jacobian matrix
    std::vector<T> zeros(num_nodes * dof_per_node, 0.0);

    bool dont_zero_jac = false;
    analysis_bulk_m.jacobian(nullptr, zeros.data(), jac_bsr);
    analysis_bulk_s.jacobian(nullptr, zeros.data(), jac_bsr, dont_zero_jac,
                             node_offset);
    analysis_interface.jacobian(nullptr, zeros.data(), jac_bsr, dont_zero_jac);

#ifdef XCGD_DEBUG_MODE
    // Write Jacobian matrix to a file
    jac_bsr->write_mtx("NitscheTwoSidedApp_K.mtx");
#endif

    if (rowp) delete rowp;
    if (cols) delete cols;

    return jac_bsr;
  }

  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = dof_per_node * (mesh_m.get_num_nodes() + mesh_s.get_num_nodes());
    int node_offset = mesh_m.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

    // Add external load contributions to the right-hand size
    analysis_bulk_m.residual(nullptr, t1.data(), rhs.data());
    analysis_bulk_s.residual(nullptr, t1.data(), rhs.data(), node_offset);
    analysis_interface.residual(nullptr, t1.data(), rhs.data());
    for (int i = 0; i < rhs.size(); i++) {
      rhs[i] *= -1.0;
    }

    for (int i = 0; i < bc_dof.size(); i++) {
      rhs[bc_dof[i]] = bc_vals[i];
    }

    for (int i = 0; i < bc_dof.size(); i++) {
      t1[bc_dof[i]] = bc_vals[i];
    }

    jac_bsr->axpy(t1.data(), t2.data());
    for (int i = 0; i < bc_dof.size(); i++) {
      t2[bc_dof[i]] = 0.0;
    }

    for (int i = 0; i < rhs.size(); i++) {
      t2[i] = rhs[i] - t2[i];
    }

    // Factorize Jacobian matrix
    SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol =
        std::make_shared<SparseUtils::SparseCholesky<T>>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;

    chol->solve(sol.data());

    if (chol_out) {
      *chol_out = chol;
    }

#ifdef XCGD_DEBUG_MODE
    // Write Jacobian matrix to a file
    jac_csc->write_mtx("NitscheTwoSidedApp_K_bcs.mtx");

    // Check error
    // res = Ku - rhs
    std::vector<T> Ku(sol.size());
    jac_bsr->axpy(sol.data(), Ku.data());
    T err = 0.0;
    for (int i = 0; i < Ku.size(); i++) {
      err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
    }
    std::printf("[Debug] Linear elasticity residual:\n");
    std::printf("||Ku - f||_2: %25.15e\n", sqrt(err));
#endif

    if (jac_bsr) delete jac_bsr;
    if (jac_csc) delete jac_csc;

    return sol;
  }

  template <class... LoadAnalyses>
  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      const std::tuple<LoadAnalyses...>& load_analyses,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = dof_per_node * (mesh_m.get_num_nodes() + mesh_s.get_num_nodes());
    int node_offset = mesh_m.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

    // Add external load contributions to the right-hand size
    analysis_bulk_m.residual(nullptr, t1.data(), rhs.data());
    analysis_bulk_s.residual(nullptr, t1.data(), rhs.data(), node_offset);
    analysis_interface.residual(nullptr, t1.data(), rhs.data());
    std::apply(
        [&t1, this](auto&&... load_analysis) mutable {
          (load_analysis.residual(nullptr, t1.data(), this->rhs.data()), ...);
        },
        load_analyses);
    for (int i = 0; i < rhs.size(); i++) {
      rhs[i] *= -1.0;
    }

    for (int i = 0; i < bc_dof.size(); i++) {
      rhs[bc_dof[i]] = bc_vals[i];
    }

    for (int i = 0; i < bc_dof.size(); i++) {
      t1[bc_dof[i]] = bc_vals[i];
    }

    jac_bsr->axpy(t1.data(), t2.data());
    for (int i = 0; i < bc_dof.size(); i++) {
      t2[bc_dof[i]] = 0.0;
    }

    for (int i = 0; i < rhs.size(); i++) {
      t2[i] = rhs[i] - t2[i];
    }

    // Factorize Jacobian matrix
    SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol =
        std::make_shared<SparseUtils::SparseCholesky<T>>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;

    chol->solve(sol.data());

    if (chol_out) {
      *chol_out = chol;
    }

#ifdef XCGD_DEBUG_MODE
    // Write Jacobian matrix to a file
    jac_csc->write_mtx("K_bcs.mtx");

    // Check error
    // res = Ku - rhs
    std::vector<T> Ku(sol.size());
    jac_bsr->axpy(sol.data(), Ku.data());
    T err = 0.0;
    for (int i = 0; i < Ku.size(); i++) {
      err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
    }
    std::printf("[Debug] Linear elasticity residual:\n");
    std::printf("||Ku - f||_2: %25.15e\n", sqrt(err));
#endif

    if (jac_bsr) delete jac_bsr;
    if (jac_csc) delete jac_csc;

    return sol;
  }

  Mesh& get_primary_mesh() { return mesh_m; }
  Mesh& get_secondary_mesh() { return mesh_s; }

  QuadratureBulk& get_primary_bulk_quadrature() { return quadrature_bulk_m; }
  QuadratureBulk& get_secondary_bulk_quadrature() { return quadrature_bulk_s; }
  QuadratureInterface& get_interface_quadrature() {
    return quadrature_interface;
  }

  Basis& get_primary_basis() { return basis_m; }
  Basis& get_secondary_basis() { return basis_s; }

  AnalysisInterface& get_interface_analysis() { return analysis_interface; }

 private:
  // essential assets for analyses
  const Grid& grid;

  Mesh& mesh_m;  // primary mesh (the main mesh)
  Mesh mesh_s;   // secondary mesh (the complement mesh)

  QuadratureBulk& quadrature_bulk_m;  // primary quadratuer
  QuadratureBulk quadrature_bulk_s;   // secondary quadratuer
  QuadratureInterface
      quadrature_interface;  // Note that this is not a reference

  Basis& basis_m;
  Basis basis_s;

  PhysicsBulk& physics_bulk_m;
  PhysicsBulk& physics_bulk_s;
  PhysicsInterface& physics_interface;

  AnalysisBulk analysis_bulk_m;
  AnalysisBulk analysis_bulk_s;
  AnalysisInterface analysis_interface;

  std::vector<T> rhs;
};
