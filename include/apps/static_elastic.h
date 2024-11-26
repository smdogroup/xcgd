#include <array>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "physics/linear_elasticity.h"
#include "sparse_utils/sparse_utils.h"

#ifndef XCGD_STATIC_ELASTIC_H
#define XCGD_STATIC_ELASTIC_H

template <typename T, class Mesh, class Quadrature, class Basis, class IntFunc>
class StaticElastic final {
 public:
  using Physics = LinearElasticity<T, Basis::spatial_dim, IntFunc>;

 private:
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  StaticElastic(T E, T nu, Mesh& mesh, Quadrature& quadrature, Basis& basis,
                const IntFunc& int_func)
      : mesh(mesh),
        quadrature(quadrature),
        basis(basis),
        physics(E, nu, int_func),
        analysis(mesh, quadrature, basis, physics) {}

  ~StaticElastic() = default;

  // Compute Jacobian matrix without boundary conditions
  BSRMat* jacobian() {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Set up Jacobian matrix
    int *rowp = nullptr, *cols = nullptr;
    auto& mesh = this->mesh;
    SparseUtils::CSRFromConnectivityFunctor(
        mesh.get_num_nodes(), mesh.get_num_elements(),
        mesh.max_nnodes_per_element,
        [&mesh](int elem, int* nodes) -> int {
          return mesh.get_elem_dof_nodes(elem, nodes);
        },
        &rowp, &cols);

    int nnz = rowp[mesh.get_num_nodes()];
    BSRMat* jac_bsr = new BSRMat(mesh.get_num_nodes(), nnz, rowp, cols);

    // Compute Jacobian matrix
    std::vector<T> zeros(ndof, 0.0);
    analysis.jacobian(nullptr, zeros.data(), jac_bsr);

    if (rowp) delete rowp;
    if (cols) delete cols;

    return jac_bsr;
  }

  std::vector<T> solve(const std::vector<int>& bc_dof,
                       const std::vector<T>& bc_vals) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    std::vector<T> rhs(ndof, 0.0), t1(ndof, 0.0), t2(ndof, 0.0);

    analysis.residual(nullptr, t1.data(), rhs.data());
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
    SparseUtils::SparseCholesky<T>* chol =
        new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;
    chol->solve(sol.data());

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
    if (chol) delete chol;

    return sol;
  }

  template <class... LoadAnalyses>
  std::vector<T> solve(const std::vector<int>& bc_dof,
                       const std::vector<T>& bc_vals,
                       const std::tuple<LoadAnalyses...>& load_analyses) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    std::vector<T> rhs(ndof, 0.0), t1(ndof, 0.0), t2(ndof, 0.0);

    // Add internal load contributions to the right-hand size
    analysis.residual(nullptr, t1.data(), rhs.data());

    // Add external load contributions to the right-hand size
    std::apply(
        [&t1, &rhs](auto&&... load_analysis) mutable {
          (load_analysis.residual(nullptr, t1.data(), rhs.data()), ...);
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
    SparseUtils::SparseCholesky<T>* chol =
        new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;
    chol->solve(sol.data());

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
    if (chol) delete chol;

    return sol;
  }

  Mesh& get_mesh() { return mesh; }
  Quadrature& get_quadrature() { return quadrature; }
  Basis& get_basis() { return basis; }
  Analysis& get_analysis() { return analysis; }

 private:
  Mesh& mesh;
  Quadrature& quadrature;
  Basis& basis;

  Physics physics;
  Analysis analysis;
};

// App class for the elastic problem using a main mesh (mesh_l) and a conjugate
// mesh (mesh_r), an important assumption is that mesh_l + mesh_r = grid
template <typename T, class Mesh, class Quadrature, class Basis, class IntFuncL,
          class IntFuncR>
class StaticElasticTwoSided final {
 public:
  using PhysicsL = LinearElasticity<T, Basis::spatial_dim, IntFuncL>;
  using PhysicsR = LinearElasticity<T, Basis::spatial_dim, IntFuncR>;

 private:
  static_assert(Mesh::is_cut_mesh,
                "StaticElasticTwoSided only takes a cut mesh");
  using Grid = StructuredGrid2D<T>;
  using AnalysisL =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, PhysicsL, true>;
  using AnalysisR =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, PhysicsR, true>;
  using BSRMat = GalerkinBSRMat<T, PhysicsL::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;
  static int constexpr max_nnodes_per_element =
      2 * Mesh::max_nnodes_per_element;

 public:
  StaticElasticTwoSided(T E_l, T nu_l, T E_r, T nu_r, Mesh& mesh_l,
                        const IntFuncL& int_func_l, const IntFuncR& int_func_r)
      : grid(mesh_l.get_grid()),
        mesh_l(mesh_l),
        mesh_r(grid),
        quadrature_l(mesh_l),
        quadrature_r(mesh_r),
        basis_l(mesh_l),
        basis_r(mesh_r),
        physics_l(E_l, nu_l, int_func_l),
        physics_r(E_r, nu_r, int_func_r),
        analysis_l(mesh_l, quadrature_l, basis_l, physics_l),
        analysis_r(mesh_r, quadrature_r, basis_r, physics_r) {
    for (int i = 0; i < grid.get_num_verts(); i++) {
      mesh_r.get_lsf_dof()[i] = -mesh_l.get_lsf_dof()[i];
    }
    mesh_r.update_mesh();
  }

  ~StaticElasticTwoSided() = default;

  // Compute Jacobian matrix without boundary conditions
  BSRMat* jacobian() {
    int ndof = PhysicsL::dof_per_node * grid.get_num_verts();

    // Set up Jacobian matrix
    int *rowp = nullptr, *cols = nullptr;
    auto& mesh_l = this->mesh_l;
    auto& mesh_r = this->mesh_r;
    SparseUtils::CSRFromConnectivityFunctor(
        grid.get_num_verts(), grid.get_num_cells(), max_nnodes_per_element,
        [mesh_l, mesh_r](int cell, int* verts) -> int {
          const auto& cell_elems_l = mesh_l.get_cell_elems();
          const auto& cell_elems_r = mesh_r.get_cell_elems();

          int nverts = 0;
          int verts_work[max_nnodes_per_element];

          if (cell_elems_l.count(cell)) {
            nverts += mesh_l.get_cell_dof_verts(cell, verts_work);
          }

          if (cell_elems_r.count(cell)) {
            nverts += mesh_r.get_cell_dof_verts(cell, verts_work + nverts);
          }

          std::set<int> verts_set(verts_work, verts_work + nverts);

          int i = 0;
          for (auto it = verts_set.begin(); it != verts_set.end(); it++, i++) {
            verts[i] = *it;
          }

          nverts = verts_set.size();

          return nverts;
        },
        &rowp, &cols);

    int nnz = rowp[grid.get_num_verts()];
    BSRMat* jac_bsr = new BSRMat(grid.get_num_verts(), nnz, rowp, cols);

    // Compute Jacobian matrix
    std::vector<T> zeros(ndof, 0.0);
    analysis_l.jacobian(nullptr, zeros.data(), jac_bsr, true);
    analysis_r.jacobian(nullptr, zeros.data(), jac_bsr, false);

    if (rowp) delete rowp;
    if (cols) delete cols;

    return jac_bsr;
  }

  std::vector<T> solve(const std::vector<int>& bc_dof,
                       const std::vector<T>& bc_vals) {
    int ndof = PhysicsL::dof_per_node * grid.get_num_verts();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    std::vector<T> rhs(ndof, 0.0), t1(ndof, 0.0), t2(ndof, 0.0);

    analysis_l.residual(nullptr, t1.data(), rhs.data());
    analysis_r.residual(nullptr, t1.data(), rhs.data());
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
    SparseUtils::SparseCholesky<T>* chol =
        new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;
    chol->solve(sol.data());

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
    if (chol) delete chol;

    return sol;
  }

  template <class... LoadAnalyses>
  std::vector<T> solve(const std::vector<int>& bc_dof,
                       const std::vector<T>& bc_vals,
                       const std::tuple<LoadAnalyses...>& load_analyses) {
    int ndof = PhysicsL::dof_per_node * grid.get_num_verts();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    std::vector<T> rhs(ndof, 0.0), t1(ndof, 0.0), t2(ndof, 0.0);

    // Add internal load contributions to the right-hand size
    analysis_l.residual(nullptr, t1.data(), rhs.data());
    analysis_r.residual(nullptr, t1.data(), rhs.data());

    // Add external load contributions to the right-hand size
    std::apply(
        [&t1, &rhs](auto&&... load_analysis) mutable {
          (load_analysis.residual(nullptr, t1.data(), rhs.data()), ...);
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
    SparseUtils::SparseCholesky<T>* chol =
        new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;
    chol->solve(sol.data());

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
    if (chol) delete chol;

    return sol;
  }

  const Mesh& get_mesh_l() const { return mesh_l; }
  const Mesh& get_mesh_r() const { return mesh_r; }
  const Quadrature& get_quadrature_l() const { return quadrature_l; }
  const Quadrature& get_quadrature_r() const { return quadrature_r; }
  const Basis& get_basis_l() const { return basis_l; }
  const Basis& get_basis_r() const { return basis_r; }
  const AnalysisL& get_analysis_l() const { return analysis_l; }
  const AnalysisR& get_analysis_r() const { return analysis_r; }

 private:
  const Grid& grid;
  Mesh &mesh_l, mesh_r;
  const Quadrature quadrature_l, quadrature_r;
  const Basis basis_l, basis_r;

  const PhysicsL physics_l;
  const PhysicsR physics_r;
  const AnalysisL analysis_l;
  const AnalysisR analysis_r;
};

#endif  // XCGD_STATIC_ELASTIC_H
