#include <array>

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "physics/linear_elasticity.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/vtk.h"

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

  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

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

  template <class... LoadAnalyses>
  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      const std::tuple<LoadAnalyses...>& load_analyses,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

    // Add external load contributions to the right-hand size
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

  std::vector<T>& get_rhs() { return rhs; }

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

  std::vector<T> rhs;
};

// App class for the elastic problem using a main mesh and a conjugate
// mesh for ersatz material, where
// (conjucate mesh) U (main mesh) = grid
template <typename T, class Mesh, class Quadrature, class Basis, class IntFunc,
          class Grid_ = StructuredGrid2D<T>>
class StaticElasticErsatz final {
 public:
  using Physics = LinearElasticity<T, Basis::spatial_dim, IntFunc>;

 private:
  static_assert(Mesh::is_cut_mesh, "StaticElasticErsatz only takes a cut mesh");
  using Grid = Grid_;
  static constexpr bool grid_is_mesh = true;
  using Analysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics, grid_is_mesh>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;
  static int constexpr max_nnodes_per_element =
      2 * Mesh::max_nnodes_per_element;

 public:
  StaticElasticErsatz(T E, T nu, Mesh& mesh, Quadrature& quadrature,
                      Basis& basis, const IntFunc& int_func,
                      double ersatz_ratio = 1e-6)
      : grid(mesh.get_grid()),
        mesh_l(mesh),
        mesh_r(grid),
        quadrature_l(quadrature),
        quadrature_r(mesh_r),
        basis_l(basis),
        basis_r(mesh_r),
        physics_l(E, nu, int_func),
        physics_r(E * ersatz_ratio, nu, int_func),
        analysis_l(mesh_l, quadrature_l, basis_l, physics_l),
        analysis_r(mesh_r, quadrature_r, basis_r, physics_r) {
    for (int i = 0; i < grid.get_num_verts(); i++) {
      mesh_r.get_lsf_dof()[i] = -mesh_l.get_lsf_dof()[i];
    }
    mesh_r.update_mesh();
  }

  StaticElasticErsatz(T E_l, T nu_l, T E_r, T nu_r, Mesh& mesh,
                      Quadrature& quadrature, Basis& basis,
                      const IntFunc& int_func)
      : grid(mesh.get_grid()),
        mesh_l(mesh),
        mesh_r(grid),
        quadrature_l(quadrature),
        quadrature_r(mesh_r),
        basis_l(basis),
        basis_r(mesh_r),
        physics_l(E_l, nu_l, int_func),
        physics_r(E_r, nu_r, int_func),
        analysis_l(mesh_l, quadrature_l, basis_l, physics_l),
        analysis_r(mesh_r, quadrature_r, basis_r, physics_r) {
    for (int i = 0; i < grid.get_num_verts(); i++) {
      mesh_r.get_lsf_dof()[i] = -mesh_l.get_lsf_dof()[i];
    }
    mesh_r.update_mesh();
  }

  ~StaticElasticErsatz() = default;

  // Compute Jacobian matrix without boundary conditions
  BSRMat* jacobian() {
    int ndof = Physics::dof_per_node * grid.get_num_verts();

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

  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = Physics::dof_per_node * grid.get_num_verts();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

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

  template <class... LoadAnalyses>
  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      const std::tuple<LoadAnalyses...>& load_analyses,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = Physics::dof_per_node * grid.get_num_verts();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

    // Add external load contributions to the right-hand size
    std::apply(
        [&t1, this](auto&&... load_analysis) mutable {
          (load_analysis.residual(nullptr, t1.data(), this->rhs.data()), ...);
        },
        load_analyses);

    // Add internal load contributions to the right-hand size
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

  std::vector<T>& get_rhs() { return rhs; }

  Mesh& get_mesh() { return mesh_l; }
  Mesh& get_mesh_ersatz() { return mesh_r; }

  Quadrature& get_quadrature() { return quadrature_l; }
  Quadrature& get_quadrature_ersatz() { return quadrature_r; }
  Basis& get_basis() { return basis_l; }
  Basis& get_basis_ersatz() { return basis_r; }
  Analysis& get_analysis() { return analysis_l; }
  Analysis& get_analysis_ersatz() { return analysis_r; }

 private:
  const Grid& grid;
  Mesh &mesh_l, mesh_r;
  Quadrature &quadrature_l, quadrature_r;
  Basis &basis_l, basis_r;

  Physics physics_l, physics_r;
  Analysis analysis_l, analysis_r;

  std::vector<T> rhs;
};

#if 0
template <typename T, class Mesh, class Quadrature, class Basis, class IntFunc>
class StaticElasticNitsche final {
 public:
  using Physics = LinearElasticity<T, Basis::spatial_dim, IntFunc>;

 private:
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  StaticElasticNitsche(T E, T nu, Mesh& mesh, Quadrature& quadrature,
                       Basis& basis, const IntFunc& int_func)
      : mesh(mesh),
        quadrature(quadrature),
        basis(basis),
        physics(E, nu, int_func),
        analysis(mesh, quadrature, basis, physics) {}

  ~StaticElasticNitsche() = default;

  std::vector<T> solve() {
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


  std::vector<T> solve(
      const std::vector<int>& bc_dof, const std::vector<T>& bc_vals,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol_out = nullptr) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian();
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side (Dirichlet bcs and load)
    rhs = std::vector<T>(ndof, 0.0);
    std::vector<T> t1(ndof, 0.0), t2(ndof, 0.0);

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
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol =
        std::make_shared<SparseUtils::SparseCholesky<T>>(jac_csc);
    chol->factor();
    std::vector<T> sol = t2;
    chol->solve(sol.data());

    if (chol_out) {
      *chol_out = chol;
    }

    if (jac_bsr) delete jac_bsr;
    if (jac_csc) delete jac_csc;

    return sol;
  }

 private:
  Mesh& mesh;
  Quadrature& quadrature;
  Basis& basis;

  Physics physics;
  Analysis analysis;

  std::vector<T> rhs;
};
#endif

#endif  // XCGD_STATIC_ELASTIC_H
