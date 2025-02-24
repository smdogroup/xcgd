#include "analysis.h"
#include "physics/poisson.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/misc.h"

#pragma once

template <typename T, class Mesh, class Quadrature, class Basis,
          class SourceFunc>
class PoissonApp final {
 public:
  using Physics = PoissonPhysics<T, Basis::spatial_dim, SourceFunc>;

 private:
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  PoissonApp(Mesh& mesh, Quadrature& quadrature, Basis& basis,
             SourceFunc& source_fun)
      : mesh(mesh),
        quadrature(quadrature),
        basis(basis),
        physics(source_fun),
        analysis(mesh, quadrature, basis, physics) {}

  ~PoissonApp() = default;

  // Compute Jacobian matrix with boundary conditions
  BSRMat* jacobian(const std::vector<int>& bc_dof) {
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

    // Apply bcs
    jac_bsr->zero_rows(bc_dof.size(), bc_dof.data());

    if (rowp) delete rowp;
    if (cols) delete cols;

    return jac_bsr;
  }

  std::vector<T> solve(const std::vector<int>& bc_dof,
                       const std::vector<T>& bc_vals) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian(bc_dof);
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set the right hand side
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
    T rhs2 = 0.0;
    for (int i = 0; i < Ku.size(); i++) {
      err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
      rhs2 += rhs[i] * rhs[i];
    }
    std::printf("[Debug]Poisson residual:\n");
    std::printf("||Ku - f||_2 / ||f||_2: %25.15e\n", sqrt(err) / sqrt(rhs2));
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

template <typename T, class Mesh, class Quadrature, class Basis,
          class SourceFunc, class BCFunc>
class PoissonNitscheApp final {
 public:
  using PoissonBulk = PoissonPhysics<T, Basis::spatial_dim, SourceFunc>;
  using PoissonBCs = PoissonCutDirichlet<T, Basis::spatial_dim, BCFunc>;

 private:
  using QuadratureBulk = Quadrature;
  using QuadratureBCs = typename Quadrature::BCQuad;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PoissonBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PoissonBCs>;

  using BSRMat = GalerkinBSRMat<T, PoissonBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  PoissonNitscheApp(Mesh& mesh, Quadrature& quadrature, Basis& basis,
                    SourceFunc& source_fun, BCFunc& bc_fun, double nitsche_eta)
      : mesh(mesh),
        quadrature_bulk(quadrature),
        quadrature_bcs(mesh),
        basis(basis),
        poisson_bulk(source_fun),
        poisson_bcs(nitsche_eta, bc_fun),
        analysis_bulk(mesh, quadrature_bulk, basis, poisson_bulk),
        analysis_bcs(mesh, quadrature_bcs, basis, poisson_bcs) {}

  std::vector<T> solve() {
    int nnodes = mesh.get_num_nodes();
    int nelems = mesh.get_num_elements();

    int ndof = nnodes * PoissonBulk::dof_per_node;

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

  PoissonBulk poisson_bulk;
  PoissonBCs poisson_bcs;
  AnalysisBulk analysis_bulk;
  AnalysisBCs analysis_bcs;
};

// This app solves the Poisson's equation using the finite cell method
template <typename T, class Mesh, class Quadrature, class Basis,
          class SourceFunc, class BCFunc>
class PoissonFiniteCellApp final {
 public:
  using PoissonBulk = PoissonPhysics<T, Basis::spatial_dim, SourceFunc>;
  using PoissonBCs = PoissonCutDirichlet<T, Basis::spatial_dim, BCFunc>;

 private:
  using QuadratureBulk = Quadrature;
  using QuadratureBCs = typename Quadrature::BCQuad;

  using AnalysisBulk =
      GalerkinAnalysis<T, Mesh, QuadratureBulk, Basis, PoissonBulk>;
  using AnalysisBCs =
      GalerkinAnalysis<T, Mesh, QuadratureBCs, Basis, PoissonBCs>;

  using BSRMat = GalerkinBSRMat<T, PoissonBulk::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  PoissonFiniteCellApp(Mesh& mesh, Quadrature& quadrature, Basis& basis,
                       SourceFunc& source_fun, BCFunc& bc_fun,
                       double nitsche_eta)
      : mesh(mesh),
        quadrature_bulk(quadrature),
        quadrature_bcs(mesh),
        basis(basis),
        poisson_bulk(source_fun),
        poisson_bcs(nitsche_eta, bc_fun),
        analysis_bulk(mesh, quadrature_bulk, basis, poisson_bulk),
        analysis_bcs(mesh, quadrature_bcs, basis, poisson_bcs) {}

  std::vector<T> solve() {
    int nnodes = mesh.get_num_nodes();
    int nelems = mesh.get_num_elements();

    int ndof = nnodes * PoissonBulk::dof_per_node;

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

  PoissonBulk poisson_bulk;
  PoissonBCs poisson_bcs;
  AnalysisBulk analysis_bulk;
  AnalysisBCs analysis_bcs;
};
