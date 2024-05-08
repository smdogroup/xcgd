#include "analysis.h"
#include "physics/linear_elasticity.h"
#include "sparse_utils/sparse_utils.h"

#ifndef XCGD_STATIC_ELASTIC_H
#define XCGD_STATIC_ELASTIC_H

template <typename T, class Mesh, class Quadrature, class Basis>
class StaticElastic final {
 public:
  using Physics = LinearElasticity<T, Basis::spatial_dim>;

 private:
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  StaticElastic(T E, T nu, Mesh& mesh, Quadrature& quadrature, Basis& basis)
      : mesh(mesh),
        quadrature(quadrature),
        basis(basis),
        physics(E, nu),
        analysis(mesh, quadrature, basis, physics) {}

  ~StaticElastic() = default;

  // Compute Jacobian matrix with boundary conditions
  BSRMat* jacobian(const std::vector<int>& bc_dof) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Set up Jacobian matrix
    int *rowp = nullptr, *cols = nullptr;
    auto& mesh = this->mesh;
    SparseUtils::CSRFromConnectivityFunctor(
        mesh.get_num_nodes(), mesh.get_num_elements(), mesh.nodes_per_element,
        [&mesh](int elem, int* nodes) { mesh.get_elem_dof_nodes(elem, nodes); },
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
                       const std::vector<int>& load_dof,
                       const std::vector<T>& load_vals) {
    int ndof = Physics::dof_per_node * mesh.get_num_nodes();

    // Compute Jacobian matrix
    BSRMat* jac_bsr = jacobian(bc_dof);
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    jac_csc->zero_columns(bc_dof.size(), bc_dof.data());

    // Set right hand side
    std::vector<T> rhs(ndof, 0.0);
    for (int i = 0; i < load_dof.size(); i++) {
      rhs[load_dof[i]] = load_vals[i];
    }
    for (int i : bc_dof) {
      rhs[i] = 0.0;
    }

    // Factorize Jacobian matrix
    SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
    SparseUtils::SparseCholesky<T>* chol =
        new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
    std::vector<T> sol = rhs;
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

#endif  // XCGD_STATIC_ELASTIC_H
