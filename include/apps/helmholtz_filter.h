#ifndef XCGD_HELMHOLTZ_FILTER_H
#define XCGD_HELMHOLTZ_FILTER_H

#include "analysis.h"
#include "physics/helmholtz.h"
#include "sparse_utils/sparse_utils.h"

template <typename T, class Mesh, class Quadrature, class Basis>
class HelmholtzFilter {
 private:
  using Physics = HelmholtzPhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;

 public:
  HelmholtzFilter(T r0, Mesh& mesh, Quadrature& quadrature, Basis& basis)
      : physics(r0),
        analysis(mesh, quadrature, basis, physics),
        num_nodes(mesh.get_num_nodes()) {
    // Set up Jacobian matrix's sparsity pattern
    int *rowp = nullptr, *cols = nullptr;
    SparseUtils::CSRFromConnectivityFunctor(
        num_nodes, mesh.get_num_elements(), mesh.nodes_per_element,
        [&mesh](int elem, int* nodes) { mesh.get_elem_dof_nodes(elem, nodes); },
        &rowp, &cols);

    int nnz = rowp[num_nodes];
    BSRMat* jac_bsr = new BSRMat(num_nodes, nnz, rowp, cols);

    // Set up the Jacobian matrix - for Helmholtz problem, the Jacobian matrix
    // does not change with x, so we can set it up and factorize it only once
    std::vector<T> zeros(num_nodes, 0.0);
    analysis.jacobian(zeros.data(), zeros.data(), jac_bsr);

    // Convert it to CSC and perform Cholesky factorization
    SparseUtils::CSCMat<T>* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
    chol = new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
  }

  ~HelmholtzFilter() {
    if (chol) {
      delete chol;
      chol = nullptr;
    }
  }

  /**
   * @brief Smooth the input x
   *
   * @param x input, the raw nodal field
   * @param phi output, smoothed nodal field
   */
  void apply(const T* x, T* phi) {
    std::vector<T> zeros(num_nodes, 0.0);
    analysis.residual(x, zeros.data(), phi);
    for (int i = 0; i < num_nodes; i++) {
      phi[i] *= -1.0;
    }
    chol->solve(phi);
  }

  /**
   * @brief Get gradient w.r.t. the raw field
   *
   * @param dfdphi input, derivatives of some scalar functional w.r.t. phi
   * @param dfdx output, derivatives of the same scalar functional w.r.t. x
   */
  void applyGradient(const T* x, const T* dfdphi, T* dfdx) {
    std::vector<T> zeros(num_nodes, 0.0);
    std::vector<T> psi(dfdphi, dfdphi + num_nodes);
    chol->solve(psi.data());
    for (int i = 0; i < num_nodes; i++) {
      psi[i] *= -1.0;
    }
    analysis.jacobian_adjoint_product(x, zeros.data(), psi.data(), dfdx);
  }

  Physics physics;
  Analysis analysis;

  int num_nodes = -1;

  // Cholesky factorization
  SparseUtils::SparseCholesky<T>* chol;
};

#endif  // XCGD_HELMHOLTZ_FILTER_H