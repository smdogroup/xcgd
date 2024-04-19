#ifndef XCGD_HELMHOLTZ_FILTER_H
#define XCGD_HELMHOLTZ_FILTER_H

#include "analysis.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "physics/helmholtz.h"
#include "sparse_utils/sparse_utils.h"

/**
 * @brief A Helmholtz filter defined on a structural grid.
 *
 * @tparam T numeric type
 * @tparam Np_1d GD degree, note that this can be different from the Np_1d used
 * for the analysis, if wished
 */
template <typename T, int Np_1d>
class HelmholtzFilter final {
 public:
  using Mesh = GridMesh<T, Np_1d>;

 private:
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Grid = StructuredGrid2D<T>;
  using Basis = GDBasis2D<T, Mesh>;
  using Physics = HelmholtzPhysics<T, Basis::spatial_dim>;
  using Analysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Physics>;
  using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

 public:
  HelmholtzFilter(T r0, Grid& grid)
      : mesh(grid),
        quadrature(mesh),
        basis(mesh),
        physics(r0),
        analysis(mesh, quadrature, basis, physics),
        num_nodes(mesh.get_num_nodes()) {
    Mesh& mesh = this->mesh;

    // Set up Jacobian matrix's sparsity pattern
    int *rowp = nullptr, *cols = nullptr;
    SparseUtils::CSRFromConnectivityFunctor(
        num_nodes, mesh.get_num_elements(), mesh.nodes_per_element,
        [&mesh](int elem, int* nodes) { mesh.get_elem_dof_nodes(elem, nodes); },
        &rowp, &cols);

    int nnz = rowp[num_nodes];
    jac_bsr = new BSRMat(num_nodes, nnz, rowp, cols);
    delete[] rowp;
    delete[] cols;

    // Set up the Jacobian matrix - for Helmholtz problem, the Jacobian matrix
    // does not change with x, so we can set it up and factorize it only once
    std::vector<T> zeros(num_nodes, 0.0);
    analysis.jacobian(zeros.data(), zeros.data(), jac_bsr);

    // Convert it to CSC and perform Cholesky factorization
    CSCMat* jac_csc = SparseUtils::bsr_to_csc(jac_bsr);
    SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
    chol = new SparseUtils::SparseCholesky<T>(jac_csc);
    chol->factor();
  }

  ~HelmholtzFilter() {
    if (chol) {
      delete chol;
      chol = nullptr;
    }
    if (jac_bsr) {
      delete jac_bsr;
      jac_bsr = nullptr;
    }
  }

  int get_num_nodes() { return num_nodes; }

  /**
   * @brief Smooth the input x
   *
   * @param x input, the raw nodal field
   * @param phi output, smoothed nodal field
   */
  void apply(const T* x, T* phi) {
    std::fill(phi, phi + num_nodes, 0.0);
    std::vector<T> zeros(num_nodes, 0.0);
    analysis.residual(x, zeros.data(), phi);
    for (int i = 0; i < num_nodes; i++) {
      phi[i] *= -1.0;
    }
#ifdef XCGD_DEBUG_MODE
    std::vector<T> rhs(phi, phi + num_nodes);
#endif
    chol->solve(phi);

#ifdef XCGD_DEBUG_MODE
    // Check error
    // res = Ku - rhs
    std::vector<T> Ku(num_nodes);
    jac_bsr->axpy(phi, Ku.data());
    T err = 0.0;
    for (int i = 0; i < num_nodes; i++) {
      err += (Ku[i] - rhs[i]) * (Ku[i] - rhs[i]);
    }
    std::printf("[Debug] Helmholtz residual:\n");
    std::printf("||Ku - f||_2: %25.15e\n", sqrt(err));
#endif
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
    std::fill(dfdx, dfdx + num_nodes, 0.0);
    analysis.jacobian_adjoint_product(x, zeros.data(), psi.data(), dfdx);
  }

  Mesh& get_mesh() { return mesh; }
  // Quadrature& get_quadrature() { return quadrature; }
  // Basis& get_basis() { return basis; }
  // Analysis& get_analysis() { return analysis; }

 private:
  Mesh mesh;
  Quadrature quadrature;
  Basis basis;
  Physics physics;
  Analysis analysis;
  int num_nodes;

  // Jacobian matrix
  BSRMat* jac_bsr = nullptr;

  // Cholesky factorization
  SparseUtils::SparseCholesky<T>* chol = nullptr;
};

#endif  // XCGD_HELMHOLTZ_FILTER_H