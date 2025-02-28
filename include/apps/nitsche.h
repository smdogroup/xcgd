/*
 * Implements an PDE solution app where the Dirichlet boundary conditions are
 * specified using the Nitsche's method
 * */
#pragma once

#include "analysis.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/linalg.h"

template <typename T, class Mesh, class Quadrature, class Basis,
          class PhysicsBulk, class PhysicsBCs>
class NitscheBCsApp final {
 public:
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
