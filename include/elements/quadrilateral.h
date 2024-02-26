#ifndef XCGD_QUADRILATERAL_H
#define XCGD_QUADRILATERAL_H

#include "galerkin_difference.h"

template <typename T>
class QuadrilateralBasis final : public BasisBase<T, FEMesh<T, 2, 4>> {
 private:
  using BasisBase = BasisBase<T, FEMesh<T, 2, 4>>;

 public:
  using BasisBase::nodes_per_element;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;

  QuadrilateralBasis(Mesh& mesh) : BasisBase(mesh) {}

  void eval_basis_grad(int _, const T* pt, T* N, T* Nxi) {
    if (N) {
      N[0] = 0.25 * (1.0 - pt[0]) * (1.0 - pt[1]);
      N[1] = 0.25 * (1.0 + pt[0]) * (1.0 - pt[1]);
      N[2] = 0.25 * (1.0 - pt[0]) * (1.0 + pt[1]);
      N[3] = 0.25 * (1.0 + pt[0]) * (1.0 + pt[1]);
    }
    if (Nxi) {
      Nxi[0] = -0.25 * (1.0 - pt[1]);
      Nxi[1] = -0.25 * (1.0 - pt[0]);
      Nxi[2] = 0.25 * (1.0 - pt[1]);
      Nxi[3] = -0.25 * (1.0 + pt[0]);
      Nxi[4] = -0.25 * (1.0 + pt[1]);
      Nxi[5] = 0.25 * (1.0 - pt[0]);
      Nxi[6] = 0.25 * (1.0 + pt[1]);
      Nxi[7] = 0.25 * (1.0 + pt[0]);
    }
  }
};

class QuadrilateralQuadrature {
 public:
  static constexpr int num_quadrature_pts = 4;

  template <typename T>
  static T get_quadrature_pt(int k, T pt[]) {
    switch (k) {
      case 0:
        pt[0] = -0.5773502692;
        pt[1] = -0.5773502692;
        break;
      case 1:
        pt[0] = 0.5773502692;
        pt[1] = -0.5773502692;
        break;
      case 2:
        pt[0] = -0.5773502692;
        pt[1] = 0.5773502692;
        break;
      case 3:
        pt[0] = 0.5773502692;
        pt[1] = 0.5773502692;
        break;
    }
    return 1.0;  // quadrature weight
  }
};

#endif  // XCGD_QUADRILATERAL_H