#ifndef XCGD_TETRAHEDRAL_H
#define XCGD_TETRAHEDRAL_H

#include <vector>

#include "galerkin_difference.h"

template <typename T>
class TetrahedralBasis final : public BasisBase<T, FEMesh<T, 3, 10>> {
 private:
  using BasisBase = BasisBase<T, FEMesh<T, 3, 10>>;

 public:
  using BasisBase::nodes_per_element;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;

  TetrahedralBasis(Mesh& mesh) : BasisBase(mesh) {}

  void eval_basis_grad(int _, const T* pt, T* N, T* Nxi) {
    if (N) {
      N[0] = 2.0 * (pt[0] + pt[1] + pt[2]) * (pt[0] + pt[1] + pt[2]) -
             3.0 * (pt[0] + pt[1] + pt[2]) + 1.0;
      N[1] = -pt[0] + 2.0 * pt[0] * pt[0];
      N[2] = -pt[1] + 2.0 * pt[1] * pt[1];
      N[3] = -pt[2] + 2.0 * pt[2] * pt[2];
      N[4] = 4.0 * pt[0] * (1.0 - pt[0] - pt[1] - pt[2]);
      N[5] = 4.0 * pt[1] * (1.0 - pt[0] - pt[1] - pt[2]);
      N[6] = 4.0 * pt[2] * (1.0 - pt[0] - pt[1] - pt[2]);
      N[7] = 4.0 * pt[0] * pt[1];
      N[8] = 4.0 * pt[0] * pt[2];
      N[9] = 4.0 * pt[1] * pt[2];
    }

    // Corner node derivatives
    if (Nxi) {
      Nxi[0] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
      Nxi[1] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;
      Nxi[2] = 4.0 * pt[0] + 4.0 * pt[1] + 4.0 * pt[2] - 3.0;

      Nxi[3] = 4.0 * pt[0] - 1.0;
      Nxi[4] = 0.0;
      Nxi[5] = 0.0;
      Nxi[6] = 0.0;
      Nxi[7] = 4.0 * pt[1] - 1.0;
      Nxi[8] = 0.0;
      Nxi[9] = 0.0;
      Nxi[10] = 0.0;
      Nxi[11] = 4.0 * pt[2] - 1.0;

      // Mid node derivatives
      Nxi[12] = -4.0 * (2.0 * pt[0] + pt[1] + pt[2] - 1.0);
      Nxi[13] = -4.0 * pt[0];
      Nxi[14] = -4.0 * pt[0];

      Nxi[15] = 4.0 * pt[1];
      Nxi[16] = 4.0 * pt[0];
      Nxi[17] = 0.0;

      Nxi[18] = -4.0 * pt[1];
      Nxi[19] = -4.0 * (pt[0] + 2.0 * pt[1] + pt[2] - 1.0);
      Nxi[20] = -4.0 * pt[1];

      Nxi[21] = -4.0 * pt[2];
      Nxi[22] = -4.0 * pt[2];
      Nxi[23] = -4.0 * (pt[0] + pt[1] + 2.0 * pt[2] - 1.0);

      Nxi[24] = 4.0 * pt[2];
      Nxi[25] = 0.0;
      Nxi[26] = 4.0 * pt[0];

      Nxi[27] = 0.0;
      Nxi[28] = 4.0 * pt[2];
      Nxi[29] = 4.0 * pt[1];
    }
  }
};

class TetrahedralQuadrature {
 public:
  static constexpr int num_quadrature_pts = 5;

  template <typename T>
  static T get_quadrature_pt(int k, T pt[]) {
    if (k == 0) {
      pt[0] = 0.25;
      pt[1] = 0.25;
      pt[2] = 0.25;
      return -2.0 / 15;
    } else if (k == 1) {
      pt[0] = 1.0 / 6.0;
      pt[1] = 1.0 / 6.0;
      pt[2] = 1.0 / 6.0;
      return 3.0 / 40;
    } else if (k == 2) {
      pt[0] = 0.5;
      pt[1] = 1.0 / 6.0;
      pt[2] = 1.0 / 6.0;
      return 3.0 / 40;
    } else if (k == 3) {
      pt[0] = 1.0 / 6.0;
      pt[1] = 0.5;
      pt[2] = 1.0 / 6.0;
      return 3.0 / 40;
    } else if (k == 4) {
      pt[0] = 1.0 / 6.0;
      pt[1] = 1.0 / 6.0;
      pt[2] = 0.5;
      return 3.0 / 40;
    }
    return 0.0;
  }
};

#endif  // XCGD_TETRAHEDRAL_H