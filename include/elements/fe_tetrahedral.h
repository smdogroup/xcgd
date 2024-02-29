#ifndef XCGD_FE_TETRAHEDRAL_H
#define XCGD_FE_TETRAHEDRAL_H

#include <vector>

#include "fe_commons.h"

template <typename T>
class TetrahedralBasis final : public BasisBase<T, 5, FEMesh<T, 3, 10>> {
 private:
  using BasisBase = BasisBase<T, 5, FEMesh<T, 3, 10>>;

 public:
  using BasisBase::nodes_per_element;
  using BasisBase::num_quadrature_pts;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;

  TetrahedralBasis(Mesh& mesh) : BasisBase(mesh) {}

  void get_quadrature_pts(T pts[], T wts[]) const {
    pts[0] = 0.25;
    pts[1] = 0.25;
    pts[2] = 0.25;
    pts[3] = 1.0 / 6.0;
    pts[4] = 1.0 / 6.0;
    pts[5] = 1.0 / 6.0;
    pts[6] = 0.5;
    pts[7] = 1.0 / 6.0;
    pts[8] = 1.0 / 6.0;
    pts[9] = 1.0 / 6.0;
    pts[10] = 0.5;
    pts[11] = 1.0 / 6.0;
    pts[12] = 1.0 / 6.0;
    pts[13] = 1.0 / 6.0;
    pts[14] = 0.5;

    wts[0] = -2.0 / 15;
    wts[1] = 3.0 / 40;
    wts[2] = 3.0 / 40;
    wts[3] = 3.0 / 40;
    wts[4] = 3.0 / 40;
  }

  void eval_basis_grad(int _, const T* pts, T* N, T* Nxi) const {
    for (int q = 0; q < num_quadrature_pts; q++) {
      int offset_n = q * nodes_per_element;
      int offset_nxi = q * nodes_per_element * spatial_dim;
      if (N) {
        N[offset_n] =
            2.0 * (pts[0] + pts[1] + pts[2]) * (pts[0] + pts[1] + pts[2]) -
            3.0 * (pts[0] + pts[1] + pts[2]) + 1.0;
        N[offset_n + 1] = -pts[0] + 2.0 * pts[0] * pts[0];
        N[offset_n + 2] = -pts[1] + 2.0 * pts[1] * pts[1];
        N[offset_n + 3] = -pts[2] + 2.0 * pts[2] * pts[2];
        N[offset_n + 4] = 4.0 * pts[0] * (1.0 - pts[0] - pts[1] - pts[2]);
        N[offset_n + 5] = 4.0 * pts[1] * (1.0 - pts[0] - pts[1] - pts[2]);
        N[offset_n + 6] = 4.0 * pts[2] * (1.0 - pts[0] - pts[1] - pts[2]);
        N[offset_n + 7] = 4.0 * pts[0] * pts[1];
        N[offset_n + 8] = 4.0 * pts[0] * pts[2];
        N[offset_n + 9] = 4.0 * pts[1] * pts[2];
      }

      // Corner node derivatives
      if (Nxi) {
        Nxi[offset_nxi] = 4.0 * pts[0] + 4.0 * pts[1] + 4.0 * pts[2] - 3.0;
        Nxi[offset_nxi + 1] = 4.0 * pts[0] + 4.0 * pts[1] + 4.0 * pts[2] - 3.0;
        Nxi[offset_nxi + 2] = 4.0 * pts[0] + 4.0 * pts[1] + 4.0 * pts[2] - 3.0;

        Nxi[offset_nxi + 3] = 4.0 * pts[0] - 1.0;
        Nxi[offset_nxi + 4] = 0.0;
        Nxi[offset_nxi + 5] = 0.0;
        Nxi[offset_nxi + 6] = 0.0;
        Nxi[offset_nxi + 7] = 4.0 * pts[1] - 1.0;
        Nxi[offset_nxi + 8] = 0.0;
        Nxi[offset_nxi + 9] = 0.0;
        Nxi[offset_nxi + 10] = 0.0;
        Nxi[offset_nxi + 11] = 4.0 * pts[2] - 1.0;

        // Mid node derivatives
        Nxi[offset_nxi + 12] = -4.0 * (2.0 * pts[0] + pts[1] + pts[2] - 1.0);
        Nxi[offset_nxi + 13] = -4.0 * pts[0];
        Nxi[offset_nxi + 14] = -4.0 * pts[0];

        Nxi[offset_nxi + 15] = 4.0 * pts[1];
        Nxi[offset_nxi + 16] = 4.0 * pts[0];
        Nxi[offset_nxi + 17] = 0.0;

        Nxi[offset_nxi + 18] = -4.0 * pts[1];
        Nxi[offset_nxi + 19] = -4.0 * (pts[0] + 2.0 * pts[1] + pts[2] - 1.0);
        Nxi[offset_nxi + 20] = -4.0 * pts[1];

        Nxi[offset_nxi + 21] = -4.0 * pts[2];
        Nxi[offset_nxi + 22] = -4.0 * pts[2];
        Nxi[offset_nxi + 23] = -4.0 * (pts[0] + pts[1] + 2.0 * pts[2] - 1.0);

        Nxi[offset_nxi + 24] = 4.0 * pts[2];
        Nxi[offset_nxi + 25] = 0.0;
        Nxi[offset_nxi + 26] = 4.0 * pts[0];

        Nxi[offset_nxi + 27] = 0.0;
        Nxi[offset_nxi + 28] = 4.0 * pts[2];
        Nxi[offset_nxi + 29] = 4.0 * pts[1];
      }
    }
  }
};

#endif  // XCGD_FE_TETRAHEDRAL_H