#ifndef XCGD_FE_QUADRILATERAL_H
#define XCGD_FE_QUADRILATERAL_H

#include "fe_commons.h"

template <typename T, class Mesh_ = FEMesh<T, 2, 4>>
class QuadrilateralQuadrature final : public QuadratureBase<T, 4, Mesh_> {
 private:
  using QuadratureBase = QuadratureBase<T, 4, Mesh_>;

 public:
  using QuadratureBase::num_quadrature_pts;
  using typename QuadratureBase::Mesh;

  QuadrilateralQuadrature(const Mesh& mesh) : QuadratureBase(mesh) {}

  void get_quadrature_pts(int _, T pts[], T wts[]) const {
    pts[0] = -0.5773502692;
    pts[1] = -0.5773502692;
    pts[2] = 0.5773502692;
    pts[3] = -0.5773502692;
    pts[4] = 0.5773502692;
    pts[5] = 0.5773502692;
    pts[6] = -0.5773502692;
    pts[7] = 0.5773502692;

    wts[0] = 1.0;
    wts[1] = 1.0;
    wts[2] = 1.0;
    wts[3] = 1.0;
  }
};

template <typename T, class Quadrature_ = QuadrilateralQuadrature<T>>
class QuadrilateralBasis final : public BasisBase<T, Quadrature_> {
 private:
  using BasisBase = BasisBase<T, Quadrature_>;

 public:
  using BasisBase::nodes_per_element;
  using BasisBase::num_quadrature_pts;
  using BasisBase::spatial_dim;
  using typename BasisBase::Mesh;
  using typename BasisBase::Quadrature;

  QuadrilateralBasis(const Mesh& mesh) : BasisBase(mesh) {}

  void eval_basis_grad(int _, const T* pts, T* N, T* Nxi) const {
    for (int q = 0; q < num_quadrature_pts; q++) {
      int offset = q * spatial_dim;
      int offset_n = q * nodes_per_element;
      int offset_nxi = q * nodes_per_element * spatial_dim;
      if (N) {
        N[offset_n] = 0.25 * (1.0 - pts[offset]) * (1.0 - pts[offset + 1]);
        N[offset_n + 1] = 0.25 * (1.0 + pts[offset]) * (1.0 - pts[offset + 1]);
        N[offset_n + 2] = 0.25 * (1.0 + pts[offset]) * (1.0 + pts[offset + 1]);
        N[offset_n + 3] = 0.25 * (1.0 - pts[offset]) * (1.0 + pts[offset + 1]);
      }
      if (Nxi) {
        Nxi[offset_nxi] = -0.25 * (1.0 - pts[offset + 1]);
        Nxi[offset_nxi + 1] = -0.25 * (1.0 - pts[offset]);
        Nxi[offset_nxi + 2] = 0.25 * (1.0 - pts[offset + 1]);
        Nxi[offset_nxi + 3] = -0.25 * (1.0 + pts[offset]);
        Nxi[offset_nxi + 4] = 0.25 * (1.0 + pts[offset + 1]);
        Nxi[offset_nxi + 5] = 0.25 * (1.0 + pts[offset]);
        Nxi[offset_nxi + 6] = -0.25 * (1.0 + pts[offset + 1]);
        Nxi[offset_nxi + 7] = 0.25 * (1.0 - pts[offset]);
      }
    }
  }
};

#endif  // XCGD_FE_QUADRILATERAL_H