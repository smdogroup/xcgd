#ifndef XCGD_QUADRILATERAL_H
#define XCGD_QUADRILATERAL_H

class QuadrilateralBasis {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int nodes_per_element = 4;

  template <typename T>
  static void eval_basis_grad(const T pt[], T Nxi[]) {
    Nxi[0] = -0.25 * (1.0 - pt[1]);
    Nxi[1] = -0.25 * (1.0 - pt[0]);
    Nxi[2] = 0.25 * (1.0 + pt[1]);
    Nxi[3] = 0.25 * (1.0 + pt[0]);
    Nxi[4] = 0.25 * (1.0 - pt[1]);
    Nxi[5] = -0.25 * (1.0 + pt[0]);
    Nxi[6] = -0.25 * (1.0 + pt[1]);
    Nxi[7] = 0.25 * (1.0 - pt[0]);
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