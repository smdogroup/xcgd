

class TetrahedralBasis {
 public:
  static const int spatial_dim = 3;
  static const int nodes_per_element = 10;

  template <typename T>
  static void eval_basis_grad(const T pt[], T Nxi[]) {
    // Corner node derivatives
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

  template <typename T, int dim>
  static void eval_grad(const T pt[], const T dof[],
                        A2D::Mat<T, dim, spatial_dim>& grad) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int k = 0; k < spatial_dim * dim; k++) {
      grad[k] = 0.0;
    }

    for (int i = 0; i < nodes_per_element; i++) {
      for (int k = 0; k < dim; k++) {
        grad(k, 0) += Nxi[spatial_dim * i] * dof[dim * i + k];
        grad(k, 1) += Nxi[spatial_dim * i + 1] * dof[dim * i + k];
        grad(k, 2) += Nxi[spatial_dim * i + 2] * dof[dim * i + k];
      }
    }
  }

  template <typename T, int dim>
  static void add_grad(const T pt[], const A2D::Mat<T, dim, spatial_dim>& coef,
                       T res[]) {
    T Nxi[spatial_dim * nodes_per_element];
    eval_basis_grad(pt, Nxi);

    for (int i = 0; i < nodes_per_element; i++) {
      for (int k = 0; k < dim; k++) {
        res[dim * i + k] += (coef(k, 0) * Nxi[spatial_dim * i] +
                             coef(k, 1) * Nxi[spatial_dim * i + 1] +
                             coef(k, 2) * Nxi[spatial_dim * i + 2]);
      }
    }
  }
};

class TetrahedralQuadrature {
 public:
  static const int num_quadrature_pts = 5;

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