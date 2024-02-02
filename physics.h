#include <cmath>

template <typename T>
inline T inv3x3(const T A[], T Ainv[]) {
  T det =
      (A[8] * (A[0] * A[4] - A[3] * A[1]) - A[7] * (A[0] * A[5] - A[3] * A[2]) +
       A[6] * (A[1] * A[5] - A[2] * A[4]));
  T detinv = 1.0 / det;

  Ainv[0] = (A[4] * A[8] - A[5] * A[7]) * detinv;
  Ainv[1] = -(A[1] * A[8] - A[2] * A[7]) * detinv;
  Ainv[2] = (A[1] * A[5] - A[2] * A[4]) * detinv;

  Ainv[3] = -(A[3] * A[8] - A[5] * A[6]) * detinv;
  Ainv[4] = (A[0] * A[8] - A[2] * A[6]) * detinv;
  Ainv[5] = -(A[0] * A[5] - A[2] * A[3]) * detinv;

  Ainv[6] = (A[3] * A[7] - A[4] * A[6]) * detinv;
  Ainv[7] = -(A[0] * A[7] - A[1] * A[6]) * detinv;
  Ainv[8] = (A[0] * A[4] - A[1] * A[3]) * detinv;

  return det;
}

template <typename T>
inline void mat3x3MatMult(const T A[], const T B[], T C[]) {
  C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
  C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
  C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];

  C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
  C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
  C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];

  C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];
  C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];
  C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template <typename T>
inline T det3x3(const T A[]) {
  return (A[8] * (A[0] * A[4] - A[3] * A[1]) -
          A[7] * (A[0] * A[5] - A[3] * A[2]) +
          A[6] * (A[1] * A[5] - A[2] * A[4]));
}

template <typename T>
inline void det3x3Sens(const T A[], T Ad[]) {
  Ad[0] = A[8] * A[4] - A[7] * A[5];
  Ad[1] = A[6] * A[5] - A[8] * A[3];
  Ad[2] = A[7] * A[3] - A[6] * A[4];

  Ad[3] = A[7] * A[2] - A[8] * A[1];
  Ad[4] = A[8] * A[0] - A[6] * A[2];
  Ad[5] = A[6] * A[1] - A[7] * A[0];

  Ad[6] = A[1] * A[5] - A[2] * A[4];
  Ad[7] = A[3] * A[2] - A[0] * A[5];
  Ad[8] = A[0] * A[4] - A[3] * A[1];
}

template <typename T>
inline void addDet3x3Sens(const T s, const T A[], T Ad[]) {
  Ad[0] += s * (A[8] * A[4] - A[7] * A[5]);
  Ad[1] += s * (A[6] * A[5] - A[8] * A[3]);
  Ad[2] += s * (A[7] * A[3] - A[6] * A[4]);

  Ad[3] += s * (A[7] * A[2] - A[8] * A[1]);
  Ad[4] += s * (A[8] * A[0] - A[6] * A[2]);
  Ad[5] += s * (A[6] * A[1] - A[7] * A[0]);

  Ad[6] += s * (A[1] * A[5] - A[2] * A[4]);
  Ad[7] += s * (A[3] * A[2] - A[0] * A[5]);
  Ad[8] += s * (A[0] * A[4] - A[3] * A[1]);
}

template <typename T>
inline void det3x32ndSens(const T s, const T A[], T Ad[]) {
  // Ad[0] = s*(A[8]*A[4] - A[7]*A[5]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = 0.0;
  Ad[4] = s * A[8];
  Ad[5] = -s * A[7];
  Ad[6] = 0.0;
  Ad[7] = -s * A[5];
  Ad[8] = s * A[4];
  Ad += 9;

  // Ad[1] += s*(A[6]*A[5] - A[8]*A[3]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = -s * A[8];
  Ad[4] = 0.0;
  Ad[5] = s * A[6];
  Ad[6] = s * A[5];
  Ad[7] = 0.0;
  Ad[8] = -s * A[3];
  ;
  Ad += 9;

  // Ad[2] += s*(A[7]*A[3] - A[6]*A[4]);
  Ad[0] = 0.0;
  Ad[1] = 0.0;
  Ad[2] = 0.0;
  Ad[3] = s * A[7];
  Ad[4] = -s * A[6];
  Ad[5] = 0.0;
  Ad[6] = -s * A[4];
  Ad[7] = s * A[3];
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[3] += s*(A[7]*A[2] - A[8]*A[1]);
  Ad[0] = 0.0;
  Ad[1] = -s * A[8];
  Ad[2] = s * A[7];
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = 0.0;
  Ad[7] = s * A[2];
  Ad[8] = -s * A[1];
  Ad += 9;

  // Ad[4] += s*(A[8]*A[0] - A[6]*A[2]);
  Ad[0] = s * A[8];
  Ad[1] = 0.0;
  Ad[2] = -s * A[6];
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = -s * A[2];
  Ad[7] = 0.0;
  Ad[8] = s * A[0];
  Ad += 9;

  // Ad[5] += s*(A[6]*A[1] - A[7]*A[0]);
  Ad[0] = -s * A[7];
  Ad[1] = s * A[6];
  Ad[2] = 0.0;
  Ad[3] = 0.0;
  Ad[4] = 0.0;
  Ad[5] = 0.0;
  Ad[6] = s * A[1];
  Ad[7] = -s * A[0];
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[6] += s*(A[1]*A[5] - A[2]*A[4]);
  Ad[0] = 0.0;
  Ad[1] = s * A[5];
  Ad[2] = -s * A[4];
  Ad[3] = 0.0;
  Ad[4] = -s * A[2];
  Ad[5] = s * A[1];
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[7] += s*(A[3]*A[2] - A[0]*A[5]);
  Ad[0] = -s * A[5];
  Ad[1] = 0.0;
  Ad[2] = s * A[3];
  Ad[3] = s * A[2];
  Ad[4] = 0.0;
  Ad[5] = -s * A[0];
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
  Ad += 9;

  // Ad[8] += s*(A[0]*A[4] - A[3]*A[1]);
  Ad[0] = s * A[4];
  Ad[1] = -s * A[3];
  Ad[2] = 0.0;
  Ad[3] = -s * A[1];
  Ad[4] = s * A[0];
  Ad[5] = 0.0;
  Ad[6] = 0.0;
  Ad[7] = 0.0;
  Ad[8] = 0.0;
}

template <typename T>
class NeohookeanPhysics {
 public:
  static const int spatial_dim = 3;
  static const int dof_per_node = 3;

  T C1, D1;  // Constitutitive data

  NeohookeanPhysics(T C1, T D1) : C1(C1), D1(D1) {}

  T energy(T weight, const T J[], const T grad[]) {
    // Compute the determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the energy density for the model
    T energy_density = C1 * (I1 - 3.0 - 2.0 * std::log(detF)) +
                       D1 * (detF - 1.0) * (detF - 1.0);

    return weight * detJ * energy_density;
  }

  void residual(T weight, const T J[], const T grad[], T coef[]) {
    // Compute the determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the derivatives of the energy density wrt I1 and detF
    T bI1 = C1;
    T bdetF = -2.0 * C1 / detF + 2.0 * D1 * (detF - 1.0);

    // Add the contributions from the quadrature
    bI1 *= weight * detJ;
    bdetF *= weight * detJ;

    // Add dU0/dI1*dI1/dUx
    coef[0] = 2.0 * F[0] * bI1;
    coef[1] = 2.0 * F[1] * bI1;
    coef[2] = 2.0 * F[2] * bI1;
    coef[3] = 2.0 * F[3] * bI1;
    coef[4] = 2.0 * F[4] * bI1;
    coef[5] = 2.0 * F[5] * bI1;
    coef[6] = 2.0 * F[6] * bI1;
    coef[7] = 2.0 * F[7] * bI1;
    coef[8] = 2.0 * F[8] * bI1;

    // Add dU0/dJ*dJ/dUx
    addDet3x3Sens(bdetF, F, coef);
  }

  void jacobian(T weight, const T J[], const T grad[], const T direct[],
                T coef[]) {
    // Compute the determinant of the Jacobian matrix
    T Jinv[spatial_dim * spatial_dim];
    T detJ = inv3x3(J, Jinv);

    // Compute the derformation gradient
    T F[spatial_dim * spatial_dim];
    mat3x3MatMult(grad, Jinv, F);
    F[0] += 1.0;
    F[4] += 1.0;
    F[8] += 1.0;

    // Compute the invariants
    T detF = det3x3(F);

    // Compute tr(C) = tr(F^{T}*F) = sum_{ij} F_{ij}^2
    T I1 =
        (F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3] + F[4] * F[4] +
         F[5] * F[5] + F[6] * F[6] + F[7] * F[7] + F[8] * F[8]);

    // Compute the energy density for the model
    // T energy_density = C1 * (I1 - 3.0 - 2.0 * std::log(detF)) +
    //                    D1 * (detF - 1.0) * (detF - 1.0);

    // Compute the derivatives of the energy density detF
    T bI1 = C1;
    T bdetF = -2.0 * C1 / detF + 2.0 * D1 * (detF - 1.0);
    T b2detF = 2.0 * C1 / (detF * detF) + 2.0 * D1;

    // Add the contributions from the quadrature
    bI1 *= weight * detJ;
    bdetF *= weight * detJ;
    b2detF *= weight * detJ;

    T Jac[9 * 9];
    det3x32ndSens(bdetF, F, Jac);
    for (int i = 0; i < 9; i++) {
      Jac[10 * i] += 2.0 * bI1;
    }

    T t[9];
    det3x3Sens(F, t);

    for (int i = 0; i < 9; i++) {
      addDet3x3Sens(b2detF * t[i], F, &Jac[9 * i]);
    }

    // Compute the Jacobian-vector product
    for (int i = 0; i < 9; i++) {
      coef[i] = 0.0;
      for (int j = 0; j < 9; j++) {
        coef[i] += Jac[9 * i + j] * direct[j];
      }
    }
  }
};
