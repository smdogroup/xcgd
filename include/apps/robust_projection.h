#pragma once

/*
 * Robust projection using Heaviside function is a common technique for
 * topology optimization.
 *
 * This implementation defines the smooth Heaviside operator H:
 *
 * H(x) = (tanh(beta * eta) + tanh(beta * (x - eta))
 *      / (tanh(beta * eta) + tanh(beta * (1 - eta))
 *
 * where x: [0, 1], H(x): [0, 1]
 *
 *                  ----    ymax
 *                /
 *               /
 * M(H(x))      |
 *             /
 *            /
 *       ----               ymin
 *
 * ref: Level set topology and shape optimization by density methods using cut
 * elements with length scale control.
 * */
#include <cmath>
#include <cstdio>
#include <stdexcept>
template <typename T>
class RobustProjection {
  /**
   * @param beta [in] controls the steepness of the projection (the larger the
   *                  beta, the steeper the slope above), beta > 0
   * @param eta [in] controls the threshold value, moving from 0 to 1, the
   *                 sloped curve moves from left to right, 0 < eta < 1
   * @param size [in] the size of the vectors to operate on
   */
 public:
  RobustProjection(double beta, double eta, int size)
      : beta(beta),
        eta(eta),
        size(size),
        c1(std::tanh(beta * eta)),
        denom((std::tanh(beta * eta) + std::tanh(beta * (1.0 - eta)))) {
    if (beta <= 0.0) {
      char msg[256];
      std::snprintf(msg, 256,
                    "beta for robust projection needs to be positive, got %.5f",
                    beta);
      throw std::runtime_error(msg);
    }
    if (eta <= 0.0 or eta >= 1.0) {
      char msg[256];
      std::snprintf(
          msg, 256,
          "eta for robust projection needs to be within (0.0, 1.0), got %.5f",
          eta);
      throw std::runtime_error(msg);
    }
  }

  void apply(const T* x, T* y) {
    for (int i = 0; i < size; i++) {
      y[i] = (c1 + std::tanh(beta * (x[i] - eta))) / denom;
    }
  }

  void applyGradient(const T* x, const T* dfdy, T* dfdx) {
    for (int i = 0; i < size; i++) {
      dfdx[i] = dfdy[i] * beta / denom *
                (1.0 - std::tanh(beta * (x[i] - eta)) *
                           std::tanh(beta * (x[i] - eta)));
    }
  }

 private:
  double beta, eta;
  int size;
  double xmax, xmin, ymax, ymin;
  T c1, denom;
};
