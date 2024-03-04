#include <omp.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "quadrature_general.hpp"
#include "utils/timer.h"

template <int N>
struct Ellipsoid {
  template <typename T>
  T operator()(const algoim::uvector<T, N>& x) const {
    if constexpr (N == 2)
      return x(0) * x(0) + 4.0 * x(1) * x(1) - 1.0;
    else
      return x(0) * x(0) + 4.0 * x(1) * x(1) + 9.0 * x(2) * x(2) - 1.0;
  }

  template <typename T>
  algoim::uvector<T, N> grad(const algoim::uvector<T, N>& x) const {
    if constexpr (N == 2)
      return algoim::uvector<T, N>(2.0 * x(0), 8.0 * x(1));
    else
      return algoim::uvector<T, N>(2.0 * x(0), 8.0 * x(1), 18.0 * x(2));
  }
};

int test_quad_general() {
  std::cout << "Algoim Examples - High-order quadrature algorithms for "
               "implicitly defined domains\n\n";
  std::cout << std::fixed << std::setprecision(16);

  // Area of a 2D ellipse using automatic subdivision
  {
    std::cout << "Area of a 2D ellipse using automatic subdivision:\n";
    Ellipsoid<2> phi;
    auto q = algoim::quadGen<2>(
        phi, algoim::HyperRectangle<double, 2>(-1.1, 1.1), -1, -1, 4);
    double area = q([](const auto& x) { return 1.0; });
    std::cout << "  computed area = " << area << "\n";
    std::cout << "    (exact area = 1.5707963267948966)\n\n";
  }

  // Volume of a 3D ellipsoid using automatic subdivision
  {
    std::cout << "Volume of a 3D ellipsoid using automatic subdivision:\n";
    Ellipsoid<3> phi;
    auto q = algoim::quadGen<3>(
        phi, algoim::HyperRectangle<double, 3>(-1.1, 1.1), -1, -1, 4);
    double volume = q([](const auto& x) { return 1.0; });
    std::cout << "  computed volume = " << volume << "\n";
    std::cout << "    (exact volume = 0.6981317007977318)\n\n";
  }

  // Area of a 2D ellipse, computed via the cells of a Cartesian grid
  {
    int n = 16;
    std::cout << "Area of a 2D ellipse, computed via the cells of a " << n
              << " by " << n << " Cartesian grid:\n";
    double dx = 2.2 / n;
    Ellipsoid<2> phi;
    double area = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        algoim::uvector<double, 2> xmin{-1.1 + i * dx, -1.1 + j * dx};
        algoim::uvector<double, 2> xmax{-1.1 + i * dx + dx, -1.1 + j * dx + dx};
        area +=
            algoim::quadGen<2>(
                phi, algoim::HyperRectangle<double, 2>(xmin, xmax), -1, -1, 4)
                .sumWeights();
      }
    std::cout << "  computed area = " << area << "\n";
    std::cout << "    (exact area = 1.5707963267948966)\n\n";
  }

  // Surface area of a 3D ellipsoid using automatic subdivision
  {
    std::cout
        << "Surface area of a 3D ellipsoid using automatic subdivision:\n";
    Ellipsoid<3> phi;
    auto q = algoim::quadGen<3>(
        phi, algoim::HyperRectangle<double, 3>(-1.1, 1.1), 3, -1, 4);
    double surface_area = q.sumWeights();
    std::cout << "  computed surface area = " << surface_area << "\n";
    std::cout << "    (exact surface area = 4.4008095646649703)\n\n";
  }

  // Visualisation of a quadrature scheme in ParaView via XML VTP file
  {
    std::cout << "Visualisation of a quadrature scheme in ParaView via XML VTP "
                 "file:\n";
    Ellipsoid<3> phi;
    auto q = algoim::quadGen<3>(
        phi, algoim::HyperRectangle<double, 3>(-1.1, 1.1), -1, -1, 2);
    std::ofstream f("scheme.vtp");
    algoim::outputQuadratureRuleAsVtpXML(q, f);
    std::cout << "  scheme.vtp file written, containing " << q.nodes.size()
              << " quadrature points\n";
  }

  return 0;
}

struct Circle {
  template <typename T>
  T operator()(const algoim::uvector<T, 2>& x) const {
    return x(0) * x(0) + x(1) * x(1) - 1.0;
  }

  template <typename T>
  algoim::uvector<T, 2> grad(const algoim::uvector<T, 2>& x) const {
    return algoim::uvector<T, 2>(2.0 * x(0), 2.0 * x(1));
  }
};

double compute_pi(double xmin = -1.0, double xmax = 1.0, int nelems_x = 100,
                  int node_count = 2, int* nquads = nullptr) {
  double pi = 0.0;
  double h = (xmax - xmin) / nelems_x;
  double detJ = h * h;

  std::vector<double> X(nelems_x);
  for (int i = 0; i < nelems_x; i++) {
    X[i] = xmin + i * h;
  }

  std::vector<double> qpts(node_count);
  std::vector<double> wts(node_count);
  for (int i = 0; i < node_count; i++) {
    qpts[i] = algoim::GaussQuad::x(node_count, i) * h;
    wts[i] = algoim::GaussQuad::w(node_count, i);
  }

#pragma omp parallel for reduction(+ : pi)
  for (int i = 0; i < nelems_x; i++) {
    for (int j = 0; j < nelems_x; j++) {
      for (int ii = 0; ii < node_count; ii++) {
        for (int jj = 0; jj < node_count; jj++) {
          pi += ((X[i] + qpts[ii]) * (X[i] + qpts[ii]) +
                     (X[j] + qpts[jj]) * (X[j] + qpts[jj]) <
                 1.0) *
                wts[ii] * wts[jj];
        }
      }
    }
  }

  pi *= detJ;

  if (nquads) {
    *nquads = nelems_x * nelems_x * node_count * node_count;
  }

  return pi;
}

double compute_pi_algoim(double xmin = -1.0, double xmax = 1.0,
                         int nelems_x = 100, int node_count = 2,
                         int* nquads = nullptr) {
  if (nquads) {
    *nquads = 0;
  }
  double pi = 0.0;
  double h = (xmax - xmin) / nelems_x;

  std::vector<double> X(nelems_x);
  for (int i = 0; i < nelems_x; i++) {
    X[i] = xmin + i * h;
  }

  Circle phi;
  auto lam = [](const auto& x) { return 1.0; };
  for (int i = 0; i < nelems_x; i++) {
    for (int j = 0; j < nelems_x; j++) {
      algoim::uvector<double, 2> xmin{X[i], X[j]};
      algoim::uvector<double, 2> xmax{X[i] + h, X[j] + h};
      auto q =
          algoim::quadGen<2>(phi, algoim::HyperRectangle<double, 2>(xmin, xmax),
                             -1, -1, node_count);
      pi += q.sumWeights();
      if (nquads) {
        *nquads += q.nodes.size();
      }
    }
  }

  return pi;
}

std::vector<int> n_sequence(int start, int stop, int num) {
  std::vector<int> vec;

  double cstart = log10((double)start);
  double cend = log10((double)stop);
  double ch = (cend - cstart) / num;

  for (int i = 0; i < num; i++) {
    vec.push_back(int(pow(10, cstart + i * ch)));
  }

  return vec;
}

void test_algoim() {
  StopWatch watch;
  int q = 5, n = 100;

  double xmin = -1.0;
  double xmax = 1.0;

  double pi_exact = 3.14159265358979323846;

  std::vector<int> q_vec = {1, 2, 3, 4, 5};
  std::vector<int> n_vec = n_sequence(10, 3000, 20);

  for (int n : n_vec) {
    for (int q : q_vec) {
      double h = (xmax - xmin) / n;

      double t1 = watch.lap();
      double pi = compute_pi(xmin, xmax, n, q);
      double t2 = watch.lap();
      double err = abs(pi - pi_exact);
      printf("[native]q: %d, n: %4d, pi: %.10f, h: %9.2e, err: %.5e (%.3f s)\n",
             q, n, pi, h, err, t2 - t1);

      double pi2 = compute_pi_algoim(-1.0, 1.0, n, q);
      double err2 = abs(pi2 - pi_exact);
      double t3 = watch.lap();
      printf("[algoim]q: %d, n: %4d, pi: %.10f, h: %9.2e, err: %.5e (%.3f s)\n",
             q, n, pi2, h, err2, t3 - t2);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    test_algoim();
  }

  else {
    int n = atoi(argv[1]);
    int q = atoi(argv[2]);

    double xmin = -1.0;
    double xmax = 1.0;
    double h = (xmax - xmin) / n;

    double pi_exact = 3.14159265358979323846;

    int nquads_native = 0, nquads_algoim = 0;
    double err_native =
        abs(compute_pi(xmin, xmax, n, q, &nquads_native) - pi_exact);
    double err_algoim =
        abs(compute_pi_algoim(xmin, xmax, n, q, &nquads_algoim) - pi_exact);
    printf("%.2e, %.10e, %d, %.10e, %d\n", h, err_native, nquads_native,
           err_algoim, nquads_algoim);
  }

  return 0;
}