#include <vector>

#include "quadrature_general.hpp"
#include "quadrature_multipoly.hpp"
#include "utils/vtk.h"

template <int N = 2>
struct Ellipsoid {
  template <typename T>
  T operator()(const algoim::uvector<T, N>& x) const {
    return x(0) * x(0) + 2.0 * x(1) * x(1) - 0.3;
  }

  template <typename T>
  algoim::uvector<T, N> grad(const algoim::uvector<T, N>& x) const {
    return algoim::uvector<T, N>(2.0 * x(0), 4.0 * x(1));
  }
};

void quadratures_general() {
  using T = double;
  constexpr int spatial_dim = 2;
  constexpr int Np_1d = 10;
  Ellipsoid phi;

  auto q = algoim::quadGen<spatial_dim>(
      phi, algoim::HyperRectangle<double, spatial_dim>(0.0, 1.0), -1, -1,
      Np_1d);

  std::vector<T> pts, vals;

  for (auto node : q.nodes) {
    vals.push_back(node.w);
    for (int d = 0; d < spatial_dim; d++) {
      pts.push_back(node.x(d));
    }
  }

  FieldToVTK<T, spatial_dim> vtk("quadratures_general.vtk");
  vtk.add_scalar_field(pts, vals);
  vtk.write_vtk();
}

void quadratures_multipoly() {
  using T = double;
  constexpr int spatial_dim = 2;
  constexpr int Np_1d = 5;

  // Define the LSF functor
  auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) {
    return -Ellipsoid()(x);
  };

  // Set bounds of the hyperrectangle
  algoim::uvector<T, spatial_dim> xmin{-0.7, -0.3};
  algoim::uvector<T, spatial_dim> xmax{1.2, 0.5};

  // Evaluate the bernstein
  T data[Np_1d * Np_1d];
  algoim::xarray<T, spatial_dim> phi(
      data, algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
  algoim::bernstein::bernsteinInterpolate<spatial_dim>(
      [&](const algoim::uvector<T, spatial_dim>& x) {
        std::printf("x: (%15.5e, %15.5e)\n", x(0), x(1));
        return phi_functor(xmin + x * (xmax - xmin));
      },
      phi);

  algoim::ImplicitPolyQuadrature<spatial_dim> ipquad(phi);

  // Compute quadrature nodes
  std::vector<algoim::uvector<T, spatial_dim>> phase0,
      phase1;  // stores quadrature nodes for the 'inside' and 'outside'
  std::vector<T> w0, w1;
  ipquad.integrate(algoim::AutoMixed, Np_1d,
                   [&](const algoim::uvector<T, spatial_dim>& x, T w) {
                     if (algoim::bernstein::evalBernsteinPoly(phi, x) < 0) {
                       phase0.push_back(x);
                       w0.push_back(w);
                     } else
                       phase1.push_back(x);
                     w1.push_back(w);
                   });

  // flatten
  std::vector<T> pts;
  for (auto node : phase0) {
    for (int d = 0; d < spatial_dim; d++) {
      pts.push_back(node(d));
    }
  }
  std::vector<T> vals = w0;

  FieldToVTK<T, spatial_dim> vtk("quadratures_multipoly.vtk");
  vtk.add_scalar_field(pts, vals);
  vtk.write_vtk();
}

int main() {
  quadratures_general();
  quadratures_multipoly();
}