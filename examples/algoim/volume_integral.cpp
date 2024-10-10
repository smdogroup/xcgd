#include <vector>

#include "quadrature_general.hpp"
#include "quadrature_multipoly.hpp"
#include "utils/vtk.h"

template <int Np_1d>
void quadratures_multipoly() {
  using T = double;
  constexpr int spatial_dim = 2;

  // Define the LSF functor
  auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) -> T {
    return x(0) * x(0) + x(1) * x(1) - 1.0;
  };

  // Set bounds of the hyperrectangle
  algoim::uvector<T, spatial_dim> xmin{-1.0, -1.0};
  algoim::uvector<T, spatial_dim> xmax{1.0, 1.0};

  // Evaluate the bernstein
  T data[Np_1d * Np_1d];
  algoim::xarray<T, spatial_dim> phi(
      data, algoim::uvector<int, spatial_dim>(Np_1d, Np_1d));
  algoim::bernstein::bernsteinInterpolate<spatial_dim>(
      [&](const algoim::uvector<T, spatial_dim>& x) {
        // std::printf("x: (%15.5e, %15.5e)\n", x(0), x(1));
        return phi_functor(xmin + x * (xmax - xmin));
      },
      phi);

  algoim::ImplicitPolyQuadrature<spatial_dim> ipquad(phi);

  // Compute quadrature nodes
  std::vector<algoim::uvector<T, spatial_dim>> quad_nodes_inner,
      quad_nodes_outer;
  std::vector<T> w_inner, w_outer;
  ipquad.integrate(algoim::AutoMixed, Np_1d,
                   [&](const algoim::uvector<T, spatial_dim>& x, T w) {
                     if (algoim::bernstein::evalBernsteinPoly(phi, x) < 0) {
                       quad_nodes_inner.push_back(x);
                       w_inner.push_back(w);
                     } else {
                       quad_nodes_outer.push_back(x);
                       w_outer.push_back(w);
                     }
                   });

  std::vector<T> pts_inner;
  for (auto node : quad_nodes_inner) {
    for (int d = 0; d < spatial_dim; d++) {
      pts_inner.push_back(node(d));
    }
  }

  std::vector<T> pts_outer;
  for (auto node : quad_nodes_outer) {
    for (int d = 0; d < spatial_dim; d++) {
      pts_outer.push_back(node(d));
    }
  }

  T sum_inner = 0.0, sum_outer = 0.0;
  for (auto w : w_inner) sum_inner += w;
  for (auto w : w_outer) sum_outer += w;

  std::cout << "number of interior quadrature points: " << w_inner.size()
            << "\n";
  std::cout << "sum of interior weights: " << sum_inner << "\n";

  std::cout << "number of exterior quadrature points: " << w_outer.size()
            << "\n";
  std::cout << "sum of exterior weights: " << sum_outer << "\n";

  FieldToVTKNew<T, spatial_dim> vtk_inner("quadratures_multipoly_inner.vtk");
  vtk_inner.add_mesh(pts_inner);
  vtk_inner.write_mesh();
  vtk_inner.add_sol("w_inner", w_inner);
  vtk_inner.write_sol("w_inner");

  FieldToVTKNew<T, spatial_dim> vtk_outer("quadratures_multipoly_outer.vtk");
  vtk_outer.add_mesh(pts_outer);
  vtk_outer.write_mesh();
  vtk_outer.add_sol("w_outer", w_outer);
  vtk_outer.write_sol("w_outer");
}

int main() { quadratures_multipoly<10>(); }
