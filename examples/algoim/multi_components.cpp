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
    return (x(0) + x(1)) * (x(0) + x(1)) -
           (x(0) - 0.4 * x(1)) * (x(0) - 0.4 * x(1)) * (x(0) - 0.4 * x(1)) *
               (x(0) - 0.4 * x(1)) -
           0.3;
    // return x(0) - x(1);
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

  std::cout << "no interfaces?: " << (ipquad.k == spatial_dim) << "\n";

  // Compute quadrature nodes
  std::vector<algoim::uvector<T, spatial_dim>> quad_nodes;
  std::vector<T> quad_wts;
  ipquad.integrate(algoim::AutoMixed, Np_1d,
                   [&](const algoim::uvector<T, spatial_dim>& x, T w) {
                     printf("pt: %.5f %.5f wt: %.5f\n", x(0), x(1), w);
                     if (algoim::bernstein::evalBernsteinPoly(phi, x) > 0) {
                       quad_nodes.push_back(x);
                       quad_wts.push_back(w);
                     }
                   });

  std::vector<T> pts;
  for (auto node : quad_nodes) {
    for (int d = 0; d < spatial_dim; d++) {
      pts.push_back(node(d));
    }
  }

  T sum = 0.0;
  for (auto w : quad_wts) sum += w;

  std::cout << "number of exterior quadrature points: " << quad_wts.size()
            << "\n";
  std::cout << "sum of exterior weights: " << sum << "\n";

  FieldToVTKNew<T, spatial_dim> vtk("multi_components.vtk");
  vtk.add_mesh(pts);
  vtk.write_mesh();
  vtk.add_sol("quad_wts", quad_wts);
  vtk.write_sol("quad_wts");
}

int main() { quadratures_multipoly<4>(); }
