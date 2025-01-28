#include <vector>

#include "bernstein.hpp"
#include "quadrature_multipoly.hpp"
#include "sparkstack.hpp"
#include "utils/vtk.h"

template <typename T, int Np_1d, int spatial_dim>
void find_roots(algoim::xarray<T, spatial_dim>& phi, int k, T x0) {
  constexpr int P = Np_1d;
  T pline[P], roots[P - 1];
  algoim::uvector<T, spatial_dim - 1> xbase{x0};
  algoim::bernstein::collapseAlongAxis(phi, xbase, k, pline);
  int rcount =
      algoim::bernstein::bernsteinUnitIntervalRealRoots(pline, P, roots);

  for (int i = 0; i < rcount; i++) {
    auto x = add_component(xbase, k, roots[i]);
    std::printf("[root %d]: (%20.10e, %20.10e)\n", i, x(0), x(1));
  }
}

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

  // Compute root on left edge
  std::printf("Left edge:\n");
  find_roots<T, Np_1d, spatial_dim>(phi, 1, 0.0);
  std::printf("Right edge:\n");
  find_roots<T, Np_1d, spatial_dim>(phi, 1, 1.0);
  std::printf("Lower edge:\n");
  find_roots<T, Np_1d, spatial_dim>(phi, 0, 0.0);
  std::printf("Upper edge:\n");
  find_roots<T, Np_1d, spatial_dim>(phi, 0, 1.0);

  algoim::ImplicitPolyQuadrature<spatial_dim> ipquad(phi);
  std::cout << "direction k: " << ipquad.k << "\n";
  std::cout << "no interfaces?: " << (ipquad.k == spatial_dim) << "\n";

  // Compute quadrature nodes
  std::vector<algoim::uvector<T, spatial_dim>> quad_nodes;
  std::vector<algoim::uvector<int, spatial_dim>> quad_partitions;
  std::vector<T> quad_wts;
  ipquad.integrate(algoim::AutoMixed, Np_1d,
                   [&](const algoim::uvector<T, spatial_dim>& x, T w,
                       const algoim::uvector<int, spatial_dim>& block_index) {
                     printf("pt: %.5f %.5f wt: %.5f\n", x(0), x(1), w);
                     if (algoim::bernstein::evalBernsteinPoly(phi, x) > 0) {
                       quad_nodes.push_back(x);
                       quad_wts.push_back(w);
                       quad_partitions.push_back(block_index);
                     }
                   });

  std::vector<T> bernstein_grad;
  for (const auto& x : quad_nodes) {
    algoim::uvector<T, spatial_dim> g =
        algoim::bernstein::evalBernsteinPolyGradient(phi, x);
    for (int d = 0; d < spatial_dim; d++) {
      bernstein_grad.push_back(g(d));
    }
  }

  std::vector<T> partition;
  for (const auto& part : quad_partitions) {
    for (int d = 0; d < spatial_dim; d++) {
      partition.push_back(part(d));
    }
  }

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

  vtk.add_vec("bernstein_grad", bernstein_grad);
  vtk.write_vec("bernstein_grad");

  vtk.add_vec("partition", partition);
  vtk.write_vec("partition");
}

int main() { quadratures_multipoly<6>(); }
