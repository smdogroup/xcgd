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

template <int Np_1d>
void quadratures_multipoly() {
  using T = double;
  constexpr int spatial_dim = 2;

  // Define the LSF functor
  auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) -> T {
    return -(4.0 * (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0) -
             1.0);
  };
  // auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) -> T {
  //   return -1.0;
  // };

  // Set bounds of the hyperrectangle
  algoim::uvector<T, spatial_dim> xmin{0.0, 0.0};
  algoim::uvector<T, spatial_dim> xmax{2.0, 4.0};

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
  std::vector<algoim::uvector<T, spatial_dim>> quad_nodes_inner,
      quad_nodes_outer, quad_nodes_surf, wn_surf;
  std::vector<T> w_inner, w_outer, w_surf;
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

  ipquad.integrate_surf(algoim::AutoMixed, Np_1d,
                        [&](const algoim::uvector<T, spatial_dim>& x, T w,
                            const algoim::uvector<T, spatial_dim>& wn) {
                          quad_nodes_surf.push_back(x);
                          w_surf.push_back(w);
                          wn_surf.push_back(wn);
                          printf("wn: (%10.5f, %10.5f)\n", wn(0), wn(1));
                        });

  std::vector<T> pts_inner;
  for (auto node : quad_nodes_inner) {
    for (int d = 0; d < spatial_dim; d++) {
      pts_inner.push_back(node(d));
    }
  }

  std::vector<T> pts_surf;
  for (auto node : quad_nodes_surf) {
    for (int d = 0; d < spatial_dim; d++) {
      pts_surf.push_back(node(d));
    }
  }

  std::vector<T> wn_vec;
  for (auto node : wn_surf) {
    for (int d = 0; d < spatial_dim; d++) {
      wn_vec.push_back(node(d));
    }
  }

  T sum_inner = 0.0, sum_surf = 0.0;
  for (auto w : w_inner) sum_inner += w;
  for (auto w : w_surf) sum_surf += w;

  std::cout << "number of interior quadrature points: " << w_inner.size()
            << "\n";
  std::cout << "sum of interior weights: " << sum_inner << "\n";

  std::cout << "number of surface quadrature points: " << w_surf.size() << "\n";
  std::cout << "sum of surface weights: " << sum_surf << "\n";

  FieldToVTKNew<T, spatial_dim> vtk_inner("quadratures_multipoly_inner.vtk");

  vtk_inner.add_mesh(pts_inner);
  vtk_inner.write_mesh();
  vtk_inner.add_sol(w_inner);
  vtk_inner.write_sol("w_inner");

  FieldToVTKNew<T, spatial_dim> vtk_surf("quadratures_multipoly_surf.vtk");
  vtk_surf.add_mesh(pts_surf);
  vtk_surf.write_mesh();
  vtk_surf.add_sol(w_surf);
  vtk_surf.write_sol("w_surf");
  vtk_surf.add_vec(wn_vec);
  vtk_surf.write_vec("wn");

  std::printf("wn_vec.size(): %lu\n", wn_vec.size());
  for (int i = 0; i < wn_vec.size() / spatial_dim; i++) {
    std::printf("wn[%2d]: (%10.5f, %10.5f)\n", i, wn_vec[spatial_dim * i],
                wn_vec[spatial_dim * i + 1]);
  }
}

int main() {
  quadratures_general();
  quadratures_multipoly<10>();
}
