#include <vector>

#include "quadrature_multipoly.hpp"
#include "utils/json.h"
#include "utils/vtk.h"

template <int Np_1d>
void quadratures_multipoly() {
  using T = double;
  constexpr int spatial_dim = 2;

  // Define the LSF functor
  auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) -> T {
    return x(0) * x(0) + 2.0 * x(1) * x(1) - 1.0;
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
        return phi_functor(xmin + x * (xmax - xmin));
      },
      phi);

  algoim::ImplicitPolyQuadrature<spatial_dim> ipquad(phi);

  // Compute quadrature nodes
  std::vector<algoim::uvector<T, spatial_dim>> qpts, qns;
  std::vector<T> ws;

  ipquad.integrate_surf(
      algoim::AutoMixed, Np_1d,
      [&](const algoim::uvector<T, spatial_dim>& x, T w,
          const algoim::uvector<T, spatial_dim>& _) {
        qpts.push_back(x);
        ws.push_back(w);

        // Evaluate the gradient on the quadrature point
        // We assume that ipquad.phi.count() == 1 here
        algoim::uvector<T, spatial_dim> g =
            algoim::bernstein::evalBernsteinPolyGradient(ipquad.phi.poly(0), x);
        qns.push_back(g);

        std::printf("pt: (%10.5f, %10.5f), gradient: (%10.5f, %10.5f)\n", x(0),
                    x(1), g(0), g(1));
      });

  std::vector<T> pts, ns;
  std::vector<std::vector<T>> qpts_json(qpts.size()), qns_json(qns.size());

  for (int i = 0; i < qpts.size(); i++) {
    for (int d = 0; d < spatial_dim; d++) {
      pts.push_back(qpts[i](d));
      qpts_json[i].push_back(qpts[i](d));
    }
  }

  for (int i = 0; i < qns.size(); i++) {
    for (int d = 0; d < spatial_dim; d++) {
      ns.push_back(qns[i](d));
      qns_json[i].push_back(qns[i](d));
    }
  }

  T wsum = 0.0;
  for (auto w : ws) wsum += w;

  std::cout << "number of surface quadrature points: " << ws.size() << "\n";
  std::cout << "sum of surface weights: " << wsum << "\n";

  json j = {
      {"quad_weights", ws},
      {"quad_points", qpts_json},
      {"quad_norms", qns_json},
  };

  write_json("quadrature_data.json", j);

  FieldToVTKNew<T, spatial_dim> vtk("surface_quadratures.vtk");
  vtk.add_mesh(pts);
  vtk.write_mesh();

  vtk.add_sol(ws);
  vtk.write_sol("weights");

  vtk.add_vec(ns);
  vtk.write_vec("quadrature_surface_norms");
}

int main() { quadratures_multipoly<3>(); }
