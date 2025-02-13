#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "test_commons.h"
#include "utils/misc.h"
#include "utils/vtk.h"

template <typename T, int spatial_dim>
void quad_pts_discriminator(
    int first_k, int last_k,
    const std::vector<algoim::uvector<T, spatial_dim>>& quad_nodes,
    const std::vector<T>& quad_lsf,
    const std::vector<algoim::uvector<int, spatial_dim>>& block_indices) {
  static_assert(spatial_dim == 2, "");
  xcgd_assert(std::set<int>{0, 1} == std::set<int>{first_k, last_k},
              "first_k and last_k should cover all axes");

  int nquads = quad_nodes.size();

  std::map<int, std::vector<int>>
      partitions;  // partition index -> quad node indices

  for (int q = 0; q < nquads; q++) {
  }
}

// functor: (const algoim::uvector<T, spatial_dim>& x) -> T, -1 <= x <= 1
template <typename T, int spatial_dim, class Functor>
void compute_and_visualize_quad_pts_single_element(std::string name,
                                                   const Functor& phi_functor) {
  int constexpr Np_bernstein = 6;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, 2>;

  // Set bounds of the hyperrectangle
  algoim::uvector<T, spatial_dim> xmin{-1.0, -1.0};
  algoim::uvector<T, spatial_dim> xmax{1.0, 1.0};

  // Evaluate the bernstein
  T data[Np_bernstein * Np_bernstein];
  algoim::xarray<T, spatial_dim> phi(
      data, algoim::uvector<int, spatial_dim>(Np_bernstein, Np_bernstein));

  algoim::bernstein::bernsteinInterpolate<spatial_dim>(
      [&](const algoim::uvector<T, spatial_dim>& x) {
        // printf("bernstein field: %10f %10f\n", x(0), x(1));
        return phi_functor(xmin + x * (xmax - xmin));
      },
      phi);

  // Visualization
  int nxy[2] = {64, 64};
  T xy0[2] = {xmin(0), xmin(1)};
  T lxy[2] = {xmax(0) - xmin(0), xmax(1) - xmin(1)};
  Grid grid(nxy, lxy, xy0);
  Mesh vtk_mesh(grid);

  std::vector<T> exact_lsf;
  std::vector<T> bernstein_lsf;
  for (int i = 0; i < vtk_mesh.get_num_nodes(); i++) {
    T xloc[2];
    grid.get_vert_xloc(i, xloc);
    algoim::uvector<T, spatial_dim> x{xloc[0], xloc[1]};
    algoim::uvector<T, spatial_dim> xref{
        (xloc[0] - xmin(0)) / (xmax(0) - xmin(0)),
        (xloc[1] - xmin(1)) / (xmax(1) - xmin(1))};
    exact_lsf.push_back(phi_functor(x));
    bernstein_lsf.push_back(algoim::bernstein::evalBernsteinPoly(phi, xref));
  }

  // vtk for visualizing a single element
  ToVTK<T, Mesh> vtk(vtk_mesh, name + "_mesh.vtk");
  vtk.write_mesh();
  vtk.write_sol("bernstein_lsf", bernstein_lsf.data());
  vtk.write_sol("exact_lsf", exact_lsf.data());

  // Compute quadrature nodes
  std::vector<algoim::uvector<T, spatial_dim>> quad_nodes;
  std::vector<algoim::uvector<int, spatial_dim>> quad_block_indices;
  std::vector<T> quad_wts, quad_lsf;
  algoim::ImplicitPolyQuadrature<spatial_dim> ipquad(phi);
  std::cout << "direction k: " << ipquad.k << "\n";
  ipquad.integrate(
      algoim::AutoMixed, Np_bernstein,
      [&](const algoim::uvector<T, spatial_dim>& x, T w) {
        // const algoim::uvector<int, spatial_dim>& block_index) {
        // if (algoim::bernstein::evalBernsteinPoly(phi, x) < 0) {
        quad_lsf.push_back(algoim::bernstein::evalBernsteinPoly(phi, x));
        quad_nodes.push_back(xmin + x * (xmax - xmin));
        quad_wts.push_back(w);
        // quad_block_indices.push_back(block_index);
        // }
      });

  std::vector<T> bernstein_grad;
  for (const auto& x : quad_nodes) {
    algoim::uvector<T, spatial_dim> g =
        algoim::bernstein::evalBernsteinPolyGradient(phi, x);
    for (int d = 0; d < spatial_dim; d++) {
      bernstein_grad.push_back(g(d));
    }
  }

  std::vector<T> block_indices;
  for (const auto& part : quad_block_indices) {
    for (int d = 0; d < spatial_dim; d++) {
      block_indices.push_back(part(d));
    }
  }

  std::vector<T> pts;
  for (auto node : quad_nodes) {
    for (int d = 0; d < spatial_dim; d++) {
      pts.push_back(node(d));
    }
  }
  FieldToVTKNew<T, spatial_dim> quad_vtk(name + "_quads.vtk");

  quad_vtk.add_mesh(pts);
  quad_vtk.write_mesh();
  quad_vtk.add_sol("quad_wts", quad_wts);
  quad_vtk.write_sol("quad_wts");

  quad_vtk.add_sol("quad_lsf", quad_lsf);
  quad_vtk.write_sol("quad_lsf");

  quad_vtk.add_vec("bernstein_grad", bernstein_grad);
  quad_vtk.write_vec("bernstein_grad");

  quad_vtk.add_vec("block_index", block_indices);
  quad_vtk.write_vec("block_index");
}

TEST(algoim, DISABLED_PartitionIdentificationCase1) {
  // Define the LSF functor
  using T = double;
  int constexpr spatial_dim = 2;
  auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) -> T {
    return -((x(0) + x(1)) * (x(0) + x(1)) -
             (x(0) - 0.4 * x(1)) * (x(0) - 0.4 * x(1)) * (x(0) - 0.4 * x(1)) *
                 (x(0) - 0.4 * x(1)) -
             0.3);
  };
  compute_and_visualize_quad_pts_single_element<T, spatial_dim>("case1",
                                                                phi_functor);
}

TEST(algoim, DISABLED_PartitionIdentificationCase2) {
  // Define the LSF functor
  using T = double;
  int constexpr spatial_dim = 2;

  const algoim::uvector<T, spatial_dim> pt1{-1.0, -1.0}, pt2{1.0, 0.0},
      pt3{-1.0, 1.0};
  T r = 0.4;

  auto phi_functor = [&](const algoim::uvector<T, spatial_dim>& x) -> T {
    T region1 = algoim::dot(x - pt1, x - pt1) - r * r;
    T region2 = algoim::dot(x - pt2, x - pt2) - r * r;
    T region3 = algoim::dot(x - pt3, x - pt3) - r * r;

    return hard_min<double>({region1, region2, region3});
  };
  compute_and_visualize_quad_pts_single_element<T, spatial_dim>("case2",
                                                                phi_functor);
}

TEST(algoim, DISABLED_PartitionIdentificationCase3) {
  // Define the LSF functor
  using T = double;
  int constexpr spatial_dim = 2;

  const algoim::uvector<T, spatial_dim> pt1{-0.6, 0.6}, pt2{-0.4, -0.6},
      pt3{0.6, 0.2};
  T r = 0.4;

  auto phi_functor = [&](const algoim::uvector<T, spatial_dim>& x) -> T {
    T region1 = algoim::dot(x - pt1, x - pt1) - r * r;
    T region2 = algoim::dot(x - pt2, x - pt2) - r * r;
    T region3 = algoim::dot(x - pt3, x - pt3) - r * r;

    return hard_min<double>({region1, region2, region3});
  };
  compute_and_visualize_quad_pts_single_element<T, spatial_dim>("case3",
                                                                phi_functor);
}

TEST(algoim, DISABLED_PartitionIdentificationCase4) {
  // Define the LSF functor
  using T = double;
  int constexpr spatial_dim = 2;

  const algoim::uvector<T, spatial_dim> pt1{-1.05, -1.1}, pt2{1.05, 1.1};
  T r = 0.8;

  auto phi_functor = [&](const algoim::uvector<T, spatial_dim>& x) -> T {
    T region1 = algoim::dot(x - pt1, x - pt1) - r * r;
    T region2 = algoim::dot(x - pt2, x - pt2) - r * r;

    return -region1 * region2;
  };
  compute_and_visualize_quad_pts_single_element<T, spatial_dim>("case4",
                                                                phi_functor);
}

TEST(algoim, DISABLED_PartitionIdentificationCase5) {
  // Define the LSF functor
  using T = double;
  int constexpr spatial_dim = 2;

  auto phi_functor = [](const algoim::uvector<T, spatial_dim>& x) -> T {
    return 1.0 - (x(0) - 1.2) * (x(0) - 1.2) / 2.0 / 2.0 -
           x(1) * x(1) / 0.5 / 0.5;
  };
  compute_and_visualize_quad_pts_single_element<T, spatial_dim>("case5",
                                                                phi_functor);
}
