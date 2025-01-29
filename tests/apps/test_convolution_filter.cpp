#include "apps/convolution_filter.h"
#include "elements/lbracket_mesh.h"
#include "test_commons.h"

template <class Grid, class Mesh>
void test_partition_of_unity(const Grid& grid, const Mesh& mesh, double r0,
                             double tol = 1e-13) {
  ConvolutionFilter<double, Grid> conv_filter(r0, grid);

  std::vector<double> x(grid.get_num_verts(), 0.0);

  for (int i = 0; i < grid.get_num_verts(); i++) {
    x[i] = 0.6789;
  }

  std::vector<double> phi = x;

  conv_filter.apply(phi.data(), phi.data());

  for (int i = 0; i < grid.get_num_verts(); i++) {
    EXPECT_NEAR(x[i], phi[i], tol);
  }
}

template <typename T>
T scalar_function(std::vector<T>& phi, std::vector<double>& w) {
  T ret = 0.0;
  for (int i = 0; i < phi.size(); i++) {
    ret += phi[i] * w[i];
  }
  return ret;
}

template <class Grid, class Mesh>
void test_gradient(const Grid& grid, const Mesh& mesh, double r0,
                   double h = 1e-30, double tol = 1e-13) {
  using T = std::complex<double>;

  ConvolutionFilter<T, Grid> conv_filter(r0, grid);

  int ndv = mesh.get_num_nodes();

  std::vector<T> x(ndv, 0.0), phi(ndv, 0.0), dfdphi(ndv, 0.0), dfdx(ndv, 0.0);
  std::vector<double> w(ndv, 0.0), p(ndv, 0.0);

  for (int i = 0; i < ndv; i++) {
    w[i] = (double)rand() / RAND_MAX;
    p[i] = (double)rand() / RAND_MAX;
    x[i] = T((double)rand() / RAND_MAX, h * p[i]);
    dfdphi[i] = T(w[i], 0.0);
  }

  conv_filter.apply(x.data(), phi.data());
  conv_filter.applyGradient(dfdphi.data(), dfdx.data());

  conv_filter.apply(x.data(), phi.data());
  T s = scalar_function(phi, w);

  double dfdx_cs = s.imag() / h;
  double dfdx_exact = 0.0;
  for (int i = 0; i < ndv; i++) {
    dfdx_exact += dfdx[i].real() * p[i];
  }

  std::printf("dfdx_fd:    %25.15e\n", dfdx_cs);
  std::printf("dfdx_exact: %25.15e\n", dfdx_exact);
  EXPECT_NEAR((dfdx_cs - dfdx_exact) / dfdx_exact, 0.0, tol);
}

// Create nodal design variables for a domain with periodic holes
template <typename T, class Mesh>
std::vector<T> create_initial_topology(const Mesh& mesh, int nholes_x = 6,
                                       int nholes_y = 6, double r = 0.05,
                                       bool cell_center = true) {
  auto grid = mesh.get_grid();
  const T* lxy = grid.get_lxy();
  int nverts = grid.get_num_verts();
  std::vector<T> lsf(nverts, 0.0);
  for (int i = 0; i < nverts; i++) {
    T xloc[Mesh::spatial_dim];
    grid.get_vert_xloc(i, xloc);
    T x = xloc[0];
    T y = xloc[1];

    std::vector<T> lsf_vals;
    for (int ix = 0; ix < nholes_x; ix++) {
      for (int iy = 0; iy < nholes_y; iy++) {
        if (cell_center) {
          T x0 = lxy[0] / nholes_x / 2.0 * (2.0 * ix + 1.0);
          T y0 = lxy[1] / nholes_y / 2.0 * (2.0 * iy + 1.0);
          lsf_vals.push_back(r -
                             sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
        } else {
          T x0 = lxy[0] / (nholes_x - 1.0) * ix;
          T y0 = lxy[1] / (nholes_y - 1.0) * iy;
          lsf_vals.push_back(r -
                             sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
        }
      }
    }
    lsf[i] = hard_max(lsf_vals);
  }

  return lsf;
}

TEST(apps, ConvolutionzFilterRegularGrid) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d>;

  int nxy[2] = {32, 64};
  T lxy[2] = {0.6, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  T r0 = 0.123;

  test_partition_of_unity(grid, mesh, r0);
  test_gradient(grid, mesh, r0);
}

TEST(apps, ConvolutionzFilterLbracketGrid) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = LbracketGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d, Grid>;

  int nxy[2] = {96, 96};
  double lxy[2] = {1.0, 1.0};
  double lbracket_frac = 0.4;

  int nx1 = nxy[0];
  int nx2 = static_cast<int>(nxy[0] * lbracket_frac);
  int ny1 = static_cast<int>(nxy[1] * lbracket_frac);
  int ny2 = nxy[1] - ny1;
  double lx1 = lxy[0];
  double ly1 = lxy[1] * lbracket_frac;
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);

  Mesh mesh(grid);
  std::vector<T> x = create_initial_topology<T>(mesh);
  ToVTK<T, Mesh> vtk(mesh, "conv_filter_lbracket.vtk");
  vtk.write_mesh();
  vtk.write_sol("x", x.data());

  for (T r0 : std::vector<T>{0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5}) {
    test_partition_of_unity(grid, mesh, r0);
    test_gradient(grid, mesh, r0);
    ConvolutionFilter<T, Grid> conv_filter(r0, grid);
    std::vector<T> phi(x.size(), 0.0);
    conv_filter.apply(x.data(), phi.data());

    vtk.write_sol("phi_r0_" + std::to_string(r0), phi.data());
  }
}
