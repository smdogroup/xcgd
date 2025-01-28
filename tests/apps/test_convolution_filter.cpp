#include "apps/convolution_filter.h"
#include "test_commons.h"

TEST(apps, ConvolutionzFilterPartitionOfUnity) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d>;

  int nxy[2] = {64, 64};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  T r0 = 0.05;

  ConvolutionFilter<T> conv_filter(r0, grid);

  std::vector<T> x(grid.get_num_verts(), 0.0), phi(grid.get_num_verts());

  for (int i = 0; i < grid.get_num_verts(); i++) {
    x[i] = 0.6789;
  }

  conv_filter.apply(x.data(), phi.data());

  for (int i = 0; i < grid.get_num_verts(); i++) {
    EXPECT_NEAR(x[i], phi[i], 1e-15);
  }
}

TEST(apps, ConvolutionzFilterVisualize) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Mesh = GridMesh<T, Np_1d>;

  int nxy[2] = {64, 64};
  T lxy[2] = {1.0, 1.0};

  Grid grid(nxy, lxy);
  Mesh mesh(grid);

  T r0 = 0.05;

  ConvolutionFilter<T> conv_filter(r0, grid);

  std::vector<T> x(grid.get_num_verts(), 0.0), phi(grid.get_num_verts());

  for (int i = 0; i < grid.get_num_verts(); i++) {
    x[i] = (double)rand() / RAND_MAX;
  }

  conv_filter.apply(x.data(), phi.data());

  ToVTK<T, Mesh> vtk(mesh, "conv_filter.vtk");

  vtk.write_mesh();
  vtk.write_sol("x", x.data());
  vtk.write_sol("phi", phi.data());
}
