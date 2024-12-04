#include "analysis.h"
#include "apps/helmholtz_filter.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "lbracket_mesh.h"
#include "physics/volume.h"
#include "test_commons.h"
#include "utils/vtk.h"

// Create nodal design variables for a domain with periodic holes
template <typename T, class Grid>
std::vector<T> create_initial_topology(const Grid& grid, int nholes_x,
                                       int nholes_y, double r,
                                       bool cell_center = true) {
  const T* lxy = grid.get_lxy();
  int nverts = grid.get_num_verts();
  std::vector<T> lsf(nverts, 0.0);
  for (int i = 0; i < nverts; i++) {
    T xloc[Grid::spatial_dim];
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

  // // Normalize lsf so values are within [-1, 1]
  // T lsf_max = hard_max(lsf);
  // T lsf_min = hard_min(lsf);
  // for (int i = 0; i < nverts; i++) {
  //   if (lsf[i] < 0.0) {
  //     lsf[i] /= -lsf_min;
  //   } else {
  //     lsf[i] /= lsf_max;
  //   }
  // }

  return lsf;
}

int main() {
  using T = double;
  using Grid = LbracketGrid2D<T>;
  int constexpr Np_1d = 2;

  using Quadrature = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid>;
  using Mesh = CutMesh<T, Np_1d, Grid>;
  using Basis = GDBasis2D<T, Mesh>;

  using Filter = HelmholtzFilter<T, Np_1d, Grid>;

  int nxy[2] = {96, 96};
  T lxy[2] = {1.0, 1.0};
  double lbracket_frac = 0.4;
  int nholes_x = 7;
  int nholes_y = 7;
  T rhole = 0.04;

  int nx1(nxy[0]);
  int nx2(static_cast<int>(nxy[0] * lbracket_frac));
  int ny1(static_cast<int>(nxy[1] * lbracket_frac));
  int ny2(nxy[1] - ny1);
  T lx1(lxy[0]);
  T ly1(lxy[1] * lbracket_frac);
  Grid grid(nx1, nx2, ny1, ny2, lx1, ly1);

  Mesh mesh(grid);
  Basis basis(mesh);
  Quadrature quadrature(mesh);

  T r0 = 0.1 / (2.0 * sqrt(3.0));
  Filter filter(r0, grid);

  std::vector<T> raw =
      create_initial_topology<T, Grid>(grid, nholes_x, nholes_y, rhole);
  std::vector<T> rho(raw.size(), 0.0);
  filter.apply(raw.data(), rho.data());

  ToVTK<T, typename Filter::Mesh> vtk(filter.get_mesh(),
                                      "helmholtz_behavior.vtk");
  vtk.write_mesh();
  vtk.write_sol("raw", raw.data());
  vtk.write_sol("rho", rho.data());
}
