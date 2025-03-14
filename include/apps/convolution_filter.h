#pragma once

#include "elements/gd_mesh.h"

/**
 * @brief A convolution type filter defined on a structured grid
 *
 * @tparam T numeric type
 */
template <typename T, class Grid_ = StructuredGrid2D<double>>
class ConvolutionFilter final {
 public:
  using Grid = Grid_;

  ConvolutionFilter(double r0, const Grid& grid) : grid(grid) {
    int constexpr spatial_dim = Grid::spatial_dim;
    static_assert(spatial_dim == 2);

    int stencil_bounds[spatial_dim];
    const double* h = grid.get_h();

    for (int d = 0; d < spatial_dim; d++) {
      stencil_bounds[d] = int(r0 / h[d]);
    }

    for (int i = -stencil_bounds[0]; i <= stencil_bounds[0]; i++) {
      for (int j = -stencil_bounds[1]; j <= stencil_bounds[1]; j++) {
        double d2 = i * i * h[0] * h[0] + j * j * h[1] * h[1];
        if (d2 <= r0 * r0) {
          vert_set.push_back({i, j});
          wts.push_back(r0 - sqrt(d2));
        }
      }
    }
  }

  void apply(const T* x, T* phi) {
    int nverts = grid.get_num_verts();

    const T* xin = x;
    std::vector<T> xdata;
    if (x == phi) {
      xdata = std::vector<T>(x, x + nverts);
      xin = xdata.data();
    }
    std::fill(phi, phi + nverts, 0.0);

    int nset = vert_set.size();
    for (int i = 0; i < nverts; i++) {
      int ixy[2] = {-1, -1};
      grid.get_vert_coords(i, ixy);

      T denom = 0.0;
      for (int index = 0; index < nset; index++) {
        int jxy[2] = {ixy[0] + vert_set[index].first,
                      ixy[1] + vert_set[index].second};
        if (grid.is_valid_vert(jxy[0], jxy[1])) {
          int j = grid.get_coords_vert(jxy);
          denom += wts[index];
          phi[i] += wts[index] * xin[j];
        }
      }
      phi[i] /= denom;
    }
  }

  void applyGradient(const T* dfdphi, T* dfdx) {
    int nverts = grid.get_num_verts();

    const T* dfdphi_in = dfdphi;
    std::vector<T> dfdphi_data;
    if (dfdphi == dfdx) {
      dfdphi_data = std::vector<T>(dfdphi, dfdphi + nverts);
      dfdphi_in = dfdphi_data.data();
    }
    std::fill(dfdx, dfdx + nverts, 0.0);

    int nset = vert_set.size();
    for (int i = 0; i < nverts; i++) {
      int ixy[2] = {-1, -1};
      grid.get_vert_coords(i, ixy);

      T denom = 0.0;
      for (int index = 0; index < nset; index++) {
        int jxy[2] = {ixy[0] + vert_set[index].first,
                      ixy[1] + vert_set[index].second};
        if (grid.is_valid_vert(jxy[0], jxy[1])) {
          denom += wts[index];
        }
      }

      for (int index = 0; index < nset; index++) {
        int jxy[2] = {ixy[0] + vert_set[index].first,
                      ixy[1] + vert_set[index].second};
        if (grid.is_valid_vert(jxy[0], jxy[1])) {
          int j = grid.get_coords_vert(jxy);
          dfdx[j] += wts[index] * dfdphi_in[i] / denom;
        }
      }
    }
  }

 private:
  const Grid& grid;

  // Neighborhood set of vertices within filter radius, stored as the vertex
  // coordinate offset (di, dj)
  std::vector<std::pair<int, int>> vert_set;

  // contributions of each vertex from the neighborhood set to the center vertex
  std::vector<double> wts;
};
