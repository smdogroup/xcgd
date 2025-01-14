#pragma once

#include <array>

#include "elements/gd_mesh.h"

template <typename T>
class LbracketGrid2D {
 public:
  static constexpr int spatial_dim = 2;
  static constexpr int nverts_per_cell = 4;

  /**
   * @brief
   *
   *     nx2
   *   -------
   *  |       |
   *  | grid2 | ny2
   *  |       |
   *  |        ------
   *  |              |  ny1, ly1
   *  |    grid1     |
   *  |              |
   *   --------------
   *       nx1, lx1
   */
  LbracketGrid2D(int nx1, int nx2, int ny1, int ny2, T lx1, T ly1)
      : nx1(nx1),
        nx2(nx2),
        ny1(ny1),
        ny2(ny2),
        lx1(lx1),
        ly1(ly1),
        grid1(std::array<int, spatial_dim>{nx1, ny1}.data(),
              std::array<T, spatial_dim>{lx1, ly1}.data()),
        grid2(std::array<int, spatial_dim>{nx2, ny2}.data(),
              std::array<T, spatial_dim>{lx1 * (T)nx2 / (T)nx1,
                                         ly1 * T(ny2) / T(ny1)}
                  .data()),
        cell_offset(grid1.get_num_cells()),
        vert_offset(grid1.get_num_verts() - nx2 - 1) {
    h[0] = lx1 / T(nx1);
    h[1] = ly1 / T(ny1);

    xy0[0] = 0.0;
    xy0[1] = 0.0;

    lxy[0] = lx1;
    lxy[1] = ly1 * T(ny1 + ny2) / T(ny1);
  }

  inline int get_num_verts() const {
    return grid1.get_num_verts() + grid2.get_num_verts() - nx2 - 1;
  }

  inline int get_num_cells() const {
    return grid1.get_num_cells() + grid2.get_num_cells();
  }

  inline bool is_valid_vert(int ix, int iy) const {
    if (iy <= ny1) {
      return (ix >= 0 and ix <= nx1 and iy >= 0);
    } else {
      return (ix >= 0 and ix <= nx2 and iy <= ny1 + ny2);
    }
  }

  // coordinates -> vert
  inline int get_coords_vert(int ix, int iy) const {
    if (iy <= ny1 + 1) {
      return grid1.get_coords_vert(ix, iy);
    } else {
      return grid2.get_coords_vert(ix, iy - ny1) + vert_offset;
    }
  }
  inline int get_coords_vert(const int* ixy) const {
    if (ixy[1] <= ny1 + 1) {
      return grid1.get_coords_vert(ixy);
    } else {
      return grid2.get_coords_vert(ixy[0], ixy[1] - ny1) + vert_offset;
    }
  }

  // vert -> coordinates
  inline void get_vert_coords(int vert, int* ixy) const {
    if (vert < grid1.get_num_verts()) {
      grid1.get_vert_coords(vert, ixy);
    } else {
      grid2.get_vert_coords(vert - vert_offset, ixy);
      ixy[1] += ny1;
    }
  }

  // coordinates -> cell
  inline int get_coords_cell(int ex, int ey) const {
    if (ey < ny1) {
      return grid1.get_coords_cell(ex, ey);
    } else {
      return grid2.get_coords_cell(ex, ey - ny1) + cell_offset;
    }
  }

  inline int get_coords_cell(const int* exy) const {
    if (exy[1] < ny1) {
      return grid1.get_coords_cell(exy);
    } else {
      return grid2.get_coords_cell(exy[0], exy[1] - ny1) + cell_offset;
    }
  }

  // cell -> coordinates
  inline void get_cell_coords(int cell, int* exy) const {
    if (cell < grid1.get_num_cells()) {
      return grid1.get_cell_coords(cell, exy);
    } else {
      grid2.get_cell_coords(cell - grid1.get_num_cells(), exy);
      exy[1] += ny1;
      return;
    }
  }

  // cell -> verts
  void get_cell_verts(int cell, int* verts) const {
    if (cell < grid1.get_num_cells()) {
      grid1.get_cell_verts(cell, verts);
    } else {
      grid2.get_cell_verts(cell - grid1.get_num_cells(), verts);
      verts[2] += vert_offset;
      verts[3] += vert_offset;
      if (verts[0] >= nx2 + 1) {
        verts[0] += vert_offset;
        verts[1] += vert_offset;
      } else {
        verts[0] += (nx1 + 1) * ny1;
        verts[1] += (nx1 + 1) * ny1;
      }
    }
  }

  // cell -> xloc for all verts
  void get_cell_verts_xloc(int cell, T* xloc) const {
    if (xloc) {
      int exy[spatial_dim];
      get_cell_coords(cell, exy);

      xloc[0] = xy0[0] + h[0] * exy[0];
      xloc[1] = xy0[1] + h[1] * exy[1];

      xloc[2] = xy0[0] + h[0] * (exy[0] + 1);
      xloc[3] = xy0[1] + h[1] * exy[1];

      xloc[4] = xy0[0] + h[0] * (exy[0] + 1);
      xloc[5] = xy0[1] + h[1] * (exy[1] + 1);

      xloc[6] = xy0[0] + h[0] * exy[0];
      xloc[7] = xy0[1] + h[1] * (exy[1] + 1);
    }
  }

  void get_cell_vert_ranges(int cell, T* xloc_min, T* xloc_max) const {
    int verts[4];
    get_cell_verts(cell, verts);

    get_vert_xloc(verts[0], xloc_min);
    get_vert_xloc(verts[2], xloc_max);
  }

  void get_vert_xloc(int vert, T* xloc) const {
    if (xloc) {
      int ixy[spatial_dim];
      get_vert_coords(vert, ixy);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xy0[d] + h[d] * T(ixy[d]);
      }
    }
  }

  // Get the xloc of the centroid of the cell
  void get_cell_xloc(int cell, T* xloc) const {
    if (xloc) {
      int ixy[spatial_dim];
      get_cell_coords(cell, ixy);
      for (int d = 0; d < spatial_dim; d++) {
        xloc[d] = xy0[d] + h[d] * (T(ixy[d]) + 0.5);
      }
    }
  }

  const T* get_lxy() const { return lxy; };
  const T* get_xy0() const { return xy0; };
  const T* get_h() const { return h; };

  template <int Np_1d>
  bool get_cell_ground_stencil(int cell, int* verts) const {
    bool is_stencil_regular = true;
    constexpr int q = Np_1d / 2;
    int exy[spatial_dim];
    get_cell_coords(cell, exy);

    if (exy[0] < q - 1) {
      exy[0] = q - 1;
      is_stencil_regular = false;
    }

    if (exy[1] < q - 1) {
      exy[1] = q - 1;
      is_stencil_regular = false;
    }

    if (exy[0] > nx1 - q) {
      exy[0] = nx1 - q;
      is_stencil_regular = false;
    }

    if (exy[1] > ny1 + ny2 - q) {
      exy[1] = ny1 + ny2 - q;
      is_stencil_regular = false;
    }

    int t1 = exy[0] - nx2 + q;
    int t2 = exy[1] - ny1 + q;
    if (t1 > 0 and t2 > 0) {
      if (t1 > t2) {
        exy[1] = ny1 - q;
      } else {
        exy[0] = nx2 - q;
      }
      is_stencil_regular = false;
    }

    int index = 0;
    for (int j = 0; j < Np_1d; j++) {
      for (int i = 0; i < Np_1d; i++, index++) {
        verts[index] = get_coords_vert(exy[0] - q + 1 + i, exy[1] - q + 1 + j);
      }
    }
    return is_stencil_regular;
  }

  template <int Np_1d>
  void check_grid_compatibility() const {
    int nxy_min[spatial_dim];
    nxy_min[0] = nx2;
    nxy_min[1] = ny1;
    for (int d = 0; d < spatial_dim; d++) {
      if (nxy_min[d] < Np_1d - 1) {
        char msg[256];
        std::snprintf(
            msg, 256,
            "too few elements (%d) for Np_1d (%d) along %d-th dimension",
            nxy_min[d], Np_1d, d);
        throw std::runtime_error(msg);
      }
    }
  }

 private:
  int nx1, nx2, ny1, ny2;
  T lx1, ly1, ly2;
  T h[spatial_dim], xy0[spatial_dim], lxy[spatial_dim];
  StructuredGrid2D<T> grid1, grid2;
  int cell_offset, vert_offset;
};
