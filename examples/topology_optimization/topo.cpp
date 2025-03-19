#include <mpi.h>
#include <sys/types.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>

#include "ParOptOptimizer.h"
#include "analysis.h"
#include "apps/convolution_filter.h"
#include "apps/helmholtz_filter.h"
#include "apps/robust_projection.h"
#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "elements/lbracket_mesh.h"
#include "physics/grad_penalization.h"
#include "physics/stress.h"
#include "physics/volume.h"
#include "snoptProblem.hpp"
#include "utils/argparser.h"
#include "utils/exceptions.h"
#include "utils/json.h"
#include "utils/loggers.h"
#include "utils/misc.h"
#include "utils/timer.h"
#include "utils/vtk.h"

#define PI 3.141592653589793

using fspath = std::filesystem::path;

template <typename T, int Np_1d, class Grid_, bool use_finite_cell_mesh>
class ProbMeshBase {
 public:
  using Grid = Grid_;
  using Mesh = typename std::conditional<use_finite_cell_mesh,
                                         FiniteCellMesh<T, Np_1d, Grid>,
                                         CutMesh<T, Np_1d, Grid>>::type;

  virtual T get_domain_area() = 0;
  virtual int get_nvars() = 0;
  virtual Grid& get_grid() = 0;
  virtual Mesh& get_erode_mesh() = 0;
  virtual Mesh& get_dilate_mesh() = 0;

  virtual std::set<int> get_loaded_cells() { return {}; };
  virtual std::set<int> get_loaded_verts() { return {}; };

  // Protected verts are verts where we would like material on
  // dv on protected verts are constrained <= 0.5 by the optimizer
  // dv < 0.5 is material
  virtual std::set<int> get_protected_verts() { return {}; };

  // Void verts are verts where we would like void on
  // dv on protected verts are constrained >= 0.5 by the optimizer
  // dv > 0.5 is void
  virtual std::set<int> get_void_verts() { return {}; }

  // Return the boundary condition verts
  // vert index -> [(dim1, val1), (dim2, val2), ...]
  // for example, 0 -> [(0, 0), (1, 0), (2, 0)] is clamping vert 0 in all x y
  // and z directions
  virtual std::map<int, std::vector<std::pair<int, T>>> get_bc() = 0;

  // === If there is non-design verts, implement the following methods

  // Convert between full vector and reduced vector
  // expand xr -> x
  virtual std::vector<T> expand(std::vector<T> x, T non_design_val) {
    return x;
  };

  // reduce x -> xr
  virtual std::vector<T> reduce(std::vector<T> x) { return x; };

  // Convert between the index in full vector and reduced vector
  // return -1 means ir is not a reduced index
  virtual int expand_index(int ir) { return ir; };
  virtual int reduce_index(int i) { return i; };

  virtual std::set<int> get_non_design_verts() { return {}; };
};

template <typename T, int Np_1d, bool use_finite_cell_mesh>
class CantileverMesh final
    : public ProbMeshBase<T, Np_1d, StructuredGrid2D<T>, use_finite_cell_mesh> {
 private:
  using Base =
      ProbMeshBase<T, Np_1d, StructuredGrid2D<T>, use_finite_cell_mesh>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  CantileverMesh(std::array<int, Grid::spatial_dim> nxy,
                 std::array<T, Grid::spatial_dim> lxy, double loaded_frac)
      : nxy(nxy),
        lxy(lxy),
        grid(nxy.data(), lxy.data()),
        erode_mesh(grid),
        dilate_mesh(grid),
        loaded_frac(loaded_frac) {
    // Find loaded cells
    for (int iy = 0; iy < nxy[1]; iy++) {
      T xloc[Grid::spatial_dim];
      int c = this->grid.get_coords_cell(nxy[0] - 1, iy);
      this->grid.get_cell_xloc(c, xloc);
      if (xloc[1] >= lxy[1] * (1.0 - loaded_frac) / 2.0 and
          xloc[1] <= lxy[1] * (1.0 + loaded_frac) / 2.0) {
        loaded_cells.insert(c);
      }
    }

    // Find loaded verts
    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      loaded_verts.insert(verts[1]);
      loaded_verts.insert(verts[2]);
    }

    // Find protected verts
    for (int v : loaded_verts) {
      int ixy[2] = {-1, -1};
      grid.get_vert_coords(v, ixy);

      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          if (grid.is_valid_vert(ixy[0] + dx, ixy[1] + dy)) {
            protected_verts.insert(
                grid.get_coords_vert(ixy[0] + dx, ixy[1] + dy));
          }
        }
      }
    }

    // Set left verts as bc verts and constrain all DOFs
    for (int iy = 0; iy < nxy[1] + 1; iy++) {
      int vert = this->grid.get_coords_vert(0, iy);
      for (int d = 0; d < Grid::spatial_dim; d++) {
        bc[vert].push_back({d, 0.0});
      }
    }
  }

  int get_nvars() { return this->grid.get_num_verts(); }
  T get_domain_area() { return lxy[0] * lxy[1]; }

  std::set<int> get_loaded_cells() { return loaded_cells; }
  std::set<int> get_loaded_verts() { return loaded_verts; }
  std::set<int> get_protected_verts() { return protected_verts; }

  std::map<int, std::vector<std::pair<int, T>>> get_bc() { return bc; }

  Grid& get_grid() { return grid; }
  Mesh& get_erode_mesh() { return erode_mesh; }
  Mesh& get_dilate_mesh() { return dilate_mesh; }

 private:
  std::array<int, Grid::spatial_dim> nxy;
  std::array<T, Grid::spatial_dim> lxy;
  Grid grid;
  Mesh erode_mesh, dilate_mesh;

  double loaded_frac;
  std::map<int, std::vector<std::pair<int, T>>> bc;
  std::set<int> loaded_cells, loaded_verts, protected_verts;
};

/*
 * Material anchor problem mesh.
 *
 *           ┌─────────── l1(C) ───────────┐
 *
 *     ┌     ┌─────────────────────────────┐→
 *     │     │                             │→
 *     │     │                             │→
 *     │     │                             │→  (D)
 * 0.5 l1 (A)│                             │→
 *     │  ┌  ├──...                        │→
 *     │l2(B)│                             │→
 *     └  └  ------------ (E) --------------→  Symmetry axis
 *
 *
 *    - Symmetry conditions are applied to E
 *    - prescribed displacement is applied to D
 *    - B is clamped
 *    - C is constrained in y direction and free to slide in x direction
 *    - l2 / l1 = clamped ratio
 *
 * */
template <typename T, int Np_1d, bool use_finite_cell_mesh>
class AnchorMesh final
    : public ProbMeshBase<T, Np_1d, StructuredGrid2D<T>, use_finite_cell_mesh> {
 private:
  using Base =
      ProbMeshBase<T, Np_1d, StructuredGrid2D<T>, use_finite_cell_mesh>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  AnchorMesh(std::array<int, Grid::spatial_dim> nxy,
             std::array<T, Grid::spatial_dim> lxy, double clamped_frac, T D_ux)
      : nxy(nxy),
        lxy(lxy),
        grid(nxy.data(), lxy.data()),
        erode_mesh(grid),
        dilate_mesh(grid),
        clamped_frac(clamped_frac) {
    // // depth of protection in number of verts, inclusive
    // int protect_levels = 2;

    // Find clamped verts
    for (int iy = 0; iy < nxy[1] + 1; iy++) {
      int vert = this->grid.get_coords_vert(0, iy);
      T xloc[Grid::spatial_dim];
      this->grid.get_vert_xloc(vert, xloc);
      if (xloc[1] <= lxy[1] * clamped_frac) {
        for (int d = 0; d < Grid::spatial_dim; d++) {
          bc[vert].push_back({d, 0.0});
        }
      }

      // Set protected verts
      // 0.9 is a magic shrinking parameter
      if (xloc[1] <= lxy[1] * clamped_frac * 0.9) {
        protected_verts.insert(vert);
      }

      // Set void verts
      // 1.1 is a magic expanding parameter
      if (xloc[1] >= lxy[1] * clamped_frac * 1.1) {
        void_verts.insert(vert);
      }
    }

    // Set bc for edge (E)
    for (int ix = 1 /*we skip the first vert*/; ix < nxy[0] + 1; ix++) {
      int vert = this->grid.get_coords_vert(ix, 0);
      bc[vert].push_back({1, 0.0});  // constrain y-movement
    }

    // Set bc for edge (C)
    for (int ix = 0; ix < nxy[0] + 1; ix++) {
      int vert = this->grid.get_coords_vert(ix, nxy[1]);
      bc[vert].push_back({1, 0.0});  // constrain y-movement

      // Set void verts
      void_verts.insert(vert);
    }

    // Set bc for edge (D)
    for (int iy = 0; iy < nxy[1] + 1; iy++) {
      int vert = this->grid.get_coords_vert(nxy[0], iy);
      bc[vert].push_back({0, D_ux});  // prescribe the x-displacement

      // Set void verts
      void_verts.insert(vert);
    }
  }

  int get_nvars() { return this->grid.get_num_verts(); }
  T get_domain_area() { return lxy[0] * lxy[1]; }

  std::set<int> get_protected_verts() { return protected_verts; }
  std::set<int> get_void_verts() { return void_verts; }

  std::map<int, std::vector<std::pair<int, T>>> get_bc() { return bc; }

  Grid& get_grid() { return grid; }
  Mesh& get_erode_mesh() { return erode_mesh; }
  Mesh& get_dilate_mesh() { return dilate_mesh; }

 private:
  std::array<int, Grid::spatial_dim> nxy;
  std::array<T, Grid::spatial_dim> lxy;
  Grid grid;
  Mesh erode_mesh, dilate_mesh;

  double clamped_frac;
  std::map<int, std::vector<std::pair<int, T>>> bc;
  std::set<int> protected_verts, void_verts;
};

template <typename T, int Np_1d, bool use_finite_cell_mesh>
class LbracketMesh final
    : public ProbMeshBase<T, Np_1d, StructuredGrid2D<T>, use_finite_cell_mesh> {
 private:
  using Base =
      ProbMeshBase<T, Np_1d, StructuredGrid2D<T>, use_finite_cell_mesh>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  LbracketMesh(std::array<int, Grid::spatial_dim> nxy,
               std::array<T, Grid::spatial_dim> lxy, double loaded_frac,
               double lbracket_frac)
      : nxy(nxy),
        lxy(lxy),
        grid(nxy.data(), lxy.data()),
        erode_mesh(grid),
        dilate_mesh(grid),
        loaded_frac(loaded_frac),
        domain_area(lxy[0] * lxy[1] *
                    (1.0 - (1.0 - lbracket_frac) * (1.0 - lbracket_frac))) {
    T ty_offset = 0.15;
    T ty = lxy[1] * lbracket_frac;

    // Find loaded cells
    for (int iy = 0; iy < nxy[1]; iy++) {
      T xloc[Grid::spatial_dim];
      int c = this->grid.get_coords_cell(nxy[0] - 1, iy);
      this->grid.get_cell_xloc(c, xloc);
      if (xloc[1] >= ty - lxy[1] * loaded_frac - ty_offset and
          xloc[1] <= ty - ty_offset) {
        loaded_cells.insert(c);
      }
    }

    // Find loaded verts
    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      loaded_verts.insert(verts[1]);
      loaded_verts.insert(verts[2]);
    }

    // Set top verts as bc verts and constrain all DOFs
    for (int ix = 0; ix < nxy[0] + 1; ix++) {
      int vert = this->grid.get_coords_vert(ix, nxy[0]);
      for (int d = 0; d < Grid::spatial_dim; d++) {
        bc[vert].push_back({d, 0.0});
      }
    }

    // Find protected verts
    for (int v : loaded_verts) {
      int ixy[2] = {-1, -1};
      grid.get_vert_coords(v, ixy);
      for (auto [dx, dy] : std::vector<std::pair<int, int>>{
               {0, 0}, {0, -1}, {-1, 0}, {-1, -1}}) {
        if (grid.is_valid_vert(ixy[0] + dx, ixy[1] + dy)) {
          protected_verts.insert(
              grid.get_coords_vert(ixy[0] + dx, ixy[1] + dy));
        };
      }
    }

    // Find non-design verts
    T tx = lxy[0] * lbracket_frac;
    for (int v = 0; v < this->grid.get_num_verts(); v++) {
      T xloc[Grid::spatial_dim];
      this->grid.get_vert_xloc(v, xloc);
      if (xloc[0] > tx and xloc[1] > ty) {
        non_design_verts.insert(v);
      }
    }

    N = this->grid.get_num_verts();
    Nr = N - non_design_verts.size();

    // Find mappings
    int ir = 0;
    for (int i = 0; i < N; i++) {
      if (non_design_verts.count(i)) {
        reduce_mapping[i] = -1;
      } else {
        expand_mapping[ir] = i;
        reduce_mapping[i] = ir;
        ir++;
      }
    }

    xcgd_assert(
        ir == Nr,
        "an error has occurred when constructing the expand-reduce mappings");
  }

  // int get_nvars() { return Nr; }  // TODO: revert
  int get_nvars() { return N; }  // TODO: delete

  // T get_domain_area() { return domain_area; }  // TODO: revert
  T get_domain_area() { return lxy[0] * lxy[1]; }  // TODO: delete

  std::set<int> get_loaded_cells() { return loaded_cells; }
  std::set<int> get_loaded_verts() { return loaded_verts; }
  // std::set<int> get_non_design_verts() { return non_design_verts; }  // TODO:
  // revert
  std::set<int> get_non_design_verts() { return {}; }  // TODO: delete
  std::set<int> get_protected_verts() { return protected_verts; }

  std::map<int, std::vector<std::pair<int, T>>> get_bc() { return bc; }

  // TODO: delete
  std::vector<T> expand(std::vector<T> xr, T _) { return xr; }
  std::vector<T> reduce(std::vector<T> x) { return x; }

  int expand_index(int ir) { return ir; }
  int reduce_index(int i) { return i; }

  // TODO: revert
  /*
  std::vector<T> expand(std::vector<T> xr, T non_design_val) {
    std::vector<T> x(N, non_design_val);
    for (int ir = 0; ir < Nr; ir++) {
      int i = expand_mapping[ir];
      x[i] = xr[ir];
    }
    return x;
  }

  std::vector<T> reduce(std::vector<T> x) {
    std::vector<T> xr(Nr);
    for (int ir = 0; ir < Nr; ir++) {
      int i = expand_mapping[ir];
      xr[ir] = x[i];
    }
    return xr;
  }

  int expand_index(int ir) { return expand_mapping[ir]; }
  int reduce_index(int i) { return reduce_mapping[i]; }
  */

  Grid& get_grid() { return grid; }
  Mesh& get_erode_mesh() { return erode_mesh; }
  Mesh& get_dilate_mesh() { return dilate_mesh; }

 private:
  std::array<int, Grid::spatial_dim> nxy;
  std::array<T, Grid::spatial_dim> lxy;
  Grid grid;
  Mesh erode_mesh, dilate_mesh;
  double loaded_frac, domain_area;
  std::map<int, std::vector<std::pair<int, T>>> bc;
  std::set<int> loaded_cells, loaded_verts, protected_verts;
  std::set<int> non_design_verts;
  int N, Nr;
  std::map<int, int> reduce_mapping, expand_mapping;
};

// load_top: true to put load on the top of the lbracket arm, false to put load
// on the side
template <typename T, int Np_1d, bool load_top = false,
          bool use_finite_cell_mesh = false>
class LbracketGridMesh final
    : public ProbMeshBase<T, Np_1d, LbracketGrid2D<T>, use_finite_cell_mesh> {
 private:
  using Base = ProbMeshBase<T, Np_1d, LbracketGrid2D<T>, use_finite_cell_mesh>;

 public:
  using typename Base::Grid;
  using typename Base::Mesh;

  LbracketGridMesh(std::array<int, Grid::spatial_dim> nxy,
                   std::array<T, Grid::spatial_dim> lxy, double loaded_frac,
                   double lbracket_frac)
      : nx1(nxy[0]),
        nx2(static_cast<int>(nxy[0] * lbracket_frac)),
        ny1(static_cast<int>(nxy[1] * lbracket_frac)),
        ny2(nxy[1] - ny1),
        lx1(lxy[0]),
        ly1(lxy[1] * lbracket_frac),
        grid(nx1, nx2, ny1, ny2, lx1, ly1),
        erode_mesh(grid),
        dilate_mesh(grid),
        loaded_frac(loaded_frac),
        domain_area(lxy[0] * lxy[1] *
                    (1.0 - (1.0 - lbracket_frac) * (1.0 - lbracket_frac))) {
    // Find loaded cells
    if constexpr (load_top) {
      for (int ix = 0; ix < nx1; ix++) {
        T xloc[Grid::spatial_dim];
        int c = this->grid.get_coords_cell(ix, ny1 - 1);
        this->grid.get_cell_xloc(c, xloc);
        if (xloc[0] >= lx1 * (1.0 - loaded_frac) and xloc[0] <= lx1) {
          loaded_cells.insert(c);
        }
      }
    } else {
      for (int iy = 0; iy < ny1; iy++) {
        T xloc[Grid::spatial_dim];
        int c = this->grid.get_coords_cell(nx1 - 1, iy);
        this->grid.get_cell_xloc(c, xloc);
        if (xloc[1] >= ly1 - lxy[1] * loaded_frac and xloc[1] <= ly1) {
          loaded_cells.insert(c);
        }
      }
    }

    // Find loaded verts
    for (int cell : loaded_cells) {
      int verts[Grid::nverts_per_cell];
      this->grid.get_cell_verts(cell, verts);
      if constexpr (load_top) {
        loaded_verts.insert(verts[2]);
        loaded_verts.insert(verts[3]);
      } else {
        loaded_verts.insert(verts[1]);
        loaded_verts.insert(verts[2]);
      }
    }

    // Find protected verts
    for (int v : loaded_verts) {
      int ixy[2] = {-1, -1};
      grid.get_vert_coords(v, ixy);
      for (auto [dx, dy] : std::vector<std::pair<int, int>>{
               {0, 0}, {0, 1}, {0, -1}, {-1, 0}, {-1, -1}}) {
        if (grid.is_valid_vert(ixy[0] + dx, ixy[1] + dy)) {
          protected_verts.insert(
              grid.get_coords_vert(ixy[0] + dx, ixy[1] + dy));
        };
      }
    }

    // Set top verts as bc verts and constrain all DOFs
    for (int ix = 0; ix < nx2 + 1; ix++) {
      int vert = this->grid.get_coords_vert(ix, ny1 + ny2);
      for (int d = 0; d < Grid::spatial_dim; d++) {
        bc[vert].push_back({d, 0.0});
      }
    }
  }

  int get_nvars() { return this->grid.get_num_verts(); }

  T get_domain_area() { return domain_area; }

  std::set<int> get_loaded_cells() { return loaded_cells; }
  std::set<int> get_loaded_verts() { return loaded_verts; }
  std::set<int> get_non_design_verts() { return {}; }
  std::set<int> get_protected_verts() { return protected_verts; }

  std::map<int, std::vector<std::pair<int, T>>> get_bc() { return bc; }

  // Dummy methods, since we don't have non-design nodes for this mesh
  std::vector<T> expand(std::vector<T> xr, T _) { return xr; }
  std::vector<T> reduce(std::vector<T> x) { return x; }

  int expand_index(int ir) { return ir; }
  int reduce_index(int i) { return i; }

  Grid& get_grid() { return grid; }
  Mesh& get_erode_mesh() { return erode_mesh; }
  Mesh& get_dilate_mesh() { return dilate_mesh; }

 private:
  int nx1, nx2, ny1, ny2;
  T lx1, ly1;
  Grid grid;
  Mesh erode_mesh, dilate_mesh;
  double loaded_frac, domain_area;
  std::map<int, std::vector<std::pair<int, T>>> bc;
  std::set<int> loaded_cells, loaded_verts, protected_verts;
};

enum class TwoMaterial { OFF, ERSATZ, NITSCHE };

template <typename T, int Np_1d, int Np_1d_filter,
          TwoMaterial two_material_method_, class Grid_,
          bool use_finite_cell_mesh>
class TopoAnalysis {
 public:
  static constexpr TwoMaterial two_material_method = two_material_method_;
  static constexpr int get_spatial_dim() { return Grid_::spatial_dim; }
  static constexpr int dof_per_node =
      get_spatial_dim();  // FIXME: this is not the best practice, should get
                          // from Physics

 private:
  using ProbMesh = ProbMeshBase<T, Np_1d, Grid_, use_finite_cell_mesh>;
  using Grid = typename ProbMesh::Grid;
  using Mesh = typename ProbMesh::Mesh;
  using Quadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid, Np_1d, Mesh>;
  using SurfQuadrature =
      GDLSFQuadrature2D<T, Np_1d, QuadPtType::SURFACE, Grid, Np_1d, Mesh>;
  using Basis = GDBasis2D<T, Mesh>;
  using HFilter = HelmholtzFilter<T, Np_1d_filter, Grid>;
  using CFilter = ConvolutionFilter<T, Grid>;

  constexpr static auto int_func =
      [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        return A2D::Vec<T, Basis::spatial_dim>{};
      };
  constexpr static auto load_func =
      [](const A2D::Vec<T, Basis::spatial_dim>& xloc) {
        A2D::Vec<T, Basis::spatial_dim> ret;
        ret(1) = -1.0;
        return ret;
      };

  using ElasticVanilla =
      StaticElastic<T, Mesh, Quadrature, Basis, typeof(int_func)>;
  using ElasticErsatz =
      StaticElasticErsatz<T, Mesh, Quadrature, Basis, typeof(int_func), Grid>;
  using ElasticNitsche = StaticElasticNitscheTwoSided<T, Mesh, Quadrature,
                                                      Basis, typeof(int_func)>;

  using Elastic = typename std::conditional<
      two_material_method == TwoMaterial::OFF, ElasticVanilla,
      typename std::conditional<two_material_method == TwoMaterial::ERSATZ,
                                ElasticErsatz, ElasticNitsche>::type>::type;

  using Volume = VolumePhysics<T, Basis::spatial_dim>;
  using Penalization = GradPenalization<T, Basis::spatial_dim>;
  using Stress = LinearElasticity2DVonMisesStress<T>;
  using StressKS = LinearElasticity2DVonMisesStressAggregation<T>;
  using SurfStress = LinearElasticity2DSurfStress<T>;
  using SurfStressKS = LinearElasticity2DSurfStressAggregation<T>;
  using VolAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Volume>;
  using PenalizationAnalysis =
      GalerkinAnalysis<T, typename HFilter::Mesh, typename HFilter::Quadrature,
                       typename HFilter::Basis, Penalization>;
  using StressAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, Stress>;
  using SurfStressAnalysis =
      GalerkinAnalysis<T, Mesh, SurfQuadrature, Basis, SurfStress>;

  static constexpr bool from_to_grid_mesh =
      (two_material_method == TwoMaterial::ERSATZ);

  using StressKSAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, StressKS, from_to_grid_mesh>;
  using SurfStressKSAnalysis =
      GalerkinAnalysis<T, Mesh, SurfQuadrature, Basis, SurfStressKS,
                       from_to_grid_mesh>;

  using StrainStress = LinearElasticity2DStrainStress<T>;
  using StrainStressAnalysis =
      GalerkinAnalysis<T, Mesh, Quadrature, Basis, StrainStress,
                       from_to_grid_mesh>;
  using BulkInt = BulkIntegration<T, Grid::spatial_dim>;
  using BulkIntAnalysis = GalerkinAnalysis<T, Mesh, Quadrature, Basis, BulkInt>;

  using LoadPhysics =
      ElasticityExternalLoad<T, Basis::spatial_dim, typeof(load_func)>;
  using LoadQuadratureRight =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::RIGHT, Mesh>;
  using LoadAnalysisRight =
      GalerkinAnalysis<T, Mesh, LoadQuadratureRight, Basis, LoadPhysics,
                       from_to_grid_mesh>;
  using LoadQuadratureTop =
      GDGaussQuadrature2D<T, Np_1d, QuadPtType::SURFACE, SurfQuad::TOP, Mesh>;
  using LoadAnalysisTop = GalerkinAnalysis<T, Mesh, LoadQuadratureTop, Basis,
                                           LoadPhysics, from_to_grid_mesh>;

  int constexpr static spatial_dim = Basis::spatial_dim;

 public:
  TopoAnalysis(ProbMesh& prob_mesh, std::string prefix,
               const ConfigParser& parser)
      : prob_mesh(prob_mesh),
        prefix(prefix),
        parser(parser),
        grid(prob_mesh.get_grid()),
        erode_mesh(prob_mesh.get_erode_mesh()),
        dilate_mesh(prob_mesh.get_dilate_mesh()),
        erode_quadrature(erode_mesh),
        dilate_quadrature(dilate_mesh),
        erode_surf_quadrature(erode_mesh),
        erode_basis(erode_mesh),
        dilate_basis(dilate_mesh),
        projector_blueprint(parser.get_double_option("robust_proj_beta"), 0.5,
                            grid.get_num_verts()),
        projector_dilate(
            parser.get_double_option("robust_proj_beta"),
            0.5 + parser.get_double_option("robust_proj_delta_eta"),
            grid.get_num_verts()),
        projector_erode(parser.get_double_option("robust_proj_beta"),
                        0.5 - parser.get_double_option("robust_proj_delta_eta"),
                        grid.get_num_verts()),
        hfilter(parser.get_double_option("filter_r0"), grid),
        cfilter(parser.get_double_option("filter_r0"), grid),
        vol_analysis(dilate_mesh, dilate_quadrature, dilate_basis, vol),
        pen(parser.get_double_option("grad_penalty_coeff")),
        pen_analysis(hfilter.get_mesh(), hfilter.get_quadrature(),
                     hfilter.get_basis(), pen),
        stress(parser.get_double_option("E"), parser.get_double_option("nu")),
        stress_analysis(erode_mesh, erode_quadrature, erode_basis, stress),
        stress_ks(parser.get_double_option("stress_ksrho_init"),
                  parser.get_double_option("E"), parser.get_double_option("nu"),
                  parser.get_double_option("yield_stress"), 1.0,
                  parser.get_bool_option("stress_use_discrete_ks")),
        stress_ks_analysis(erode_mesh, erode_quadrature, erode_basis,
                           stress_ks),
        surf_stress(parser.get_double_option("E"),
                    parser.get_double_option("nu"), SurfStressType::normal),
        surf_stress_analysis(erode_mesh, erode_surf_quadrature, erode_basis,
                             surf_stress),
        surf_stress_ks(parser.get_double_option("stress_ksrho_init"),
                       parser.get_double_option("E"),
                       parser.get_double_option("nu"),
                       parser.get_double_option("surf_yield_stress"), 1.0,
                       SurfStressType::normal),
        surf_stress_ks_analysis(erode_mesh, erode_surf_quadrature, erode_basis,
                                surf_stress_ks),
        bulk_int_analysis(erode_mesh, erode_quadrature, erode_basis, bulk_int),
        phi_erode(erode_mesh.get_lsf_dof()),
        phi_dilate(dilate_mesh.get_lsf_dof()),
        phi_blueprint(phi_dilate.size(), 0.0),
        cache({{"x", {}}, {"sol", {}}, {"chol", nullptr}}) {
    // Get loaded cells
    loaded_cells = prob_mesh.get_loaded_cells();

    // Instantiate the elastic app
    if constexpr (two_material_method == TwoMaterial::OFF) {
      elastic = std::make_shared<Elastic>(
          parser.get_double_option("E"), parser.get_double_option("nu"),
          erode_mesh, erode_quadrature, erode_basis, int_func);
    } else if constexpr (two_material_method == TwoMaterial::ERSATZ) {
      double E = parser.get_double_option("E");
      double nu = parser.get_double_option("nu");
      double E2 = parser.get_double_option("E2");
      double nu2 = parser.get_double_option("nu2");
      elastic =
          std::make_shared<Elastic>(E, nu, erode_mesh, erode_quadrature,
                                    erode_basis, int_func, E2 / E, nu2 / nu);
    } else if constexpr (two_material_method == TwoMaterial::NITSCHE) {
      double nitsche_eta = parser.get_double_option("nitsche_eta");
      double E1 = parser.get_double_option("E");
      double nu1 = parser.get_double_option("nu");
      double E2 = parser.get_double_option("E2");
      double nu2 = parser.get_double_option("nu2");
      elastic =
          std::make_shared<Elastic>(nitsche_eta, E1, nu1, E2, nu2, erode_mesh,
                                    erode_quadrature, erode_basis, int_func);
    } else {
      throw std::runtime_error("unknown two_material_method");
    }
  }

  // construct the initial design using sine waves
  std::vector<T> create_initial_topology_sine(int period_x, int period_y,
                                              double offset) {
    const T* lxy = grid.get_lxy();
    int nverts = grid.get_num_verts();
    std::vector<T> dvs(nverts, 0.0);
    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      grid.get_vert_xloc(i, xloc);
      T x = xloc[0];
      T y = xloc[1];
      dvs[i] = -cos(x / lxy[0] * 2.0 * PI * period_x) *
                   cos(y / lxy[1] * 2.0 * PI * period_y) -
               offset;
      dvs[i] = dvs[i] / (1.0 + offset) / 2.0 +
               0.5;  // scale dvs uniformly so it's in [0, 1]
    }
    return dvs;
  }

  // create a quarter of ellipse centered at (0, 0)
  // a: semi-major axis
  // b: semi-minor axis
  std::vector<T> create_initial_topology_anchor(double a, double b) {
    int nverts = grid.get_num_verts();
    std::vector<T> dvs(nverts, 0.0);
    const T* lxy = grid.get_lxy();

    T dv_max = lxy[0] * lxy[0] / a / a + lxy[1] * lxy[1] / b / b - 1.0;
    T t = dv_max > 1.0 ? dv_max : 1.0;
    t = 2.01 * t;

    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      grid.get_vert_xloc(i, xloc);
      T x = xloc[0];
      T y = xloc[1];
      dvs[i] = (x * x / a / a + y * y / b / b - 1.0) / t +
               0.5;  // in [0, 1] and dvs == 0.5 is the cut boundary
    }

    return dvs;
  }

  // Create nodal design variables for a domain with periodic holes,
  // specifically designed for the L-bracket case
  std::vector<T> create_initial_topology_lbracket(int nholes_1d = 6,
                                                  double r = 0.15) {
    const T* lxy = grid.get_lxy();
    int nverts = grid.get_num_verts();
    std::vector<T> dvs(nverts, 0.0);

    double dx = lxy[0] / (nholes_1d - 1);
    double dy = lxy[1] / (nholes_1d - 1);

    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      grid.get_vert_xloc(i, xloc);
      T x = xloc[0];
      T y = xloc[1];

      std::vector<T> dvs_vals;
      for (int ix = 0; ix < nholes_1d; ix++) {
        for (int iy = 0; iy < nholes_1d; iy++) {
          if ((ix + iy) % 2) {
            continue;
          }

          T x0 = ix * dx;
          T y0 = iy * dy;

          dvs_vals.push_back(r -
                             sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
        }
      }
      dvs[i] = hard_max(dvs_vals);
    }

    // This is a (conservative) maximum possible magnitude of the
    // signed-distance function
    T l_max = sqrt(lxy[0] * lxy[0] / nholes_1d / nholes_1d +
                   lxy[1] * lxy[1] / nholes_1d / nholes_1d);

    // scale dv values so they're in [0, 1]
    for (int i = 0; i < nverts; i++) {
      dvs[i] = (dvs[i] + l_max) / 2.0 / l_max;
    }

    return dvs;
  }

  // Create nodal design variables for a domain with periodic holes
  std::vector<T> create_initial_topology(
      int nholes_x, int nholes_y, double r, bool cell_center = true,
      int shrink_level = 0,  // number of verts we shrink
      bool is_lbracket = false, double lbracket_frac = 0.4,
      double loaded_frac = 0.05) {
    const double* lxy = grid.get_lxy();
    const double* h = grid.get_h();
    double dh = 0.5 * (h[0] + h[1]);
    int nverts = grid.get_num_verts();
    std::vector<T> dvs(nverts, 0.0);
    for (int i = 0; i < nverts; i++) {
      T xloc[Mesh::spatial_dim];
      grid.get_vert_xloc(i, xloc);
      T x = xloc[0];
      T y = xloc[1];

      std::vector<T> dvs_vals;
      for (int ix = 0; ix < nholes_x; ix++) {
        for (int iy = 0; iy < nholes_y; iy++) {
          if (cell_center) {
            T x0 = lxy[0] / nholes_x / 2.0 * (2.0 * ix + 1.0);
            T y0 = lxy[1] / nholes_y / 2.0 * (2.0 * iy + 1.0);
            dvs_vals.push_back(r -
                               sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
          } else {
            T x0 = lxy[0] / (nholes_x - 1.0) * ix;
            T y0 = lxy[1] / (nholes_y - 1.0) * iy;
            dvs_vals.push_back(r -
                               sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0)));
          }
        }
      }
      dvs[i] = hard_max(dvs_vals);

      // Shrink domain on the boundaries
      int ixy[2];
      grid.get_vert_coords(i, ixy);

      // if (ixy[0] < shrink_level or ixy[1] < shrink_level) {
      //   dvs[i] = (shrink_level + 0.5) * dh;
      // }

      /**
       * @brief
       *        |
       *        |-> b1
       *        |
       *     nx2
       *   -------      ||
       *  |       |  b3<||> b4
       *  | grid2 | ny2 ||
       *  |       |                      ^
       *  |        ------                |  b2
       *  |              |  ny1, ly1    ----
       *  |    grid1     |              ----
       *  |              | ---           |  b5
       *   --------------   | b7
       *       nx1, lx1
       *    |
       *  <-| b6
       */
      if (is_lbracket) {
        bool b1 = ixy[0] * h[0] >
                  lbracket_frac * lxy[0] - (shrink_level - 0.5) * h[0];
        bool b2 = ixy[1] * h[1] >
                  lbracket_frac * lxy[1] - (shrink_level - 0.5) * h[1];
        bool b3 = ixy[0] * h[0] < lxy[0] * (1.0 - 1.5 * loaded_frac) -
                                      (shrink_level - 0.5) * h[0];
        bool b4 = ixy[0] * h[0] < lxy[0] - (shrink_level - 0.5) * h[0];
        bool b5 =
            (ixy[1] + 1.0) * h[1] < (lbracket_frac - loaded_frac) * lxy[1] -
                                        (shrink_level - 0.5) * h[1];

        bool b6 = ixy[0] * h[0] < (shrink_level - 0.5) * h[0];
        bool b7 = ixy[1] * h[1] < (shrink_level - 0.5) * h[1];

        if (b1 and b2 and b3) {
          dvs[i] = (shrink_level + 0.5) * dh;
        }
        if (!b4 and b5) {
          dvs[i] = (shrink_level + 0.5) * dh;
        }
        if (b6 or b7) {
          dvs[i] = (shrink_level + 0.5) * dh;
        }
      }
    }

    // This is a (conservative) maximum possible magnitude of the
    // signed-distance function
    T l_max = sqrt(lxy[0] * lxy[0] / nholes_x / nholes_x +
                   lxy[1] * lxy[1] / nholes_y / nholes_y);

    // scale dv values so they're in [0, 1]
    for (int i = 0; i < nverts; i++) {
      dvs[i] = (dvs[i] + l_max) / 2.0 / l_max;
    }

    return dvs;
  }

  // Map filtered x -> phi, recall that
  // x  -> filtered x -> phi
  // where x, filtered x are in [0, 1], and phi in h * fx - 0.5h, h = element
  // size
  void design_mapping_apply(const T* x, T* y) {
    double elem_h = 0.5 * (grid.get_h()[0] + grid.get_h()[1]);
    int ndv = phi_blueprint.size();
    for (int i = 0; i < ndv; i++) {
      y[i] = elem_h * (x[i] - 0.5);
    }
  }

  void design_mapping_apply_gradient(const T* dfdy, T* dfdx) {
    double elem_h = 0.5 * (grid.get_h()[0] + grid.get_h()[1]);
    int ndv = phi_blueprint.size();
    for (int i = 0; i < ndv; i++) {
      dfdx[i] = elem_h * dfdy[i];
    }
  }

  std::vector<T> update_mesh(const std::vector<T>& x) {
    if (x.size() != phi_blueprint.size()) {
      throw std::runtime_error("sizes don't match");
    }

    // Get options
    bool use_helmholtz_filter = parser.get_bool_option("use_helmholtz_filter");
    int num_conv_filter_apply = parser.get_int_option("num_conv_filter_apply");
    bool use_robust_projection =
        parser.get_bool_option("use_robust_projection");

    // Apply filter on design variables
    if (use_helmholtz_filter) {
      hfilter.apply(x.data(), phi_blueprint.data());
    } else {
      cfilter.apply(x.data(), phi_blueprint.data());
      for (int i = 0; i < num_conv_filter_apply - 1; i++) {
        cfilter.apply(phi_blueprint.data(), phi_blueprint.data());
      }
    }

    // Apply projection
    if (use_robust_projection) {
      projector_dilate.apply(phi_blueprint.data(), phi_dilate.data());
      projector_erode.apply(phi_blueprint.data(), phi_erode.data());
      projector_blueprint.apply(phi_blueprint.data(), phi_blueprint.data());
    } else {
      phi_dilate = phi_blueprint;
      phi_erode = phi_blueprint;
    }

    // Save H(F(x))
    std::vector<T> HFx = phi_blueprint;

    // Apply design mapping
    design_mapping_apply(phi_blueprint.data(), phi_blueprint.data());
    design_mapping_apply(phi_dilate.data(), phi_dilate.data());
    design_mapping_apply(phi_erode.data(), phi_erode.data());

    // Update the mesh given new phi value
    erode_mesh.update_mesh();

    // Update dilate mesh used for volume
    dilate_mesh.update_mesh();

    // Update main mesh for elastic analysis
    elastic->update_mesh();

    // Update bc dof for elastic
    bc_dof.clear();
    bc_vals.clear();
    std::map<int, std::vector<std::pair<int, T>>> bc = prob_mesh.get_bc();

    auto& bc_mesh = elastic->get_mesh();

    for (auto& [vert, dim_val_v] : bc) {
      for (auto& dim_val : dim_val_v) {
        int d = dim_val.first;
        T val = dim_val.second;
        if constexpr (two_material_method == TwoMaterial::OFF) {
          if (bc_mesh.get_vert_nodes().count(vert)) {
            int n = bc_mesh.get_vert_nodes().at(vert);
            bc_dof.push_back(spatial_dim * n + d);
            bc_vals.push_back(val);
          }
        } else if constexpr (two_material_method == TwoMaterial::ERSATZ) {
          bc_dof.push_back(spatial_dim * vert + d);
          bc_vals.push_back(val);
        } else if constexpr (two_material_method == TwoMaterial::NITSCHE) {
          auto bc_mesh_ersatz = elastic->get_mesh_ersatz();
          int node_offset = bc_mesh.get_num_nodes();

          if (bc_mesh.get_vert_nodes().count(vert)) {
            int n = bc_mesh.get_vert_nodes().at(vert);
            bc_dof.push_back(spatial_dim * n + d);
            bc_vals.push_back(val);
          }

          if (bc_mesh_ersatz.get_vert_nodes().count(vert)) {
            int n = bc_mesh_ersatz.get_vert_nodes().at(vert) + node_offset;
            bc_dof.push_back(spatial_dim * n + d);
            bc_vals.push_back(val);
          }

        } else {
          throw std::runtime_error("unknown two_material_method");
        }
      }
    }

    xcgd_assert(bc_dof.size() == bc_vals.size(),
                "bc_dof and bc_vals have different sizes (" +
                    std::to_string(bc_dof.size()) + ", " +
                    std::to_string(bc_vals.size()) + ")");

    return HFx;
  }

  std::vector<T> update_mesh_and_solve(
      const std::vector<T>& x, std::vector<T>& HFx,
      std::shared_ptr<SparseUtils::SparseCholesky<T>>* chol = nullptr) {
    // Solve the static problem
    HFx = update_mesh(x);

    VandermondeCondLogger::clear();
    VandermondeCondLogger::enable();

    try {
      LoadPhysics load_physics(load_func);
      std::set<int> load_elements;
      auto& elastic_mesh = elastic->get_mesh();
      int elastic_nelems = elastic_mesh.get_num_elements();
      for (int i = 0; i < elastic_nelems; i++) {
        if (loaded_cells.count(elastic_mesh.get_elem_cell(i))) {
          load_elements.insert(i);
        }
      }

      std::vector<T> sol;

      bool load_top = parser.get_bool_option("load_top");
      if (load_top) {
        LoadQuadratureTop load_quadrature(elastic_mesh, load_elements);
        LoadAnalysisTop load_analysis(elastic_mesh, load_quadrature,
                                      elastic->get_basis(), load_physics);
        sol = elastic->solve(bc_dof, bc_vals,
                             std::tuple<LoadAnalysisTop>(load_analysis), chol);
      } else {
        LoadQuadratureRight load_quadrature(elastic_mesh, load_elements);
        LoadAnalysisRight load_analysis(elastic_mesh, load_quadrature,
                                        elastic->get_basis(), load_physics);
        sol =
            elastic->solve(bc_dof, bc_vals,
                           std::tuple<LoadAnalysisRight>(load_analysis), chol);
      }

      return sol;
    } catch (const StencilConstructionFailed& e) {
      std::printf(
          "StencilConstructionFailed error has been caught when calling "
          "update_mesh_and_solve(), dumping debug info in a vtk and "
          "throwing...\n");
      auto cut_mesh = elastic->get_mesh();
      ToVTK<T, Mesh> err_vtk(
          cut_mesh, fspath(prefix) / "stencil_construction_failed.vtk");
      err_vtk.write_mesh();

      std::vector<double> failed_elem(cut_mesh.get_num_elements(), 0.0);
      failed_elem[e.get_elem_index()] = 1.0;
      err_vtk.write_cell_sol("failed_element", failed_elem.data());

      throw e;
    }

    VandermondeCondLogger::disable();
  }

  auto eval_obj_con(const std::vector<T>& x) {
    // Get options
    bool stress_use_discrete_ks =
        parser.get_bool_option("stress_use_discrete_ks");

    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol;
    std::vector<T> HFx;
    std::vector<T> sol = update_mesh_and_solve(x, HFx, &chol);

    T comp = std::inner_product(sol.begin(), sol.end(),
                                elastic->get_rhs().begin(), T(0.0));

    std::vector<T> vol_dummy(vol_analysis.get_mesh().get_num_nodes(), 0.0);
    T area = vol_analysis.energy(nullptr, vol_dummy.data());

    T pterm = pen_analysis.energy(nullptr, phi_blueprint.data());

    auto [xloc_q, stress_q] = eval_stress(sol);
    xcgd_assert(stress_q.size() > 0,
                "error has occured on quadrature stress evaluation");
    T max_stress = *std::max_element(stress_q.begin(), stress_q.end());
    T max_stress_ratio = max_stress / stress_ks.get_yield_stress();
    stress_ks.set_max_stress_ratio(max_stress_ratio);

    T ks_energy = stress_ks_analysis.energy(nullptr, sol.data());

    // Evaluate continuous or discrete ks aggregation
    T ks_stress_ratio = 0.0;
    if (stress_use_discrete_ks) {
      ks_stress_ratio =
          max_stress_ratio + log(ks_energy) / stress_ks.get_ksrho();
    } else {
      ks_stress_ratio =
          max_stress_ratio + log(ks_energy / (prob_mesh.get_domain_area())) /
                                 stress_ks.get_ksrho();
    }

    auto [xloc_surf_q, stress_surf_q] = eval_surf_stress(sol);
    xcgd_assert(stress_surf_q.size() > 0,
                "error has occured on surface quadrature stress evaluation");
    T max_surf_stress =
        *std::max_element(stress_surf_q.begin(), stress_surf_q.end());
    T max_surf_stress_ratio =
        max_surf_stress / surf_stress_ks.get_yield_stress();

    T surf_ks_energy = surf_stress_ks_analysis.energy(nullptr, sol.data());

    T surf_ks_stress_ratio = max_surf_stress_ratio +
                             log(surf_ks_energy) / surf_stress_ks.get_ksrho();

    // Save information
    cache["x"] = x;
    cache["sol"] = sol;
    cache["chol"] = chol;
    cache["area"] = area;
    cache["ks_energy"] = ks_energy;

    return std::make_tuple(HFx, comp, area, pterm, max_stress, max_stress_ratio,
                           ks_stress_ratio, surf_ks_stress_ratio, sol, xloc_q,
                           stress_q, xloc_surf_q, stress_surf_q);
  }

  // only useful if ersatz material is used
  template <int dim>
  std::vector<T> grid_dof_to_cut_dof(const Mesh& mesh, const std::vector<T> u) {
    if (u.size() != mesh.get_grid().get_num_verts() * dim) {
      throw std::runtime_error(
          "grid_dof_to_cut_dof() only takes dof vectors of size nverts * dim "
          "(" +
          std::to_string(mesh.get_grid().get_num_verts() * dim) + "), got " +
          std::to_string(u.size()));
    }

    int nnodes = mesh.get_num_nodes();
    std::vector<T> u0(dim * nnodes);
    for (int i = 0; i < nnodes; i++) {
      int vert = mesh.get_node_vert(i);
      for (int d = 0; d < dim; d++) {
        u0[dim * i + d] = u[dim * vert + d];
      }
    }
    return u0;
  }

  template <int dim>
  std::vector<T> cut_dof_to_grid_dof(const Mesh& mesh,
                                     const std::vector<T> u0) {
    if (u0.size() != mesh.get_num_nodes() * dim) {
      throw std::runtime_error(
          "cut_dof_to_grid_dof() only takes dof vectors of size nnodes * dim "
          "(" +
          std::to_string(mesh.get_num_nodes() * dim) + "), got " +
          std::to_string(u0.size()));
    }

    int nnodes = mesh.get_num_nodes();
    std::vector<T> u(dim * mesh.get_grid().get_num_verts(), T(0.0));
    for (int i = 0; i < nnodes; i++) {
      int vert = mesh.get_node_vert(i);
      for (int d = 0; d < dim; d++) {
        u[dim * vert + d] = u0[dim * i + d];
      }
    }
    return u;
  }

  std::pair<std::vector<T>, std::vector<T>> eval_stress(
      const std::vector<T>& u) {
    if constexpr (two_material_method == TwoMaterial::ERSATZ) {
      return stress_analysis.interpolate_energy(
          grid_dof_to_cut_dof<spatial_dim>(stress_analysis.get_mesh(), u)
              .data());
    } else {
      return stress_analysis.interpolate_energy(
          u.data());  // FIXME: this is hacky, if two_material_method ==
                      // TwoMaterial::NITSCHE, u is really concat(u_primary,
                      // u_secondary), and we're effectly using the primary part
                      // of it because pass-by-pointer does the trick
    }
  }

  std::pair<std::vector<T>, std::vector<T>> eval_surf_stress(
      const std::vector<T>& u) {
    if constexpr (two_material_method == TwoMaterial::ERSATZ) {
      return surf_stress_analysis.interpolate_energy(
          grid_dof_to_cut_dof<spatial_dim>(surf_stress_analysis.get_mesh(), u)
              .data());
    } else {
      return surf_stress_analysis.interpolate_energy(
          u.data());  // FIXME: this is hacky, if two_material_method ==
                      // TwoMaterial::NITSCHE, u is really concat(u_primary,
                      // u_secondary), and we're effectly using the primary part
                      // of it because pass-by-pointer does the trick
    }
  }

  void eval_obj_con_gradient(const std::vector<T>& x, std::vector<T>& gcomp,
                             std::vector<T>& garea, std::vector<T>& gpen,
                             std::vector<T>& gstress) {
    // Get options
    bool use_helmholtz_filter = parser.get_bool_option("use_helmholtz_filter");
    int num_conv_filter_apply = parser.get_int_option("num_conv_filter_apply");
    bool use_robust_projection =
        parser.get_bool_option("use_robust_projection");
    bool stress_use_discrete_ks =
        parser.get_bool_option("stress_use_discrete_ks");

    T ks_energy = 0.0;
    std::vector<T> sol;
    std::shared_ptr<SparseUtils::SparseCholesky<T>> chol;
    if (x == std::get<std::vector<T>>(cache["x"])) {
      sol = std::get<std::vector<T>>(cache["sol"]);
      chol = std::get<std::shared_ptr<SparseUtils::SparseCholesky<T>>>(
          cache["chol"]);
      ks_energy = std::get<T>(cache["ks_energy"]);
    } else {
      std::vector<T> HFx;
      sol = update_mesh_and_solve(x, HFx, &chol);
      ks_energy = stress_ks_analysis.energy(nullptr, sol.data());
      std::vector<T> vol_dummy(vol_analysis.get_mesh().get_num_nodes(), 0.0);
    }

    // Compliance function is self-adjoint with a sign flip
    std::vector<T> psi_comp = sol;
    for (T& p : psi_comp) p *= -1.0;

    // Evaluate the rhs of the adjoint equation for stress
    // adjoint equation is K * psi = -∂s/∂u
    std::vector<T> psi_stress(sol.size(), T(0.0));
    stress_ks_analysis.residual(nullptr, sol.data(), psi_stress.data());

    // Apply boundary conditions to the rhs
    for (int i : bc_dof) {
      psi_stress[i] = 0.0;
    }

    // Compute stress adjoints
    chol->solve(psi_stress.data());

    // Apply boundary conditions again to the adjoint variables
    for (int i : bc_dof) {
      psi_stress[i] = 0.0;
    }

    std::vector<T> psi_stress_neg = psi_stress;
    for (T& p : psi_stress) p *= -1.0;

    gcomp.resize(x.size());
    std::fill(gcomp.begin(), gcomp.end(), 0.0);
    elastic->get_analysis().LSF_jacobian_adjoint_product(
        sol.data(), psi_comp.data(), gcomp.data());
    if constexpr (two_material_method == TwoMaterial::ERSATZ) {
      elastic->get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), sol.data() /*this is effectively -psi*/, gcomp.data());
    } else if constexpr (two_material_method == TwoMaterial::NITSCHE) {
      int node_offset = elastic->get_mesh().get_num_nodes();
      elastic->get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), sol.data() /*this is effectively -psi*/, gcomp.data(),
          node_offset);
      elastic->get_analysis_interface().LSF_jacobian_adjoint_product(
          sol.data(), psi_comp.data(), gcomp.data());
    }

    std::vector<T> Fx(x.size(), 0.0);
    if (use_helmholtz_filter) {
      hfilter.apply(x.data(), Fx.data());
    } else {
      cfilter.apply(x.data(), Fx.data());
    }

    design_mapping_apply_gradient(gcomp.data(), gcomp.data());
    if (use_robust_projection) {
      projector_erode.applyGradient(Fx.data(), gcomp.data(), gcomp.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(gcomp.data(), gcomp.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(gcomp.data(), gcomp.data());
      }
    }

    garea.resize(x.size());
    std::fill(garea.begin(), garea.end(), 0.0);
    vol_analysis.LSF_volume_derivatives(garea.data());

    design_mapping_apply_gradient(garea.data(), garea.data());
    if (use_robust_projection) {
      projector_dilate.applyGradient(Fx.data(), garea.data(), garea.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(garea.data(), garea.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(garea.data(), garea.data());
      }
    }

    gpen.resize(x.size());
    std::fill(gpen.begin(), gpen.end(), 0.0);
    pen_analysis.residual(nullptr, phi_blueprint.data(), gpen.data());

    design_mapping_apply_gradient(gpen.data(), gpen.data());
    if (use_robust_projection) {
      projector_blueprint.applyGradient(Fx.data(), gpen.data(), gpen.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(gpen.data(), gpen.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(gpen.data(), gpen.data());
      }
    }

    gstress.resize(x.size());
    std::fill(gstress.begin(), gstress.end(), 0.0);

    // Explicit partials
    stress_ks_analysis.LSF_energy_derivatives(sol.data(), gstress.data(),
                                              stress_use_discrete_ks);

    // Implicit derivatives via the adjoint variables
    elastic->get_analysis().LSF_jacobian_adjoint_product(
        sol.data(), psi_stress.data(), gstress.data());
    if constexpr (two_material_method == TwoMaterial::ERSATZ) {
      elastic->get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), psi_stress_neg.data(), gstress.data());
    } else if constexpr (two_material_method == TwoMaterial::NITSCHE) {
      int node_offset = elastic->get_mesh().get_num_nodes();
      elastic->get_analysis_ersatz().LSF_jacobian_adjoint_product(
          sol.data(), psi_stress_neg.data(), gstress.data(), node_offset);
      elastic->get_analysis_interface().LSF_jacobian_adjoint_product(
          sol.data(), psi_stress.data(), gstress.data());
    }

    design_mapping_apply_gradient(gstress.data(), gstress.data());
    if (use_robust_projection) {
      projector_erode.applyGradient(Fx.data(), gstress.data(), gstress.data());
    }
    if (use_helmholtz_filter) {
      hfilter.applyGradient(gstress.data(), gstress.data());
    } else {
      for (int i = 0; i < num_conv_filter_apply; i++) {
        cfilter.applyGradient(gstress.data(), gstress.data());
      }
    }

    // For now we have denergy/dx, next, compute dks/dx:
    double denom = ks_energy * stress_ks.get_ksrho();
    for (int i = 0; i < gstress.size(); i++) {
      gstress[i] /= denom;
    }
  }

  void write_grid_vtk(const std::string vtk_path, const std::vector<T>& x,
                      std::map<std::string, std::vector<T>&> node_sols = {},
                      std::map<std::string, std::vector<T>&> cell_sols = {},
                      std::map<std::string, std::vector<T>&> node_vecs = {},
                      std::map<std::string, std::vector<T>&> cell_vecs = {}) {
    ToVTK<T, typename HFilter::Mesh> vtk(hfilter.get_mesh(), vtk_path);
    vtk.write_mesh();

    // Node solutions
    for (auto [name, vals] : node_sols) {
      if (vals.size() != grid.get_num_verts()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]node sol size doesn't match "
            "number of verts");
      }
      vtk.write_sol(name, vals.data());
    }

    vtk.write_sol("x", x.data());
    vtk.write_sol("phi_blueprint", phi_blueprint.data());
    vtk.write_sol("phi_dilate", phi_dilate.data());
    vtk.write_sol("phi_erode", phi_erode.data());

    // Node vectors
    for (auto [name, vals] : node_vecs) {
      if (vals.size() != spatial_dim * grid.get_num_verts()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]node vec size doesn't match "
            "number of verts * spatial_dim");
      }
      vtk.write_vec(name, vals.data());
    }

    // Cell solutions
    for (auto [name, vals] : cell_sols) {
      if (vals.size() != grid.get_num_cells()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]cell sol size doesn't match "
            "number of cells");
      }
      vtk.write_cell_sol(name, vals.data());
    }

    std::vector<T> loaded_cells_v(grid.get_num_cells(), 0.0);
    for (int c : loaded_cells) loaded_cells_v[c] = 1.0;
    vtk.write_cell_sol("loaded_cells", loaded_cells_v.data());

    // Cell vectors
    for (auto [name, vals] : cell_vecs) {
      if (vals.size() != spatial_dim * grid.get_num_cells()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_grid_vtk()]cell vec size doesn't match "
            "number of cells * spatial_dim");
      }
      vtk.write_cell_vec(name, vals.data());
    }
  }

  template <bool is_secondary_mesh = false>
  Mesh& get_mesh_for_vtk() {
    if constexpr (is_secondary_mesh) {
      return elastic->get_mesh_ersatz();
    } else {
      return elastic->get_mesh();
    }
  }

  void write_cut_vtk(const std::string vtk_path, const std::vector<T>& x,
                     std::map<std::string, std::vector<T>&> node_sols = {},
                     std::map<std::string, std::vector<T>&> cell_sols = {},
                     std::map<std::string, std::vector<T>&> node_vecs = {},
                     std::map<std::string, std::vector<T>&> cell_vecs = {}) {}

  void write_cut_vtk(Mesh& mesh, const std::string vtk_path,
                     std::map<std::string, std::vector<T>&> node_sols = {},
                     std::map<std::string, std::vector<T>&> cell_sols = {},
                     std::map<std::string, std::vector<T>&> node_vecs = {},
                     std::map<std::string, std::vector<T>&> cell_vecs = {}) {
    ToVTK<T, Mesh> vtk(mesh, vtk_path);
    vtk.write_mesh();

    // Node solutions
    {
      for (auto [name, vals] : node_sols) {
        xcgd_assert(vals.size() == mesh.get_num_nodes(),
                    "node sol size doesn't match number of nodes");
        vtk.write_sol(name, vals.data());
      }

      vtk.write_sol("phi_blueprint", mesh.get_lsf_nodes(phi_blueprint).data());
      vtk.write_sol("phi_dilate", mesh.get_lsf_nodes(phi_dilate).data());
      vtk.write_sol("phi_erode", mesh.get_lsf_nodes(phi_erode).data());
    }

    // Node vectors
    {
      for (auto [name, vals] : node_vecs) {
        xcgd_assert(
            vals.size() == spatial_dim * mesh.get_num_nodes(),
            "node vec size doesn't match number of nodes * spatial_dim");
        vtk.write_vec(name, vals.data());
      }
    }

    // Cell solutions
    {
      for (auto [name, vals] : cell_sols) {
        xcgd_assert(vals.size() == mesh.get_num_elements(),
                    "cell sol size doesn't match number of elements");
        vtk.write_cell_sol(name, vals.data());
      }
    }

    // Cell vectors
    {
      for (auto [name, vals] : cell_vecs) {
        xcgd_assert(
            vals.size() == mesh.get_num_elements() * spatial_dim,
            "cell vec size doesn't match numbre of elements * spatial_dim");
        vtk.write_cell_vec(name, vals.data());
      }
    }
  }

  template <bool is_secondary_mesh = false>
  void write_cut_vtk_deprecated(
      const std::string vtk_path, const std::vector<T>& x,
      std::map<std::string, std::vector<T>&> node_sols = {},
      std::map<std::string, std::vector<T>&> cell_sols = {},
      std::map<std::string, std::vector<T>&> node_vecs = {},
      std::map<std::string, std::vector<T>&> cell_vecs = {}) {
    if constexpr (is_secondary_mesh) {
      static_assert(two_material_method == TwoMaterial::NITSCHE,
                    "only supported for Nitsche formulation");
    }

    Mesh& mesh = get_mesh_for_vtk<is_secondary_mesh>();

    ToVTK<T, Mesh> vtk(mesh, vtk_path);
    vtk.write_mesh();

    // Node solutions
    for (auto [name, vals] : node_sols) {
      if (vals.size() != mesh.get_num_nodes()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_cut_vtk()]node sol size doesn't match "
            "number of nodes");
      }
      vtk.write_sol(name, vals.data());
    }
    vtk.write_sol("x", mesh.get_lsf_nodes(x).data());
    vtk.write_sol("phi_blueprint", mesh.get_lsf_nodes(phi_blueprint).data());
    vtk.write_sol("phi_dilate", mesh.get_lsf_nodes(phi_dilate).data());
    vtk.write_sol("phi_erode", mesh.get_lsf_nodes(phi_erode).data());

    // Node vectors
    for (auto [name, vals] : node_vecs) {
      if (vals.size() != spatial_dim * mesh.get_num_nodes()) {
        throw std::runtime_error("[TopoAnalysis::write_cut_vtk()]node vec " +
                                 name +
                                 " size doesn't match "
                                 "number of nodes * spatial_dim");
      }
      vtk.write_vec(name, vals.data());
    }

    std::vector<T> bc_dof_v(dof_per_node * mesh.get_num_nodes(), 0.0);
    if constexpr (is_secondary_mesh) {
      int dof_offset = dof_per_node * elastic->get_mesh().get_num_nodes();
      for (int val : bc_dof) {
        bc_dof_v[val - dof_offset] = 1.0;
      }
    } else {
      for (int val : bc_dof) {
        bc_dof_v[val] = 1.0;
      }
    }

    vtk.write_vec("bc", bc_dof_v.data());

    // Cell solutions
    int nelems = mesh.get_num_elements();
    std::vector<T> is_cut_elem_v(nelems, 0.0);
    for (int i = 0; i < nelems; i++) {
      if (mesh.is_cut_elem(i)) {
        is_cut_elem_v[i] = 1.0;
      }
    }
    vtk.write_cell_sol("is_cut_element", is_cut_elem_v.data());
    for (auto [name, vals] : cell_sols) {
      if (vals.size() != mesh.get_num_elements()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_cut_vtk()]cell sol size doesn't match "
            "number of elements");
      }
      vtk.write_cell_sol(name, vals.data());
    }

    std::vector<double> conds(mesh.get_num_elements());
    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      conds[elem] = VandermondeCondLogger::get_conds().at(elem);
    }
    vtk.write_cell_sol("cond", conds.data());

    auto& stencils = mesh.get_elem_nodes();
    std::map<int, std::vector<int>> degenerate_stencils;
    std::vector<double> nstencils(stencils.size(), -1);
    for (auto& [elem, stencil] : stencils) {
      int num = stencil.size();
      nstencils[elem] = num;
      if (num < Mesh::max_nnodes_per_element) {
        degenerate_stencils[elem] = stencil;
      }
    }
    vtk.write_cell_sol("nstencils", nstencils.data());

    // Cell vectors
    for (auto [name, vals] : cell_vecs) {
      if (vals.size() != mesh.get_num_elements()) {
        throw std::runtime_error(
            "[TopoAnalysis::write_cut_vtk()]cell vec size doesn't match "
            "number of elements * spatial_dim");
      }
      vtk.write_cell_vec(name, vals.data());
    }

    // Save stencils and degenerate stencils
    if constexpr (two_material_method != TwoMaterial::NITSCHE) {
      auto [base, suffix] = split_path(vtk_path);
      StencilToVTK<T, Mesh> stencil_vtk(mesh, base + "_stencils" + suffix);
      stencil_vtk.write_stencils(stencils);
      StencilToVTK<T, Mesh> degen_stencil_vtk(
          mesh, base + "_degen_stencils" + suffix);
      degen_stencil_vtk.write_stencils(degenerate_stencils);
    }
  }

  void write_prob_json(const std::string json_path, const ConfigParser& parser,
                       const std::vector<T>& x) {
    json j;
    j["cfg"] = parser.get_options();
    j["phi_blueprint"] = phi_blueprint;
    j["bc_dof"] = bc_dof;
    j["bc_vals"] = bc_vals;
    j["loaded_cells"] = loaded_cells;
    j["dvs"] = x;
    write_json(json_path, j);
  }

  std::vector<T>& get_rhs() { return elastic->get_rhs(); }
  ProbMesh& get_prob_mesh() { return prob_mesh; }

  void set_stress_ksrho(double ksrho) { stress_ks.set_ksrho(ksrho); }

  auto& get_elastic() { return elastic; }

 private:
  ProbMesh& prob_mesh;
  std::string prefix;
  const ConfigParser& parser;

  Grid& grid;
  Mesh& erode_mesh;
  Mesh& dilate_mesh;
  Quadrature erode_quadrature, dilate_quadrature;
  SurfQuadrature erode_surf_quadrature;
  Basis erode_basis, dilate_basis;

  RobustProjection<T> projector_blueprint, projector_dilate, projector_erode;
  HFilter hfilter;
  CFilter cfilter;
  std::shared_ptr<Elastic> elastic;

  Volume vol;
  VolAnalysis vol_analysis;

  Penalization pen;
  PenalizationAnalysis pen_analysis;

  Stress stress;
  StressAnalysis stress_analysis;

  StressKS stress_ks;
  StressKSAnalysis stress_ks_analysis;

  SurfStress surf_stress;
  SurfStressAnalysis surf_stress_analysis;

  SurfStressKS surf_stress_ks;
  SurfStressKSAnalysis surf_stress_ks_analysis;

  BulkInt bulk_int;
  BulkIntAnalysis bulk_int_analysis;

  // level-set values for blueprint, dilate and erode design
  std::vector<T>& phi_erode;
  std::vector<T>& phi_dilate;
  std::vector<T> phi_blueprint;

  std::vector<int> bc_dof;
  std::vector<T> bc_vals;

  std::set<int> loaded_cells;

  std::map<std::string,
           std::variant<T, std::vector<T>,
                        std::shared_ptr<SparseUtils::SparseCholesky<T>>>>
      cache;
};

template <typename T, class TopoAnalysis>
class TopoProb {
 public:
  TopoProb(TopoAnalysis& topo, std::string prefix, const ConfigParser& parser)
      : topo(topo),
        prefix(prefix),
        parser(parser),
        domain_area(topo.get_prob_mesh().get_domain_area()),
        area_frac_init(parser.get_double_option("area_frac_init")),
        area_frac_final(parser.get_double_option("area_frac_final")),
        area_frac_decrease_every(
            parser.get_int_option("area_frac_decrease_every")),
        area_frac_decrease_rate(
            parser.get_double_option("area_frac_decrease_rate")),
        stress_ksrho_init(parser.get_double_option("stress_ksrho_init")),
        stress_ksrho_final(parser.get_double_option("stress_ksrho_final")),
        stress_ksrho_increase_every(
            parser.get_int_option("stress_ksrho_increase_every")),
        stress_ksrho_increase_rate(
            parser.get_double_option("stress_ksrho_increase_rate")),
        stress_ratio_ub_init(
            parser.get_double_option("stress_ratio_upper_bound_init")),
        stress_ratio_ub_final(
            parser.get_double_option("stress_ratio_upper_bound_final")),
        stress_ratio_ub_decay_rate(
            parser.get_double_option("stress_ratio_upper_bound_decay_rate")),
        has_stress_constraint(parser.get_bool_option("has_stress_constraint")),
        has_compliance_constraint(
            parser.get_bool_option("has_compliance_constraint")),
        compliance_constraint_upper_bound(
            parser.get_double_option("compliance_constraint_upper_bound")),
        has_stress_objective(parser.get_bool_option("has_stress_objective")),
        compliance_objective_scalar(
            parser.get_double_option("compliance_objective_scalar")),
        stress_objective_scalar(
            parser.get_double_option("stress_objective_scalar")),
        stress_objective_theta(
            parser.get_double_option("stress_objective_theta")),
        is_volume_objective(parser.get_bool_option("is_volume_objective")),
        nvars(topo.get_prob_mesh().get_nvars()),
        ncon(0),
        HFx_old(nvars, 0.0) {
    // Set number of constraints

    if (not is_volume_objective) {  // volume constraint
      ncon++;
    }
    if (has_stress_constraint) {
      ncon++;
    }
    if (has_compliance_constraint) {
      ncon++;
    }

    if (!std::filesystem::is_directory(prefix)) {
      std::filesystem::create_directory(prefix);
    }

    reset_counter();
  }

  int get_nvars() { return nvars; }
  int get_ncon() { return ncon; }

  void print_progress(T obj, T comp, T reg, T pterm, T vol_frac, T max_stress,
                      T max_stress_ratio, double ksrho, T ks_stress_ratio,
                      T ks_stress_ratio_ub, T HFx_change,
                      int header_every = 10) {
    std::ofstream progress_file(fspath(prefix) / fspath("optimization.log"),
                                std::ios::app);
    if (minor_counter % header_every == 0) {
      char line[2048];
      std::snprintf(
          line, 2048,
          "\n%6s%6s%13s%13s%13s%13s%9s%9s%13s%13s%9s%13s%9s%13s%13s%15s\n",
          "major", "minor", "obj", "comp", "x_regular", "grad_pen", "vol(\%)",
          "vub(\%)", "max(vm)", "max(vm/y)", "ksrho", "ks(vm/y)", "kserr(\%)",
          "ks_ub", "|dx|_infty", "uptime(hms)");
      std::cout << line;
      progress_file << line;
    }
    char line[2048];

    std::string major_it = "";
    if (counter != prev_counter) {
      prev_counter = counter;
      major_it = std::to_string(counter);
    }
    std::snprintf(
        line, 2048,
        "%6s%6d%13.3e%13.3e%13.3e%13.3e%9.3f%9.3f%13.3e%13.3e%9.3f%13.3e%9.3f"
        "%13.3e%13.3e%15s\n",
        major_it.c_str(), minor_counter, obj, comp, reg, pterm,
        100.0 * vol_frac, 100.0 * area_frac, max_stress, max_stress_ratio,
        ksrho, ks_stress_ratio,
        (ks_stress_ratio - max_stress_ratio) / max_stress_ratio * 100.0,
        ks_stress_ratio_ub, HFx_change, watch.format_time(watch.lap()).c_str());
    std::cout << line;
    progress_file << line;
    progress_file.close();
  }

  void reset_counter() {
    counter = 0;
    minor_counter = 0;
  }
  void inc_counter() { counter++; }
  int get_counter() { return counter; }

  void getVarsAndBounds(T* xr, T* lb, T* ub) {
    // Set initial design
    std::vector<T> x0;
    std::string init_topo_json_path =
        parser.get_str_option("init_topology_from_json");
    if (!init_topo_json_path.empty()) {
      json j = read_json(init_topo_json_path);
      json j_this(parser.get_options());

      // Sanity checks to make sure the input json has the same basic settings
      // as the current case

      for (std::string key :
           std::vector<std::string>{"instance", "nx", "ny", "lx", "ly"}) {
        xcgd_assert(j["cfg"][key] == j_this[key], key + "mismatch");
      }

      x0 = std::vector<T>(j["dvs"]);
      xcgd_assert(x0.size() == nvars,
                  "design variable dimension mismatch, expect " +
                      std::to_string(nvars) + ", got " +
                      std::to_string(x0.size()));

    }

    else {
      std::string init_topology_method =
          parser.get_str_option("init_topology_method");

      if (init_topology_method == "circles") {
        x0 = topo.create_initial_topology(
            parser.get_int_option("init_topology_nholes_x"),
            parser.get_int_option("init_topology_nholes_y"),
            parser.get_double_option("init_topology_r"),
            parser.get_bool_option("init_topology_cell_center"),
            parser.get_int_option("init_topology_shrink_level"),
            parser.get_str_option("instance") == "lbracket",
            parser.get_double_option("lbracket_frac"),
            parser.get_double_option("loaded_frac"));
      } else if (init_topology_method == "sinusoidal") {
        x0 = topo.create_initial_topology_sine(
            parser.get_int_option("init_topology_sine_period_x"),
            parser.get_int_option("init_topology_sine_period_y"),
            parser.get_double_option("init_topology_sine_offset"));
      } else if (init_topology_method == "lbracket") {
        x0 = topo.create_initial_topology_lbracket(
            parser.get_int_option("init_topology_lbracket_nholes_1d"),
            parser.get_double_option("init_topology_lbracket_r"));
      } else if (init_topology_method == "anchor") {
        x0 = topo.create_initial_topology_anchor(
            parser.get_double_option("init_topology_anchor_a"),
            parser.get_double_option("init_topology_anchor_b"));
      }

      else {
        throw std::runtime_error("init_topology_method = " +
                                 init_topology_method + " is not supported");
      }
    }

    auto& prob_mesh = topo.get_prob_mesh();

    // update mesh and bc dof, but don't perform the linear solve
    std::vector<T> x0r = prob_mesh.reduce(x0);
    x0 = prob_mesh.expand(x0r,
                          1.0);  // set non-design values to 1.0
    topo.update_mesh(x0);

    for (int i = 0; i < nvars; i++) {
      xr[i] = x0r[i];
      lb[i] = 0.0;
      ub[i] = 1.0;
    }

    const auto& protected_verts = topo.get_prob_mesh().get_protected_verts();
    for (int i : protected_verts) {
      int ir = prob_mesh.reduce_index(i);
      if (ir >= 0) {
        ub[i] = 0.499;  // we prescribe x to be sufficient low for protected
                        // verts such that material will be placed at those
                        // locations
      }
    }

    const auto& void_verts = topo.get_prob_mesh().get_void_verts();
    for (int i : void_verts) {
      int ir = prob_mesh.reduce_index(i);
      if (ir >= 0) {
        lb[i] = 0.501;  // we prescribe x to be sufficient high for void
                        // verts such that material will not be placed at
                        // those locations
      }
    }

    std::vector<T> lb_v(nvars), ub_v(nvars);
    for (int i = 0; i < nvars; i++) {
      lb_v[i] = (T)lb[i];
      ub_v[i] = (T)ub[i];
    }

    std::vector<T> lb_expand_v =
        prob_mesh.expand(lb_v, std::numeric_limits<T>::quiet_NaN());
    std::vector<T> ub_expand_v =
        prob_mesh.expand(ub_v, std::numeric_limits<T>::quiet_NaN());

    topo.write_grid_vtk(fspath(prefix) / fspath("init_and_bounds.vtk"), x0,
                        {{"lb", lb_expand_v}, {"ub", ub_expand_v}}, {}, {}, {});
  }

  int evalObjCon(T* xptr, T* fobj, T* cons) {
    std::vector<T> xr(xptr, xptr + nvars);
    std::vector<T> x =
        topo.get_prob_mesh().expand(xr, 1.0);  // set non-design values to 1.0

    // Update KS parameter
    double ksrho =
        stress_ksrho_init +
        stress_ksrho_increase_rate * int(counter / stress_ksrho_increase_every);
    if (ksrho > stress_ksrho_final) {
      ksrho = stress_ksrho_final;
    }
    if (is_gradient_check) {
      ksrho = 6.789;
    }
    topo.set_stress_ksrho(ksrho);

    // Update area constraint
    area_frac = area_frac_init - area_frac_decrease_rate *
                                     int(counter / area_frac_decrease_every);
    if (area_frac < area_frac_final) {
      area_frac = area_frac_final;
    }

    // Update stress constraint
    stress_ratio_ub =
        stress_ratio_ub_init - stress_ratio_ub_decay_rate * counter;
    if (stress_ratio_ub < stress_ratio_ub_final) {
      stress_ratio_ub = stress_ratio_ub_final;
    }

    if (is_gradient_check) {
      stress_ratio_ub = 1.234;
    }

    // Regularization term
    double reg_coeff = parser.get_double_option("regularization_coeff") / nvars;
    T reg = 0.0;
    for (int i = 0; i < nvars; i++) {
      reg += xr[i] * (1.0 - xr[i]);
    }
    reg *= reg_coeff;

    // Save the elastic problem instance to json
    if (counter % parser.get_int_option("save_prob_json_every") == 0) {
      std::string json_path = fspath(prefix) / fspath("json") /
                              ((is_gradient_check ? "fdcheck_" : "opt_") +
                               std::to_string(counter) + ".json");
      topo.write_prob_json(json_path, parser, x);
    }

    auto [HFx, comp, area, pterm, max_stress, max_stress_ratio, ks_stress_ratio,
          surf_ks_stress_ratio, u, xloc_q, stress_q, xloc_surf_q,
          stress_surf_q] = topo.eval_obj_con(x);

    // Evaluate the design change ||phi_new - phi_old||_infty
    std::transform(
        HFx_old.begin(), HFx_old.end(), HFx.begin(), HFx_old.begin(),
        [](T a, T b) { return abs(b - a); });  // HFx_old = abs(HFx_old - HFx)
    T HFx_change = *std::max_element(HFx_old.begin(), HFx_old.end());
    HFx_old = HFx;

    if (has_stress_objective) {
      *fobj =
          (1.0 - stress_objective_theta) * compliance_objective_scalar * comp +
          stress_objective_theta * stress_objective_scalar * ks_stress_ratio +
          pterm + reg;
    } else if (is_volume_objective) {
      *fobj = area / domain_area + pterm + reg;
    } else {
      *fobj = compliance_objective_scalar * comp + pterm + reg;
    }

    int con_index = 0;
    if (not is_volume_objective) {
      cons[con_index] = 1.0 - area / (domain_area * area_frac);  // >= 0
      con_index++;
    }

    if (has_stress_constraint) {
      cons[con_index] = 1.0 - ks_stress_ratio / stress_ratio_ub;  // >= 0
      con_index++;
    }

    if (has_compliance_constraint) {
      cons[con_index] = 1.0 - comp / compliance_constraint_upper_bound;
      con_index++;
    }

    if (counter % parser.get_int_option("write_vtk_every") == 0) {
      // Write design to vtk
      std::string vtk_path =
          fspath(prefix) / fspath("grid_" + std::to_string(counter) + ".vtk");
      std::vector<double> protected_verts_v(x.size(), 0.0);
      const auto& protected_verts = topo.get_prob_mesh().get_protected_verts();
      for (auto v : protected_verts) {
        protected_verts_v[v] = 1.0;
      }

      if constexpr (TopoAnalysis::two_material_method == TwoMaterial::ERSATZ) {
        topo.write_grid_vtk(vtk_path, x,
                            {{"protected_verts", protected_verts_v}}, {},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else if constexpr (TopoAnalysis::two_material_method ==
                               TwoMaterial::OFF or
                           TopoAnalysis::two_material_method ==
                               TwoMaterial::NITSCHE) {
        topo.write_grid_vtk(
            vtk_path, x, {{"protected_verts", protected_verts_v}}, {}, {}, {});
      }

      // Write cut mesh to vtk
      vtk_path =
          fspath(prefix) / fspath("cut_" + std::to_string(counter) + ".vtk");
      if constexpr (TopoAnalysis::two_material_method == TwoMaterial::ERSATZ) {
        topo.write_cut_vtk(vtk_path, x, {}, {}, {}, {});
      } else if constexpr (TopoAnalysis::two_material_method ==
                           TwoMaterial::OFF) {
        topo.write_cut_vtk(vtk_path, x, {}, {}, {{"displacement", u}}, {});
      } else if constexpr (TopoAnalysis::two_material_method ==
                           TwoMaterial::NITSCHE) {
        std::string vtk_path_primary =
            fspath(prefix) /
            fspath("cut_primary_" + std::to_string(counter) + ".vtk");
        std::string vtk_path_secondary =
            fspath(prefix) /
            fspath("cut_secondary_" + std::to_string(counter) + ".vtk");

        constexpr int dof_per_node = TopoAnalysis::dof_per_node;
        int dof_offset =
            topo.get_elastic()->get_mesh().get_num_nodes() * dof_per_node;

        std::vector<T> u_primary(u.begin(), u.begin() + dof_offset);
        std::vector<T> u_secondary(u.begin() + dof_offset, u.end());

        xcgd_assert((u_primary.size() + u_secondary.size()) == u.size(),
                    "incompatible");

        topo.write_cut_vtk(topo.get_elastic()->get_mesh(), vtk_path_primary, {},
                           {}, {{"displacement", u_primary}}, {});

        topo.write_cut_vtk(topo.get_elastic()->get_mesh_ersatz(),
                           vtk_path_secondary, {}, {},
                           {{"displacement", u_secondary}}, {});
      }

      // Write bulk quadrature-level data
      {
        vtk_path =
            fspath(prefix) / fspath("quad_" + std::to_string(counter) + ".vtk");
        FieldToVTKNew<T, TopoAnalysis::get_spatial_dim()> field_vtk(vtk_path);
        field_vtk.add_mesh(xloc_q);
        field_vtk.write_mesh();
        field_vtk.add_sol("VonMises", stress_q);
        field_vtk.write_sol("VonMises");
      }

      // Write surface quadrature-level data
      {
        vtk_path = fspath(prefix) /
                   fspath("surfquad_" + std::to_string(counter) + ".vtk");
        FieldToVTKNew<T, TopoAnalysis::get_spatial_dim()> field_vtk(vtk_path);
        field_vtk.add_mesh(xloc_surf_q);
        field_vtk.write_mesh();

        field_vtk.add_sol("SurfStress", stress_surf_q);
        field_vtk.write_sol("SurfStress");
      }
    }

    // write quadrature to vtk for gradient check
    if (is_gradient_check) {
      std::string vtk_name = "fdcheck_quad_" + std::to_string(counter) + ".vtk";

      vtk_name = "fdcheck_grid_" + std::to_string(counter) + ".vtk";

      std::vector<T> vert_loaded_or_not(
          topo.get_prob_mesh().get_grid().get_num_verts(), 0.0);
      for (int i : topo.get_prob_mesh().get_loaded_verts()) {
        vert_loaded_or_not[i] = 1.0;
      }

      std::vector<T> cell_loaded_or_not(
          topo.get_prob_mesh().get_grid().get_num_cells(), 0.0);
      for (int i : topo.get_prob_mesh().get_loaded_cells()) {
        cell_loaded_or_not[i] = 1.0;
      }

      if constexpr (TopoAnalysis::two_material_method == TwoMaterial::ERSATZ) {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            {{"loaded_verts", vert_loaded_or_not}},
                            {{"loaded_cells", cell_loaded_or_not}},
                            {{"displacement", u}, {"rhs", topo.get_rhs()}}, {});
      } else {
        topo.write_grid_vtk(fspath(prefix) / fspath(vtk_name), x,
                            {{"loaded_verts", vert_loaded_or_not}},
                            {{"loaded_cells", cell_loaded_or_not}}, {}, {});
      }

      vtk_name = "fdcheck_cut_" + std::to_string(counter) + ".vtk";
      if constexpr (TopoAnalysis::two_material_method == TwoMaterial::ERSATZ) {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, {}, {}, {},
                           {});
      } else if constexpr (TopoAnalysis::two_material_method ==
                           TwoMaterial::OFF) {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, {}, {},
                           {{"displacement", u}}, {});
      } else if constexpr (TopoAnalysis::two_material_method ==
                           TwoMaterial::NITSCHE) {
        topo.write_cut_vtk(fspath(prefix) / fspath(vtk_name), x, {}, {}, {},
                           {});
      }
    }

    // print optimization progress
    print_progress(*fobj, comp, reg, pterm, area / domain_area, max_stress,
                   max_stress_ratio, ksrho, ks_stress_ratio, stress_ratio_ub,
                   HFx_change);
    minor_counter++;
    return 0;
  }

  int evalObjConGradient(T* xptr, T* g, std::vector<T*> gc) {
    std::vector<T> xr(xptr, xptr + nvars);
    std::vector<T> x =
        topo.get_prob_mesh().expand(xr, 1.0);  // set non-design values to 1.0

    std::vector<T> gcomp, garea, gpen, gstress;
    topo.eval_obj_con_gradient(x, gcomp, garea, gpen, gstress);

    std::vector<T> gcompr = topo.get_prob_mesh().reduce(gcomp);
    std::vector<T> garear = topo.get_prob_mesh().reduce(garea);
    std::vector<T> gpenr = topo.get_prob_mesh().reduce(gpen);
    std::vector<T> gstressr;

    if (has_stress_objective or has_stress_constraint) {
      gstressr = topo.get_prob_mesh().reduce(gstress);
    }

    std::vector<T> greg(nvars, 0.0);
    double reg_coeff = parser.get_double_option("regularization_coeff") / nvars;
    for (int i = 0; i < nvars; i++) {
      greg[i] = reg_coeff * (1.0 - 2.0 * xr[i]);
    }

    // Populate objective gradients
    if (has_stress_objective) {
      T a = (1.0 - stress_objective_theta) * compliance_objective_scalar;
      T b = stress_objective_theta * stress_objective_scalar;
      for (int i = 0; i < nvars; i++) {
        g[i] = a * gcompr[i] + b * gstressr[i] + gpenr[i] + greg[i];
      }
    } else if (is_volume_objective) {
      for (int i = 0; i < nvars; i++) {
        g[i] = garear[i] / domain_area + gpenr[i] + greg[i];
      }
    } else {
      for (int i = 0; i < nvars; i++) {
        g[i] = compliance_objective_scalar * gcompr[i] + gpenr[i] + greg[i];
      }
    }

    // Populate constraint gradients

    int con_index = 0;

    if (not is_volume_objective) {
      for (int i = 0; i < nvars; i++) {
        gc[con_index][i] = -garear[i] / (domain_area * area_frac);
      }
      con_index++;
    }

    if (has_stress_constraint) {
      for (int i = 0; i < nvars; i++) {
        gc[con_index][i] = -gstressr[i] / stress_ratio_ub;
      }
      con_index++;
    }

    if (has_compliance_constraint) {
      for (int i = 0; i < nvars; i++) {
        gc[con_index][i] = -gcompr[i] / compliance_constraint_upper_bound;
      }
      con_index++;
    }

    return 0;
  }

  void gradient_check_on() { is_gradient_check = true; }
  void gradient_check_off() { is_gradient_check = false; }

 private:
  TopoAnalysis& topo;
  std::string prefix;
  const ConfigParser& parser;
  double domain_area = 0.0;
  double area_frac = 0.0;
  double area_frac_init = 0.0;
  double area_frac_final = 0.0;
  int area_frac_decrease_every = 0.0;
  double area_frac_decrease_rate = 0.0;
  double stress_ksrho_init = 0.0;
  double stress_ksrho_final = 0.0;
  int stress_ksrho_increase_every = 0;
  double stress_ksrho_increase_rate = 0.0;
  double stress_ratio_ub_init = 0.0;
  double stress_ratio_ub_final = 0.0;
  double stress_ratio_ub_decay_rate = 0.0;
  double stress_ratio_ub = 0.0;
  bool has_stress_constraint = false;
  bool has_compliance_constraint = false;
  double compliance_constraint_upper_bound = 0.0;
  bool has_stress_objective = false;
  double compliance_objective_scalar = 0.0;
  double stress_objective_scalar = 0.0;
  double stress_objective_theta = 0.0;
  bool is_volume_objective = false;
  int nvars = -1;
  int ncon = -1;

  std::vector<T> HFx_old;

  int prev_counter = -1;
  int counter = -1;  // major iteration counter
  int minor_counter =
      -1;  // minor iteration counter, increased at each function evaluation
  StopWatch watch;
  bool is_gradient_check = false;
};

template <typename T, class TopoProb>
class TopoProbParOpt : public ParOptProblem {
 public:
  TopoProbParOpt(TopoProb& prob) : ParOptProblem(MPI_COMM_SELF), prob(prob) {
    setProblemSizes(prob.get_nvars(), prob.get_ncon(), 0);
    setNumInequalities(prob.get_ncon(), 0);
  }

  void check_gradients(double dh) {
    prob.gradient_check_on();
    checkGradients(dh);
    prob.gradient_check_off();
    prob.reset_counter();
    VandermondeCondLogger::clear();
  }

  void getVarsAndBounds(ParOptVec* xvec, ParOptVec* lbvec, ParOptVec* ubvec) {
    T *xr, *lb, *ub;
    xvec->getArray(&xr);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);
    prob.getVarsAndBounds(xr, lb, ub);
  }

  int evalObjCon(ParOptVec* xvec, T* fobj, T* cons) {
    T* xptr;
    xvec->getArray(&xptr);
    int ret = prob.evalObjCon(xptr, fobj, cons);
    prob.inc_counter();
    return ret;
  }

  int evalObjConGradient(ParOptVec* xvec, ParOptVec* gvec, ParOptVec** Ac) {
    T* xptr;
    xvec->getArray(&xptr);

    T* g;
    gvec->getArray(&g);
    gvec->zeroEntries();

    std::vector<T*> gc(prob.get_ncon(), nullptr);

    for (int i = 0; i < prob.get_ncon(); i++) {
      Ac[i]->getArray(&gc[i]);
      Ac[i]->zeroEntries();
    }
    return prob.evalObjConGradient(xptr, g, gc);
  }

  // Dummy method
  ParOptQuasiDefMat* createQuasiDefMat() {
    int nwblock = 0;
    return new ParOptQuasiDefBlockMat(this, nwblock);
  }

 private:
  TopoProb& prob;
};

template <typename T, class TopoProb>
class TopoProbSNOPT : public snoptProblemC {
 public:
  using TopoProbSNOPT_s = TopoProbSNOPT<T, TopoProb>;
  inline static TopoProbSNOPT_s* instance = nullptr;
  TopoProbSNOPT(TopoProb& prob) : snoptProblemC("SNOPT_problem"), prob(prob) {}

  // This function is called at each major iteration
  static void snstop(int* iAbort, int KTcond[], int* MjrPrt, int* minimz,
                     int* m, int* maxS, int* n, int* nb, int* nnCon0,
                     int* nnCon, int* nnObj0, int* nnObj, int* nS, int* itn,
                     int* nMajor, int* nMinor, int* nSwap, double* condHz,
                     int* iObj, double* sclObj, double* ObjAdd, double* fObj,
                     double* fMrt, double* PenNrm, double* step, double* prInf,
                     double* duInf, double* vimax, double* virel, int hs[],
                     int* ne, int* nlocJ, int locJ[], int indJ[], double Jcol[],
                     int* negCon, double Ascale[], double bl[], double bu[],
                     double Fx[], double fCon[], double gCon[], double gObj[],
                     double yCon[], double pi[], double rc[], double rg[],
                     double x[], char cu[], int* lencu, int iu[], int* leniu,
                     double ru[], int* lenru, char cw[], int* lencw, int iw[],
                     int* leniw, double rw[], int* lenrw) {
    instance->prob.inc_counter();
  }

  static void usrfun(int* mode, int* nnObj, int* nnCon, int* nnJac, int* nnL,
                     int* negCon, double x[], double* fObj, double gObj[],
                     double fCon[], double gCon[], int* Status, char* cu,
                     int* lencu, int iu[], int* leniu, double ru[],
                     int* lenru) {
    // Evaluate objective and constraint values
    if (*mode == 0 || *mode == 2) {
      instance->prob.evalObjCon(x, fObj, fCon);
    }

    // Evaluate objective gradient and constraint Jacobian
    if (*mode == 1 || *mode == 2) {
      int nvars = instance->prob.get_nvars();
      int ncon = instance->prob.get_ncon();

      std::vector<T*> gc(ncon, nullptr);
      std::vector<std::vector<T>> gc_data(ncon, std::vector<T>(nvars, 0.0));
      for (int i = 0; i < ncon; i++) {
        gc[i] = gc_data[i].data();
      }

      instance->prob.evalObjConGradient(x, gObj, gc);

      for (int col = 0, index = 0; col < nvars; col++) {
        for (int row = 0; row < ncon; row++, index++) {
          gCon[index] = gc_data[row][col];
        }
      }
    }
  }

  TopoProb& prob;
};

template <typename T, class TopoProb>
void optimize_paropt(bool smoke_test, std::string prefix, ConfigParser& parser,
                     TopoProb& prob) {
  TopoProbParOpt<T, TopoProb>* paropt_prob =
      new TopoProbParOpt<T, TopoProb>(prob);
  paropt_prob->incref();

  double dh = parser.get_double_option("grad_check_fd_h");
  paropt_prob->check_gradients(dh);

  if (parser.get_bool_option("check_grad_and_exit")) {
    return;
  }

  // Set options
  ParOptOptions* options = new ParOptOptions;
  options->incref();
  ParOptOptimizer::addDefaultOptions(options);

  int max_it = smoke_test ? 10 : parser.get_int_option("max_it");

  options->setOption("algorithm",
                     parser.get_str_option("paropt_algorithm").c_str());

  // Interior-point solver options
  options->setOption("output_file",
                     (fspath(prefix) / fspath("paropt.out")).c_str());
  options->setOption("starting_point_strategy",
                     parser.get_str_option("starting_point_strategy").c_str());
  options->setOption("barrier_strategy",
                     parser.get_str_option("barrier_strategy").c_str());
  options->setOption("abs_res_tol", parser.get_double_option("abs_res_tol"));
  options->setOption("use_line_search",
                     parser.get_bool_option("use_line_search"));
  options->setOption("max_major_iters",
                     parser.get_int_option("max_major_iters"));
  options->setOption("penalty_gamma",
                     parser.get_double_option("penalty_gamma"));
  options->setOption("qn_subspace_size",
                     parser.get_int_option("qn_subspace_size"));
  options->setOption("qn_type", parser.get_str_option("qn_type").c_str());
  options->setOption("qn_diag_type",
                     parser.get_str_option("qn_diag_type").c_str());

  // Trust-region options
  options->setOption("tr_max_iterations", max_it);
  options->setOption("tr_init_size", parser.get_double_option("tr_init_size"));
  options->setOption("tr_min_size", parser.get_double_option("tr_min_size"));
  options->setOption("tr_max_size", parser.get_double_option("tr_max_size"));
  options->setOption("tr_eta", parser.get_double_option("tr_eta"));
  options->setOption("tr_infeas_tol",
                     parser.get_double_option("tr_infeas_tol"));
  options->setOption("tr_l1_tol", parser.get_double_option("tr_l1_tol"));
  options->setOption("tr_linfty_tol",
                     parser.get_double_option("tr_linfty_tol"));
  options->setOption("tr_adaptive_gamma_update",
                     parser.get_bool_option("tr_adaptive_gamma_update"));
  options->setOption("tr_output_file",
                     (fspath(prefix) / fspath("paropt.tr")).c_str());

  // MMA options
  options->setOption("mma_max_iterations", max_it);
  options->setOption("mma_init_asymptote_offset",
                     parser.get_double_option("mma_init_asymptote_offset"));
  options->setOption("mma_move_limit",
                     parser.get_double_option("mma_move_limit"));
  options->setOption("mma_output_file",
                     (fspath(prefix) / fspath("paropt.mma")).c_str());

  ParOptOptimizer* opt = new ParOptOptimizer(paropt_prob, options);
  opt->incref();

  opt->optimize();

  paropt_prob->decref();
  options->decref();
  opt->decref();
}
template <typename T, class TopoProb>
void optimize_snopt(bool smoke_test, std::string prefix, ConfigParser& parser,
                    TopoProb& prob) {
  constexpr double SNOPT_INF = 1e30;

  using TopoProbSNOPT_s = TopoProbSNOPT<T, TopoProb>;
  TopoProbSNOPT_s* snopt_prob = new TopoProbSNOPT_s(prob);
  TopoProbSNOPT_s::instance = snopt_prob;

  snopt_prob->initialize("", 1);
  snopt_prob->setIntParameter(
      "Verify level",
      0);  // Perform a cheap finite-difference gradient check
  snopt_prob->setIntParameter(
      "Derivative level",
      3);  // All All objective and constraint gradients are known.
  snopt_prob->setIntParameter("Print file", 18);
  snopt_prob->setIntParameter("Summary file", 19);

  // Options from config
  int max_it = smoke_test ? 10 : parser.get_int_option("max_it");
  snopt_prob->setIntParameter("Major iterations limit", max_it);
  snopt_prob->setIntParameter("Iterations limit",
                              parser.get_int_option("snopt_minor_iter_limit"));
  snopt_prob->setRealParameter(
      "Major feasibility tolerance",
      parser.get_double_option("snopt_major_feas_tol"));
  snopt_prob->setRealParameter("Major optimality tolerance",
                               parser.get_double_option("snopt_major_opt_tol"));
  snopt_prob->setIntParameter("Major print level",
                              parser.get_int_option("snopt_major_print_level"));
  snopt_prob->setIntParameter("Minor print level",
                              parser.get_int_option("snopt_minor_print_level"));
  snopt_prob->setRealParameter(
      "Major step limit",
      parser.get_double_option("snopt_major_step_size_limit"));

  int Cold = 0;  // cold start
  int n = prob.get_nvars();
  int m = prob.get_ncon();
  int ne = m * n;       // number of non-zeros in constraint Jacobian
  int negCon = m * n;   // size of gCon
  int nnCon = m;        // number of nonlinear constraints
  int nnObj = n;        // number of nonlinear dvs for the nonlinear objective
  int nnJac = n;        // number of nonlinear dvs for the nonlinear constraints
  int iObj = -1;        // no such row in A containing a linear objective vector
  double ObjAdd = 0.0;  // a constant that will be added to the objective for
                        // printing purposes.

  // Dense constraint Jacobian in CSR format
  std::vector<T> Avals(ne, 0.0);   // constants of the A, all zero as we
                                   // eval them through usrfun
  std::vector<int> Arows(ne, -1);  // row index of each non-zero entry of A
  std::vector<int> Aloc(n + 1);    // pointers to beginning of each column
  for (int col = 0, i = 0; col < n; col++) {
    Aloc[col] = col * m;
    for (int row = 0; row < m; row++, i++) {
      Arows[i] = row;
    }
  }
  Aloc[n] = ne;

  std::vector<T> bl(n + m, 0.0);        // lower bounds of dvs and constraints
  std::vector<T> bu(n + m, SNOPT_INF);  // upper bounds of dvs and constraints
  std::vector<int> hs(n + m, 0);        // initial states, just set them to 0
  std::vector<T> x0(n + m);  // initial values of dvs, we only need first n

  // Populate initial dvs and bounds of dvs
  prob.getVarsAndBounds(x0.data(), bl.data(), bu.data());

  std::vector<T> pi(m, 0.0),
      rc(n + m, 0.0);  // initial guess of multipliers? Just set to 0 because we
                       // perform cold start

  // On exit
  int nS = 0;  // not needed for cold start
  int nInf;
  double objective, sInf;

  // Optimize
  snopt_prob->setSTOP(TopoProbSNOPT_s::snstop);
  snopt_prob->solve(Cold, m, n, ne, negCon, nnCon, nnObj, nnJac, iObj, ObjAdd,
                    TopoProbSNOPT_s::usrfun, Avals.data(), Arows.data(),
                    Aloc.data(), bl.data(), bu.data(), hs.data(), x0.data(),
                    pi.data(), rc.data(), nS, nInf, sInf, objective);
}

template <int Np_1d, TwoMaterial two_material_method, bool use_lbracket_grid,
          bool use_finite_cell_mesh>
void execute(int argc, char* argv[]) {
  constexpr int Np_1d_filter = Np_1d > 2 ? 4 : 2;
  MPI_Init(&argc, &argv);

  using T = double;
  using Grid = typename std::conditional<use_lbracket_grid, LbracketGrid2D<T>,
                                         StructuredGrid2D<T>>::type;
  using TopoAnalysis = TopoAnalysis<T, Np_1d, Np_1d_filter, two_material_method,
                                    Grid, use_finite_cell_mesh>;
  using TopoProb = TopoProb<T, TopoAnalysis>;

  bool smoke_test = false;
  if (argc > 2 and "--smoke" == std::string(argv[2])) {
    std::printf("This is a smoke test\n");
    smoke_test = true;
  }

  // Create config parser
  std::string cfg_path{argv[1]};
  ConfigParser parser{cfg_path};

  std::string prefix = parser.get_str_option("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
    if (smoke_test) {
      prefix = "smoke_" + prefix;
    } else {
      prefix = "opt_" + prefix;
    }
  }

  if (!std::filesystem::is_directory(prefix)) {
    std::filesystem::create_directory(prefix);
  }

  std::string json_dir = fspath(prefix) / fspath("json");
  if (!std::filesystem::is_directory(json_dir)) {
    std::filesystem::create_directory(json_dir);
  }

  std::filesystem::copy(
      cfg_path,
      fspath(prefix) / fspath(std::filesystem::absolute(cfg_path).filename()));

  std::string init_topo_json_path =
      parser.get_str_option("init_topology_from_json");
  if (!init_topo_json_path.empty()) {
    std::filesystem::copy(
        init_topo_json_path,
        fspath(prefix) /
            fspath(std::filesystem::absolute(init_topo_json_path).filename()));
  }

  // Set up grid
  std::array<int, 2> nxy = {parser.get_int_option("nx"),
                            parser.get_int_option("ny")};
  if (smoke_test) {
    nxy[0] = 9;
    nxy[1] = 5;
  }
  std::array<T, 2> lxy = {parser.get_double_option("lx"),
                          parser.get_double_option("ly")};

  // Set up analysis
  std::string instance = parser.get_str_option("instance");
  if (!(instance == "lbracket" or instance == "cantilever" or
        instance == "anchor")) {
    throw std::runtime_error(
        "expect lbracket or cantilever for option instance, got " + instance);
  }

  std::shared_ptr<ProbMeshBase<T, Np_1d, Grid, use_finite_cell_mesh>> prob_mesh;
  if constexpr (use_lbracket_grid) {
    double loaded_frac = parser.get_double_option("loaded_frac");
    double lbracket_frac = parser.get_double_option("lbracket_frac");
    bool load_top = parser.get_bool_option("load_top");
    if (load_top) {
      prob_mesh = std::make_shared<
          LbracketGridMesh<T, Np_1d, true, use_finite_cell_mesh>>(
          nxy, lxy, loaded_frac, lbracket_frac);
    } else {
      prob_mesh = std::make_shared<
          LbracketGridMesh<T, Np_1d, false, use_finite_cell_mesh>>(
          nxy, lxy, loaded_frac, lbracket_frac);
    }
  } else {
    if (instance == "cantilever") {
      double loaded_frac = parser.get_double_option("loaded_frac");
      prob_mesh =
          std::make_shared<CantileverMesh<T, Np_1d, use_finite_cell_mesh>>(
              nxy, lxy, loaded_frac);
    } else if (instance == "anchor") {
      double clamped_frac = parser.get_double_option("anchor_clamped_frac");
      double D_ux = parser.get_double_option("anchor_D_ux");
      prob_mesh = std::make_shared<AnchorMesh<T, Np_1d, use_finite_cell_mesh>>(
          nxy, lxy, clamped_frac, D_ux);
    } else if (instance == "lbracket") {
      double loaded_frac = parser.get_double_option("loaded_frac");
      double lbracket_frac = parser.get_double_option("lbracket_frac");
      prob_mesh =
          std::make_shared<LbracketMesh<T, Np_1d, use_finite_cell_mesh>>(
              nxy, lxy, loaded_frac, lbracket_frac);
    } else {
      throw std::runtime_error("invalid instance " + instance);
    }
  }

  TopoAnalysis topo{*prob_mesh, prefix, parser};
  TopoProb prob{topo, prefix, parser};

  if (parser.get_str_option("optimizer") == "paropt") {
    optimize_paropt<T>(smoke_test, prefix, parser, prob);
  } else if (parser.get_str_option("optimizer") == "snopt") {
    optimize_snopt<T>(smoke_test, prefix, parser, prob);
  } else {
    throw std::runtime_error("unsupported optimizer " +
                             parser.get_str_option("optimizer"));
  }

  MPI_Finalize();
}

template <TwoMaterial two_material_method, bool use_lbracket_grid,
          bool use_finite_cell_mesh>
void execute_1(int argc, char* argv[], int Np_1d) {
  if (Np_1d == 2) {
    execute<2, two_material_method, use_lbracket_grid, use_finite_cell_mesh>(
        argc, argv);
  }
  // else if (Np_1d == 4) {
  //   execute<4, two_material_method, use_lbracket_grid, use_finite_cell_mesh>(
  //       argc, argv);
  // }
  // else if (Np_1d == 6) {
  //   execute<6, two_material_method, use_lbracket_grid, use_finite_cell_mesh>(
  //       argc, argv);
  // }
  else {
    throw std::runtime_error("Np_1d = " + std::to_string(Np_1d) +
                             " not precompiled");
  }
}

template <bool use_lbracket_grid, bool use_finite_cell_mesh>
void execute_2(int argc, char* argv[], int Np_1d,
               TwoMaterial two_material_method) {
  if (two_material_method == TwoMaterial::OFF) {
    execute_1<TwoMaterial::OFF, use_lbracket_grid, use_finite_cell_mesh>(
        argc, argv, Np_1d);
  } else if (two_material_method == TwoMaterial::ERSATZ) {
    execute_1<TwoMaterial::ERSATZ, use_lbracket_grid, use_finite_cell_mesh>(
        argc, argv, Np_1d);
  } else {
    execute_1<TwoMaterial::NITSCHE, use_lbracket_grid, use_finite_cell_mesh>(
        argc, argv, Np_1d);
  }
}

template <bool use_finite_cell_mesh>
void execute_3(int argc, char* argv[], int Np_1d,
               TwoMaterial two_material_method, bool use_lbracket_grid) {
  if (use_lbracket_grid) {
    execute_2<true, use_finite_cell_mesh>(argc, argv, Np_1d,
                                          two_material_method);
  } else {
    execute_2<false, use_finite_cell_mesh>(argc, argv, Np_1d,
                                           two_material_method);
  }
}

void execute_4(int argc, char* argv[], int Np_1d,
               TwoMaterial two_material_method, bool use_lbracket_grid,
               bool use_finite_cell_mesh) {
  if (use_finite_cell_mesh) {
    execute_3<true>(argc, argv, Np_1d, two_material_method, use_lbracket_grid);
  } else {
    execute_3<false>(argc, argv, Np_1d, two_material_method, use_lbracket_grid);
  }
}

int main(int argc, char* argv[]) {
  VandermondeCondLogger::enable();

  if (argc == 1) {
    std::printf("Usage: ./topo level_set.cfg [--smoke]\n");
    exit(0);
  }

  std::string cfg_path{argv[1]};
  ConfigParser parser{cfg_path};
  int Np_1d = parser.get_int_option("Np_1d");

  std::map<std::string, TwoMaterial> two_material_map = {
      {"ersatz", TwoMaterial::ERSATZ},
      {"off", TwoMaterial::OFF},
      {"nitsche", {TwoMaterial::NITSCHE}}};
  std::string two_material_method_str =
      parser.get_str_option("two_material_method");
  xcgd_assert(two_material_map.count(two_material_method_str),
              "unknown two_material_method");
  TwoMaterial two_material_method = two_material_map[two_material_method_str];

  bool use_lbracket_grid = false;
  if (parser.get_str_option("instance") == "lbracket") {
    use_lbracket_grid = parser.get_bool_option("use_lbracket_grid");
  }

  bool use_finite_cell_mesh = parser.get_bool_option("use_finite_cell_mesh");

  if (Np_1d % 2) {
    std::printf("[Error]Invalid input, expect even Np_1d, got %d\n", Np_1d);
    exit(-1);
  }

  execute_4(argc, argv, Np_1d, two_material_method, use_lbracket_grid,
            use_finite_cell_mesh);

  return 0;
}
