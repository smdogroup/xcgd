/*
 * The implementation of the Superconvergent Patch Recovery (SPR). Useful for
 * reconstructing a continuous field that is a function of the first order
 * derivatives of the solution field, such as stress field for elasticity
 * problem.
 *
 * Ref:
 *   - Zienkiewicz, O.C. and Zhu, J.Z. (1992), The superconvergent patch
 * recovery and a posteriori error estimates. Part 1: The recovery technique.
 * Int. J. Numer. Meth. Engng., 33: 1331-1364.
 * */

#pragma once

#include <vector>

#include "analysis.h"
#include "elements/gd_vandermonde.h"
#include "physics/stress.h"

template <typename T, int Np_1d, class Grid, class Mesh, class Basis,
          bool from_to_grid_mesh>
class SPRStress2D {
 private:
  static constexpr int Amat_dim = Np_1d * Np_1d;  // for 2d problem
  static constexpr int nsamples_per_elem_1d = Np_1d - 1;
  static constexpr int spatial_dim = Basis::spatial_dim;
  // TODO: don't hardcode quadrature class as GDLSFQuadrature2D
  using SPRSampler = GDLSFQuadrature2D<T, Np_1d, QuadPtType::INNER, Grid,
                                       nsamples_per_elem_1d, Mesh>;
  using StrainStress = LinearElasticity2DStrainStress<T>;
  using SPRAnalysis = GalerkinAnalysis<T, Mesh, SPRSampler, Basis, StrainStress,
                                       from_to_grid_mesh>;

 public:
  SPRStress2D(Mesh& mesh, Basis& basis, double E, double nu)
      : mesh(mesh), basis(basis), strain_stress(E, nu) {}

  // return sx_nodal, sy_nodal, sxy_nodal
  std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> apply(
      const T* sol) {
    SPRSampler spr_sampler(mesh);

    SPRAnalysis spr_analysis(mesh, spr_sampler, basis, strain_stress);

    // Evaluate stress components on all sampling points
    // Note: this implementation relies on a stable, consistency order of
    // quadrature point query
    std::map<int, std::vector<T>> samples_xloc_map =
        spr_analysis.interpolate_energy_map(sol).first;

    std::map<StrainStressType, std::map<int, std::vector<T>>> samples_scomp_map;
    std::vector<StrainStressType> stress_comps = {
        StrainStressType::sx, StrainStressType::sy, StrainStressType::sxy};

    for (auto scomp : stress_comps) {
      strain_stress.set_type(scomp);
      samples_scomp_map[scomp] =
          spr_analysis.interpolate_energy_map(sol).second;
    }

    // Allocate recovered nodal stress values
    std::vector<T> sx_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<T> sy_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<T> sxy_recovered(mesh.get_num_nodes(), 0.0);
    std::vector<int> num_recovered(mesh.get_num_nodes(), 0);

    // Loop over all patch assemply nodes
    for (int n = 0; n < mesh.get_num_nodes(); n++) {
      // Get patch elements
      std::set<int> patch_elems = mesh.get_node_patch_elems(n);

      // Skip the patch assembly node if it doesn't have enough patch elements
      int nsamples_estimate =
          patch_elems.size() * nsamples_per_elem_1d * nsamples_per_elem_1d;
      if (nsamples_estimate < Amat_dim) {
        /*
        std::printf("node %4d doesn't have enough patch elements (%d <%d)\n", n,
                    nsamples_estimate, Amat_dim);
        */
        continue;
      }

      // Get the dimensions bounding box for the element cell
      T patch_xloc_min[spatial_dim], patch_xloc_max[spatial_dim];
      mesh.get_node_patch_elems_node_ranges(n, patch_xloc_min, patch_xloc_max);
      T patch_dh[spatial_dim];
      for (int d = 0; d < spatial_dim; d++) {
        patch_dh[d] = patch_xloc_max[d] - patch_xloc_min[d];
      }

      // Get number of coordinates and values of sample points in all patch
      // elements of this assembly node
      std::vector<T> xloc_p;  // sample points in the patch frame
      std::vector<T> sx_p, sy_p, sxy_p;
      // Collect sample points from patch elements
      for (int e : patch_elems) {
        const std::vector<T>& xloc = samples_xloc_map[e];
        const std::vector<T>& sx = samples_scomp_map[StrainStressType::sx][e];
        const std::vector<T>& sy = samples_scomp_map[StrainStressType::sy][e];
        const std::vector<T>& sxy = samples_scomp_map[StrainStressType::sxy][e];

        int npts = xloc.size() / spatial_dim;

        // Convert sample points in patch frame
        for (int q = 0; q < npts; q++) {
          for (int d = 0; d < spatial_dim; d++) {
            int index = q * spatial_dim + d;
            xloc_p.push_back(
                (xloc[index] - patch_xloc_min[d]) / patch_dh[d] * 2.0 - 1.0);
          }

          sx_p.push_back(sx[q]);
          sy_p.push_back(sy[q]);
          sxy_p.push_back(sxy[q]);
        }
      }

      // Get number of sample points in all patch elements of this assembly node
      int nsamples = xloc_p.size() / spatial_dim;

      // Assemble the least-square system
      std::vector<T> Amat(Amat_dim * Amat_dim,
                          0.0);  // matrix entries stored column by column
      for (int s = 0; s < nsamples; s++) {
        // Get P entries for a sample point
        T x = xloc_p[spatial_dim * s];
        T y = xloc_p[spatial_dim * s + 1];
        std::array<T, Np_1d> xpows, ypows;
        for (int t = 0; t < Np_1d; t++) {
          xpows[t] = pow(x, t);
          ypows[t] = pow(y, t);
        }

        // Add PPT to A
        // for 2d problem, PT = [1, x, x^2, ..., y, yx, yx^2, ...]
        for (int col = 0; col < Amat_dim; col++) {
          int P_ix = col % Np_1d;
          int P_iy = col / Np_1d;
          for (int row = 0; row < Amat_dim; row++) {
            int PT_ix = row % Np_1d;
            int PT_iy = row / Np_1d;
            Amat[col * Amat_dim + row] +=
                xpows[P_ix] * ypows[P_iy] * xpows[PT_ix] * ypows[PT_iy];
          }
        }
      }

      // Factor A
      DirectSolve<T> Afact(Amat_dim, Amat.data());

      // rhs for the SPR least square linear system at assemply node n
      std::vector<T> bx(Amat_dim, 0.0), by(Amat_dim, 0.0), bxy(Amat_dim, 0.0);

      // Evaluate stress components on sampling points
      for (int s = 0; s < nsamples; s++) {
        // Get P entries for a sample point
        T x = xloc_p[spatial_dim * s];
        T y = xloc_p[spatial_dim * s + 1];
        std::array<T, Np_1d> xpows, ypows;
        for (int t = 0; t < Np_1d; t++) {
          xpows[t] = pow(x, t);
          ypows[t] = pow(y, t);
        }

        for (int row = 0; row < Amat_dim; row++) {
          int PT_ix = row % Np_1d;
          int PT_iy = row / Np_1d;
          T pv = xpows[PT_ix] * ypows[PT_iy];
          bx[row] += sx_p[s] * pv;
          by[row] += sy_p[s] * pv;
          bxy[row] += sxy_p[s] * pv;
        }
      }

      // Apply invA to stress components
      Afact.apply(bx.data());
      Afact.apply(by.data());
      Afact.apply(bxy.data());

      // Get all nodes we add value to
      std::set<int> patch_nodes;
      for (int e : patch_elems) {
        int nodes[Mesh::corner_nodes_per_element];
        mesh.get_elem_corner_nodes(e, nodes);
        for (int i = 0; i < Mesh::corner_nodes_per_element; i++) {
          patch_nodes.insert(nodes[i]);
        }
      }

      // Add nodal contributions to nodal recovered values
      for (int pn : patch_nodes) {
        // Get nodal coordinates in the patch frame
        T pn_xloc[spatial_dim];
        mesh.get_node_xloc(pn, pn_xloc);
        T x = (pn_xloc[0] - patch_xloc_min[0]) / patch_dh[0] * 2.0 - 1.0;
        T y = (pn_xloc[1] - patch_xloc_min[1]) / patch_dh[1] * 2.0 - 1.0;
        std::array<T, Np_1d> xpows, ypows;
        for (int t = 0; t < Np_1d; t++) {
          xpows[t] = pow(x, t);
          ypows[t] = pow(y, t);
        }

        // val_recovered = PTa
        for (int row = 0; row < Amat_dim; row++) {
          int PT_ix = row % Np_1d;
          int PT_iy = row / Np_1d;
          T pv = xpows[PT_ix] * ypows[PT_iy];

          sx_recovered[pn] += bx[row] * pv;
          sy_recovered[pn] += by[row] * pv;
          sxy_recovered[pn] += bxy[row] * pv;
        }
        num_recovered[pn]++;
      }
    }

    // Adjust recovered values by taking the average of multiple contributions
    int nnodes = mesh.get_num_nodes();
    std::vector<T> vm_recovered(nnodes, 0.0);
    for (int i = 0; i < nnodes; i++) {
      sx_recovered[i] /= num_recovered[i];
      sy_recovered[i] /= num_recovered[i];
      sxy_recovered[i] /= num_recovered[i];

      vm_recovered[i] = sqrt(sx_recovered[i] * sx_recovered[i] -
                             sx_recovered[i] * sy_recovered[i] +
                             sy_recovered[i] * sy_recovered[i] +
                             3.0 * sxy_recovered[i] * sxy_recovered[i]);
    }

    return {sx_recovered, sy_recovered, sxy_recovered};
  }

  void applyGradient() {}

 private:
  Mesh& mesh;
  Basis& basis;
  StrainStress strain_stress;
};
