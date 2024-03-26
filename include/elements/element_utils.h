#ifndef XCGD_ELEMENT_UTILS_H
#define XCGD_ELEMENT_UTILS_H

#include <vector>

#include "element_commons.h"
#include "gd_mesh.h"
#include "physics/physics_commons.h"
#include "utils/vtk.h"

template <typename T, class Basis>
void get_element_xloc(const typename Basis::Mesh& mesh, int e,
                      T element_xloc[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    mesh.get_node_xloc(nodes[j], element_xloc);
    element_xloc += spatial_dim;
  }
}

template <typename T, int dim, class Basis>
void get_element_vars(const typename Basis::Mesh& mesh, int e, const T dof[],
                      T element_dof[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    for (int k = 0; k < dim; k++, element_dof++) {
      element_dof[0] = dof[dim * nodes[j] + k];
    }
  }
}

template <typename T, int dim, class Basis>
void add_element_res(const typename Basis::Mesh& mesh, int e,
                     const T element_res[], T res[]) {
  int constexpr nodes_per_element = Basis::nodes_per_element;
  int constexpr spatial_dim = Basis::spatial_dim;
  int nodes[nodes_per_element];
  mesh.get_elem_dof_nodes(e, nodes);
  for (int j = 0; j < nodes_per_element; j++) {
    for (int k = 0; k < dim; k++, element_res++) {
      res[dim * nodes[j] + k] += element_res[0];
    }
  }
}

template <typename T, int Np_1d, int samples_1d>
class GDSampler2D final : public QuadratureBase<T> {
 private:
  using Mesh =
      GDMesh2D<T, Np_1d>;  // Np_1d and samples_1d can be different, which is
                           // the whole point of using this Sampler clas
  static constexpr int spatial_dim = Mesh::spatial_dim;
  static constexpr int samples = samples_1d * samples_1d;

 public:
  GDSampler2D(const Mesh& mesh) : mesh(mesh) {}

  int get_quadrature_pts(int elem, std::vector<T>& pts,
                         std::vector<T>& _) const {
    pts.resize(spatial_dim * samples);

    T xymin[2], xymax[2];
    get_computational_coordinates_limits(mesh, elem, xymin, xymax);

    T lxy[2], xy0[2];
    for (int d = 0; d < spatial_dim; d++) {
      xy0[d] = xymin[d] + 0.05 * (xymax[d] - xymin[d]);
      lxy[d] = 0.95 * (xymax[d] - xymin[d]);
    }
    int nxy[2] = {samples_1d - 1, samples_1d - 1};
    StructuredGrid2D<T> grid(nxy, lxy, xy0);

    T* pts_ptr = pts.data();
    for (int i = 0; i < samples; i++) {
      grid.get_vert_xloc(i, pts_ptr);
      pts_ptr += spatial_dim;
    }

    return samples;
  }

 private:
  const Mesh& mesh;
};

/**
 * @brief Given a mesh and dof, interpolate the field.
 *
 * Note: This is useful for sanity check and debugging.
 */
template <typename T, class Sampler, class Basis>
class Interpolator final {
  using Mesh = typename Basis::Mesh;

  static int constexpr dof_per_node = 1;
  static int constexpr data_per_node = 0;
  static int constexpr spatial_dim = Mesh::spatial_dim;

  using Physics = PhysicsBase<T, spatial_dim, data_per_node, dof_per_node>;

 public:
  Interpolator(const Mesh& mesh, const Sampler& sampler, const Basis& basis)
      : mesh(mesh), basis(basis), sampler(sampler) {}

  void to_vtk(const std::string name, T* dof) const {
    FieldToVTK<T, spatial_dim> field_vtk(name);

    for (int elem = 0; elem < mesh.get_num_elements(); elem++) {
      std::vector<T> element_dof(Mesh::nodes_per_element);
      get_element_vars<T, dof_per_node, Basis>(mesh, elem, dof,
                                               element_dof.data());

      std::vector<T> element_xloc(Mesh::nodes_per_element * Basis::spatial_dim);
      get_element_xloc<T, Basis>(mesh, elem, element_xloc.data());

      std::vector<T> pts, wts;
      int nsamples = sampler.get_quadrature_pts(elem, pts, wts);

      std::vector<T> N, Nxi;
      basis.eval_basis_grad(elem, pts, N, Nxi);

      std::vector<T> vals(nsamples);
      std::vector<T> ptx(nsamples * Basis::spatial_dim);

      for (int i = 0; i < nsamples; i++) {
        int offset_n = i * Basis::nodes_per_element;
        T val = 0.0;
        interp_val_grad<T, Basis>(elem, element_dof.data(), &N[offset_n],
                                  nullptr, &val, nullptr);
        vals[i] = val;
        A2D::Vec<T, Basis::spatial_dim> xloc;
        interp_val_grad<T, Basis, Basis::spatial_dim>(
            elem, element_xloc.data(), &N[offset_n], nullptr, &xloc, nullptr);
        for (int d = 0; d < Basis::spatial_dim; d++) {
          ptx[i * Basis::spatial_dim + d] = xloc[d];
        }
      }
      field_vtk.add_scalar_field(ptx, vals);
    }
    field_vtk.write_vtk();
  }

 private:
  const Mesh& mesh;
  const Basis& basis;
  const Sampler& sampler;
  Physics physics;
};

#endif  // XCGD_ELEMENT_UTILS_H