#ifndef XCGD_VOLUME_H
#define XCGD_VOLUME_H

#include "physics_commons.h"

template <typename T, int spatial_dim_>
class VolumePhysics final : public PhysicsBase<T, spatial_dim_, 0, 1> {
 private:
  using PhysicsBase_s = PhysicsBase<T, spatial_dim_, 0, 1>;

 public:
  using PhysicsBase_s::data_per_node;
  using PhysicsBase_s::dof_per_node;
  using PhysicsBase_s::spatial_dim;

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ;
  }
};

template <typename T, int spatial_dim>
class BulkIntegration final : public PhysicsBase<T, spatial_dim, 0, 1> {
 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& ___,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ____) const {
    T detJ;
    A2D::MatDet(J, detJ);
    return weight * detJ * val;
  }
};

template <typename T, int spatial_dim>
class SurfaceIntegration final : public PhysicsBase<T, spatial_dim, 0, 1> {
  static_assert(spatial_dim == 2,
                "This part is not yet implemented properly for 3D");

 public:
  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& __,
           A2D::Vec<T, spatial_dim>& nrm_ref,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& ___) const {
    T dt_val[spatial_dim] = {nrm_ref[1], -nrm_ref[0]};

    A2D::Mat<T, spatial_dim, spatial_dim> JTJ;
    A2D::Vec<T, spatial_dim> dt(dt_val);
    A2D::Vec<T, spatial_dim> JTJdt;

    T scale;
    A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(J, J, JTJ);
    A2D::MatVecMult(JTJ, dt, JTJdt);
    A2D::VecDot(dt, JTJdt, scale);
    scale = sqrt(scale);

    return weight * scale * val;
  }
};

#endif  // XCGD_VOLUME_H
