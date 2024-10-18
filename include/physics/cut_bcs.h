#pragma once

#include "physics_commons.h"

/**
 * @brief The Dirichlet boundary condition applied on the cut boundary using
 * Nitsche's method.
 */
template <typename T, int spatial_dim>
class CutDirichlet final : public PhysicsBase<T, spatial_dim, 0, 1> {
 private:
  using PhysicsBase = PhysicsBase<T, spatial_dim, 0, 1>;

 public:
  using PhysicsBase::data_per_node;
  using PhysicsBase::dof_per_node;
  using PhysicsBase::spatial_dim;

  CutDirichlet() {}

  T energy(T weight, T _, A2D::Vec<T, spatial_dim>& xloc,
           A2D::Mat<T, spatial_dim, spatial_dim>& J, T& val,
           A2D::Vec<T, spatial_dim>& grad) const {}
};
