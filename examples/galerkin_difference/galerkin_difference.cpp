#include "elements/galerkin_difference.h"

#include <string>

#include "analysis.h"
#include "elements/tetrahedral.h"
#include "physics/poisson.h"
#include "utils/mesh.h"

int main(int argc, char *argv[]) {
  using T = double;
  constexpr int spatial_dim = 2;
  int constexpr Np_1d = 4;
  int constexpr nx = 5, ny = 5;  // number of elements along x and y directions
  using Grid = StructuredGrid2D<T>;
  using Mesh = GDMesh2D<T, Np_1d>;
  using Basis = GDBasis2D<T, Np_1d>;
  using Quadrature = GDQuadrature2D<Np_1d>;
  using Physics = PoissonPhysics<T, spatial_dim>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  // Set the number of degrees of freedom
  int ndof = spatial_dim * (nx + 1) * (ny + 1);

  // Allocate space for the degrees of freedom
  T *dof = new T[ndof];
  T *res = new T[ndof];
  T *Jp = new T[ndof];
  T *direction = new T[ndof];
  for (int i = 0; i < ndof; i++) {
    dof[i] = 0.01 * rand() / RAND_MAX;
    res[i] = 0.0;
    Jp[i] = 0.0;
    direction[i] = 1.0;
  }

  // Allocate the physics
  Physics physics;

  int nxy[2] = {nx, ny};
  T lxy[2] = {1.0, 1.0};
  Grid grid(nxy, lxy);
  Mesh mesh(grid);
  Basis basis(mesh);

  Analysis analysis(basis);

  // Allocate space for the residual
  T energy = analysis.energy_new(physics, dof);

  std::cout << energy << std::endl;

  return 0;
}