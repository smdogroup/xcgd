#include <string>

#include "analysis.h"
#include "elements/tetrahedral.h"
#include "physics/neohookean.h"
#include "utils/mesh.h"

int main(int argc, char *argv[]) {
  using T = double;
  using Basis = TetrahedralBasis;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  // Load in the mesh
  std::string filename("../../input/Tensile.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes, &element_nodes, &xloc);

  // Set the number of degrees of freeom
  int ndof = 3 * num_nodes;

  // Allocate space for the degrees of freeom
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
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);

  // Allocate space for the residual
  T energy = Analysis::energy(physics, num_elements, element_nodes, xloc, dof);
  Analysis::residual(physics, num_elements, element_nodes, xloc, dof, res);
  Analysis::jacobian_product(physics, num_elements, element_nodes, xloc, dof,
                             direction, Jp);

  std::cout << energy << std::endl;

  return 0;
}