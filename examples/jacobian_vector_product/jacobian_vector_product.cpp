#include <string>

#include "analysis.h"
#include "elements/fe_tetrahedral.h"
#include "physics/neohookean.h"
#include "utils/mesh.h"

int main(int argc, char *argv[]) {
  using T = double;
  using Basis = TetrahedralBasis<T>;
  using Mesh = Basis::Mesh;
  using Quadrature = TetrahedralQuadrature<T>;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  // Load in the mesh
  std::string filename("../../input/Tensile.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes, &element_nodes, &xloc);

  Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Quadrature quadrature;
  Basis basis(mesh, quadrature);

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
    direction[i] = (T)rand() / RAND_MAX;
  }

  // Allocate the physics
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);
  Analysis analysis(basis, physics);

  // Allocate space for the residual
  T energy = analysis.energy(dof);
  analysis.residual(dof, res);
  analysis.jacobian_product(dof, direction, Jp);

  T pres = 0.0, pJp = 0.0;
  for (int i = 0; i < ndof; i++) {
    pres += res[i] * direction[i];
    pJp += Jp[i] * direction[i];
  }

  std::printf("energy: %20.10e\n", energy);
  std::printf("res   : %20.10e\n", pres);
  std::printf("Jp    : %20.10e\n", pJp);

  return 0;
}