#include <string>

#include "analysis.h"
#include "elements/tetrahedral.h"
#include "physics/neohookean.h"
#include "utils/mesh.h"

int main(int argc, char *argv[]) {
  using T = double;
  using Basis = TetrahedralBasis<T>;
  using Mesh = Basis::Mesh;
  using Quadrature = TetrahedralQuadrature;
  using Physics = NeohookeanPhysics<T>;
  using Analysis = FEAnalysis<T, Basis, Quadrature, Physics>;

  int num_elements, num_nodes;
  int *element_nodes;
  T *xloc;

  // Load in the mesh
  std::string filename("../../input/Tensile.inp");
  load_mesh<T>(filename, &num_elements, &num_nodes, &element_nodes, &xloc);

  Mesh mesh(num_elements, num_nodes, element_nodes, xloc);
  Basis basis(mesh);

  // Set the number of degrees of freeom
  int ndof = 3 * num_nodes;

  // Allocate space for the degrees of freeom
  T *dof = new T[ndof];
  T *res = new T[ndof];
  T *res_new = new T[ndof];
  T *Jp = new T[ndof];
  T *Jp_new = new T[ndof];
  T *direction = new T[ndof];
  for (int i = 0; i < ndof; i++) {
    dof[i] = 0.01 * rand() / RAND_MAX;
    res[i] = 0.0;
    res_new[i] = 0.0;
    Jp[i] = 0.0;
    Jp_new[i] = 0.0;
    direction[i] = (T)rand() / RAND_MAX;
  }

  // Allocate the physics
  T C1 = 0.01;
  T D1 = 0.5;
  Physics physics(C1, D1);
  Analysis analysis(basis, physics);

  // Allocate space for the residual
  T energy = analysis.energy(physics, num_elements, element_nodes, xloc, dof);
  T energy_new = analysis.energy_new(dof);
  analysis.residual(physics, num_elements, element_nodes, xloc, dof, res);
  analysis.residual_new(dof, res_new);
  analysis.jacobian_product(physics, num_elements, element_nodes, xloc, dof,
                            direction, Jp);
  analysis.jacobian_product_new(dof, direction, Jp_new);

  T pres = 0.0, pres_new = 0.0, pJp = 0.0, pJp_new = 0.0;
  for (int i = 0; i < ndof; i++) {
    pres += res[i] * direction[i];
    pres_new += res_new[i] * direction[i];
    pJp += Jp[i] * direction[i];
    pJp_new += Jp_new[i] * direction[i];
  }

  T energy_err = (energy - energy_new) / energy;
  T pres_err = (pres - pres_new) / pres;
  T pJp_err = (pJp - pJp_new) / pJp;
  std::printf("energy: %20.10e, energy_new: %20.10e, err: %20.10e\n", energy,
              energy_new, energy_err);
  std::printf("res   : %20.10e, res_new   : %20.10e, err: %20.10e\n", pres,
              pres_new, pres_err);
  std::printf("Jp    : %20.10e, Jp_new    : %20.10e, err: %20.10e\n", pJp,
              pJp_new, pJp_err);

  return 0;
}