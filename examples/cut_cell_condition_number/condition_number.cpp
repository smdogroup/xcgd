#include <string>
#include <vector>

#include "apps/static_elastic.h"
#include "elements/element_utils.h"
#include "elements/gd_vandermonde.h"
#include "sparse_utils/sparse_utils.h"
#include "utils/argparser.h"
#include "utils/vtk.h"

template <int Np_1d>
void generate_stiffness_matrix(int n, int l, double x0, double y0, double r,
                               std::string prefix) {
  using T = double;
  using Grid = StructuredGrid2D<T>;
  using Mesh = CutMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Quadrature = GDLSFQuadrature2D<T, Np_1d>;
  using Elastic = StaticElastic<T, Mesh, Quadrature, Basis>;
  using BSRMat = GalerkinBSRMat<T, Elastic::Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;
  using Interpolator = Interpolator<T, Quadrature, Basis>;

  int nxy[2];
  nxy[0] = n;
  nxy[1] = n;
  T lxy[2];
  lxy[0] = l;
  lxy[1] = l;
  Grid grid(nxy, lxy);

  // Create the mesh with a hole
  Mesh mesh(grid, [&r, &x0, &y0](T* xloc) {
    return r * r - (xloc[0] - x0) * (xloc[0] - x0) -
           (xloc[1] - y0) * (xloc[1] - y0);
  });

  Quadrature quadrature(mesh);
  Basis basis(mesh);

  T E = 1.0, nu = 0.3;
  Elastic elastic(E, nu, mesh, quadrature, basis);
  // Apply bcs
  std::vector<int> bc_nodes = mesh.get_left_boundary_nodes();
  std::vector<int> bc_dof;
  for (int node : bc_nodes) {
    bc_dof.push_back(2 * node);
    bc_dof.push_back(2 * node + 1);
  }
  BSRMat* bsr_mat = elastic.jacobian(bc_dof);
  CSCMat* csc_mat = SparseUtils::bsr_to_csc(bsr_mat);
  csc_mat->zero_columns(bc_dof.size(), bc_dof.data());
  csc_mat->write_mtx(std::filesystem::path(prefix) /
                     std::filesystem::path("stiffness_matrix.mtx"));

  // Export quadratures and mesh
  Interpolator interp(mesh, quadrature, basis);
  interp.to_vtk(std::filesystem::path(prefix) /
                std::filesystem::path("quad.vtk"));
  ToVTK<T, Mesh> vtk(
      mesh, std::filesystem::path(prefix) / std::filesystem::path("mesh.vtk"));
  vtk.write_mesh();
  vtk.write_sol("lsf", mesh.get_lsf_nodes().data());
  std::vector<T> bcs(mesh.get_num_nodes() * 2, 0.0);
  for (int i : bc_dof) {
    bcs[i] = 1.0;
  }
  vtk.write_vec("bcs", bcs.data());

  if (bsr_mat) delete bsr_mat;
  if (csc_mat) delete csc_mat;
}

int main(int argc, char* argv[]) {
  ArgParser p;
  p.add_argument<int>("--Np_1d", 2);
  p.add_argument<int>("--n", 33);
  p.add_argument<double>("--l", 1.0);
  p.add_argument<double>("--x0", 0.45);
  p.add_argument<double>("--y0", 0.42);
  p.add_argument<double>("--r", 0.35);
  p.add_argument<std::string>("--prefix", {});
  p.parse_args(argc, argv);

  std::string prefix = p.get<std::string>("prefix");
  if (prefix.empty()) {
    prefix = get_local_time();
  }

  p.write_args_to_file(std::filesystem::path(prefix) /
                       std::filesystem::path("args.txt"));

  int Np_1d = p.get<int>("Np_1d");
  int n = p.get<int>("n");
  double l = p.get<double>("l");
  double x0 = p.get<double>("x0");
  double y0 = p.get<double>("y0");
  double r = p.get<double>("r");

  switch (Np_1d) {
    case 2:
      generate_stiffness_matrix<2>(n, l, x0, y0, r, prefix);
      break;
    case 4:
      generate_stiffness_matrix<4>(n, l, x0, y0, r, prefix);
      break;
    case 6:
      generate_stiffness_matrix<6>(n, l, x0, y0, r, prefix);
      break;
    case 8:
      generate_stiffness_matrix<8>(n, l, x0, y0, r, prefix);
      break;
    default:
      printf("Unsupported Np_1d (%d), exiting...\n", Np_1d);
      exit(-1);
  }

  return 0;
}
