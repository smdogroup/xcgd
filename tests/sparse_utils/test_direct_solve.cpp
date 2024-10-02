#include <string>
#include <vector>

#include "apps/poisson_app.h"
#include "elements/gd_mesh.h"
#include "elements/gd_vandermonde.h"
#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/json.h"
#include "utils/linalg.h"
#include "utils/misc.h"

template <typename T>
void verify_solve(SparseUtils::CSCMat<T>& csc_mat, const std::vector<T>& rhs,
                  const std::vector<T>& sol_expected, double tol = 1e-15) {
  assert(csc_mat.nrows == csc_mat.ncols);
  assert(csc_mat.ncols == rhs.size());
  assert(csc_mat.ncols == sol_expected.size());

  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T> chol(&csc_mat);
  chol.factor();
  std::vector<T> sol = rhs;
  chol.solve(sol.data());
  EXPECT_VEC_NEAR(sol.size(), sol, sol_expected, tol);
}

template <typename T, int block_size>
void test_solve_explicit_mat(json inputs, std::string name) {
  using BSRMat = GalerkinBSRMat<T, block_size>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int nbrows = inputs["nrows"];
  int nnz = inputs["nnz"];
  std::vector<int> rowp((inputs["rowp"]));
  std::vector<int> cols((inputs["cols"]));
  std::vector<T> vals((inputs["vals"]));

  std::vector<T> rhs(inputs["rhs"]);
  std::vector<T> sol_expected(inputs["sol"]);

  BSRMat bsr(nbrows, nnz, rowp.data(), cols.data(), vals.data());
  CSCMat* csc = SparseUtils::bsr_to_csc(&bsr);

  verify_solve<T>(*csc, rhs, sol_expected);

  bsr.write_mtx(name + "_bsr.mtx");
  csc->write_mtx(name + "_csc.mtx");

  if (csc) {
    delete csc;
    csc = nullptr;
  }
}

TEST(sparse_utils, ExplicitCSR) {
  constexpr int max_block_size = 5;
  json data = read_json("data.json");

  for (auto it : data.items()) {
    std::string key = it.key();
    json d = it.value();

    // This is c++20, we're only c++17
    auto f = [&d, &key]<int block_size>() {
      test_solve_explicit_mat<double, block_size>(d, key);
    };
    switcher<max_block_size>::run(f, d["block_size"]);
  }
}

TEST(sparse_utils, Poisson) {
  using T = double;
  int constexpr Np_1d = 2;
  using Grid = StructuredGrid2D<T>;
  using Quadrature = GDGaussQuadrature2D<T, Np_1d>;
  using Mesh = GridMesh<T, Np_1d>;
  using Basis = GDBasis2D<T, Mesh>;
  using Poisson = PoissonApp<T, Mesh, Quadrature, Basis>;

  using BSRMat = GalerkinBSRMat<T, Poisson::Physics::dof_per_node>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int nxy[2] = {32, 32};
  T lxy[2] = {2.0, 2.0};
  T xy0[2] = {-1.0, -1.0};
  Grid grid(nxy, lxy, xy0);
  Mesh mesh(grid);
  Quadrature quadrature(mesh);
  Basis basis(mesh);

  Poisson poisson(mesh, quadrature, basis);

  std::vector<int> dof_bcs;
  double tol = 1e-6, xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;

  for (int i = 0; i < mesh.get_num_nodes(); i++) {
    T xloc[Basis::spatial_dim];
    mesh.get_node_xloc(i, xloc);
    if (freal(xloc[0]) < xmin + tol or freal(xloc[1]) < ymin + tol or
        freal(xloc[0]) > freal(xmax) - tol or
        freal(xloc[1]) > freal(ymax) - tol) {
      dof_bcs.push_back(i);
    }
  }

  BSRMat* bsr = poisson.jacobian(dof_bcs);
  CSCMat* csc = SparseUtils::bsr_to_csc(bsr);
  csc->zero_columns(dof_bcs.size(), dof_bcs.data());

  csc->write_mtx("poisson_csc.mtx");

  std::vector<T> sol_expected(csc->ncols, 0.0), rhs(csc->ncols, 0.0);
  for (int i = 0; i < csc->ncols; i++) {
    sol_expected[i] = T(rand()) / RAND_MAX;
  }

  for (int dof : dof_bcs) {
    sol_expected[dof] = 0.0;
  }

  bsr->axpy(sol_expected.data(), rhs.data());

  verify_solve<T>(*csc, rhs, sol_expected, 1e-12);

  if (bsr) {
    delete bsr;
    bsr = nullptr;
  }
  if (csc) {
    delete csc;
    csc = nullptr;
  }
}
