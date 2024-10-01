#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sparse_utils/sparse_utils.h"
#include "test_commons.h"
#include "utils/json.h"
#include "utils/linalg.h"
#include "utils/misc.h"

// template <typename T, class Physics>
// void factor_Jacobian_mat() {
//   using BSRMat = GalerkinBSRMat<T, Physics::dof_per_node>;
// }
//

template <typename T, int block_size>
void test_factorization(json inputs, std::string name) {
  using BSRMat = GalerkinBSRMat<T, block_size>;
  using CSCMat = SparseUtils::CSCMat<T>;

  int nbrows = inputs["nrows"];
  int nnz = inputs["nnz"];
  std::vector<int> rowp((inputs["rowp"]));
  std::vector<int> cols((inputs["cols"]));
  std::vector<T> vals((inputs["vals"]));

  std::vector<T> rhs(inputs["rhs"]);
  std::vector<T> sol_exact(inputs["sol"]);

  BSRMat bsr(nbrows, nnz, rowp.data(), cols.data(), vals.data());

  CSCMat* csc = SparseUtils::bsr_to_csc(&bsr);

  SparseUtils::CholOrderingType order = SparseUtils::CholOrderingType::ND;
  SparseUtils::SparseCholesky<T>* chol =
      new SparseUtils::SparseCholesky<T>(csc);
  chol->factor();
  std::vector<T> sol = rhs;
  chol->solve(sol.data());

  EXPECT_VEC_NEAR(sol.size(), sol, sol_exact);

  bsr.write_mtx(name + "_bsr.mtx");
  csc->write_mtx(name + "_csc.mtx");
}

TEST(sparse_utils, ExplicitCSR) {
  constexpr int max_block_size = 5;
  json data = read_json("data.json");

  for (auto it : data.items()) {
    std::string key = it.key();
    json d = it.value();
    auto f = [&d, &key]<int block_size>() {
      test_factorization<double, block_size>(d, key);
    };
    switcher<max_block_size>::run(f, d["block_size"]);
  }
}
