#include <omp.h>

#include "test_commons.h"

TEST(openmp, CanCompileAndRun) {
  std::cout << omp_get_num_threads() << "\n";
#pragma omp parallel
  {
    std::cout << omp_get_num_threads() << "\n";
    std::cout << "Hello World" << std::endl;
  }
}
