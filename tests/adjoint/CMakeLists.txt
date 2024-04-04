add_executable(test_adjoint test_adjoint.cpp)
add_executable(test_quadrature_grad test_quadrature_grad.cpp)

target_include_directories(test_adjoint PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_quadrature_grad PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_adjoint PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_quadrature_grad PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_adjoint)
gtest_discover_tests(test_quadrature_grad)
