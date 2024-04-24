add_executable(test_level_set_adjoint test_level_set_adjoint.cpp)
add_executable(test_helmholtz_adjoint test_helmholtz_adjoint.cpp)
add_executable(test_quadrature_grad test_quadrature_grad.cpp)


target_include_directories(test_level_set_adjoint PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_helmholtz_adjoint PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_quadrature_grad PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

# Treat algoim as system library to supress warnings
target_include_directories(test_level_set_adjoint SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_helmholtz_adjoint SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_quadrature_grad SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_level_set_adjoint PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_helmholtz_adjoint PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_quadrature_grad PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_level_set_adjoint)
gtest_discover_tests(test_helmholtz_adjoint)
gtest_discover_tests(test_quadrature_grad)
