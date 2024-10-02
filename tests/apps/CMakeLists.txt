add_executable(test_helmholtz_filter test_helmholtz_filter.cpp)
add_executable(test_robust_projection test_robust_projection.cpp)
add_executable(test_poisson_app test_poisson_app.cpp)

target_include_directories(test_helmholtz_filter PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_robust_projection PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_poisson_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_helmholtz_filter SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_robust_projection SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_poisson_app SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_helmholtz_filter PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_robust_projection PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_poisson_app PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_helmholtz_filter)
gtest_discover_tests(test_robust_projection)
gtest_discover_tests(test_poisson_app)
