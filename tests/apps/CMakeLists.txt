add_executable(test_helmholtz_filter test_helmholtz_filter.cpp)
add_executable(test_robust_projection test_robust_projection.cpp)
add_executable(test_poisson_app test_poisson_app.cpp)
add_executable(test_elastic_app test_elastic_app.cpp)
add_executable(test_convolution_filter test_convolution_filter.cpp)

configure_file(${CMAKE_SOURCE_DIR}/tests/apps/plot_robust_projection.py plot_robust_projection.py COPYONLY)

target_include_directories(test_helmholtz_filter PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_robust_projection PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_poisson_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_elastic_app PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_convolution_filter PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_helmholtz_filter SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_robust_projection SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_poisson_app SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_elastic_app SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_convolution_filter SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_helmholtz_filter PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_robust_projection PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_poisson_app PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_elastic_app PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_convolution_filter PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_helmholtz_filter)
gtest_discover_tests(test_robust_projection)
gtest_discover_tests(test_poisson_app)
gtest_discover_tests(test_elastic_app)
gtest_discover_tests(test_convolution_filter)
