add_executable(test_partition_identification test_partition_identification.cpp)
add_executable(test_algoim_gradient_1d test_algoim_gradient_1d.cpp)
add_executable(test_algoim_gradient_2d test_algoim_gradient_2d.cpp)

target_include_directories(test_partition_identification PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_algoim_gradient_1d PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_algoim_gradient_2d PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_partition_identification SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_algoim_gradient_1d SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_algoim_gradient_2d SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_partition_identification PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_algoim_gradient_1d PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_algoim_gradient_2d PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_partition_identification)
gtest_discover_tests(test_algoim_gradient_1d)
gtest_discover_tests(test_algoim_gradient_2d)
