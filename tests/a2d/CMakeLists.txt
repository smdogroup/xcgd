add_executable(test_a2d test_a2d.cpp)

target_include_directories(test_a2d PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_a2d SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_a2d PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_a2d)
