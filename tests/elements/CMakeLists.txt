add_executable(test_elements test_elements.cpp)

target_include_directories(test_elements PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_elements PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_elements)
