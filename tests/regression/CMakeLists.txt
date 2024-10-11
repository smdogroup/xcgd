add_executable(test_regression_static test_regression_static.cpp)

target_include_directories(test_regression_static PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_regression_static SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_regression_static PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/tests/regression/data_static.json data_static.json COPYONLY)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_regression_static)
