add_executable(test_openmp test_openmp.cpp)

target_include_directories(test_openmp PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

# Link to the default main from Google Test
target_link_libraries(test_openmp PRIVATE gtest_main A2D::A2D)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_openmp)
