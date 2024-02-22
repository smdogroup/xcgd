add_executable(test_linalg test_linalg.cpp)

target_include_directories(test_linalg PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

# Link to the default main from Google Test
target_link_libraries(test_linalg PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_linalg)
