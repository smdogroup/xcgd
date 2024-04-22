add_executable(test_parser test_parser.cpp)

target_include_directories(test_parser PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

# Link to the default main from Google Test
target_link_libraries(test_parser PRIVATE gtest_main A2D::A2D)

configure_file(${CMAKE_SOURCE_DIR}/tests/utils/test_parser.cfg test_parser.cfg COPYONLY)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_parser)
