add_executable(test_parser test_parser.cpp)
add_executable(test_json test_json.cpp)

target_include_directories(test_parser PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_json PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

# Link to the default main from Google Test
target_link_libraries(test_parser PRIVATE gtest_main A2D::A2D)
target_link_libraries(test_json PRIVATE gtest_main A2D::A2D)

configure_file(${CMAKE_SOURCE_DIR}/tests/utils/test_parser.cfg test_parser.cfg COPYONLY)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_parser)
gtest_discover_tests(test_json)
