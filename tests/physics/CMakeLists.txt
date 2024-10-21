add_executable(test_physics test_physics.cpp)
add_executable(test_physics_fd test_physics_fd.cpp)

target_include_directories(test_physics PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_physics_fd PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_physics SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_physics_fd SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_physics PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_physics_fd PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_physics)
gtest_discover_tests(test_physics_fd)
