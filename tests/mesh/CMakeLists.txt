configure_file(${CMAKE_SOURCE_DIR}/tests/mesh/lsf_dof.json lsf_dof.json COPYONLY)

add_executable(test_gd_mesh test_gd_mesh.cpp)

target_include_directories(test_gd_mesh PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_gd_mesh SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_gd_mesh PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_gd_mesh)
