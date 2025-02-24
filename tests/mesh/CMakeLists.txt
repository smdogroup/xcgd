configure_file(${CMAKE_SOURCE_DIR}/tests/mesh/lsf_dof.json lsf_dof.json COPYONLY)

add_executable(test_gd_mesh test_gd_mesh.cpp)
add_executable(test_lbracket_mesh test_lbracket_mesh.cpp)
add_executable(test_finite_cell_mesh test_finite_cell_mesh.cpp)

target_include_directories(test_gd_mesh PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_lbracket_mesh PRIVATE
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_finite_cell_mesh PRIVATE
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_gd_mesh SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_lbracket_mesh SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_finite_cell_mesh SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_gd_mesh PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_lbracket_mesh PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_finite_cell_mesh PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_gd_mesh)
gtest_discover_tests(test_lbracket_mesh)
gtest_discover_tests(test_finite_cell_mesh)
