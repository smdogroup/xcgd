find_package(MPI REQUIRED)

configure_file(${CMAKE_SOURCE_DIR}/examples/topology_optimization/topo.cfg
    topo.cfg COPYONLY)

add_executable(topo topo.cpp)
add_executable(test_lbracket_mesh test_lbracket_mesh.cpp)
add_executable(test_topo test_topo.cpp)
add_executable(paropt_example paropt_example.cpp)
add_executable(test_filter test_filter.cpp)

# target_compile_options(topo PRIVATE -fsanitize=address)
# target_link_options(topo PRIVATE -fsanitize=address)

target_include_directories(topo PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_lbracket_mesh PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_topo PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(paropt_example PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_filter PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(topo SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_lbracket_mesh SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_topo SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(paropt_example SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_filter SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

target_link_directories(topo PRIVATE ${XCGD_PAROPT_DIR}/lib)
target_link_directories(test_lbracket_mesh PRIVATE ${XCGD_PAROPT_DIR}/lib)
target_link_directories(test_topo PRIVATE ${XCGD_PAROPT_DIR}/lib)
target_link_directories(paropt_example PRIVATE ${XCGD_PAROPT_DIR}/lib)
target_link_directories(test_filter PRIVATE ${XCGD_PAROPT_DIR}/lib)

target_link_libraries(topo PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX)
target_link_libraries(test_lbracket_mesh PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX gtest_main)
target_link_libraries(test_topo PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX gtest_main)
target_link_libraries(paropt_example PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX gtest_main)
target_link_libraries(test_filter PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX gtest_main)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_lbracket_mesh)
gtest_discover_tests(test_topo)
