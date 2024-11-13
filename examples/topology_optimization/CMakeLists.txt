find_package(MPI REQUIRED)

configure_file(${CMAKE_SOURCE_DIR}/examples/topology_optimization/topo.cfg
    topo.cfg COPYONLY)

add_executable(topo topo.cpp)
add_executable(test_lbracket_mesh test_lbracket_mesh.cpp)

target_include_directories(topo PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include)
target_include_directories(test_lbracket_mesh PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include)

target_include_directories(topo SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_lbracket_mesh SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

target_link_directories(topo PRIVATE ${XCGD_PAROPT_DIR}/lib)
target_link_directories(test_lbracket_mesh PRIVATE ${XCGD_PAROPT_DIR}/lib)

target_link_libraries(topo PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX)
target_link_libraries(test_lbracket_mesh PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX)
