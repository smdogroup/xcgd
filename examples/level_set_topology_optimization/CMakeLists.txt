find_package(MPI REQUIRED)

configure_file(${CMAKE_SOURCE_DIR}/examples/level_set_topology_optimization/level_set.cfg
    level_set.cfg COPYONLY)

add_executable(level_set level_set.cpp)

target_include_directories(level_set PRIVATE ${XCGD_PAROPT_DIR}/src
    ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/include)
target_include_directories(level_set SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

target_link_directories(level_set PRIVATE ${XCGD_PAROPT_DIR}/lib)

target_link_libraries(level_set PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX)
