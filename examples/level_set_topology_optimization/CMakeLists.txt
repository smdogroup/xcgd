find_package(MPI REQUIRED)

add_executable(level_set level_set.cpp)
add_executable(paropt_example paropt_example.cpp)

target_include_directories(level_set PRIVATE ${XCGD_PAROPT_DIR}/src)
target_include_directories(paropt_example PRIVATE ${XCGD_PAROPT_DIR}/src)

target_include_directories(level_set PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(paropt_example PRIVATE ${MPI_INCLUDE_PATH})

target_link_directories(level_set PRIVATE ${XCGD_PAROPT_DIR}/lib)
target_link_directories(paropt_example PRIVATE ${XCGD_PAROPT_DIR}/lib)

target_include_directories(level_set PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(paropt_example PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)

target_link_libraries(level_set PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX)
target_link_libraries(paropt_example PRIVATE A2D::A2D SparseUtils::SparseUtils paropt MPI::MPI_CXX)
