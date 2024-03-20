add_executable(linear_elasticity linear_elasticity.cpp)
target_include_directories(linear_elasticity PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(linear_elasticity PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/python/visualize_mtx.py visualize_mtx.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/check_eigenvalues.py check_eigenvalues.py COPYONLY)
