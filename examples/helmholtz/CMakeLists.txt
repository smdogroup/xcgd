add_executable(helmholtz helmholtz.cpp)
target_include_directories(helmholtz PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(helmholtz SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(helmholtz PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/python/visualize_mtx.py visualize_mtx.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/check_eigenvalues.py check_eigenvalues.py COPYONLY)
