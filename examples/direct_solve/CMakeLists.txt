add_executable(direct_solve direct_solve.cpp)
target_include_directories(direct_solve PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(direct_solve PRIVATE A2D::A2D SparseUtils::SparseUtils)

# Copy over file
configure_file(${CMAKE_SOURCE_DIR}/examples/direct_solve/visualize_mtx.py visualize_mtx.py COPYONLY)