add_executable(write_vtk write_vtk.cpp)
target_include_directories(write_vtk PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(write_vtk SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(write_vtk PRIVATE A2D::A2D SparseUtils::SparseUtils)
