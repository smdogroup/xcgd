add_executable(structured_tet_mesh structured_tet_mesh.cpp)
target_include_directories(structured_tet_mesh PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(structured_tet_mesh PRIVATE A2D::A2D SparseUtils::SparseUtils)
