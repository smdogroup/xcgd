add_executable(cut_cell cut_cell.cpp)
target_include_directories(cut_cell PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(cut_cell PRIVATE A2D::A2D SparseUtils::SparseUtils)
