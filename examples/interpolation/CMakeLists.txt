add_executable(interpolation interpolation.cpp)
target_include_directories(interpolation PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(interpolation PRIVATE A2D::A2D SparseUtils::SparseUtils)
