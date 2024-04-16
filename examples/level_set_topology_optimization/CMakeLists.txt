add_executable(level_set level_set.cpp)
target_include_directories(level_set PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(level_set PRIVATE A2D::A2D SparseUtils::SparseUtils)
