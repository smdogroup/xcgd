add_executable(volume_integral volume_integral.cpp)
target_include_directories(volume_integral PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(volume_integral SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(volume_integral PRIVATE A2D::A2D SparseUtils::SparseUtils)
