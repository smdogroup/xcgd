add_executable(use_a2d use_a2d.cpp)

target_compile_options(use_a2d PRIVATE -fsanitize=address)
target_link_options(use_a2d PRIVATE -fsanitize=address)

target_include_directories(use_a2d PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(use_a2d SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(use_a2d PRIVATE A2D::A2D SparseUtils::SparseUtils)
