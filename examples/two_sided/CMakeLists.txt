add_executable(two_sided two_sided.cpp)

# target_compile_options(two_sided PRIVATE -fsanitize=address)
# target_link_options(two_sided PRIVATE -fsanitize=address)

target_include_directories(two_sided PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(two_sided SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(two_sided PRIVATE A2D::A2D SparseUtils::SparseUtils)
