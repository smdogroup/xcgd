add_executable(galerkin_difference galerkin_difference.cpp)
target_include_directories(galerkin_difference PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(galerkin_difference PRIVATE A2D::A2D SparseUtils::SparseUtils)
