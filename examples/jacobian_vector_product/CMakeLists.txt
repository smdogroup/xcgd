add_executable(jacobian_vector_product jacobian_vector_product.cpp)
target_include_directories(jacobian_vector_product PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(jacobian_vector_product SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(jacobian_vector_product PRIVATE A2D::A2D SparseUtils::SparseUtils)