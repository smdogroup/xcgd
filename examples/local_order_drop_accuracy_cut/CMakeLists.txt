add_executable(cut_poisson_order_drop cut_poisson_order_drop.cpp)
target_include_directories(cut_poisson_order_drop PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(cut_poisson_order_drop SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(cut_poisson_order_drop PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/examples/local_order_drop_accuracy/driver.py driver.py COPYONLY)
