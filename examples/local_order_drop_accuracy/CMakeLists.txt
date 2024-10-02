add_executable(poisson_order_drop poisson_order_drop.cpp)
target_include_directories(poisson_order_drop PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(poisson_order_drop SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(poisson_order_drop PRIVATE A2D::A2D SparseUtils::SparseUtils)


configure_file(${CMAKE_SOURCE_DIR}/examples/local_order_drop_accuracy/driver.py driver.py COPYONLY)
