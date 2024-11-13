add_executable(order_drop_study order_drop_study.cpp)
target_include_directories(order_drop_study PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(order_drop_study SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(order_drop_study PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/examples/local_order_drop_accuracy/driver.py driver.py COPYONLY)
