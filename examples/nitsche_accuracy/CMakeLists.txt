add_executable(nitsche_accuracy nitsche_accuracy.cpp)
target_include_directories(nitsche_accuracy PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(nitsche_accuracy SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(nitsche_accuracy PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/examples/local_order_drop_accuracy/driver.py driver.py COPYONLY)
