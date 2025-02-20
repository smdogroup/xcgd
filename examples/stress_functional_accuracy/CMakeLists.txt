add_executable(stress_functional_accuracy stress_functional_accuracy.cpp)
target_include_directories(stress_functional_accuracy PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(stress_functional_accuracy SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(stress_functional_accuracy PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/examples/stress_functional_accuracy/driver.py driver.py COPYONLY)
