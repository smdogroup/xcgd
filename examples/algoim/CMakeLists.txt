add_executable(volume_integral volume_integral.cpp)
add_executable(surface_integral surface_integral.cpp)

target_include_directories(volume_integral PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(surface_integral PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_include_directories(volume_integral SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(surface_integral SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

target_link_libraries(volume_integral PRIVATE A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(surface_integral PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/examples/algoim/driver.py driver.py COPYONLY)
