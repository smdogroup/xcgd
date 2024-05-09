add_executable(condition_number condition_number.cpp)
target_include_directories(condition_number PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(condition_number SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(condition_number PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_SOURCE_DIR}/python/visualize_mtx.py visualize_mtx.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/check_eigenvalues.py check_eigenvalues.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/python/check_condition_number.py check_condition_number.py COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/examples/cut_cell_condition_number/driver.py driver.py COPYONLY)
