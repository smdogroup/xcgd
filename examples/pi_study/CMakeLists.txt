add_executable(pi_study pi_study.cpp)
add_executable(pi_study_mesh pi_study_mesh.cpp)

target_include_directories(pi_study PRIVATE
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(pi_study_mesh PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

target_include_directories(pi_study SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(pi_study_mesh SYSTEM PRIVATE
    ${XCGD_ALGOIM_DIR}/algoim)

find_package(OpenMP REQUIRED)
target_link_libraries(pi_study PRIVATE A2D::A2D SparseUtils::SparseUtils OpenMP::OpenMP_CXX)
target_link_libraries(pi_study_mesh PRIVATE A2D::A2D SparseUtils::SparseUtils OpenMP::OpenMP_CXX)

# Copy over file
configure_file(${CMAKE_SOURCE_DIR}/examples/pi_study/driver.py driver.py COPYONLY)
