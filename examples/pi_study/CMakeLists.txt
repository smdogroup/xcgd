add_executable(pi_study pi_study.cpp)
target_include_directories(pi_study PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${XCGD_ALGOIM_DIR}/algoim)

find_package(LAPACKE REQUIRED PATHS /opt/homebrew/opt/lapack)
find_package(OpenMP REQUIRED)
target_link_libraries(pi_study PRIVATE A2D::A2D SparseUtils::SparseUtils lapacke OpenMP::OpenMP_CXX)

# Copy over file
configure_file(${CMAKE_SOURCE_DIR}/examples/pi_study/driver.py driver.py COPYONLY)
