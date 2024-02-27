# Specify cmake version needed
cmake_minimum_required(VERSION 3.12)

project(XCGD
  VERSION 1.0.0
  DESCRIPTION "Numerical discretization using continuous Galerkin difference method"
  HOMEPAGE_URL "https://github.com/smdogroup/xcgd"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(XCGD_SPARSE_UTILS_DIR $ENV{HOME}/installs/sparse-utils)
set(XCGD_A2D_DIR $ENV{HOME}/installs/a2d)
set(XCGD_ALGOIM_DIR $ENV{HOME}/git/algoim)
option(XCGD_BUILD_TESTS "Build unit tests or not" ON)

# Dependencies
include(cmake/utils.cmake)
check_path_and_inform(a2d ${XCGD_A2D_DIR} XCGD_A2D_DIR)
check_path_and_inform(SparseUtils ${XCGD_SPARSE_UTILS_DIR} XCGD_SPARSE_UTILS_DIR)
check_path_and_inform(algoim ${XCGD_ALGOIM_DIR} XCGD_ALGOIM_DIR)
find_package(A2D REQUIRED PATHS ${XCGD_A2D_DIR})
find_package(SparseUtils REQUIRED PATHS ${XCGD_SPARSE_UTILS_DIR})

# Set warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|GNU")
  add_compile_options(
    -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare
  )
endif()

# Copy over inputs
file(COPY ${PROJECT_SOURCE_DIR}/input DESTINATION ${PROJECT_BINARY_DIR})

if(XCGD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(examples)
