# Specify cmake version needed
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Set project name, language and c++ std
project(XCGD LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path variables
set(XCGD_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Dependencies
if(NOT DEFINED XCGD_A2D_DIR)
  set(XCGD_A2D_DIR $ENV{HOME}/git/a2d)

  if(NOT EXISTS ${XCGD_A2D_DIR})
    message(FATAL_ERROR "${XCGD_A2D_DIR} does not exist, please specify the path by a2d using -DXCGD_A2D_DIR=...")
  endif()
endif()

# Set warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|GNU")
  add_compile_options(
    -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare
  )
endif()

# Copy over inputs
file(COPY ${XCGD_ROOT_DIR}/input DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Build unit tests
# Download google test (git tag needs to be updated frequently)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929 # release-1.10.0
)
FetchContent_MakeAvailable(googletest)
enable_testing()
add_subdirectory(${XCGD_ROOT_DIR}/tests)

# Build examples
add_subdirectory(${XCGD_ROOT_DIR}/examples)
