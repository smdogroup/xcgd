# Specify cmake version needed
cmake_minimum_required(VERSION 3.12)

project("XCGD"
  VERSION 1.0.0
  DESCRIPTION "Numerical discretization using continuous Galerkin difference method"
  HOMEPAGE_URL "https://github.com/smdogroup/xcgd"
  LANGUAGES CXX)

# Export the compile commands that clangd LS uses
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ 17 is required for XCGD
set(CMAKE_CXX_STANDARD 17)

set(XCGD_SPARSE_UTILS_DIR $ENV{HOME}/installs/sparse-utils)
set(XCGD_A2D_DIR $ENV{HOME}/installs/a2d)
set(XCGD_ALGOIM_DIR $ENV{HOME}/git/algoim)
set(XCGD_PAROPT_DIR $ENV{HOME}/git/paropt)

option(XCGD_BUILD_TESTS "Build unit tests or not" ON)
option(XCGD_BUILD_EXAMPLES "Build examples or not" ON)
option(XCGD_USE_OPENMP "use openmp or not" OFF)

# If in debug mode, set the preprocessor definition
if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DXCGD_DEBUG_MODE)
endif()

# add_compile_options(-Og)

# Dependencies
include(cmake/utils.cmake)
check_path_and_inform(FATAL_ERROR A2D ${XCGD_A2D_DIR} XCGD_A2D_DIR)
check_path_and_inform(FATAL_ERROR SparseUtils ${XCGD_SPARSE_UTILS_DIR} XCGD_SPARSE_UTILS_DIR)
check_path_and_inform(FATAL_ERROR algoim ${XCGD_ALGOIM_DIR} XCGD_ALGOIM_DIR)
check_path_and_inform(WARNING paropt ${XCGD_PAROPT_DIR} XCGD_PAROPT_DIR)
find_package(A2D REQUIRED PATHS ${XCGD_A2D_DIR})
find_package(SparseUtils REQUIRED PATHS ${XCGD_SPARSE_UTILS_DIR})

if(XCGD_USE_OPENMP)
find_package(OpenMP REQUIRED)
endif()

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

if(XCGD_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# === Install xcgd as a header-only library ===

# Change the default value of CMAKE_INSTALL_PREFIX
set(XCGD_INSTALL_DIR $ENV{HOME}/installs/xcgd)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${XCGD_INSTALL_DIR} CACHE PATH "The default\
  path to install xcgd to" FORCE)
  message("The directory to install xcgd is set to default:\
  ${XCGD_INSTALL_DIR}")
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(XCGD_INSTALL_LIBRARY "Enable installation" ${PROJECT_IS_TOP_LEVEL})

# Interface makes the target header-only that does not need to be compiled
add_library(${PROJECT_NAME} INTERFACE)

# Pass along c++17 requirements to projects using this library
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Dependencies
target_include_directories(${PROJECT_NAME} INTERFACE
  ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(${PROJECT_NAME} INTERFACE A2D::A2D SparseUtils::SparseUtils)

if (XCGD_INSTALL_LIBRARY)
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

  # install cmake files
  install(
    EXPORT ${PROJECT_NAME}_Targets
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  )
  install(
    FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  )

  # Install headers
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
endif()
