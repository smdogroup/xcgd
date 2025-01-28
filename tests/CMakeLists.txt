include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59 # release-1.15.2
)
FetchContent_MakeAvailable(googletest)

if (XCGD_USE_OPENMP)
  link_libraries(OpenMP::OpenMP_CXX)
  add_subdirectory(openmp)
endif()

add_subdirectory(physics)
add_subdirectory(linalg)
add_subdirectory(elements)
add_subdirectory(mesh)
add_subdirectory(adjoint)
add_subdirectory(apps)
add_subdirectory(utils)
add_subdirectory(stencil)
add_subdirectory(regression)
add_subdirectory(sparse_utils)
add_subdirectory(analysis)
add_subdirectory(algoim)
