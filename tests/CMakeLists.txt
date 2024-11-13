include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929 # release-1.10.0
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
