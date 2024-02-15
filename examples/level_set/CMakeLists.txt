add_executable(level_set level_set.cpp)
target_include_directories(level_set PRIVATE
    ${PROJECT_SOURCE_DIR}/include $ENV{HOME}/git/algoim/algoim)

find_package(LAPACKE REQUIRED PATHS /opt/homebrew/opt/lapack)
find_package(OpenMP REQUIRED)
target_link_libraries(level_set PRIVATE A2D::A2D SparseUtils::SparseUtils lapacke OpenMP::OpenMP_CXX)
