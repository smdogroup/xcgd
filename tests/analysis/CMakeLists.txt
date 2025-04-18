configure_file(${CMAKE_SOURCE_DIR}/tests/analysis/step_size_study.py step_size_study.py COPYONLY)

add_executable(test_analysis test_analysis.cpp)
add_executable(test_quad_level_derivs test_quad_level_derivs.cpp)

target_include_directories(test_analysis PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_quad_level_derivs PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_analysis SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_quad_level_derivs SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_analysis PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_quad_level_derivs PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_analysis)
gtest_discover_tests(test_quad_level_derivs)
