add_executable(test_elements test_elements.cpp)
add_executable(test_interpolation test_interpolation.cpp)
add_executable(test_integration test_integration.cpp)
add_executable(test_element_misc test_element_misc.cpp)
add_executable(test_quadrature test_quadrature.cpp)

target_include_directories(test_elements PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_interpolation PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_integration PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_element_misc PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
target_include_directories(test_quadrature PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test_elements SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_interpolation SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_integration SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_element_misc SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_include_directories(test_quadrature SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)

# Link to the default main from Google Test
target_link_libraries(test_elements PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_interpolation PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_integration PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_element_misc PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)
target_link_libraries(test_quadrature PRIVATE gtest_main A2D::A2D SparseUtils::SparseUtils)

# Make tests auto-testable with CMake ctest
include(GoogleTest)
gtest_discover_tests(test_elements)
gtest_discover_tests(test_interpolation)
gtest_discover_tests(test_integration)
gtest_discover_tests(test_element_misc)
gtest_discover_tests(test_quadrature)
