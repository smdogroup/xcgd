add_executable(nitsche_accuracy nitsche_accuracy.cpp)

# target_compile_options(nitsche_accuracy PRIVATE -fsanitize=address)
# target_link_options(nitsche_accuracy PRIVATE -fsanitize=address)

target_include_directories(nitsche_accuracy PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(nitsche_accuracy SYSTEM PRIVATE ${XCGD_ALGOIM_DIR}/algoim)
target_link_libraries(nitsche_accuracy PRIVATE A2D::A2D SparseUtils::SparseUtils)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/driver.py ${CMAKE_CURRENT_BINARY_DIR}/driver.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lsf_creator.py ${CMAKE_CURRENT_BINARY_DIR}/lsf_creator.py COPYONLY)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/images)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/images/stanford_dragon.png ${CMAKE_CURRENT_BINARY_DIR}/images/stanford_dragon.png COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/images/buzz.png ${CMAKE_CURRENT_BINARY_DIR}/images/buzz.png COPYONLY)
