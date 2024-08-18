cmake_minimum_required(VERSION 3.18)

project(playground VERSION "0.1")

# Dependencies
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake" ${CMAKE_MODULE_PATH})
find_package(Threads)
find_package(CUDAToolkit)

# Library definition
add_library(playground "")

target_sources(playground
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/pinned_vector.cpp

    ${CMAKE_CURRENT_LIST_DIR}/include/playground/pinned_vector.h
)
target_link_libraries(playground
  PUBLIC
    CUDA::cudart
)
target_compile_features(playground PUBLIC cxx_std_17)
target_include_directories(playground
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
add_library(ThatSzucs::playground ALIAS playground)
