cmake_minimum_required(VERSION 3.11)
project(playground)

# Dependencies
find_package(Pybind11)
if(NOT TARGET ThatSzucs::playground)
  find_package(playground CONFIG REQUIRED)
endif()

# This module needs to be named the same as in the cpp file
pybind11_add_module(playground_bindings "")

target_sources(playground_bindings
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src/bindings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pyarray_utils.h
)

target_link_libraries(playground_bindings
  PUBLIC
    ThatSzucs::playground
)
