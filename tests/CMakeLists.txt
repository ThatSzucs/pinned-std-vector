cmake_minimum_required(VERSION 3.8)
project(playgroundTests)

# Dependencies
enable_testing()
if(NOT TARGET ThatSzucs::playground)
  find_package(playground CONFIG REQUIRED)
endif()

# Speed up Catch2 compilation.
# Source: https://github.com/catchorg/Catch2/blob/master/docs/slow-compiles.md
add_library(catch_test_main ${CMAKE_CURRENT_LIST_DIR}/catch_initializer.cpp)

target_include_directories(catch_test_main
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/
)

# Function for setting up executible for benchmarks
# - add private source sharing name with target
# - link libraries
# - add test
function(set_up_test target_name)

  # Target definition
  add_executable(${target_name} "")

  target_sources(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/${target_name}.cpp
  )

  target_link_libraries(${target_name}
    PRIVATE
      ThatSzucs::playground
      catch_test_main
  )

  add_test(
    NAME
      ${target_name}
    COMMAND
      $<TARGET_FILE:${target_name}>
  )
endfunction()

set_up_test(test_pinned_vector)
