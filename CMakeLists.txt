cmake_minimum_required(VERSION 3.17)
project(playground)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Used by find_package() to find CMake onfiguration and FindXXX.cmake files.
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# option(PIP_INSTALL "Running pip install." OFF)

if(PIP_INSTALL)
    # Project code
    # add_subdirectory(src)

    # # Python bindings
    # add_subdirectory(python)
else()
    # Project code
    add_subdirectory(src)
    
    # Unit tests
    enable_testing()
    add_subdirectory(tests)

    # # Python bindings
    # add_subdirectory(python)

    # Misc. mumbo-jumbo
    add_subdirectory(apps)
endif()
