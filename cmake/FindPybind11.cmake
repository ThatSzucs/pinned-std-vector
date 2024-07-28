include(FetchContent)

# Fetch pybind11
FetchContent_Declare(
  pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.5.0
)

FetchContent_MakeAvailable(pybind11)
