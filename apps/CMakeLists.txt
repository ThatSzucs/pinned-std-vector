# CUDA setup for this standalone executable
find_package(CUDAToolkit)
enable_language(CUDA)


add_executable(app_bandwidth_tester "")

target_sources(app_bandwidth_tester
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/app_bandwidth_tester.cu
)

set_target_properties(app_bandwidth_tester
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
