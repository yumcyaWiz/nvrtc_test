cmake_minimum_required(VERSION 3.27)
project(nvrtc-test LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ version and std
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# CUDA C++ version and std
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

add_executable(main "src/main.cpp")
target_link_libraries(main PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
)