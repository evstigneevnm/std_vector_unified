cmake_minimum_required(VERSION 3.20)
set(project_name universal_allocator)
project(${project_name} LANGUAGES CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 35 52 61 70 75 80)
set(CMAKE_CONFIGURATION_TYPES Debug Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(std_vector_custom_allocator.bin std_vector_custom_allocator.cpp)
add_executable(std_vector_cuda.bin std_vector_cuda.cu)

