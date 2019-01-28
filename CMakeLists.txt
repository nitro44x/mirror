cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(mirror LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "-arch=sm_61")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-G")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)