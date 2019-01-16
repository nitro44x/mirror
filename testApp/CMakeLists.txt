cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(srcs main.cpp 
         kernels.cuh
         kernels.cu
)

set(headers simt_vector.hpp
            simt_allocator.hpp
			simt_macros.hpp
)

add_executable(containerApp ${srcs} ${headers})