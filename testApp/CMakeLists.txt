cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(srcs main.cpp)
set(cu_srcs kernels.cuh
            kernels.cu
)

set(headers simt_vector.hpp
            simt_allocator.hpp
            simt_macros.hpp
)


set_property(SOURCE ${srcs} PROPERTY LANGUAGE CUDA)
add_executable(containerApp ${srcs} ${cu_srcs} ${headers})
target_compile_features(containerApp PUBLIC cxx_std_11)