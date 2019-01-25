cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(srcs main.cpp
         kernels2.cpp
)

set(cu_srcs kernels.cu
)

set(headers kernels.cuh
            kernels2.hpp
)

set(simt_headers simt_vector.hpp
                 simt_allocator.hpp
                 simt_macros.hpp
                 simt_serialization.hpp
                 simt_utilities.hpp
)

set(simt_srcs simt_utilities.cpp)


set_property(SOURCE ${srcs} PROPERTY LANGUAGE CUDA)
set_property(SOURCE ${simt_srcs} PROPERTY LANGUAGE CUDA)
add_executable(containerApp ${headers} ${srcs} ${cu_srcs} ${simt_headers} ${simt_srcs})
target_compile_features(containerApp PUBLIC cxx_std_11)
set_target_properties(containerApp PROPERTIES POSIION_INDEPENDENT_CODE ON)
set_target_properties(containerApp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(containerApp PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)