cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(srcs main.cpp)

set(cuda_cpp_srcs kernels2.cpp
                  Particle.cpp
)

set(cu_srcs kernels.cu
            particle_test.cu
)

set(headers kernels.hpp
            kernels2.hpp
            particle_test.hpp
            Particle.hpp
)


set_property(SOURCE ${cuda_cpp_srcs} PROPERTY LANGUAGE CUDA)
add_executable(testApp ${headers} ${srcs} ${cuda_cpp_srcs} ${cu_srcs})
target_link_libraries(testApp PRIVATE libmirror)
target_compile_features(testApp PUBLIC cxx_std_11)
set_target_properties(testApp PROPERTIES POSIION_INDEPENDENT_CODE ON)
set_target_properties(testApp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)