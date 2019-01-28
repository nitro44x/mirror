cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(srcs main.cpp)

set(cuda_cpp_srcs kernels2.cpp
                  Particle.cpp
                  allocator_tests.cpp
                  vector_tests.cpp
                  serialization_tests.cpp
)

set(cu_srcs kernels.cu
            particle_test.cu
)

set(headers kernels.hpp
            kernels2.hpp
            particle_test.hpp
            Particle.hpp
)

set(app_name mirrorTests)
set_property(SOURCE ${cuda_cpp_srcs} PROPERTY LANGUAGE CUDA)

add_executable(${app_name} ${headers} ${srcs} ${cuda_cpp_srcs} ${cu_srcs})
target_include_directories(${app_name} PRIVATE ${CMAKE_SOURCE_DIR}/external/catch2)
target_link_libraries(${app_name} PRIVATE libmirror)
target_compile_features(${app_name} PUBLIC cxx_std_11)
set_target_properties(${app_name} PROPERTIES POSIION_INDEPENDENT_CODE ON)
set_target_properties(${app_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)