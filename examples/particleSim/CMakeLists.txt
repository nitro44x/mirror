cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

find_package(OpenMP REQUIRED)

set(cuda_cpp_srcs Particle.cpp
                  particleSim.cpp
)

set(headers Particle.hpp)

set(srcs main.cpp)

set(app_name particleSim)
set_property(SOURCE ${cuda_cpp_srcs} PROPERTY LANGUAGE CUDA)

message(STATUS ${CMAKE_CUDA_FLAGS})
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/openmp")
message(STATUS ${CMAKE_CUDA_FLAGS})

add_executable(${app_name} ${headers} ${cuda_cpp_srcs} ${srcs})
target_include_directories(${app_name} PRIVATE ${CMAKE_SOURCE_DIR}/external/catch2)
target_link_libraries(${app_name} PRIVATE libmirror)
target_link_libraries(${app_name} PRIVATE OpenMP::OpenMP_CXX)
target_compile_features(${app_name} PUBLIC cxx_std_11)
set_target_properties(${app_name} PROPERTIES POSIION_INDEPENDENT_CODE ON)
set_target_properties(${app_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${app_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>)