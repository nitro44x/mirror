cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(include_dir ${CMAKE_SOURCE_DIR}/include)

set(headers ${include_dir}/mirror/simt_vector.hpp
            ${include_dir}/mirror/simt_allocator.hpp
            ${include_dir}/mirror/simt_macros.hpp
            ${include_dir}/mirror/simt_serialization.hpp
            ${include_dir}/mirror/simt_utilities.hpp
)

set(cuda_srcs simt_utilities.cpp)

set(cpp_force_srcs force.cpp)

set(cuda_force_srcs force.cu)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)
add_library(libmirror STATIC ${headers} ${cuda_srcs} ${cpp_force_srcs})
set_target_properties(libmirror PROPERTIES DEBUG_POSTFIX d)
target_include_directories(libmirror PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                                             $<INSTALL_INTERFACE:include>)
target_compile_options(libmirror PUBLIC $<IF:$<CONFIG:Debug>,-D_ITERATOR_DEBUG_LEVEL=2 -D_DEBUG, >)
set_target_properties(libmirror PROPERTIES PUBLIC_HEADER "${headers}") # must use "" around the headers, otherwise only the first file is copied
target_compile_features(libmirror PUBLIC cxx_std_11)
set_target_properties(libmirror PROPERTIES POSIION_INDEPENDENT_CODE ON)
set_target_properties(libmirror PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(libmirror PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(TARGETS libmirror
        EXPORT MirrorConfig
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
        PUBLIC_HEADER DESTINATION include/mirror
)

export(TARGETS libmirror
       NAMESPACE Mirror::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/MirrorConfig.cmake")

install(EXPORT
        MirrorConfig
        DESTINATION cmake
        NAMESPACE Mirror::
)