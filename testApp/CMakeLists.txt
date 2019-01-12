cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(srcs main.cu)

add_executable(containerApp ${srcs} ${headers})