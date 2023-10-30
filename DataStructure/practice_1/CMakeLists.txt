cmake_minimum_required(VERSION 3.0)
project(practice_1 C)

include_directories(lib)

set(CMAKE_C_STANDARD 11)

add_executable(practice_1 main.c
        lib/list/list.h)
