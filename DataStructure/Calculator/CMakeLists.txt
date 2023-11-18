cmake_minimum_required(VERSION 3.0)
project(Calculator C)

include_directories(lib)

set(CMAKE_C_STANDARD 11)

add_executable(Calculator src/main.c lib/list/list.h
        lib/algorithm/ExpressionConverter.h
        lib/system/system.h)

target_link_libraries(Calculator m)

target_compile_options(Calculator PRIVATE -w)