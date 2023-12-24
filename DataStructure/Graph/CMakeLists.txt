cmake_minimum_required(VERSION 3.26)
project(Graph C)

include_directories(lib)

set(CMAKE_C_STANDARD 11)

add_executable(Graph main.c
        lib/list/list.h
        lib/graph/graph.h
        lib/graph/graph_expansion.h)
