cmake_minimum_required(VERSION 3.0)

project(BinarySearchTree C)
set(CMAKE_C_STANDARD 11)
include_directories(lib)

add_executable(BinarySearchTree src/main.c lib/bst/bst.h lib/bst/bst_expansion.h)
