cmake_minimum_required(VERSION 3.11)

project(big_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(../ big_int_lib)


# add example executable
add_executable(example_app main.cpp)
target_link_libraries(example_app PRIVATE big_int)
