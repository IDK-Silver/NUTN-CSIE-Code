cmake_minimum_required(VERSION 3.16)

project(openmp_cmake C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

add_executable(parallel_for_cpu_demo src/parallel_for_cpu_demo.c)
target_link_libraries(parallel_for_cpu_demo PRIVATE OpenMP::OpenMP_C)

add_executable(linear_search_compare src/linear_search_compare.c)
target_link_libraries(linear_search_compare PRIVATE OpenMP::OpenMP_C)

add_executable(prime_schedule_benchmark src/prime_schedule_benchmark.c)
target_link_libraries(prime_schedule_benchmark PRIVATE OpenMP::OpenMP_C)

add_executable(matrix_multiply_compare src/matrix_multiply_compare.c)
target_link_libraries(matrix_multiply_compare PRIVATE OpenMP::OpenMP_C)
