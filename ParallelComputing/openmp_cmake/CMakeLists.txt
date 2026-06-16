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

add_executable(sum_reduction_atomic_compare src/sum_reduction_atomic_compare.c)
target_link_libraries(sum_reduction_atomic_compare PRIVATE OpenMP::OpenMP_C)

add_executable(collapse_nested_loops_compare src/collapse_nested_loops_compare.c)
target_link_libraries(collapse_nested_loops_compare PRIVATE OpenMP::OpenMP_C)

add_executable(task_linked_list_compare src/task_linked_list_compare.c)
target_link_libraries(task_linked_list_compare PRIVATE OpenMP::OpenMP_C)

add_executable(fibonacci_task_compare src/fibonacci_task_compare.c)
target_link_libraries(fibonacci_task_compare PRIVATE OpenMP::OpenMP_C)
