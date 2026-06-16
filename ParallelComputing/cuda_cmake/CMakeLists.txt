cmake_minimum_required(VERSION 3.18)

project(cuda_cmake LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(vector_add src/vector_add.cu)
target_link_libraries(vector_add PRIVATE CUDA::cudart)

add_executable(add_by_block src/add_by_block.cu)
target_link_libraries(add_by_block PRIVATE CUDA::cudart)

add_executable(add_by_thread src/add_by_thread.cu)
target_link_libraries(add_by_thread PRIVATE CUDA::cudart)

add_executable(add_block_size_benchmark src/add_block_size_benchmark.cu)
target_link_libraries(add_block_size_benchmark PRIVATE CUDA::cudart)

add_executable(add_arbitrary_size src/add_arbitrary_size.cu)
target_link_libraries(add_arbitrary_size PRIVATE CUDA::cudart)

add_executable(parallel_kernels src/parallel_kernels.cu)
target_link_libraries(parallel_kernels PRIVATE CUDA::cudart)

add_executable(stencil_serial src/stencil_serial.cu)
target_link_libraries(stencil_serial PRIVATE CUDA::cudart)

add_executable(stencil_global src/stencil_global.cu)
target_link_libraries(stencil_global PRIVATE CUDA::cudart)

add_executable(stencil_shared src/stencil_shared.cu)
target_link_libraries(stencil_shared PRIVATE CUDA::cudart)

add_executable(device_query src/device_query.cu)
target_link_libraries(device_query PRIVATE CUDA::cudart)

add_executable(hello_world_magic src/hello_world_magic.cu)
target_link_libraries(hello_world_magic PRIVATE CUDA::cudart)
