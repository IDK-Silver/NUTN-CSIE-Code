cmake_minimum_required(VERSION 3.5)

set(executable_name ElectricVehicleMotor)
project(${executable_name} LANGUAGES CXX)
include_directories(src)

add_library(core STATIC src/core_algorithm.cpp src/core_process_io.cpp src/core_structure.cpp src/core_algorithm.hpp)
set(
    source_code
    src/main.cpp
)

add_executable(${executable_name} ${source_code})

target_link_libraries(${executable_name} PRIVATE core)
