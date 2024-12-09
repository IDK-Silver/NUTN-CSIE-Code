cmake_minimum_required(VERSION 3.5)

set(executable_name TreeSearch)
project(${executable_name} LANGUAGES CXX)
include_directories(src/lib)

# add_library(core STATIC src/trees/bst.hp)
set(
    source_code
    src/main.cpp
    src/lib/token/token.cpp
)

add_executable(${executable_name} ${source_code})

# target_link_libraries(${executable_name} PRIVATE core)
