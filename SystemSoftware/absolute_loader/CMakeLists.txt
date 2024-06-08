cmake_minimum_required(VERSION 3.0)

project(absolute_loader C)

set(CMAKE_C_STANDARD 11)

include_directories(lib)


add_executable(
    absolute_loader
    src/main.c

    lib/utility/file.c
    lib/utility/numeric.c

    lib/core/loader.h
    lib/core/loader.c
    lib/argument/argument_options.c
    lib/argument/argument_parser.c
)

target_link_libraries(absolute_loader m)