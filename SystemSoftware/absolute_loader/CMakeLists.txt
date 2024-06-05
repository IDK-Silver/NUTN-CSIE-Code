cmake_minimum_required(VERSION 3.0)

project(absolute_loader C)

set(CMAKE_C_STANDARD 11)

include_directories(lib)


add_executable(
        absolute_loader
        src/main.c
#        lib/argument/argument_parser.h
#        lib/argument/argument_options.h
        lib/argument/argument_options.c
        lib/argument/argument_parser.c
#        lib/utility/file.h
        lib/utility/file.c
#        lib/utility/numeric.h
        lib/utility/numeric.c
        lib/core/loader.h
        lib/core/loader.c
        src/absolute_loader_config.h
        #        lib/core/loader_core.h
)
target_link_libraries(absolute_loader m)