cmake_minimum_required(VERSION 3.0)

project(OJ)

add_executable(Solution main_2.cpp)
# add_executable(Solution main.cpp)

# add_executable(Run_SPJ src/blank_target.cpp)
# add_dependencies(Run_SPJ run_spj)

add_custom_target(
        Run_Solution
        DEPENDS Solution
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/in
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/in ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_SOURCE_DIR}/out
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/Solution < ${CMAKE_CURRENT_BINARY_DIR}/in > ${CMAKE_CURRENT_SOURCE_DIR}/out
        COMMENT "Copy 1.in & Run Solution -> 1.out"
)
