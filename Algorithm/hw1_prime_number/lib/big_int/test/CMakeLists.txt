cmake_minimum_required(VERSION 3.11)

project(big_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(src)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror")

# add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(../ big_int_lib)

enable_testing()

# add test executable
add_executable(test_big_int test.cpp)
target_link_libraries(test_big_int PRIVATE big_int gtest_main)

# add test
include(GoogleTest)
gtest_discover_tests(test_big_int)