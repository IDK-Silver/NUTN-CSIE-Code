cmake_minimum_required(VERSION 3.11)

project(prime_algorithms_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 big_int 子目錄
add_subdirectory(lib/big_int)

# 創建 prime_algorithm 庫
add_library(prime_algorithm STATIC lib/prime/src/prime.cpp)

# 為 prime_algorithm 設置包含目錄
target_include_directories(prime_algorithm PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/prime/src
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/big_int/src
)

# 將 big_int 庫鏈接到 prime_algorithm
target_link_libraries(prime_algorithm PUBLIC big_int)


set(
    CoreSourceCode
    src/core/prime_runner.hpp
    src/core/config.hpp
    src/core/config.cpp
)


add_executable(fermat_vs_miller  src/fermat_vs_miller.cpp ${CoreSourceCode})
target_link_libraries(fermat_vs_miller PRIVATE prime_algorithm)
target_include_directories(fermat_vs_miller PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/prime/src
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/big_int/src
)


add_executable(all_algorithm_cost_time  src/all_algorithm_cost_time.cpp ${CoreSourceCode})
target_link_libraries(all_algorithm_cost_time PRIVATE prime_algorithm)
target_include_directories(all_algorithm_cost_time PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/prime/src
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/big_int/src
)

add_executable(find_mersenne_prime_number  src/find_mersenne_prime_number.cpp ${CoreSourceCode})
target_link_libraries(find_mersenne_prime_number PRIVATE prime_algorithm)
target_include_directories(find_mersenne_prime_number PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/prime/src
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/big_int/src
)