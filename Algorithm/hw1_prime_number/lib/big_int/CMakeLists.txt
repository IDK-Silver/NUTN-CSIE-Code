cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(big_int SHARED src/big_int.cpp)

target_include_directories(big_int PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

# 安裝目標
install(TARGETS big_int
    EXPORT big_int_Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安裝頭文件
install(DIRECTORY src/ DESTINATION include)

# 配置導出
install(EXPORT big_int_Targets
    FILE big_int_Config.cmake
    NAMESPACE BigInt::
    DESTINATION lib/cmake/big_int
)
