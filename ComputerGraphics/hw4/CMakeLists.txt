# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.5)

# Set the name of the executable
set(exe_name hw4)

project(${exe_name} CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLM CONFIG REQUIRED)

include_directories(/opt/homebrew/Cellar/glm/1.0.1/include/glm)
# Set the source files
set(SOURCE_FILES
        src/main.cpp
)

add_executable(${exe_name} ${SOURCE_FILES})
target_link_libraries(${exe_name}  PRIVATE GLUT::GLUT OpenGL::GL)
