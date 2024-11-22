# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.12)

# Set the name of the executable
set(exe_name hw2)

project(${exe_name})

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenGL and GLFW libraries
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)


# Set the source files
set(
    SOURCE_FILES
    src/main.cpp
)

# Add the executable
add_executable( ${exe_name} ${SOURCE_FILES})

# Link the libraries
target_link_libraries(${exe_name} OpenGL::GL glfw GLU)
