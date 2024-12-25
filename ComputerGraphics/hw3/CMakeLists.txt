# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.5)

# Set the name of the executable
set(exe_name hw3)

project(${exe_name})

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/libs CACHE PATH "FetchContent base directory" FORCE)

# Find OpenGL package (without GLU)
find_package(OpenGL REQUIRED)

# Find GLFW
find_package(GLUT REQUIRED)

# Find GLM
find_package(glm REQUIRED)

# Find Freetype
find_package(Freetype REQUIRED)

# Fetch Assimp library
FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
        GIT_TAG master
)
# Configure Assimp options
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)

# Fetch GLAD library
FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
)
# Set GLAD options
set(GLAD_PROFILE "core" CACHE STRING "OpenGL profile")
set(GLAD_API "gl=" CACHE STRING "API type/version pairs, like \"gl=3.2,gles=\", no version means latest")
set(GLAD_GENERATOR "c" CACHE STRING "Language to generate the binding for")

FetchContent_GetProperties(glad)
if(NOT glad_POPULATED)
    FetchContent_Populate(glad)
    add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR})
endif()

# Set the source files
set(SOURCE_FILES
        src/main.cpp
        src/widget/menu/main_menu.cpp
        # List other source files here
)

# Add the executable
add_executable(${exe_name} ${SOURCE_FILES})

# Link libraries
target_link_libraries(${exe_name} PRIVATE
        glad
        OpenGL::GL
        glut
        GLU
        assimp
        glm::glm
        Freetype::Freetype
)
