# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.5)

# Set the name of the executable
set(exe_name bowl-is-all-you-need)

project(${exe_name})

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#find_package(GLUT 3.3 REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLM  REQUIRED)
find_package(assimp REQUIRED)
find_package(Freetype REQUIRED)

include_directories(/opt/homebrew/Cellar/glm/1.0.1/include/glm)
include_directories(/opt/homebrew/Cellar/assimp/5.4.3/include)
include_directories(libs/)
include_directories(libs/stb)
include_directories(libs/glad/include)
include_directories(/opt/homebrew/Cellar/freetype/2.13.3/include/freetype2/)
include_directories(/opt/homebrew/Cellar/glew/2.2.0_1/include/)
include_directories(/opt/homebrew/Cellar/sfml/3.0.0/include/)

set(SOURCE_FILES
        src/main.cpp
        libs/glad/include/glad/glad.h
        libs/glad/src/glad.c
        libs/learnopengl/shader.h
        libs/learnopengl/shader_m.h
        libs/learnopengl/camera.h
        libs/learnopengl/model.h
        libs/learnopengl/mesh.h
        libs/stb/stb_image.h
        src/object/axis/axis.h
        src/object/bowling.h
        src/object/vector.h
        src/object/floor.h
)

add_executable(${exe_name} ${SOURCE_FILES})

set(
        SFML_LIBRARIES
        /opt/homebrew/Cellar/sfml/3.0.0/lib/libsfml-audio.dylib
        /opt/homebrew/Cellar/sfml/3.0.0/lib/libsfml-graphics.dylib
        /opt/homebrew/Cellar/sfml/3.0.0/lib/libsfml-system.dylib
        /opt/homebrew/Cellar/sfml/3.0.0/lib/libsfml-window.dylib
        /opt/homebrew/Cellar/sfml/3.0.0/lib/libsfml-network.dylib
)

target_link_libraries(
        ${exe_name} PRIVATE
        glfw OpenGL::GL
        /opt/homebrew/Cellar/assimp/5.4.3/lib/libassimp.dylib
        ${FREETYPE_LIBRARIES}
        /opt/homebrew/Cellar/glew/2.2.0_1/lib/libGLEW.2.2.0.dylib
        ${SFML_LIBRARIES}
)
