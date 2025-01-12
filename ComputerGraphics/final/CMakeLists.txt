# Specify the minimum required version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project name and set the C++ standard to C++17
project(bowling-is-all-you-need)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the required packages: OpenGL and GLFW (version 3.3)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# MacOS-specific configuration
If (APPLE)
    # Display messages indicating the platform and required setup
    message("The platform is MacOS")
    message("Ensure your are use brew to install library")

    # Define a function to retrieve the prefix path of a Homebrew-installed formula
    function(brew_prefix formula outVar)
        execute_process(
                COMMAND brew --prefix ${formula}  # Get the prefix path
                OUTPUT_VARIABLE _tmp
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        # If the formula is not found, display an error message
        if (NOT _tmp)
            message(FATAL_ERROR "Could NOT find prefix for '${formula}' via 'brew --prefix'!
        (請先 brew link ${formula})")
        endif()
        # Set the prefix path to the specified output variable
        set(${outVar} "${_tmp}" PARENT_SCOPE)
    endfunction()

    # Retrieve and store prefix paths for required libraries
    brew_prefix(glm   GLM_PREFIX)
    brew_prefix(glew  GLEW_PREFIX)
    brew_prefix(sfml  SFML_PREFIX)
    brew_prefix(assimp ASSIMP_PREFIX)

    # Display the retrieved prefix paths
    message(STATUS "GLM_PREFIX  = ${GLM_PREFIX}")
    message(STATUS "GLEW_PREFIX = ${GLEW_PREFIX}")
    message(STATUS "SFML_PREFIX = ${SFML_PREFIX}")
    message(STATUS "ASSIMP_PREFIX = ${ASSIMP_PREFIX}")

    # Include directories for MacOS-specific dependencies
    include_directories(
            ${GLM_PREFIX}/include
            ${GLEW_PREFIX}/include
            ${SFML_PREFIX}/include
    )

    # Set the list of additional library dependencies
    set(
            OTHER_LIBS
            ${ASSIMP_PREFIX}/lib/libassimp.dylib
            ${FREETYPE_LIBRARIES}
            ${GLEW_PREFIX}/lib/libGLEW.dylib
            ${SFML_PREFIX}/lib/libsfml-audio.dylib
            ${SFML_PREFIX}/lib/libsfml-graphics.dylib
            ${SFML_PREFIX}/lib/libsfml-system.dylib
            ${SFML_PREFIX}/lib/libsfml-window.dylib
            ${SFML_PREFIX}/lib/libsfml-network.dylib
    )
endif ()

# Include additional directories for headers and libraries
include_directories(
        libs/
        libs/stb
        libs/glad/include
)

# Define the source files for the project
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

# Create the executable using the defined source files
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link the executable with the required libraries
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        glfw
        OpenGL::GL
        ${OTHER_LIBS}
)