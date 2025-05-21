#---------------------------------------------------------------
#  CMakeLists.txt — cross-platform (macOS + Windows/vcpkg)
#---------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)

project(bowling-is-all-you-need LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#---------------------------#
# Common (platform-agnostic)
#---------------------------#
find_package(OpenGL REQUIRED)          # macOS 走內建框架；Windows 由 vcpkg 提供
find_package(glfw3 CONFIG REQUIRED)    # vcpkg 與 Homebrew 都帶 CONFIG
                                       
#------------ macOS (Homebrew) --------------------------------#
if(APPLE)
    message(STATUS "Building on macOS — libraries expected via Homebrew")

    # 透過 brew --prefix 取得安裝路徑
    function(brew_prefix formula outVar)
        execute_process(
            COMMAND brew --prefix ${formula}
            OUTPUT_VARIABLE _tmp OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT _tmp)
            message(FATAL_ERROR
                "Could NOT find prefix for '${formula}' via 'brew --prefix'!\n"
                "(請先 brew install / brew link ${formula})")
        endif()
        set(${outVar} "${_tmp}" PARENT_SCOPE)
    endfunction()

    brew_prefix(glm    GLM_PREFIX)
    brew_prefix(glew   GLEW_PREFIX)
    brew_prefix(sfml   SFML_PREFIX)
    brew_prefix(assimp ASSIMP_PREFIX)

    message(STATUS "GLM_PREFIX   = ${GLM_PREFIX}")
    message(STATUS "GLEW_PREFIX  = ${GLEW_PREFIX}")
    message(STATUS "SFML_PREFIX  = ${SFML_PREFIX}")
    message(STATUS "ASSIMP_PREFIX= ${ASSIMP_PREFIX}")

    include_directories(
        ${GLM_PREFIX}/include
        ${GLEW_PREFIX}/include
        ${SFML_PREFIX}/include
    )

    # 需手動列出 dylib
    set(OTHER_LIBS
        ${ASSIMP_PREFIX}/lib/libassimp.dylib
        ${GLEW_PREFIX}/lib/libGLEW.dylib
        ${SFML_PREFIX}/lib/libsfml-audio.dylib
        ${SFML_PREFIX}/lib/libsfml-graphics.dylib
        ${SFML_PREFIX}/lib/libsfml-system.dylib
        ${SFML_PREFIX}/lib/libsfml-window.dylib
        ${SFML_PREFIX}/lib/libsfml-network.dylib
    )
endif()

#------------ Windows (vcpkg) ---------------------------------#
if(WIN32)
    message(STATUS "Building on Windows — expecting vcpkg toolchain integration")

    # 這些 find_package 會由 vcpkg 的 *CONFIG* 包解析出 imported targets
    find_package(GLEW      REQUIRED)   # GLEW::GLEW
    find_package(glm      CONFIG REQUIRED)   # glm::glm
    find_package(SFML CONFIG REQUIRED COMPONENTS
        Graphics
        Audio
        Window
        Network
        System
    )
    find_package(assimp   CONFIG REQUIRED)   # assimp::assimp
    find_package(Freetype  REQUIRED)   # freetype::freetype
    find_package(glad CONFIG REQUIRED)
    
endif()

#---------------------------#
# Local (header-only/third-party in repo)
#---------------------------#
if(APPLE)
    # macOS 下保留原路徑
    include_directories(
        libs/
        libs/stb
        libs/glad/include
    )
elseif(WIN32)
    # Windows 下切換到 libs/win/glad/include
    include_directories(
        libs/
        libs/stb
    )
else()
    # 其他平台（可選）
    include_directories(
        libs/
        libs/stb
        libs/glad/include
    )
endif()
#---------------------------#
#   Sources
#---------------------------#
# 先把通用的來源檔案列出來（不含 glad）
set(SOURCE_FILES
    src/main.cpp
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

# macOS (Apple) 才加入 glad 的 header 和 source
if(APPLE)
    list(APPEND SOURCE_FILES
        libs/glad/include/glad/glad.h
        libs/glad/src/glad.c
    )
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

#---------------------------#
#   Link libraries
#---------------------------#
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        glfw
        OpenGL::GL
        ${OTHER_LIBS}
    )
elseif(WIN32)


    target_link_libraries(${PROJECT_NAME} PRIVATE
        glfw
        OpenGL::GL
        GLEW::GLEW
        glm::glm
        # SFML modules must be linked via their SFML:: targets
        SFML::Graphics
        SFML::Audio
        SFML::Window
        SFML::Network
        SFML::System
        assimp::assimp
        Freetype::Freetype
         glad::glad
    )

endif()
