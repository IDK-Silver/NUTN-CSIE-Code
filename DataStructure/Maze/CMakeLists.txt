cmake_minimum_required(VERSION 3.26)

project(Maze)

# include project lib
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Set C++ Std 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt Lib
find_package(QT NAMES Qt6 COMPONENTS Widgets REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets REQUIRED)

set(QT_Widget
        src/widget/MainWindow/MainWindow.ui
        src/widget/MainWindow/MainWindow.cpp
        src/widget/MainWindow/MainWindow.h

)

set(LibSourceCode
        lib/Maze/Maze.cpp
        lib/Maze/Maze.h
        lib/Maze/Qt/MazeQtGraphicsView.cpp
        lib/Maze/Qt/MazeQtGraphicsView.h
        lib/Maze/Qt/MazeQtGraphicsPixmapItem.cpp
        lib/Maze/Qt/MazeQtGraphicsPixmapItem.h
        lib/Maze/MazeAlgorithm.cpp
        lib/Maze/MazeAlgorithm.h
)

set(ProjectSourceCode
        src/main.cpp
        ${QT_Widget}
        ${LibSourceCode}
)


IF (WIN32)
    # non-console for Windows
    add_executable(Maze WIN32 ${ProjectSourceCode})
ELSE()
    add_executable(Maze ${ProjectSourceCode})
ENDIF()


# link library
target_link_libraries(Maze PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)


# set build output folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# copy file command
add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/res
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/res
        COMMENT "Copying resources to build directory"
)

# copy res file
add_custom_target(CopyResources ALL DEPENDS ${PROJECT_NAME})