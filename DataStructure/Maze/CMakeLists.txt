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

set(ProjectSourceCode

        src/main.cpp
        ${QT_Widget}
        lib/Maze/Maze.cpp
        lib/Maze/Maze.h
        lib/Maze/Qt/MazeQtGraphicsView.cpp
        lib/Maze/Qt/MazeQtGraphicsView.h
        lib/Maze/Qt/MazeQtGraphicsPixmapItem.cpp
        lib/Maze/Qt/MazeQtGraphicsPixmapItem.h
        lib/Maze/MazeAlgorithm.cpp
        lib/Maze/MazeAlgorithm.h
)

add_executable(Maze ${ProjectSourceCode})

target_link_libraries(Maze PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)