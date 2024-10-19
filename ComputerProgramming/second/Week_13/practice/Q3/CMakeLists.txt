cmake_minimum_required(VERSION 3.0)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/thirdparty/SQLiteCpp)

Project(APP)
add_executable(APP Q3.cpp)

target_link_libraries(APP
  SQLiteCpp
  sqlite3
  pthread
  dl
  )





