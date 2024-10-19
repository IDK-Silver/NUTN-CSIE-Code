Project(App)
set (CMAKE_CXX_STANDARD 11)
include_directories(lib)

add_library(DigitalTime SHARED 
    lib/DigitalTime.hpp
    lib/DigitalTime.cpp
)

add_executable(App Q1.cpp)

target_link_libraries(App DigitalTime)



