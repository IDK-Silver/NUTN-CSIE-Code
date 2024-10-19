Project(APP)

add_library(SELF_MATH 
    self_math.h
    self_math.cpp
)


add_executable(APP Q3.cpp)

target_link_libraries(APP SELF_MATH)

