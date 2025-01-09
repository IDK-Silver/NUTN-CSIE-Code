//
// Created by 黃毓峰 on 2025/1/1.
//

#ifndef OBJECT_VECTOR_H
#define OBJECT_VECTOR_H
#include <glm/vec3.hpp>
#include <GL/gl.h>

struct object_vector_angle {
    float x;
    float y;
    float z;
};

#define object_vector_angle_set(_v_obj_, x, y, z) {\
    _v_obj_.x = x\
    _v_obj_.y = y\
    _v_obj_.z = z\
}



#endif //OBJECT_VECTOR_H
