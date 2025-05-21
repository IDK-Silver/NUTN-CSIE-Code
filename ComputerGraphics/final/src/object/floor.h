//
// Created by 黃毓峰 on 2025/1/1.
//

#ifndef FLOOR_H
#define FLOOR_H
#include <learnopengl/model.h>


struct map_floor {
    int id;
    float x;
    float y;
    float z;
    float scale;
    glm::vec3 angle;
    Model* model;
    bool visible;
    glm::vec3 acceleration;
    glm::vec3 velocity;
};

#define map_floor_init(bowling) { \
    bowling.id = current_id;\
    current_id += 1;\
    bowling.x = 0;\
    bowling.y = 0;\
    bowling.z = 0;\
    bowling.model = new Model("../res/obj/map_floor/map_floor.obj");\
    bowling.scale = 1;\
    bowling.visible = true;\
    bowling.velocity = glm::vec3(0.0f, 0, 0); \
    bowling.acceleration = glm::vec3(0.0f, 0, 0); \
}



glm::mat4 map_floor_get_model_matrix(map_floor &bowling) {
    glm::mat4 model = glm::mat4(1.0f); \
    model = glm::translate(model, glm::vec3(bowling.x, bowling.y, bowling.z)); \
    model = glm::scale(model, glm::vec3(bowling.scale, bowling.scale, bowling.scale));\
    return model;
}

void drawMapFloor(map_floor &bowling, Shader &ourShader)
{
    if (!bowling.visible) return;
    ourShader.setMat4("model", map_floor_get_model_matrix(bowling));
    bowling.model->Draw(ourShader);
}




#endif //BOWLING_H
