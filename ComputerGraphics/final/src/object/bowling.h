//
// Created by 黃毓峰 on 2025/1/1.
//

#ifndef BOWLING_H
#define BOWLING_H

struct bowling {
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

#define bowling_init(bowling) { \
    bowling.id = current_id;\
    current_id += 1;\
    bowling.x = 0;\
    bowling.y = 0;\
    bowling.z = 0;\
    bowling.model = new Model("../res/obj/bowling/tree/tree.obj");\
    bowling.scale = 1;\
    bowling.visible = true;\
    bowling.velocity = glm::vec3(0.0f, 0, 0); \
    bowling.acceleration = glm::vec3(0.0f, 0, 0); \
}



glm::mat4 bowling_get_model_matrix(bowling &bowling) {
    glm::mat4 model = glm::mat4(1.0f); \
    model = glm::translate(model, glm::vec3(bowling.x, bowling.y, bowling.z)); \
    model = glm::rotate(model, glm::radians(bowling.angle[0]), glm::vec3(1.0f, 0.0f, 0.0f)); \
    model = glm::rotate(model, glm::radians(bowling.angle[1]), glm::vec3(0.0f, 1.0f, 0.0f)); \
    model = glm::rotate(model, glm::radians(bowling.angle[2]), glm::vec3(0.0f, 0.0f, 1.0f)); \
    model = glm::scale(model, glm::vec3(bowling.scale, bowling.scale, bowling.scale));\
    return model;
}

void drawBowling(bowling &bowling, Shader &ourShader)
{
    if (!bowling.visible) return;
    ourShader.setMat4("model", bowling_get_model_matrix(bowling));
    bowling.model->Draw(ourShader);
}

bool is_bowling_fly(bowling &bowling) {
    return bowling.visible && bowling.y > 0.2;
}


std::pair<glm::vec3, glm::vec3> getWorldPos(bowling &bowling) {

    auto minAABB = bowling.model->minAABB;
    auto maxAABB = bowling.model->maxAABB;
    // 取得在本地座標下 AABB 的 8 個 corner
    std::vector<glm::vec3> corners {
                { minAABB.x, minAABB.y, minAABB.z },
                { minAABB.x, minAABB.y, maxAABB.z },
                { minAABB.x, maxAABB.y, minAABB.z },
                { minAABB.x, maxAABB.y, maxAABB.z },
                { maxAABB.x, minAABB.y, minAABB.z },
                { maxAABB.x, minAABB.y, maxAABB.z },
                { maxAABB.x, maxAABB.y, minAABB.z },
                { maxAABB.x, maxAABB.y, maxAABB.z },
            };

    glm::vec3 worldMin( std::numeric_limits<float>::max() );
    glm::vec3 worldMax( -std::numeric_limits<float>::max() );


    for (auto &corner : corners)
    {
        glm::vec4 worldPos = bowling_get_model_matrix(bowling) * glm::vec4(corner, 1.0);
        worldMin.x = std::min(worldMin.x, worldPos.x);
        worldMin.y = std::min(worldMin.y, worldPos.y);
        worldMin.z = std::min(worldMin.z, worldPos.z);

        worldMax.x = std::max(worldMax.x, worldPos.x);
        worldMax.y = std::max(worldMax.y, worldPos.y);
        worldMax.z = std::max(worldMax.z, worldPos.z);
    }

    // cout << worldMax.x << " " << worldMax.y << " " << worldMax.z << "\t";
    // cout << worldMin.x << " " << worldMin.y << " " << worldMin.z << endl;

    return std::make_pair(worldMin, worldMax);
}


bool isColliding(std::pair<glm::vec3, glm::vec3>b1, std::pair<glm::vec3, glm::vec3> b2) {
    auto box1Min = b1.first;
    auto box1Max = b1.second;
    auto box2Min = b2.first;
    auto box2Max = b2.second;
    return (box1Max.x >= box2Min.x) && (box1Min.x <= box2Max.x) &&
    (box1Max.y >= box2Min.y) && (box1Min.y <= box2Max.y) &&
    (box1Max.z >= box2Min.z) && (box1Min.z <= box2Max.z);
}

void updateBowlingPhysics(bowling &bowling, float deltaTime)
{
    if (!bowling.visible)
        return;

    glm::vec3 gravity(0.0f, -9.81f, 0.0f);

    if (bowling.y > 0)
        bowling.velocity += (bowling.acceleration + gravity) * deltaTime;

    bowling.x += bowling.velocity.x * deltaTime;
    bowling.y += bowling.velocity.y * deltaTime;
    bowling.z += bowling.velocity.z * deltaTime;

    float floorY = 0.0f;
    if (bowling.y < floorY)
    {
        bowling.y = floorY;

        float bounceFactor = 0.4f;
        if (bowling.velocity.y < 0.0f)
        {
            bowling.velocity.y = -bowling.velocity.y * bounceFactor;
        }
    }

    float damping = 0.98f;
    bowling.velocity.x *= damping;
    bowling.velocity.z *= damping;
}


#endif //BOWLING_H
