#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

#include <iostream>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SFML/Audio/Music.hpp>

#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include "object/axis/axis.h"
#include "object/bowling.h"
#include "object/floor.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 960;
const unsigned int SCR_HEIGHT = 800;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// object id
int current_id = 0;

// random objecy
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0, 1.0);

// the bowling's max move range
constexpr float MAX_MAP_SIZE = 10;

int main()
{
    glfwSetErrorCallback([](int error, const char* desc){
    std::cerr << "GLFW Error (" << error << "): " << desc << std::endl;
    });

    // load sound that when build-tree(刻意的中式英文) is fly to sky
    sf::Music music;
    if (!music.openFromFile("../res/sound/tree_fly_sound.mp3"))
        return -1;

    if (!glfwInit()) {
    std::cerr << "glfwInit failed\n";
    return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    // floor f;
    std::vector<bowling> all_bowling(30);

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Bowling is all you need", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    glEnable(GL_DEPTH_TEST);

    Shader ourShader("./res/shader/model_loading_vertex_shader.glsl", "./res/shader/model_loading_fragment_shader.glsl");

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(glm::vec3(0, 0, 0)));


    object::initAxis();

    {
        std::uniform_real_distribution<> dis(-static_cast<double>(MAX_MAP_SIZE), static_cast<double>(MAX_MAP_SIZE));
        for (auto& _bowling : all_bowling) {
            bowling_init(_bowling);
            _bowling.x = dis(gen);
            _bowling.z = dis(gen);
            _bowling.scale = 0.125;
        }
    }




    map_floor floor;
    map_floor_init(floor);
    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // fixed camera y poistion
        camera.Position.y = 0.25f;

        // per-frame time logic
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // random move
        static float time_to_next_move = 0.0f;
        time_to_next_move -= deltaTime;
        if (time_to_next_move <= 0.0f)
        {
            constexpr float move_target_secound = 1.0f;
            for (auto& _bowling : all_bowling)
            {
                if (_bowling.y > 0.1f) continue;
                float randX = dis(gen);
                float randZ = dis(gen);
                float speed = 2.0f;
                glm::vec3 dir = glm::normalize(glm::vec3(randX, 0, randZ));
                _bowling.velocity = dir * speed;
            }
            time_to_next_move = move_target_secound;
        }

        // process input
        processInput(window);

        // render
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // enable shader before setting uniforms
        ourShader.use();

        // view/projection transformations
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);


        // proces bowling hit physics
        for (auto& _bowling : all_bowling) {

            // check is fly
            if (is_bowling_fly(_bowling))
                _bowling.angle = _bowling.angle + glm::vec3(10, 10, 10);
            else
                _bowling.angle = glm::vec3(0, 0, 0);

            // ensure in map that allow to move
            if (_bowling.x < -MAX_MAP_SIZE) { _bowling.x = -MAX_MAP_SIZE; _bowling.velocity.x *= -1; }
            if (_bowling.x >  MAX_MAP_SIZE) { _bowling.x =  MAX_MAP_SIZE; _bowling.velocity.x *= -1; }
            if (_bowling.z < -MAX_MAP_SIZE) { _bowling.z = -MAX_MAP_SIZE; _bowling.velocity.z *= -1; }
            if (_bowling.z >  MAX_MAP_SIZE) { _bowling.z =  MAX_MAP_SIZE; _bowling.velocity.z *= -1; }


            // get user hit box
            auto user_box = std::pair<glm::vec3 , glm::vec3>(
                camera.Position - glm::vec3(0.5f, 0.5f, 0.5f),
                camera.Position + glm::vec3(0.5f, 0.5f, 0.5f)
            );

            // check hit user
            if (
                isColliding(getWorldPos(_bowling), user_box)
                && !is_bowling_fly(_bowling)
            )
            {
                music.play();

                // set applied force
                glm::vec3 push_dir = glm::normalize(glm::vec3(_bowling.x, _bowling.y, _bowling.z) - camera.Position);
                push_dir += glm::vec3(0, 0.7, 0);
                float push_strength = 20.0f;
                _bowling.velocity += push_dir * push_strength;
            }
        }

        // display and update physics
        for (auto& _bowling : all_bowling)
        {
            updateBowlingPhysics(_bowling, deltaTime);
            drawBowling(_bowling, ourShader);
        }

        // set the map floor
        floor.y = -1.0f;
        floor.scale = 1;
        drawMapFloor(floor, ourShader);



        // object::drawAxis(view, projection);

        // glfw swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}