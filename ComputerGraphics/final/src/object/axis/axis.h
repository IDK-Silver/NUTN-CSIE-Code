//
// Created by 黃毓峰 on 2025/1/1.
//

#ifndef AXIS_H
#define AXIS_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace object {

    // 軸線長度
    static float axisLength = 1000.0f;

    // 頂點資料 (位置 + 顏色)
    static float axesVertices[] = {
        // X 軸 (紅)
         0.0f,       0.0f,       0.0f,  1.0f, 0.0f, 0.0f,   // 原點 (R)
         axisLength, 0.0f,       0.0f,  1.0f, 0.0f, 0.0f,   // X 軸端點 (R)

        // Y 軸 (綠)
         0.0f,       0.0f,       0.0f,  0.0f, 1.0f, 0.0f,   // 原點 (G)
         0.0f,       axisLength, 0.0f,  0.0f, 1.0f, 0.0f,   // Y 軸端點 (G)

        // Z 軸 (藍)
         0.0f,       0.0f,       0.0f,  0.0f, 0.0f, 1.0f,   // 原點 (B)
         0.0f,       0.0f,       axisLength, 0.0f, 0.0f, 1.0f  // Z 軸端點 (B)
    };

    // 預先宣告 VAO, VBO 與 shader program
    static unsigned int axesVAO = 0;
    static unsigned int axesVBO = 0;
    static unsigned int axesShaderProgram = 0;

    // 讀取外部檔案中的 Shader 原始碼
    static std::string readShaderCodeFromFile(const char* shaderPath)
    {
        std::ifstream shaderFile(shaderPath);
        if (!shaderFile.is_open())
        {
            std::cerr << "Failed to open shader file: " << shaderPath << std::endl;
            return std::string{};
        }
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        return shaderStream.str();
    }

    // 建立並回傳 Shader Program
    static unsigned int createShaderProgram(const char* vertexPath, const char* fragmentPath)
    {
        // 讀取並編譯 Vertex Shader
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        std::string vertexCode = readShaderCodeFromFile(vertexPath);
        const char* vShaderCode = vertexCode.c_str();
        glShaderSource(vertexShader, 1, &vShaderCode, NULL);
        glCompileShader(vertexShader);

        // 可以加上錯誤檢查
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if(!success)
        {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // 讀取並編譯 Fragment Shader
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        std::string fragmentCode = readShaderCodeFromFile(fragmentPath);
        const char* fShaderCode = fragmentCode.c_str();
        glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
        glCompileShader(fragmentShader);

        // 可以加上錯誤檢查
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if(!success)
        {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // Link shaders
        unsigned int shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // 可以加上錯誤檢查
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if(!success)
        {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }

        // 刪除用不到的 shader 物件
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return shaderProgram;
    }

    // 初始化：配置 VAO, VBO 與著色器
    static void initAxis()
    {
        // 產生 VAO / VBO
        glGenVertexArrays(1, &axesVAO);
        glGenBuffers(1, &axesVBO);

        glBindVertexArray(axesVAO);
        glBindBuffer(GL_ARRAY_BUFFER, axesVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(axesVertices), axesVertices, GL_STATIC_DRAW);

        // Position 屬性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // Color 屬性
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // 建立著色器程式
        axesShaderProgram = createShaderProgram("./res/shader/axes_vertex_shader.glsl", "./res/shader/axes_fragment_shader.glsl");
    }

    // 繪製坐標軸
    static void drawAxis(const glm::mat4 &view, const glm::mat4 &projection)
    {
        // 如果還沒初始化，就先初始化
        if (axesVAO == 0 || axesVBO == 0 || axesShaderProgram == 0)
        {
            initAxis();
        }

        // 啟用 shader
        glUseProgram(axesShaderProgram);

        // 模型矩陣：畫坐標軸時通常就是單位矩陣
        glm::mat4 axesModel = glm::mat4(1.0f);

        // 傳入 uniform
        int modelLoc = glGetUniformLocation(axesShaderProgram, "model");
        int viewLoc  = glGetUniformLocation(axesShaderProgram, "view");
        int projLoc  = glGetUniformLocation(axesShaderProgram, "projection");

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(axesModel));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // 繪製
        glBindVertexArray(axesVAO);
        glDrawArrays(GL_LINES, 0, 6); // 3 條線 (X, Y, Z)，每條線 2 個頂點，共 6 個頂點
        glBindVertexArray(0);
    }

} // namespace object

#endif // AXIS_H