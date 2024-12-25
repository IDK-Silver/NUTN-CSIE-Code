// main.cpp

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// 定義結構來存儲頂點信息
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
};

// 定義結構來代表一個材質(目前僅支持基本顏色)
struct Material {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
};

// 網格結構
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Material material;

    unsigned int VAO, VBO, EBO;

    // 初始化網格的OpenGL數據
    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex),
                     vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                     indices.data(), GL_STATIC_DRAW);

        // 位置屬性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              (void*)offsetof(Vertex, Position));
        glEnableVertexAttribArray(0);
        // 法線屬性
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                              (void*)offsetof(Vertex, Normal));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    // 繪製網格
    void Draw(unsigned int shaderProgram) {
        // 設置材質屬性 (如果需要在著色器中使用)
        // glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, &material.diffuse[0]);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()),
                       GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

// 函數聲明
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
unsigned int loadShaders(const char* vertexPath, const char* fragmentPath);

int main()
{
    // 初始化 GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // 設置 GLFW 上下文版本和配置
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // 針對 macOS
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // 創建窗口
    GLFWwindow* window = glfwCreateWindow(800, 600, "3D Object Viewer with ASSIMP", NULL, NULL);
    if (window == NULL)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 設置回調
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    // 啟用深度測試
    glEnable(GL_DEPTH_TEST);

    // 加載並編譯著色器
    unsigned int shaderProgram = loadShaders("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
    if (!shaderProgram)
    {
        std::cerr << "Failed to load shaders\n";
        return -1;
    }

    // 使用 ASSIMP 載入 OBJ 文件
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile("Pin.obj",
        aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_JoinIdenticalVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cerr << "ASSIMP Error: " << importer.GetErrorString() << "\n";
        return -1;
    }

    // 我們僅處理第一個網格
    std::vector<Mesh> meshes;
    for (unsigned int m = 0; m < scene->mNumMeshes; m++)
    {
        aiMesh* ai_mesh = scene->mMeshes[m];
        Mesh mesh;

        // 處理頂點
        for (unsigned int v = 0; v < ai_mesh->mNumVertices; v++)
        {
            Vertex vertex;
            vertex.Position = glm::vec3(
                ai_mesh->mVertices[v].x,
                ai_mesh->mVertices[v].y,
                ai_mesh->mVertices[v].z
            );

            if (ai_mesh->HasNormals())
            {
                vertex.Normal = glm::vec3(
                    ai_mesh->mNormals[v].x,
                    ai_mesh->mNormals[v].y,
                    ai_mesh->mNormals[v].z
                );
            }
            else
            {
                vertex.Normal = glm::vec3(0.0f, 0.0f, 0.0f);
            }

            mesh.vertices.push_back(vertex);
        }

        // 處理索引
        for (unsigned int f = 0; f < ai_mesh->mNumFaces; f++)
        {
            aiFace face = ai_mesh->mFaces[f];
            for (unsigned int i = 0; i < face.mNumIndices; i++)
                mesh.indices.push_back(face.mIndices[i]);
        }

        // 處理材質 (如果有的話)
        if (ai_mesh->mMaterialIndex >= 0)
        {
            aiMaterial* material = scene->mMaterials[ai_mesh->mMaterialIndex];
            aiColor3D color(0.0f, 0.0f, 0.0f);

            // 獲取漫反射顏色
            material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
            mesh.material.diffuse = glm::vec3(color.r, color.g, color.b);

            // 獲取鏡面反射顏色
            material->Get(AI_MATKEY_COLOR_SPECULAR, color);
            mesh.material.specular = glm::vec3(color.r, color.g, color.b);

            // 獲取環境光顏色
            material->Get(AI_MATKEY_COLOR_AMBIENT, color);
            mesh.material.ambient = glm::vec3(color.r, color.g, color.b);

            // 獲取材質的 shininess
            float shininess;
            if (material->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS)
                mesh.material.shininess = shininess;
            else
                mesh.material.shininess = 32.0f; // 默認值
        }
        else
        {
            // 默認材質
            mesh.material.ambient = glm::vec3(1.0f, 0.5f, 0.31f);
            mesh.material.diffuse = glm::vec3(1.0f, 0.5f, 0.31f);
            mesh.material.specular = glm::vec3(0.5f, 0.5f, 0.5f);
            mesh.material.shininess = 32.0f;
        }

        // 設置網格的 OpenGL 數據
        mesh.setupMesh();

        meshes.push_back(mesh);
    }

    // 設置視圖和投影矩陣
    glm::mat4 model = glm::mat4(1.0f);
    // 可選：旋轉模型以改善視覺
    model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(1.0, 0.0, 0.0f));

    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -5.0f));

    glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                           800.0f / 600.0f, 0.1f, 100.0f);

    // 獲取 uniform 位置
    glUseProgram(shaderProgram);
    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
    unsigned int projLoc  = glGetUniformLocation(shaderProgram, "projection");

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // 設置燈光和視點位置
    unsigned int lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    unsigned int viewPosLoc  = glGetUniformLocation(shaderProgram, "viewPos");
    unsigned int objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
    unsigned int lightColorLoc  = glGetUniformLocation(shaderProgram, "lightColor");

    glUniform3f(lightPosLoc, 5.0f, 5.0f, 5.0f);
    glUniform3f(viewPosLoc, 0.0f, 0.0f, 5.0f);
    glUniform3f(objectColorLoc, 1.0f, 0.5f, 0.31f);
    glUniform3f(lightColorLoc,  1.0f, 1.0f, 1.0f);

    // 渲染循環
    while (!glfwWindowShouldClose(window))
    {
        // 處理輸入
        processInput(window);

        // 清除顏色和深度緩衝
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 更新模型矩陣（可選：旋轉動畫）
        model = glm::rotate(model, glm::radians(0.1f), glm::vec3(1.0, 1.0, 0.0f));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        // 繪製所有網格
        glUseProgram(shaderProgram);
        for (auto &mesh : meshes)
            mesh.Draw(shaderProgram);

        // 交換緩衝和輪詢事件
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 釋放資源
    for (auto &mesh : meshes)
    {
        glDeleteVertexArrays(1, &mesh.VAO);
        glDeleteBuffers(1, &mesh.VBO);
        glDeleteBuffers(1, &mesh.EBO);
    }
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

// 處理窗口尺寸變化
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// 處理輸入
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// 加載著色器函數
unsigned int loadShaders(const char* vertexPath, const char* fragmentPath)
{
    // 讀取頂點著色器源碼
    std::string vertexCode;
    std::ifstream vShaderFile(vertexPath);
    if (!vShaderFile)
    {
        std::cerr << "Failed to open " << vertexPath << "\n";
        return 0;
    }
    std::stringstream vShaderStream;
    vShaderStream << vShaderFile.rdbuf();
    vertexCode = vShaderStream.str();
    vShaderFile.close();

    // 讀取片段著色器源碼
    std::string fragmentCode;
    std::ifstream fShaderFile(fragmentPath);
    if (!fShaderFile)
    {
        std::cerr << "Failed to open " << fragmentPath << "\n";
        return 0;
    }
    std::stringstream fShaderStream;
    fShaderStream << fShaderFile.rdbuf();
    fragmentCode = fShaderStream.str();
    fShaderFile.close();

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    // 編譯頂點著色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vShaderCode, NULL);
    glCompileShader(vertexShader);

    // 檢查編譯錯誤
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << "\n";
        return 0;
    }

    // 編譯片段著色器
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
    glCompileShader(fragmentShader);

    // 檢查編譯錯誤
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << "\n";
        return 0;
    }

    // 連結著色器程序
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 檢查連結錯誤
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << "\n";
        return 0;
    }

    // 刪除著色器對象
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}