#version 330 core
layout (location = 0) in vec3 aPos;      // 頂點位置
layout (location = 1) in vec3 aNormal;   // 法線
layout (location = 2) in vec2 aTexCoord; // 貼圖座標

out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
}