#version 330 core
layout(location = 0) in vec3 aPos;    // Vertex position
layout(location = 1) in vec3 aColor;  // Vertex color

out vec3 vertexColor;  // Pass to fragment shader

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vertexColor = aColor;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}