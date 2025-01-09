#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec2 TexCoord;

// 若有使用貼圖
uniform sampler2D ourTexture;

void main()
{
    // 簡單顯示貼圖
    // 若不需要貼圖，可改成常數顏色
    FragColor = texture(ourTexture, TexCoord);

    // 如果想要簡單光照，可再加個 lightDir, diff 等運算
    // ...
}