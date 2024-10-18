#include <GLFW/glfw3.h>
#include <iostream>

// Set the clear color to black
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

GLFWwindow* createWindow(int width, int height, const char* title) {
    // Set window hints to prevent resizing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    // Set the current OpenGL context to this window
    // This tells OpenGL which window to draw on
    glfwMakeContextCurrent(window);

    // Set up a callback function for window resize events
    // This ensures the viewport is adjusted when the window size changes
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize OpenGL settings
    // Set the background color to black (R=0, G=0, B=0, A=1)
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    // Set the drawing color to white (R=1, G=1, B=1)
    glColor3f(1.0f, 1.0f, 1.0f);
    // Switch to the projection matrix
    glMatrixMode(GL_PROJECTION);
    // Reset the projection matrix
    glLoadIdentity();
    // Set up a 2D orthographic projection
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    return window;
}

// Draw a white square
void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
        glVertex2f(-0.5, -0.5);
        glVertex2f(0.5, -0.5);
        glVertex2f(0.5, 0.5);
        glVertex2f(-0.5, 0.5);
    glEnd();
    glFlush();
}

int main(int argc, char** argv) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    const int WIDTH = 500;
    const int HEIGHT = 500;
    const char* TITLE = "S11159005 黃毓峰";

    auto window = createWindow(WIDTH, HEIGHT, TITLE);
    if (!window) {
        return -1;
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
