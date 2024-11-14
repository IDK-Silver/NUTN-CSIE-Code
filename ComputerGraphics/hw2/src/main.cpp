#include <GLFW/glfw3.h>
#include <iostream>

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int num_of_tetrahedron = 0;
GLfloat menu_rectangles[6][4] = {{0}};
bool should_draw_menu = false;

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
    // Set the background color to white (R=1, G=1, B=1, A=1)
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    // Set the drawing color to black (R=0, G=0, B=0)
    glColor3f(0.0f, 0.0f, 0.0f);
    // Switch to the projection matrix
    glMatrixMode(GL_PROJECTION);
    // Reset the projection matrix
    glLoadIdentity();
    // Set up a 2D orthographic projection with range -2 to 2
    glOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
    // Switch back to the modelview matrix
    glMatrixMode(GL_MODELVIEW);

    glfwSetMouseButtonCallback(window, mouse_button_callback);

    return window;
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        // Handle menu selection
        if (should_draw_menu) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            
            // Convert window coordinates to OpenGL coordinates
            int width, height;
            glfwGetWindowSize(window, &width, &height);
            GLfloat x = (xpos / width) * 4.0 - 2.0;
            GLfloat y = -(ypos / height) * 4.0 + 2.0;
            
            // Check if click is within any rectangle
            bool clicked_menu = false;
            for(int i = 0; i < 6; i++) {
                if (x >= menu_rectangles[i][0] && x <= menu_rectangles[i][0] + menu_rectangles[i][2] &&
                    y >= menu_rectangles[i][1] && y <= menu_rectangles[i][1] + menu_rectangles[i][3]) {
                    if(i < 5) {
                        num_of_tetrahedron = i;
                    } else {
                        glfwSetWindowShouldClose(window, true);
                    }
                    clicked_menu = true;
                    should_draw_menu = false;
                    break;
                }
            }
            
            if (!clicked_menu) {
                should_draw_menu = false;
            }
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        // Get cursor position
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        
        // Convert window coordinates to OpenGL coordinates
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        // Convert to range [-2, 2]
        GLfloat x = (xpos / width) * 4.0 - 2.0;
        // Convert to range [-2, 2] and flip y 
        GLfloat y = -(ypos / height) * 4.0 + 2.0;  
        
        // Create menu rectangles
        for(int i = 0; i < 6; i++) {
            // x position
            menu_rectangles[i][0] = x - 0.1f;  
            // y position (vertical spacing)
            menu_rectangles[i][1] = y - 0.1f - (i * 0.2f); 
            // width 
            menu_rectangles[i][2] = 0.15f;
            // height
            menu_rectangles[i][3] = 0.15f;  
        }
        
        should_draw_menu = true;
    }
}



void draw_triangle_2d(GLfloat *p1, GLfloat *p2, GLfloat *p3) {
    glVertex3fv(p1);
    glVertex3fv(p2);
    glVertex3fv(p3);
}


void draw_triangle_3d(GLfloat *p1, GLfloat *p2, GLfloat *p3, GLfloat *p4) {
    glColor3f(1,1,1);
    draw_triangle_2d(p1,p2,p3);
    glColor3f(1,0,0);
    draw_triangle_2d(p1,p3,p4);
    glColor3f(0,1,0);
    draw_triangle_2d(p2,p3,p4);
    glColor3f(0,0,1);
    draw_triangle_2d(p1,p2,p4);
}


GLfloat* midpoint(const GLfloat a[], const GLfloat b[]) {
    GLfloat *mid = new GLfloat[3];
    for(int i = 0; i < 3; ++i) {
        mid[i] = (a[i] + b[i]) / 2.0f;
    }
    return mid;
}

void tetrahedron(GLfloat v1[], GLfloat v2[], GLfloat v3[], GLfloat v4[], int n)
{
    if(n > 0)
    {
        GLfloat *v_12 = midpoint(v1, v2);
        GLfloat *v_23 = midpoint(v2, v3);
        GLfloat *v_31 = midpoint(v3, v1);
        GLfloat *v_14 = midpoint(v1, v4);
        GLfloat *v_24 = midpoint(v2, v4);
        GLfloat *v_34 = midpoint(v3, v4);

        tetrahedron(v1, v_12, v_31, v_14, n - 1);
        tetrahedron(v_12, v2, v_23, v_24, n - 1);
        tetrahedron(v_31, v_23, v3, v_34, n - 1);
        tetrahedron(v_14, v_24, v_34, v4, n - 1);
    }
    else
        draw_triangle_3d(v1, v2, v3, v4);
}

// Add this new function
void draw_menu() {
    if (should_draw_menu) {
        const GLfloat colors[6][3] = {
            {1.0f, 0.0f, 0.0f},  // Red
            {0.0f, 1.0f, 0.0f},  // Green
            {0.0f, 0.0f, 1.0f},  // Blue
            {1.0f, 1.0f, 0.0f},  // Yellow
            {1.0f, 0.0f, 1.0f},  // Magenta
            {0.0f, 0.0f, 0.0f}   // Black
        };
        
        for(int i = 0; i < 6; i++) {
            glColor3fv(colors[i]);  // Set different color for each rectangle
            glBegin(GL_QUADS);
            glVertex2f(menu_rectangles[i][0], menu_rectangles[i][1]);  // Bottom left
            glVertex2f(menu_rectangles[i][0] + menu_rectangles[i][2], menu_rectangles[i][1]);  // Bottom right
            glVertex2f(menu_rectangles[i][0] + menu_rectangles[i][2], menu_rectangles[i][1] + menu_rectangles[i][3]);  // Top right
            glVertex2f(menu_rectangles[i][0], menu_rectangles[i][1] + menu_rectangles[i][3]);  // Top left
            glEnd();
        }
    }
}

// Modify display function
void display() {
    GLfloat start_points[4][3] = {
        {-0.65,-0.5, 0.5},
        { 0.65,-0.5, 0.5},
        { 0  , 0.6, 0.5},
        { 0  ,-0.05,-0.5},
    };
    glClear(GL_COLOR_BUFFER_BIT);
    
    glBegin(GL_TRIANGLES);
    tetrahedron(start_points[0], start_points[1], start_points[2], start_points[3], num_of_tetrahedron);
    glEnd();
    
    draw_menu();  // Call the new function here
    
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
