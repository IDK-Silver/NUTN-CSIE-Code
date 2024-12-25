#include <GL/glut.h>
#include <iostream>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <GL/glut.h>
#include <cmath>

#define FPS 50
#define ONE_FPS_TIME_MS (1000.0f / FPS)

int num_of_tetrahedron = 0;  
int rotationAxis = 0;
int rotationDirection = 0;
float rotationAngle = 0.0f;
auto lastUpFrameTime = std::chrono::high_resolution_clock::now();

int prevAxis = 0;

float theta = 0.0f;    
float phi = 0.01f;     
float radius = 2.0f;   

int mouseButtonPressed = 0;
int prevMouseX = 0;
int prevMouseY = 0;



#define EXIT_MENU 5

#define AXIS_X 6
#define AXIS_Y 7

#define DIRECTION_STOP 9
#define DIRECTION_CLOCKWISE 10
#define DIRECTION_COUNTERCLOCKWISE 11
#define AXIS_Z 12
void init() {

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  
    glColor3f(0.0f, 0.0f, 0.0f);         

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(100.0, 1.0, 0.1, 100.0); 
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST); 
}

void draw_triangle_2d(GLfloat *p1, GLfloat *p2, GLfloat *p3) {
    glVertex3fv(p1);
    glVertex3fv(p2);
    glVertex3fv(p3);
}

void draw_triangle_3d(GLfloat *p1, GLfloat *p2, GLfloat *p3, GLfloat *p4) {
    glColor3f(0,0,0);
    draw_triangle_2d(p1,p2,p3);
    glColor3f(1,0,0);
    draw_triangle_2d(p1,p3,p4);
    glColor3f(0,1,0);
    draw_triangle_2d(p2,p3,p4);
    glColor3f(0,0,1);
    draw_triangle_2d(p1,p2,p4);
}


GLfloat* midpoint(const GLfloat a[], const GLfloat b[]) {
    auto *mid = new GLfloat[3];
    for(int i = 0; i < 3; ++i) {
        mid[i] = (a[i] + b[i]) / 2.0f;
    }
    return mid;
}

void tetrahedron(GLfloat v1[], GLfloat v2[], GLfloat v3[], GLfloat v4[], const int n)
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

        delete[] v_12;
        delete[] v_23;
        delete[] v_31;
        delete[] v_14;
        delete[] v_24;
        delete[] v_34;
    }
    else
        draw_triangle_3d(v1, v2, v3, v4);
}


void idle() {
    auto currentTime = std::chrono::high_resolution_clock::now();
    if (currentTime - lastUpFrameTime > std::chrono::milliseconds(int(ONE_FPS_TIME_MS))) {
        lastUpFrameTime = currentTime;
        if (rotationDirection != 0) {
            rotationAngle += rotationDirection * 0.5f; 
            if (rotationAngle > 360.0f) rotationAngle -= 360.0f;
            if (rotationAngle < -360.0f) rotationAngle += 360.0f;
            glutPostRedisplay(); 
        }
    }
}

void display() {
    GLfloat start_points[4][3] = {
            {-0.65,-0.5, 0.5},
            { 0.65,-0.5, 0.5},
            { 0  , 0.6, 0.5},
            { 0  ,-0.05,-0.5},
    };
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system 

    float cX = radius * sinf(phi) * cosf(theta);
    float cY = radius * cosf(phi);
    float cZ = radius * sinf(phi) * sinf(theta);

    gluLookAt(cX, cY, cZ,    
              0.0f, 0.0f, 0.0f,   
              0.0f, 1.f, 0.0f);  

    if (rotationAxis == AXIS_X) {
        glRotatef(rotationAngle, 1.0f, 0.0f, 0.0f); 
    } else if (rotationAxis == AXIS_Y) {
        glRotatef(rotationAngle, 0.0f, 1.0f, 0.0f); 
    }
    else if (rotationAxis == AXIS_Z) {
        glRotatef(rotationAngle, 0.0f, 0.0f, 1.0f); 
    }

    glBegin(GL_TRIANGLES);
    tetrahedron(start_points[0], start_points[1], start_points[2], start_points[3], num_of_tetrahedron);
    glEnd();

    glFlush();
}

void processKeyboardPress(unsigned char key, int x, int y) {
    switch (key)
    {
      case 'q':
      case 'Q':
        exit(0);
      break;
    }

}


void processSubdivisionLevelMenu(int choice) {
    if (choice >= 0 && choice <= 4) {
        num_of_tetrahedron = choice;
        glutPostRedisplay(); 
    }
}

void processAxisMenu(int choice) {
    rotationAxis = choice;
}

void processDirectionMenu(int choice) {
    if (choice == DIRECTION_STOP) {
        rotationDirection = 0;
    } else if (choice == DIRECTION_CLOCKWISE) {
        rotationDirection = -1; 
    } else if (choice == DIRECTION_COUNTERCLOCKWISE) {
        rotationDirection = 1; 
    }
}

int createSubdivisionLevelSubmenu() {
    int subdivisionSubmenu = glutCreateMenu(processSubdivisionLevelMenu);
    glutAddMenuEntry("Level 0", 0);
    glutAddMenuEntry("Level 1", 1);
    glutAddMenuEntry("Level 2", 2);
    glutAddMenuEntry("Level 3", 3);
    glutAddMenuEntry("Level 4", 4);
    return subdivisionSubmenu;
}

int createAxisSubMenu() {
    int axisSubMenu = glutCreateMenu(processAxisMenu);
    glutAddMenuEntry("X", AXIS_X);
    glutAddMenuEntry("Y", AXIS_Y);
    glutAddMenuEntry("Z", AXIS_Z);
    return axisSubMenu;
}

int createDirectionSubMenu() {
    int directionSubMenu = glutCreateMenu(processDirectionMenu);
    glutAddMenuEntry("Stop", DIRECTION_STOP);
    glutAddMenuEntry("Clockwise", DIRECTION_CLOCKWISE);
    glutAddMenuEntry("Counter-Clockwise", DIRECTION_COUNTERCLOCKWISE);
    return directionSubMenu;
}

int createRotationSubMenu() {
        glutDetachMenu(GLUT_RIGHT_BUTTON);
    int axisSubMenu = createAxisSubMenu();
    int directionSubMenu = createDirectionSubMenu();

    int rotationSubMenu = glutCreateMenu(NULL);
    glutAddSubMenu("Axis", axisSubMenu);
    glutAddSubMenu("Direction", directionSubMenu);

    return rotationSubMenu;
}

void processMainMenu(int choice) {
    if (choice == EXIT_MENU) {
        exit(0);  
    }
}

void createMenu() {
    int subdivisionSubmenu = createSubdivisionLevelSubmenu();
    int rotationSubmenu = createRotationSubMenu();
    glutCreateMenu(processMainMenu);
    glutAddSubMenu("Subdivision Level", subdivisionSubmenu);
    glutAddSubMenu("Rotation", rotationSubmenu);
    glutAddMenuEntry("Exit", EXIT_MENU);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

}



void mouseButton(int button, int state, int x, int y) {
    prevMouseX = x;
    if(state == GLUT_DOWN) {
        // Set the bit corresponding to the pressed button
        mouseButtonPressed |= 1 << button;
        prevMouseY = y;
        glutDetachMenu(GLUT_RIGHT_BUTTON);
    } else if(state == GLUT_UP) {
        // Clear the bit corresponding to the released button
        mouseButtonPressed &= ~(1 << button);
        glutAttachMenu(GLUT_RIGHT_BUTTON);
    }
}


void mouseMotion(int x, int y) {

    int dx = x - prevMouseX;
    int dy = y - prevMouseY;

    if((mouseButtonPressed & (1 << GLUT_LEFT_BUTTON)) != 0 &&
       (mouseButtonPressed & (1 << GLUT_RIGHT_BUTTON)) == 0) {
        theta += dx * 0.005f; 
        phi   -= dy * 0.005f;


        if(phi <= 0.01f) phi = 0.01f;
        if(phi >= 3.13f) phi = 3.13f;
       }

    if((mouseButtonPressed & (1 << GLUT_LEFT_BUTTON)) != 0 &&
       (mouseButtonPressed & (1 << GLUT_RIGHT_BUTTON)) != 0) {
        radius += dy * 0.01f;

        if(radius < 1.0f) radius = 1.0f;
        if(radius > 10.0f) radius = 10.0f;
       }

    prevMouseX = x;
    prevMouseY = y;

    glutPostRedisplay(); 
}
int main(int argc, char** argv) {
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);

    glutInit(&argc, argv); 
    // glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB); 
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH); 
    const int WIDTH = 640;
    const int HEIGHT = 640;
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("S11159005 Yu-feng");
    init();
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    createMenu();
    glutKeyboardFunc(&processKeyboardPress);
    glutMainLoop();
    return 0;
}

