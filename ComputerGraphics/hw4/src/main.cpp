#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#include <GLUT/glut.h>
#include <iostream>
#include <chrono>
#include <glm/glm.hpp>
#include <fstream>

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
#define LIGHT_0_ENABLE 13
#define LIGHT_0_DISABLE 14
#define LIGHT_1_ENABLE 15
#define LIGHT_1_DISABLE 16
bool light0_status = 0;
bool light1_status = 1;

void createMenu();
void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glColor3f(0.0f, 0.0f, 0.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(100.0, 1.0, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

    GLfloat light_ambient[]   = {0.0f, 0.0f, 0.0f, 1.0f};
    GLfloat light_diffuse[]   = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat slight_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};

    glLightfv(GL_LIGHT0, GL_AMBIENT,   light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, slight_specular);
    glLightfv(GL_LIGHT1, GL_AMBIENT,   light_ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,   light_diffuse);
    glLightfv(GL_LIGHT1, GL_SPECULAR, slight_specular);
}


void draw_triangle_with_normal(GLfloat p1[], GLfloat p2[], GLfloat p3[]) {
    GLfloat u[3], v[3], normal[3];
    for (int i = 0; i < 3; ++i) {
        u[i] = p2[i] - p1[i];
        v[i] = p3[i] - p1[i];
    }

    normal[0] = u[1]*v[2] - u[2]*v[1];
    normal[1] = u[2]*v[0] - u[0]*v[2];
    normal[2] = u[0]*v[1] - u[1]*v[0];

    GLfloat length = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    for (int i = 0; i < 3; ++i) normal[i] /= length;

    glNormal3fv(normal);

    glBegin(GL_TRIANGLES);
    glVertex3fv(p1);
    glVertex3fv(p2);
    glVertex3fv(p3);
    glEnd();
}


GLfloat* midpoint(const GLfloat a[], const GLfloat b[]) {
    auto *mid = new GLfloat[3];
    for(int i = 0; i < 3; ++i) {
        mid[i] = (a[i] + b[i]) / 2.0f;
    }
    return mid;
}

void draw_triangle_3d(GLfloat *p1, GLfloat *p2, GLfloat *p3, GLfloat *p4) {
    glColor3f(1,0,1);
    draw_triangle_with_normal(p1,p2,p3);
    glColor3f(1,0,0);
    draw_triangle_with_normal(p1,p3,p4);
    glColor3f(0,1,0);
    draw_triangle_with_normal(p2,p3,p4);
    glColor3f(0,0,1);
    draw_triangle_with_normal(p1,p2,p4);
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
        {-0.65f, -0.5f,  0.5f},
        { 0.65f, -0.5f,  0.5f},
        { 0.0f,   0.6f,  0.5f},
        { 0.0f,  -0.05f, -0.5f},
    };

    if (light0_status) {
        glEnable(GL_LIGHT0);
    }
    else {
        glDisable(GL_LIGHT0);
    }

    if (light1_status) {
        glEnable(GL_LIGHT1);
    }
    else {
        glDisable(GL_LIGHT1);
    }

    GLfloat light0_position[] = {-0.0f, 0.f, 0.65f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
    GLfloat light1_position[] = {-0.3, -0.25f, -0.5f, 0.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);

    glutPostRedisplay();


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

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
    } else if (rotationAxis == AXIS_Z) {
        glRotatef(rotationAngle, 0.0f, 0.0f, 1.0f);
    }


    tetrahedron(start_points[0], start_points[1], start_points[2], start_points[3], num_of_tetrahedron);

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


void processLight0Menu(int choice) {
    if (choice == LIGHT_0_ENABLE) {
        light0_status = 1;
    }
    if (choice == LIGHT_0_DISABLE) {
        light0_status = 0;
    }
    createMenu();
}

void processLight1Menu(int choice) {
    if (choice == LIGHT_1_ENABLE) {
        light1_status = 1;
    }
    if (choice == LIGHT_1_DISABLE) {
        light1_status = 0;
    }
    createMenu();
}

int createL0SubMenu() {
    int light0_menu = glutCreateMenu(processLight0Menu);
    glutAddMenuEntry("On", LIGHT_0_ENABLE);
    glutAddMenuEntry("Off", LIGHT_0_DISABLE);
    return light0_menu;
}

int createL1SubMenu() {

    int light1_menu = glutCreateMenu(processLight1Menu);
    glutAddMenuEntry("On", LIGHT_1_ENABLE);
    glutAddMenuEntry("Off", LIGHT_1_DISABLE);
    return light1_menu;
}

int createLightingSubMenu() {

    int L0SubMenu = createL0SubMenu();
    int L1SubMenu = createL1SubMenu();

    int createLightingMenu = glutCreateMenu(NULL);
    glutAddSubMenu("Light - 0", L0SubMenu);
    glutAddSubMenu("Light - 1", L1SubMenu);

    return createLightingMenu;
}

void createMenu() {
    int subdivisionSubmenu = createSubdivisionLevelSubmenu();
    int rotationSubmenu = createRotationSubMenu();
    int lightingSubmenu = createLightingSubMenu();
    glutCreateMenu(processMainMenu);
    glutAddSubMenu("Subdivision Level", subdivisionSubmenu);
    glutAddSubMenu("Rotation", rotationSubmenu);
    glutAddSubMenu("Lighting", lightingSubmenu);
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
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
    const int WIDTH = 640;
    const int HEIGHT = 640;
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("S11159005 黃毓峰");
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
