#ifndef Visualizer_H
#define Visualizer_H

#include <memory>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "LBMSolver.h"
#include "Utility/StopWatch.h"
#include "glfwFramework/Program.h"

typedef unsigned int uint;

typedef unsigned long long ulint;

class Visualizer
{
private:

    static GLFWwindow* gWindow;
    static tdogl::Program* gProgram;

    static uint nx;
    static uint ny;

    static float pxPerVertex;
    
    static uint vertexArrayID;
    static uint vertexBufferID;
    static uint elementBufferID;

    static std::vector<float> vertices;
    static std::vector<uint>  elements;

    static lbmSolverPtr solver;

    static char lbModel;

    static bool isMouseButtonPressed;
    static bool geoModified;
    static bool delelteGeo;

    static uint xIdxLast;
    static uint yIdxLast;

    static char postProcessingType;

    static uint timeStepsPerFrame;

    static StopWatchPtr stopWatch;

    static double nups;
    static double fps;

    Visualizer();

public:

    static void initialize( uint nx, uint ny, 
                            float pxPerVertex, uint timeStepsPerFrame,
                            lbmSolverPtr solver );

    static void installShaders();

    static void generateVertices();

    static void generateElements();

    static void run();

    // Callback Functions
    static void displayCall();

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    static void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos);

    static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // get/set
    static uint getVertexBufferID();
};

#endif