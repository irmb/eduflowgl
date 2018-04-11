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

    GLFWwindow* gWindow;
    tdogl::Program* gProgram;
    
    uint nx;
    uint ny;
    uint lref;
    
    float pxPerVertex;
    
    uint vertexArrayID;
    uint vertexBufferID;
    uint elementBufferID;
    
    std::vector<float> vertices;
    std::vector<uint>  elements;
    
    lbmSolverPtr solver;
    
    char lbModel;
    
    bool isMouseButtonPressed;
    bool geoModified;
    bool delelteGeo;
    
    uint xIdxLast;
    uint yIdxLast;
    
    char postProcessingType;
    
    uint timeStepsPerFrame;
    
    StopWatchPtr stopWatch;
    
    double nups;
    double fps;

public:

    Visualizer( uint nx, uint ny,  
                float pxPerVertex, uint timeStepsPerFrame,
                lbmSolverPtr solver );
    
    void installShaders();
    
    void generateVertices();
    
    void generateElements();
    
    void run();

    void displayCall();

    void drawFlowField();
    
    // Callback Function Wrapper
    static void mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods);
    static void mouseMotionCallbackWrapper(GLFWwindow* window, double xpos, double ypos);
    static void keyboardCallbackWrapper   (GLFWwindow* window, int key, int scancode, int action, int mods);
    
    // Callback Functions called by the Wrappers
    void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos);
    void keyboardCallback   (GLFWwindow* window, int key, int scancode, int action, int mods);

    // get/set
    uint getVertexBufferID();
};

typedef std::shared_ptr<Visualizer> VisualizerPtr;

#endif