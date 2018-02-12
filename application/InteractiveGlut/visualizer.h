#ifndef Visualizer_H
#define Visualizer_H

#include <memory>
#include <vector>

#include "lbm.h"
#include "Utility/StopWatch.h"

typedef unsigned int uint;

typedef unsigned long long ulint;

class Visualizer
{
private:

    static uint nx;
    static uint ny;

    static uint pxPerVertex;
    
    static uint vertexBufferID;
    static uint elementBufferID;

    static std::vector<float> vertices;
    static std::vector<uint>  elements;

    static lbmSolverPtr solver;

    static char lbModel;

    static bool clicked;
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

    static void initialize( int argc, char *argv[], 
                            uint nx, uint ny, 
                            uint pxPerVertex, uint timeStepsPerFrame,
                            lbmSolverPtr solver );

    static void installShaders();

    static void generateVertices();

    static void generateElements();

    static void run();

    // Callback Functions
    static void displayCall();

    static void click(int button, int updown, int x, int y);

    static void motion(int x, int y);

    static void keyboard(unsigned char key, int x, int y);

    // get/set
    static uint getVertexBufferID();
};

#endif