#ifndef Visualizer_H
#define Visualizer_H

#include <memory>
#include <vector>

#include "lbm.h"

typedef unsigned int uint;

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

    static bool clicked;
    static bool geoModified;
    static bool delelteGeo;

    static uint xIdxLast;
    static uint yIdxLast;

    static char postProcessingType;

    Visualizer();

public:

    static void initialize( int argc, char *argv[], 
                            uint nx, uint ny, 
                            uint pxPerVertex, lbmSolverPtr solver );

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