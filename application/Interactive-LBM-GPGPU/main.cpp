#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <cmath>
#include <cstdlib>

#include "src/tests/Test.h"
#include "src/LBMSolver.h"
#include "src/visualizer.h"

//////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

//////////////////////////////////////////////////////////////////////////

int ff;
int NX = 1000;//420*2*2 ;//1500;//300*5;
int NY = 600;//170*2*2*2 ;//1000;//100*5;
int Lref=(12*NY)/30;

float pxPerNode = 1.65;

const uint timeStepsPerFrame = 300;

const float U = 0.08f; 
const float V = 0.0f;

const float nu    = 0.1;//U * Lref / 400;
const float omega = 2.0f / ( 6.0f * nu + 1.0f );

//////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    

    if (argc > 2 && strcmp(argv[1], "test") == 0) {
        
        float threshold = atof(argv[2]);
        return Test::runTest(threshold);
    }
    // if (argc>1) pxPerNode=atof(argv[1]);
    // if (argc>=3){
    //     NX=atoi(argv[2]);
    //     NY=atoi(argv[3]);
    //     //if (argc>3) {
    //     // Lref=NY*atof(argv[3]);
    //     //}
    // }
    LBMSolverPtr solver = std::make_shared<LBMSolver>( NX, NY, omega, U, V );

    solver->initializeDistributions();

    solver->initializeGeo();

    solver->setRefLength((Lref>0 )? Lref: NY);

    //////////////////////////////////////////////////////////////////////////

    VisualizerPtr visualizer = std::make_shared<Visualizer>( NX, NY, pxPerNode, timeStepsPerFrame, solver );

    visualizer->installShaders();

    visualizer->generateVertices();

    visualizer->generateElements();

    solver->connectVertexBuffer( visualizer->getVertexBufferID() );

    //////////////////////////////////////////////////////////////////////////

 
    int r = Lref/2;

    for( int x = -r; x <= r; x++ ){
        for( int y = -r; y <= r; y++ ){
            if( sqrt( x * x + y * y ) > r ) continue;
            solver->setGeo(1.9*r + x, NY/2 + y+50,GEO_SOLID);
        }
    }
 
    //////////////////////////////////////////////////////////////////////////

    visualizer->run();

    return 0;
}


