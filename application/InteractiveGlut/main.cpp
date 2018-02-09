#include <vector>
#include <chrono>
#include <iostream>
#include <memory>

#include "lbm.h"
#include "Visualizer.h"

//////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

//////////////////////////////////////////////////////////////////////////

const uint NX = 1600;
const uint NY = 1000;

const uint pxPerNode = 1;

const uint timeStepsPerFrame = 50;

const float U = 0.025f;
const float V = 0.000f;

const float omega = 1.99f;

//////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    lbmSolverPtr solver = std::make_shared<lbmSolver>( NX, NY, omega, U, V );

    solver->initializeDistributions();

    //////////////////////////////////////////////////////////////////////////

    Visualizer::initialize( argc, argv, NX, NY, pxPerNode, timeStepsPerFrame, solver );

    Visualizer::installShaders();

    Visualizer::generateVertices();

    Visualizer::generateElements();

    solver->connectVertexBuffer( Visualizer::getVertexBufferID() );

    //////////////////////////////////////////////////////////////////////////

    for( int x = -250; x <= 250; x++ ){
        for( int y = -250; y <= 250; y++ ){
            if( sqrt( x * x + y * y ) > 250 ) continue;
            solver->setGeo(NY/2 + x, NY/2 + y,1);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    Visualizer::run();

    return 0;
}


