#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <cmath>

#include "src/LBMSolver.h"
#include "src/Visualizer.h"

//////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

//////////////////////////////////////////////////////////////////////////

const int NX = 1500;//300*5;
const int NY = 1000;//100*5;

const float pxPerNode = 1;

const uint timeStepsPerFrame = 50;

const float U = 0.025f; 
const float V = 0.0f;

const float nu    = U * NY/4 / 10000;
const float omega = 2.0f / ( 6.0f * nu + 1.0f );

//////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    lbmSolverPtr solver = std::make_shared<lbmSolver>( NX, NY, omega, U, V );

    solver->initializeDistributions();

    solver->initializeGeo();

    //////////////////////////////////////////////////////////////////////////

    Visualizer::initialize( NX, NY, pxPerNode, timeStepsPerFrame, solver );

    Visualizer::installShaders();

    Visualizer::generateVertices();

    Visualizer::generateElements();

    solver->connectVertexBuffer( Visualizer::getVertexBufferID() );

    //////////////////////////////////////////////////////////////////////////

    int r = NY/4;

    for( int x = -r; x <= r; x++ ){
        for( int y = -r; y <= r; y++ ){
            if( sqrt( x * x + y * y ) > r ) continue;
            solver->setGeo(2*r + x, 2*r + y,1);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    Visualizer::run();

    return 0;
}
