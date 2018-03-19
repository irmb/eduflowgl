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

const int NX = 420*2*2;//1500;//300*5;
const int NY = 170*2*2*2;//1000;//100*5;
const int Lref=(12*NY)/30;

const float pxPerNode = 0.5;

const uint timeStepsPerFrame = 300;

const float U = 0.08f; 
const float V = 0.0f;

const float nu    = U * Lref / 400;
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

    int r = Lref/2;

    for( int x = -r; x <= r; x++ ){
        for( int y = -r; y <= r; y++ ){
            if( sqrt( x * x + y * y ) > r ) continue;
            solver->setGeo(1.9*r + x, NY/2 + y+50,1);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    Visualizer::run();

    return 0;
}

